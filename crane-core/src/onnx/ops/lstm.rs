// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `LSTM` as a native eval op, with `forward` and
//! `bidirectional` direction support.

use candle_core::{DType, Device, Result, Tensor, bail};
use candle_nn::ops::sigmoid;

use crate::onnx::proto::{self, NodeProto};

/// Inputs to an ONNX `LSTM` node, bundled to keep [`lstm`]'s argument count
/// reasonable. Optional fields are `None` when the corresponding ONNX input
/// is absent (an empty name in the node's input list).
#[derive(Clone, Copy)]
pub(crate) struct LstmInputs<'a> {
    /// `X`: `[seq_length, batch_size, input_size]`.
    pub input: &'a Tensor,
    /// `W`: `[num_directions, 4*hidden_size, input_size]`, `iofc` gate order.
    pub weight: &'a Tensor,
    /// `R`: `[num_directions, 4*hidden_size, hidden_size]`, `iofc` gate order.
    pub recurrence_weight: &'a Tensor,
    /// `B`: `[num_directions, 8*hidden_size]`; defaults to all-zero.
    pub bias: Option<&'a Tensor>,
    /// `sequence_lens`: `[batch_size]`; only the default (`seq_length` for
    /// every sequence) is supported.
    pub seq_lens: Option<&'a Tensor>,
    /// `initial_h`: `[num_directions, batch_size, hidden_size]`; defaults to
    /// all-zero.
    pub initial_h: Option<&'a Tensor>,
    /// `initial_c`: `[num_directions, batch_size, hidden_size]`; defaults to
    /// all-zero.
    pub initial_c: Option<&'a Tensor>,
    /// `P` (peephole weights): `[num_directions, 3*hidden_size]`; only the
    /// default (absent / all-zero) is supported.
    pub peephole: Option<&'a Tensor>,
}

/// Outputs produced by an ONNX `LSTM` node. Each field is `None` when the
/// corresponding ONNX output slot is omitted (empty name) on the node.
#[derive(Debug)]
pub(crate) struct LstmOutputs {
    /// The full sequence of hidden states, shape `[seq_length,
    /// num_directions, batch_size, hidden_size]`.
    pub y: Option<Tensor>,
    /// The final hidden state, shape `[num_directions, batch_size,
    /// hidden_size]`.
    pub y_h: Option<Tensor>,
    /// The final cell state, shape `[num_directions, batch_size,
    /// hidden_size]`.
    pub y_c: Option<Tensor>,
}

/// Validated `LSTM` attributes relevant to shape computation.
struct LstmAttrs {
    num_directions: usize,
    hidden_size: i64,
}

/// Per-direction weights and initial state, already sliced out of the
/// `[num_directions, ...]` inputs and reordered from ONNX's `iofc` gate
/// order to the `ifco` order used by [`run_direction`].
struct DirectionParams {
    /// `[4*hidden_size, input_size]`, `ifco` order.
    weight: Tensor,
    /// `[4*hidden_size, hidden_size]`, `ifco` order.
    recurrence_weight: Tensor,
    /// `[4*hidden_size]`, `ifco` order.
    input_bias: Tensor,
    /// `[4*hidden_size]`, `ifco` order.
    recurrent_bias: Tensor,
    /// `[batch_size, hidden_size]`.
    initial_h: Tensor,
    /// `[batch_size, hidden_size]`.
    initial_c: Tensor,
}

/// A direction's hidden and cell state.
struct DirectionState {
    h: Tensor,
    c: Tensor,
}

/// ONNX `LSTM`: a single-layer LSTM over a `[seq_length, batch_size,
/// input_size]` input, supporting `direction` values `"forward"` and
/// `"bidirectional"`.
///
/// `weight`/`recurrence_weight`/`bias`/`initial_h`/`initial_c`/`peephole`
/// carry a leading `num_directions` dimension per the ONNX spec (1 for
/// `"forward"`, 2 for `"bidirectional"`). `bias`, `initial_h`, and
/// `initial_c` default to all-zero tensors when omitted. `seq_lens`
/// (variable-length sequences) and `peephole` are only supported when
/// absent or equal to their default (`seq_length` and all-zero,
/// respectively).
///
/// For `"bidirectional"`, the backward direction runs the same forward
/// recurrence over the time-reversed input; its `Y` output is un-reversed
/// before concatenation. Its final `Y_h`/`Y_c` need no such correction --
/// the last state of a forward pass over the reversed sequence already is
/// the ONNX-defined backward-direction final state.
///
/// # Errors
///
/// Returns an error if:
/// - `direction` is anything other than `"forward"` or `"bidirectional"`.
/// - `hidden_size` is missing or non-positive.
/// - `input_forget` is non-zero.
/// - `activations` is present and not the default `(Sigmoid, Tanh, Tanh)`.
/// - the `clip` attribute is present.
/// - `layout` is non-zero.
/// - `seq_lens` is present and not the default (`seq_length` for every
///   sequence).
/// - `peephole` (`P`) is present and non-zero.
pub(crate) fn lstm(node: &NodeProto, inputs: LstmInputs) -> Result<LstmOutputs> {
    let LstmInputs {
        input,
        weight,
        recurrence_weight,
        bias,
        seq_lens,
        initial_h,
        initial_c,
        peephole,
    } = inputs;

    let attrs = validate_attributes(node)?;
    let (seq_length, batch_size, _input_size) = input.dims3()?;
    validate_unsupported_inputs(node, seq_lens, peephole, seq_length)?;

    // hidden_size is a small architecture dimension (RNN width), always
    // well within usize::MAX and non-negative for any real ONNX model.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let hidden_size = attrs.hidden_size as usize;
    let (bias, initial_h, initial_c) = resolve_optional_state(
        bias,
        initial_h,
        initial_c,
        attrs.num_directions,
        batch_size,
        hidden_size,
        input.device(),
    )?;

    let output_y = node.output.first().is_some_and(|name| !name.is_empty());
    let output_y_h = node.output.get(1).is_some_and(|name| !name.is_empty());
    let output_y_c = node.output.get(2).is_some_and(|name| !name.is_empty());

    let forward_params = extract_direction_params(
        weight,
        recurrence_weight,
        &bias,
        &initial_h,
        &initial_c,
        0,
        attrs.hidden_size,
    )?;
    let (y_forward, state_forward) = run_direction(input, &forward_params, output_y)?;

    let (y, y_h, y_c) = if attrs.num_directions == 2 {
        let backward_params = extract_direction_params(
            weight,
            recurrence_weight,
            &bias,
            &initial_h,
            &initial_c,
            1,
            attrs.hidden_size,
        )?;
        let input_reversed = input.flip(&[0])?;
        let (y_backward_raw, state_backward) =
            run_direction(&input_reversed, &backward_params, output_y)?;
        combine_bidirectional(y_forward, y_backward_raw, state_forward, state_backward)?
    } else {
        single_direction_outputs(y_forward, &state_forward)?
    };

    Ok(LstmOutputs {
        y: if output_y { y } else { None },
        y_h: if output_y_h { Some(y_h) } else { None },
        y_c: if output_y_c { Some(y_c) } else { None },
    })
}

/// Parses and validates the `LSTM` attributes that affect shape
/// computation, bailing on any attribute value the evaluator doesn't
/// implement (non-default `activations`, non-zero `input_forget`, `clip`,
/// non-zero `layout`).
fn validate_attributes(node: &NodeProto) -> Result<LstmAttrs> {
    let direction = string_attribute(node, "direction")?.unwrap_or("forward");
    if direction != "forward" && direction != "bidirectional" {
        bail!(
            "LSTM node '{}' currently only supports direction \"forward\" or \"bidirectional\", got {direction:?}",
            node.name
        );
    }
    let num_directions = if direction == "bidirectional" { 2 } else { 1 };

    let hidden_size = required_int_attribute(node, "hidden_size")?;
    if hidden_size <= 0 {
        bail!(
            "LSTM node '{}' has non-positive hidden_size {hidden_size}",
            node.name
        );
    }
    let input_forget = int_attribute(node, "input_forget", 0)?;
    if input_forget != 0 {
        bail!(
            "LSTM node '{}' currently only supports input_forget == 0",
            node.name
        );
    }

    let mut expected_activations = vec![
        "Sigmoid".to_string(),
        "Tanh".to_string(),
        "Tanh".to_string(),
    ];
    if num_directions == 2 {
        let per_direction = expected_activations.clone();
        expected_activations.extend(per_direction);
    }
    if let Some(activations) = strings_attribute(node, "activations")?
        && activations != expected_activations
    {
        bail!(
            "LSTM node '{}' currently only supports default activations ({expected_activations:?})",
            node.name
        );
    }
    // activation_alpha and activation_beta don't apply to (Sigmoid, Tanh, Tanh) so ignoring them is okay.
    if node
        .attribute
        .iter()
        .any(|attribute| attribute.name == "clip")
    {
        bail!(
            "LSTM node '{}' does not currently support the clip attribute",
            node.name
        );
    }
    let layout = int_attribute(node, "layout", 0)?;
    if layout != 0 {
        bail!(
            "LSTM node '{}' currently only supports layout == 0",
            node.name
        );
    }

    Ok(LstmAttrs {
        num_directions,
        hidden_size,
    })
}

/// Bails if `seq_lens` or `peephole` are present and non-default; both
/// remain unimplemented (variable-length sequences and peephole
/// connections, respectively).
fn validate_unsupported_inputs(
    node: &NodeProto,
    seq_lens: Option<&Tensor>,
    peephole: Option<&Tensor>,
    seq_length: usize,
) -> Result<()> {
    if let Some(seq_lens) = seq_lens {
        // seq_length comes from the input tensor's own shape, always far
        // below i64::MAX.
        #[allow(clippy::cast_possible_wrap)]
        let seq_length_i64 = seq_length as i64;
        let seq_lens_is_default = seq_lens
            .to_vec1::<i64>()?
            .iter()
            .all(|&length| length == seq_length_i64);
        if !seq_lens_is_default {
            bail!(
                "LSTM node '{}' currently only supports the default value of seq_lens",
                node.name
            );
        }
    }
    if let Some(peephole) = peephole {
        let peephole_is_zero = peephole
            .to_vec2::<f32>()?
            .iter()
            .all(|row| row.iter().all(|&value| value == 0.0));
        if !peephole_is_zero {
            bail!(
                "LSTM node '{}' currently only supports a zero (absent) peephole weight (P)",
                node.name
            );
        }
    }
    Ok(())
}

/// Materializes `bias`/`initial_h`/`initial_c` as owned tensors, filling in
/// all-zero defaults for any that are absent.
fn resolve_optional_state(
    bias: Option<&Tensor>,
    initial_h: Option<&Tensor>,
    initial_c: Option<&Tensor>,
    num_directions: usize,
    batch_size: usize,
    hidden_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let bias = if let Some(bias) = bias {
        bias.clone()
    } else {
        Tensor::zeros((num_directions, 8 * hidden_size), DType::F32, device)?
    };
    let initial_h = if let Some(initial_h) = initial_h {
        initial_h.clone()
    } else {
        Tensor::zeros(
            (num_directions, batch_size, hidden_size),
            DType::F32,
            device,
        )?
    };
    let initial_c = if let Some(initial_c) = initial_c {
        initial_c.clone()
    } else {
        Tensor::zeros(
            (num_directions, batch_size, hidden_size),
            DType::F32,
            device,
        )?
    };
    Ok((bias, initial_h, initial_c))
}

/// Slices direction `direction_index` out of `weight`/`recurrence_weight`/
/// `bias`/`initial_h`/`initial_c` (each shaped `[num_directions, ...]`),
/// splits `bias` into its input/recurrent halves, and reorders
/// `weight`/`recurrence_weight`/the bias halves from ONNX's `iofc` gate
/// order to the `ifco` order [`run_direction`] expects.
fn extract_direction_params(
    weight: &Tensor,
    recurrence_weight: &Tensor,
    bias: &Tensor,
    initial_h: &Tensor,
    initial_c: &Tensor,
    direction_index: usize,
    hidden_size: i64,
) -> Result<DirectionParams> {
    let weight_dir = weight.get(direction_index)?; // [4*hidden_size, input_size], iofc order
    let recurrence_dir = recurrence_weight.get(direction_index)?; // [4*hidden_size, hidden_size], iofc order
    let bias_dir = bias.get(direction_index)?; // [8*hidden_size] == concat(wb[iofc], rb[iofc])
    let initial_h = initial_h.get(direction_index)?;
    let initial_c = initial_c.get(direction_index)?;

    // hidden_size is a small architecture dimension (RNN width), always
    // well within usize::MAX and non-negative for any real ONNX model.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let hidden_size = hidden_size as usize;
    let input_bias = bias_dir.narrow(0, 0, 4 * hidden_size)?;
    let recurrent_bias = bias_dir.narrow(0, 4 * hidden_size, 4 * hidden_size)?;

    Ok(DirectionParams {
        weight: reorder_iofc_to_ifco(&weight_dir, hidden_size)?,
        recurrence_weight: reorder_iofc_to_ifco(&recurrence_dir, hidden_size)?,
        input_bias: reorder_iofc_to_ifco(&input_bias, hidden_size)?,
        recurrent_bias: reorder_iofc_to_ifco(&recurrent_bias, hidden_size)?,
        initial_h,
        initial_c,
    })
}

/// Reorders `tensor`'s leading (gate) dimension from ONNX's `iofc` (input,
/// output, forget, cell) order to the `ifco` (input, forget, cell, output)
/// order used by [`run_direction`]'s gate math. Each gate occupies a
/// contiguous run of `hidden_size` rows, so this is a slice-and-concatenate
/// rather than a general permutation.
fn reorder_iofc_to_ifco(tensor: &Tensor, hidden_size: usize) -> Result<Tensor> {
    let input_gate = tensor.narrow(0, 0, hidden_size)?;
    let output_gate = tensor.narrow(0, hidden_size, hidden_size)?;
    let forget_gate = tensor.narrow(0, 2 * hidden_size, hidden_size)?;
    let cell_gate = tensor.narrow(0, 3 * hidden_size, hidden_size)?;
    Tensor::cat(&[&input_gate, &forget_gate, &cell_gate, &output_gate], 0)
}

/// Runs the standard LSTM recurrence over `input` (`[seq_length,
/// batch_size, input_size]`) for a single direction, returning the
/// optional stacked hidden-state sequence (`[seq_length, batch_size,
/// hidden_size]`, present only when `collect_y` is set) along with the
/// final [`DirectionState`].
fn run_direction(
    input: &Tensor,
    params: &DirectionParams,
    collect_y: bool,
) -> Result<(Option<Tensor>, DirectionState)> {
    let seq_length = input.dim(0)?;
    let mut state = DirectionState {
        h: params.initial_h.clone(),
        c: params.initial_c.clone(),
    };
    let mut hidden_states: Option<Vec<Tensor>> = if collect_y {
        Some(Vec::with_capacity(seq_length))
    } else {
        None
    };

    // weight/recurrence_weight are the same for every timestep, so their
    // transpose is hoisted out of the loop below.
    let weight_t = params.weight.t()?;
    let recurrence_weight_t = params.recurrence_weight.t()?;
    for t in 0..seq_length {
        let step_input = input.get(t)?;
        let gates_from_input = step_input
            .matmul(&weight_t)?
            .broadcast_add(&params.input_bias)?;
        let gates_from_hidden = state
            .h
            .matmul(&recurrence_weight_t)?
            .broadcast_add(&params.recurrent_bias)?;
        let gates = (&gates_from_input + &gates_from_hidden)?;
        let chunks = gates.chunk(4, 1)?;
        let input_gate = sigmoid(&chunks[0])?;
        let forget_gate = sigmoid(&chunks[1])?;
        let cell_candidate = chunks[2].tanh()?;
        let output_gate = sigmoid(&chunks[3])?;

        let next_c = ((&forget_gate * &state.c)? + (&input_gate * &cell_candidate)?)?;
        let next_h = (&output_gate * &next_c.tanh()?)?;
        state = DirectionState {
            h: next_h,
            c: next_c,
        };

        if let Some(states) = &mut hidden_states {
            states.push(state.h.clone());
        }
    }

    // Tensor::stack errors on an empty slice; an empty sequence has no
    // timesteps to collect, so Y is simply absent.
    let y = match hidden_states {
        Some(states) if !states.is_empty() => Some(Tensor::stack(&states, 0)?),
        _ => None,
    };
    Ok((y, state))
}

/// Assembles the `Y`/`Y_h`/`Y_c` outputs for a `"bidirectional"` node from
/// each direction's raw results: `y_backward_raw` (computed over the
/// time-reversed input) is un-reversed before stacking with `y_forward`
/// along the direction axis; final states need no un-reversal (see
/// [`lstm`]'s doc comment).
fn combine_bidirectional(
    y_forward: Option<Tensor>,
    y_backward_raw: Option<Tensor>,
    state_forward: DirectionState,
    state_backward: DirectionState,
) -> Result<(Option<Tensor>, Tensor, Tensor)> {
    let y = match (y_forward, y_backward_raw) {
        (Some(y_forward), Some(y_backward_raw)) => {
            let y_backward = y_backward_raw.flip(&[0])?;
            Some(Tensor::stack(&[y_forward, y_backward], 1)?)
        },
        _ => None,
    };
    let y_h = Tensor::stack(&[state_forward.h, state_backward.h], 0)?;
    let y_c = Tensor::stack(&[state_forward.c, state_backward.c], 0)?;
    Ok((y, y_h, y_c))
}

/// Assembles the `Y`/`Y_h`/`Y_c` outputs for a `"forward"`-only node by
/// inserting a size-1 direction axis.
fn single_direction_outputs(
    y: Option<Tensor>,
    state: &DirectionState,
) -> Result<(Option<Tensor>, Tensor, Tensor)> {
    let y = y.map(|y| y.unsqueeze(1)).transpose()?;
    let y_h = state.h.unsqueeze(0)?;
    let y_c = state.c.unsqueeze(0)?;
    Ok((y, y_h, y_c))
}

fn string_attribute<'a>(node: &'a NodeProto, name: &str) -> Result<Option<&'a str>> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(None);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::String {
        bail!(
            "LSTM node '{}' has a non-STRING '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    let value = std::str::from_utf8(&attribute.s).map_err(candle_core::Error::wrap)?;
    Ok(Some(value))
}

fn strings_attribute(node: &NodeProto, name: &str) -> Result<Option<Vec<String>>> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(None);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Strings {
        bail!(
            "LSTM node '{}' has a non-STRINGS '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    let mut values = Vec::with_capacity(attribute.strings.len());
    for bytes in &attribute.strings {
        values.push(String::from_utf8(bytes.clone()).map_err(candle_core::Error::wrap)?);
    }
    Ok(Some(values))
}

/// Looks up an `INT` attribute by name, returning `Ok(None)` if it's absent.
fn find_int_attribute(node: &NodeProto, name: &str) -> Result<Option<i64>> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(None);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Int {
        bail!(
            "LSTM node '{}' has a non-INT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(Some(attribute.i))
}

fn int_attribute(node: &NodeProto, name: &str, default: i64) -> Result<i64> {
    Ok(find_int_attribute(node, name)?.unwrap_or(default))
}

fn required_int_attribute(node: &NodeProto, name: &str) -> Result<i64> {
    match find_int_attribute(node, name)? {
        Some(value) => Ok(value),
        None => bail!(
            "LSTM node '{}' is missing the required '{}' attribute",
            node.name,
            name
        ),
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::{LstmInputs, lstm};

    fn string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::String as i32,
            s: value.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    fn int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::Int as i32,
            i: value,
            ..Default::default()
        }
    }

    fn strings_attr(name: &str, values: &[&str]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::Strings as i32,
            strings: values.iter().map(|value| value.as_bytes().to_vec()).collect(),
            ..Default::default()
        }
    }

    fn lstm_node(direction: &str, hidden_size: i64) -> NodeProto {
        NodeProto {
            name: "LSTM.0".to_string(),
            attribute: vec![
                string_attr("direction", direction),
                int_attr("hidden_size", hidden_size),
            ],
            output: vec!["y".to_string(), "y_h".to_string(), "y_c".to_string()],
            ..Default::default()
        }
    }

    fn minimal_inputs<'a>(
        input: &'a Tensor,
        weight: &'a Tensor,
        recurrence_weight: &'a Tensor,
    ) -> LstmInputs<'a> {
        LstmInputs {
            input,
            weight,
            recurrence_weight,
            bias: None,
            seq_lens: None,
            initial_h: None,
            initial_c: None,
            peephole: None,
        }
    }

    fn sigmoid_f32(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn forward_single_step_matches_hand_computed_gates() -> Result<()> {
        // A single-timestep, single-batch forward LSTM: verifies both the
        // iofc-to-ifco gate reordering and the gate math itself by comparing
        // against a hand-computed reference (zero bias/recurrence/initial
        // state keeps the arithmetic tractable).
        let hidden_size = 2i64;
        // iofc order: rows 0-1 = input gate, 2-3 = output gate, 4-5 = forget
        // gate, 6-7 = cell candidate; input_size = 1.
        let weight = Tensor::new(
            &[[1.0f32], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]],
            &Device::Cpu,
        )?
        .reshape((1, 8, 1))?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::new(&[[[2.0f32]]], &Device::Cpu)?; // [seq_length=1, batch=1, input_size=1]
        let node = lstm_node("forward", hidden_size);

        let outputs = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))?;

        let y = outputs.y.expect("Y requested");
        let y_h = outputs.y_h.expect("Y_h requested");
        let y_c = outputs.y_c.expect("Y_c requested");
        assert_eq!(y.dims(), &[1, 1, 1, 2]);
        assert_eq!(y_h.dims(), &[1, 1, 2]);
        assert_eq!(y_c.dims(), &[1, 1, 2]);

        let input_gate = [sigmoid_f32(2.0), sigmoid_f32(4.0)];
        let forget_gate = [sigmoid_f32(10.0), sigmoid_f32(12.0)];
        let cell_candidate = [14.0f32.tanh(), 16.0f32.tanh()];
        let output_gate = [sigmoid_f32(6.0), sigmoid_f32(8.0)];
        // next_c = forget_gate * c_init + input_gate * cell_candidate;
        // c_init is 0, so forget_gate doesn't affect the result, but is
        // included for documentation.
        let expected_c: Vec<f32> = input_gate
            .iter()
            .zip(cell_candidate.iter())
            .zip(forget_gate.iter())
            .map(|((i, c), f)| f * 0.0 + i * c)
            .collect();
        let expected_h: Vec<f32> = output_gate
            .iter()
            .zip(expected_c.iter())
            .map(|(o, c)| o * c.tanh())
            .collect();

        let got_h = y_h.flatten_all()?.to_vec1::<f32>()?;
        let got_c = y_c.flatten_all()?.to_vec1::<f32>()?;
        for (got, expected) in got_h.iter().zip(expected_h.iter()) {
            assert!((got - expected).abs() < 1e-5, "h: {got} vs {expected}");
        }
        for (got, expected) in got_c.iter().zip(expected_c.iter()) {
            assert!((got - expected).abs() < 1e-5, "c: {got} vs {expected}");
        }
        Ok(())
    }

    #[test]
    fn forward_multi_step_y_h_matches_last_y_timestep() -> Result<()> {
        // Y_h must equal the last timestep of Y for a forward-only LSTM.
        let hidden_size = 3i64;
        let seq_length = 4usize;
        let batch_size = 2usize;
        let input_size = 5usize;
        let weight = Tensor::randn(
            0f32,
            1f32,
            (1, 4 * hidden_size as usize, input_size),
            &Device::Cpu,
        )?;
        let recurrence_weight = Tensor::randn(
            0f32,
            1f32,
            (1, 4 * hidden_size as usize, hidden_size as usize),
            &Device::Cpu,
        )?;
        let input = Tensor::randn(
            0f32,
            1f32,
            (seq_length, batch_size, input_size),
            &Device::Cpu,
        )?;
        let node = lstm_node("forward", hidden_size);

        let outputs = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))?;

        let y = outputs.y.expect("Y requested");
        let y_h = outputs.y_h.expect("Y_h requested");
        assert_eq!(y.dims(), &[seq_length, 1, batch_size, hidden_size as usize]);
        assert_eq!(y_h.dims(), &[1, batch_size, hidden_size as usize]);

        let last_y = y
            .get(seq_length - 1)?
            .get(0)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let y_h_flat = y_h.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(last_y, y_h_flat);
        Ok(())
    }

    #[test]
    fn bidirectional_matches_two_independent_forward_passes() -> Result<()> {
        // A bidirectional LSTM must equal two independent forward passes:
        // direction 0 over x as-is, direction 1 over x reversed (with its Y
        // output un-reversed back to original time order).
        let hidden_size = 2i64;
        let weight_forward = [0.5f32, -0.3, 0.2, 0.4, 0.1, -0.6, 0.7, -0.2];
        let weight_backward = [-0.4f32, 0.6, -0.1, 0.3, -0.5, 0.2, 0.15, -0.7];
        let recurrence_forward = [
            0.1f32, -0.2, 0.05, 0.1, -0.1, 0.2, 0.05, -0.05, 0.15, -0.1, 0.2, -0.15, 0.1, 0.05,
            -0.2, 0.1,
        ];
        let recurrence_backward = [
            -0.2f32, 0.1, 0.15, -0.1, 0.05, -0.15, 0.2, 0.1, -0.1, 0.2, -0.05, 0.15, -0.15, 0.1,
            0.05, -0.2,
        ];
        let input_vals = [1.0f32, -2.0, 0.5];

        let input = Tensor::from_vec(input_vals.to_vec(), (3, 1, 1), &Device::Cpu)?;
        let weight_forward_t = Tensor::from_vec(weight_forward.to_vec(), (1, 8, 1), &Device::Cpu)?;
        let recurrence_forward_t =
            Tensor::from_vec(recurrence_forward.to_vec(), (1, 8, 2), &Device::Cpu)?;
        let weight_backward_t =
            Tensor::from_vec(weight_backward.to_vec(), (1, 8, 1), &Device::Cpu)?;
        let recurrence_backward_t =
            Tensor::from_vec(recurrence_backward.to_vec(), (1, 8, 2), &Device::Cpu)?;

        let forward_node = lstm_node("forward", hidden_size);
        let y_forward_ref = lstm(
            &forward_node,
            minimal_inputs(&input, &weight_forward_t, &recurrence_forward_t),
        )?
        .y
        .expect("Y requested")
        .flatten_all()?
        .to_vec1::<f32>()?;

        let input_reversed = input.flip(&[0])?;
        let backward_node = lstm_node("forward", hidden_size);
        let y_backward_raw = lstm(
            &backward_node,
            minimal_inputs(&input_reversed, &weight_backward_t, &recurrence_backward_t),
        )?
        .y
        .expect("Y requested");
        let y_backward_expected = y_backward_raw.flip(&[0])?.flatten_all()?.to_vec1::<f32>()?;

        let mut weight_both = weight_forward.to_vec();
        weight_both.extend_from_slice(&weight_backward);
        let mut recurrence_both = recurrence_forward.to_vec();
        recurrence_both.extend_from_slice(&recurrence_backward);
        let weight_bidi = Tensor::from_vec(weight_both, (2, 8, 1), &Device::Cpu)?;
        let recurrence_bidi = Tensor::from_vec(recurrence_both, (2, 8, 2), &Device::Cpu)?;
        let bidi_node = lstm_node("bidirectional", hidden_size);

        let outputs = lstm(
            &bidi_node,
            minimal_inputs(&input, &weight_bidi, &recurrence_bidi),
        )?;
        let y = outputs.y.expect("Y requested");
        assert_eq!(y.dims(), &[3, 2, 1, 2]);

        let y_actual_forward = y.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let y_actual_backward = y.narrow(1, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;
        for (actual, expected) in y_actual_forward.iter().zip(y_forward_ref.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "forward mismatch: {actual} vs {expected}"
            );
        }
        for (actual, expected) in y_actual_backward.iter().zip(y_backward_expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "backward mismatch: {actual} vs {expected}"
            );
        }

        let y_h = outputs.y_h.expect("Y_h requested");
        assert_eq!(y_h.dims(), &[2, 1, 2]);
        let y_c = outputs.y_c.expect("Y_c requested");
        assert_eq!(y_c.dims(), &[2, 1, 2]);
        Ok(())
    }

    #[test]
    fn default_optional_inputs_produce_zero_initial_state() -> Result<()> {
        // Omitting bias/initial_h/initial_c must behave like passing
        // all-zero tensors, not panic or error.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((3, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let node = lstm_node("forward", hidden_size);

        let outputs = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))?;

        let y = outputs.y.expect("Y requested");
        assert_eq!(y.dims(), &[3, 1, 1, 2]);
        // All-zero weights/state/input -> tanh(0)=0 cell candidate, so both
        // h and c stay exactly zero for every timestep.
        let got = y.flatten_all()?.to_vec1::<f32>()?;
        assert!(got.iter().all(|&value| value == 0.0));
        Ok(())
    }

    #[test]
    fn direction_reverse_is_rejected() -> Result<()> {
        // "reverse" is a legal ONNX LSTM direction value, but unimplemented here.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let node = lstm_node("reverse", hidden_size);

        let err = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))
            .expect_err("reverse direction should be rejected");
        assert!(err.to_string().contains("direction"));
        Ok(())
    }

    #[test]
    fn non_default_activations_are_rejected() -> Result<()> {
        // Only the default (Sigmoid, Tanh, Tanh) activation set is implemented.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let mut node = lstm_node("forward", hidden_size);
        node.attribute
            .push(strings_attr("activations", &["Relu", "Tanh", "Tanh"]));

        let err = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))
            .expect_err("non-default activations should be rejected");
        assert!(err.to_string().contains("activations"));
        Ok(())
    }

    #[test]
    fn clip_attribute_is_rejected() -> Result<()> {
        // clip is a valid ONNX LSTM attribute, but unimplemented here.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let mut node = lstm_node("forward", hidden_size);
        node.attribute.push(int_attr("clip", 0));

        let err = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))
            .expect_err("clip attribute should be rejected");
        assert!(err.to_string().contains("clip"));
        Ok(())
    }

    #[test]
    fn non_zero_layout_is_rejected() -> Result<()> {
        // layout == 1 (batch-major) is a valid ONNX LSTM attribute value, but unimplemented.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let mut node = lstm_node("forward", hidden_size);
        node.attribute.push(int_attr("layout", 1));

        let err = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))
            .expect_err("non-zero layout should be rejected");
        assert!(err.to_string().contains("layout"));
        Ok(())
    }

    #[test]
    fn non_zero_input_forget_is_rejected() -> Result<()> {
        // The coupled input/forget gate variant (input_forget != 0) is unimplemented.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let mut node = lstm_node("forward", hidden_size);
        node.attribute.push(int_attr("input_forget", 1));

        let err = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))
            .expect_err("non-zero input_forget should be rejected");
        assert!(err.to_string().contains("input_forget"));
        Ok(())
    }

    #[test]
    fn non_uniform_seq_lens_is_rejected() -> Result<()> {
        // Variable-length sequences (a seq_lens entry shorter than seq_length)
        // are unimplemented.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((3, 2, 1), candle_core::DType::F32, &Device::Cpu)?;
        let seq_lens = Tensor::from_vec(vec![3i64, 2i64], (2,), &Device::Cpu)?;
        let node = lstm_node("forward", hidden_size);
        let inputs = LstmInputs {
            input: &input,
            weight: &weight,
            recurrence_weight: &recurrence_weight,
            bias: None,
            seq_lens: Some(&seq_lens),
            initial_h: None,
            initial_c: None,
            peephole: None,
        };

        let err = lstm(&node, inputs).expect_err("non-uniform seq_lens should be rejected");
        assert!(err.to_string().contains("seq_lens"));
        Ok(())
    }

    #[test]
    fn non_zero_peephole_is_rejected() -> Result<()> {
        // Peephole connections (P) are unimplemented.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let peephole = Tensor::ones(
            (1, 3 * hidden_size as usize),
            candle_core::DType::F32,
            &Device::Cpu,
        )?;
        let node = lstm_node("forward", hidden_size);
        let inputs = LstmInputs {
            input: &input,
            weight: &weight,
            recurrence_weight: &recurrence_weight,
            bias: None,
            seq_lens: None,
            initial_h: None,
            initial_c: None,
            peephole: Some(&peephole),
        };

        let err = lstm(&node, inputs).expect_err("non-zero peephole should be rejected");
        assert!(err.to_string().contains("peephole"));
        Ok(())
    }

    #[test]
    fn omitting_y_h_and_y_c_returns_none() -> Result<()> {
        // Only Y is requested (empty output names for Y_h/Y_c); those two
        // outputs must come back None.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((2, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let mut node = lstm_node("forward", hidden_size);
        node.output = vec!["y".to_string(), String::new(), String::new()];

        let outputs = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))?;
        assert!(outputs.y.is_some());
        assert!(outputs.y_h.is_none());
        assert!(outputs.y_c.is_none());
        Ok(())
    }

    #[test]
    fn omitting_y_returns_none_but_states_still_computed() -> Result<()> {
        // Only Y_h/Y_c are requested (empty output name for Y); Y must come
        // back None while the final states are still produced.
        let hidden_size = 2i64;
        let weight = Tensor::zeros((1, 8, 1), candle_core::DType::F32, &Device::Cpu)?;
        let recurrence_weight = Tensor::zeros((1, 8, 2), candle_core::DType::F32, &Device::Cpu)?;
        let input = Tensor::zeros((2, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let mut node = lstm_node("forward", hidden_size);
        node.output = vec![String::new(), "y_h".to_string(), "y_c".to_string()];

        let outputs = lstm(&node, minimal_inputs(&input, &weight, &recurrence_weight))?;
        assert!(outputs.y.is_none());
        assert!(outputs.y_h.is_some());
        assert!(outputs.y_c.is_some());
        Ok(())
    }
}
