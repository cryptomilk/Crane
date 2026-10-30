//! Crane Added 20260731: ONNX `ConvTranspose` backed by Candle kernels.

use candle_core::{Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// Executes ONNX `ConvTranspose` with Candle's transposed-convolution kernels.
///
/// The ONNX and Candle weight layouts are identical: `[C_in, C_out / group,
/// spatial...]`.  This deliberately calls Tensor kernels directly instead of
/// constructing a `candle_nn` module for each graph evaluation.
pub(crate) fn conv_transpose(
    node: &NodeProto,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    validate_common_attributes(node)?;
    let group = int_attribute(node, "group", 1)?;
    let Some(group) = usize::try_from(group).ok().filter(|&group| group > 0) else {
        bail!(
            "ConvTranspose node '{}' has invalid group {group}",
            node.name
        );
    };

    let output = match weight.rank() {
        3 => conv_transpose1d(node, input, weight, group)?,
        4 => conv_transpose2d(node, input, weight, group)?,
        rank => bail!(
            "ConvTranspose node '{}' requires a rank-3 or rank-4 weight, got rank {rank}",
            node.name
        ),
    };
    add_bias(node, output, bias)
}

fn conv_transpose1d(
    node: &NodeProto,
    input: &Tensor,
    weight: &Tensor,
    group: usize,
) -> Result<Tensor> {
    let padding = symmetric_value(node, "pads", 0, 1)?;
    let output_padding = symmetric_value(node, "output_padding", 0, 1)?;
    let stride = symmetric_value(node, "strides", 1, 1)?;
    let dilation = symmetric_value(node, "dilations", 1, 1)?;
    input.conv_transpose1d(weight, padding, output_padding, stride, dilation, group)
}

fn conv_transpose2d(
    node: &NodeProto,
    input: &Tensor,
    weight: &Tensor,
    group: usize,
) -> Result<Tensor> {
    if group != 1 {
        bail!(
            "ConvTranspose node '{}' uses group={group}; Candle's 2D transposed-convolution kernel currently supports group=1 only",
            node.name
        );
    }
    let padding = symmetric_value(node, "pads", 0, 2)?;
    let output_padding = symmetric_value(node, "output_padding", 0, 2)?;
    let stride = symmetric_value(node, "strides", 1, 2)?;
    let dilation = symmetric_value(node, "dilations", 1, 2)?;
    input.conv_transpose2d(weight, padding, output_padding, stride, dilation)
}

fn add_bias(node: &NodeProto, output: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let Some(bias) = bias else {
        return Ok(output);
    };
    if bias.rank() != 1 {
        bail!(
            "ConvTranspose node '{}' has a rank-{} bias; expected rank 1",
            node.name,
            bias.rank()
        );
    }
    let mut shape = vec![1; output.rank()];
    shape[1] = bias.elem_count();
    output.broadcast_add(&bias.reshape(shape)?)
}

fn validate_common_attributes(node: &NodeProto) -> Result<()> {
    let auto_pad = string_attribute(node, "auto_pad")?.unwrap_or("NOTSET");
    if auto_pad != "NOTSET" {
        bail!(
            "ConvTranspose node '{}' uses auto_pad='{auto_pad}', which is not supported; export with explicit symmetric pads",
            node.name
        );
    }
    if node
        .attribute
        .iter()
        .any(|attribute| attribute.name == "output_shape")
    {
        bail!(
            "ConvTranspose node '{}' uses output_shape, which is not supported; export with pads and output_padding instead",
            node.name
        );
    }
    Ok(())
}

fn symmetric_value(
    node: &NodeProto,
    name: &str,
    default: usize,
    spatial_dims: usize,
) -> Result<usize> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(default);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Ints {
        bail!(
            "ConvTranspose node '{}' has a non-INTS '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type()
        );
    }
    let expected_len = if name == "pads" {
        spatial_dims * 2
    } else {
        spatial_dims
    };
    if attribute.ints.len() != expected_len || attribute.ints.iter().any(|&value| value < 0) {
        bail!(
            "ConvTranspose node '{}' has invalid '{}'={:?}; expected {expected_len} non-negative value(s)",
            node.name,
            name,
            attribute.ints
        );
    }
    // Every value was checked non-negative above, so the sign is never lost.
    #[allow(clippy::cast_sign_loss)]
    let value = attribute.ints[0] as usize;
    // `value` is non-negative, so equality on `i64` is equality on `usize`.
    if attribute
        .ints
        .iter()
        .any(|&candidate| candidate != attribute.ints[0])
    {
        bail!(
            "ConvTranspose node '{}' has asymmetric '{}'={:?}; the current Candle kernel requires identical values on every spatial axis",
            node.name,
            name,
            attribute.ints
        );
    }
    Ok(value)
}

fn int_attribute(node: &NodeProto, name: &str, default: i64) -> Result<i64> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(default);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Int {
        bail!(
            "ConvTranspose node '{}' has a non-INT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type()
        );
    }
    Ok(attribute.i)
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
            "ConvTranspose node '{}' has a non-STRING '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type()
        );
    }
    std::str::from_utf8(&attribute.s)
        .map(Some)
        .map_err(candle_core::Error::wrap)
}
