// SPDX-License-Identifier: MIT

use crate::models::hunyuan_dense::modeling::Gguf;
use crate::ops::linear::LinearLayer;
use crate::ops::prof::{self, Span};
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, VarBuilder, linear_no_bias};
use std::io::{Read, Seek};

/// Configuration for Mixture-of-Experts feed-forward layers.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of experts per `MoE` layer.
    pub num_experts: usize,
    /// Number of experts activated per token (top-K).
    pub num_experts_per_tok: usize,
    /// Hidden dimension of each expert's feed-forward network.
    pub moe_intermediate_size: usize,
    /// Whether to renormalize the top-K routing weights to sum to 1.
    pub norm_topk_prob: bool,
    /// Every Nth layer is `MoE`; the rest stay dense MLP.
    /// `None` means dense-only (non-MoE checkpoints).
    /// Only used by the safetensors path; the GGUF path detects
    /// MoE-vs-dense per layer by tensor presence.
    pub decoder_sparse_step: Option<usize>,
}

/// A single Mixture-of-Experts feed-forward expert.
///
/// A `SiLU`-gated MLP identical in shape to a dense Qwen3 `Mlp`, but sized to
/// `moe_intermediate_size` rather than the model's dense `intermediate_size`.
#[allow(clippy::struct_field_names)]
pub struct MoeExpert {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
}

impl MoeExpert {
    /// Create an expert from a safetensors checkpoint.
    ///
    /// # Arguments
    /// * `hidden_size` - Model hidden dimension (input/output size)
    /// * `intermediate_size` - Expert feed-forward hidden dimension
    /// * `vb` - `VarBuilder` scoped to this expert
    ///
    /// # Errors
    /// Returns an error if any of the expert's weight tensors are missing or
    /// have the wrong shape.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj: LinearLayer::Standard(gate_proj),
            up_proj: LinearLayer::Standard(up_proj),
            down_proj: LinearLayer::Standard(down_proj),
        })
    }

    /// Create an expert from a GGUF checkpoint using the per-expert tensor
    /// layout (`blk.{layer_idx}.ffn_gate.{expert_idx}.weight`, etc.).
    ///
    /// # Arguments
    /// * `gg` - GGUF reader
    /// * `layer_idx` - Decoder layer index
    /// * `expert_idx` - Expert index within the layer
    /// * `device` - Device to load the expert's weights onto (supports
    ///   expert offloading to a device other than the rest of the model)
    ///
    /// # Errors
    /// Returns an error if the expert's tensors are missing from the GGUF file.
    pub fn new_from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        layer_idx: usize,
        expert_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let gate_proj = gg.linear_on(&format!("{prefix}.ffn_gate.{expert_idx}.weight"), device)?;
        let up_proj = gg.linear_on(&format!("{prefix}.ffn_up.{expert_idx}.weight"), device)?;
        let down_proj = gg.linear_on(&format!("{prefix}.ffn_down.{expert_idx}.weight"), device)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Build an expert directly from already-loaded projections.
    ///
    /// Used by [`SparseMoeBlock::new_from_gguf`] for the packed GGUF expert
    /// layout, where the three projections come from dequantizing and
    /// narrowing a shared 3D tensor rather than loading per-expert tensors.
    fn from_layers(gate_proj: LinearLayer, up_proj: LinearLayer, down_proj: LinearLayer) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

impl Module for MoeExpert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let gate = Activation::Silu.forward(&gate)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Mixture-of-Experts routing + dispatch block.
///
/// Routes each token to its top-K experts (by router-gate logits), runs only
/// those experts, and combines their outputs weighted by routing probability.
/// Experts may live on a different device than the router and input
/// (`expert_device`), so that expert weights can be offloaded (e.g. to CPU)
/// while the rest of the model stays on GPU.
pub struct SparseMoeBlock {
    gate: LinearLayer,
    experts: Vec<MoeExpert>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    expert_device: Device,
}

impl SparseMoeBlock {
    /// Create a `MoE` block from a safetensors checkpoint.
    ///
    /// # Arguments
    /// * `config` - `MoE` layer configuration
    /// * `hidden_size` - Model hidden dimension
    /// * `vb` - `VarBuilder` scoped to this block (holds `gate` and `experts.{i}`)
    /// * `expert_device` - Device to place expert weights on
    ///
    /// # Errors
    /// Returns an error if the router or any expert's weight tensors are
    /// missing or have the wrong shape.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        config: &MoeConfig,
        hidden_size: usize,
        vb: VarBuilder,
        expert_device: &Device,
    ) -> Result<Self> {
        let gate = linear_no_bias(hidden_size, config.num_experts, vb.pp("gate"))?;
        // Router logits are always computed in F32 (see `forward`); re-wrap the
        // weight in F32 here so a BF16/F16 checkpoint doesn't hit a dtype
        // mismatch on the first router matmul.
        let gate_weight = gate.weight().to_dtype(DType::F32)?;
        let gate = Linear::new(gate_weight, None);
        let experts = (0..config.num_experts)
            .map(|i| {
                MoeExpert::new(
                    hidden_size,
                    config.moe_intermediate_size,
                    vb.pp(format!("experts.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gate: LinearLayer::Standard(gate),
            experts,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            expert_device: expert_device.clone(),
        })
    }

    /// Create a `MoE` block from a GGUF checkpoint.
    ///
    /// Supports both packed (`blk.{i}.ffn_gate_exps.weight`, all experts
    /// stacked in one 3D tensor) and per-expert
    /// (`blk.{i}.ffn_gate.{j}.weight`) tensor layouts, auto-detected by
    /// tensor presence.
    ///
    /// # Arguments
    /// * `config` - `MoE` layer configuration
    /// * `gg` - GGUF reader
    /// * `layer_idx` - Decoder layer index
    /// * `expert_device` - Device to place expert weights on (supports
    ///   expert offloading to a device other than the rest of the model)
    ///
    /// # Errors
    /// Returns an error if the router or any expert's tensors are missing
    /// from the GGUF file.
    pub fn new_from_gguf<R: Read + Seek>(
        config: &MoeConfig,
        gg: &mut Gguf<R>,
        layer_idx: usize,
        expert_device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let gate_weight = gg
            .dequant_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?
            .to_dtype(DType::F32)?;
        let gate = LinearLayer::Standard(Linear::new(gate_weight, None));

        let packed_name = format!("{prefix}.ffn_gate_exps.weight");
        let experts = if gg.contains_tensor(&packed_name) {
            Self::load_packed_experts(gg, &prefix, config.num_experts, expert_device)?
        } else {
            (0..config.num_experts)
                .map(|expert_idx| {
                    MoeExpert::new_from_gguf(gg, layer_idx, expert_idx, expert_device)
                })
                .collect::<Result<Vec<_>>>()?
        };

        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            expert_device: expert_device.clone(),
        })
    }

    /// Load the packed (Unsloth-style) `_exps` expert tensor layout: each
    /// projection is a single 3D tensor `[num_experts, out, in]` covering all
    /// experts, dequantized onto `expert_device` and narrowed per expert.
    ///
    /// Unlike the per-expert layout (kept quantized as `LinearLayer::Quantized`
    /// via [`MoeExpert::new_from_gguf`]), packed experts are dequantized to
    /// full precision up front, since candle's `QTensor` has no support for
    /// slicing a packed 3D quantized tensor per-expert without dequantizing
    /// it. This means CPU-offloading a packed-layout checkpoint's experts
    /// (`expert_device` != the router's device) costs full-precision memory
    /// per expert rather than the on-disk quantized size. A checkpoint that
    /// fits in CPU RAM quantized may not once dequantized this way.
    fn load_packed_experts<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_experts: usize,
        expert_device: &Device,
    ) -> Result<Vec<MoeExpert>> {
        let gate_all =
            gg.dequant_tensor_on(&format!("{prefix}.ffn_gate_exps.weight"), expert_device)?;
        let up_all =
            gg.dequant_tensor_on(&format!("{prefix}.ffn_up_exps.weight"), expert_device)?;
        let down_all =
            gg.dequant_tensor_on(&format!("{prefix}.ffn_down_exps.weight"), expert_device)?;

        (0..num_experts)
            .map(|i| {
                let gate = slice_packed_expert(&gate_all, i)?;
                let up = slice_packed_expert(&up_all, i)?;
                let down = slice_packed_expert(&down_all, i)?;
                Ok(MoeExpert::from_layers(
                    LinearLayer::Standard(Linear::new(gate, None)),
                    LinearLayer::Standard(Linear::new(up, None)),
                    LinearLayer::Standard(Linear::new(down, None)),
                ))
            })
            .collect()
    }
}

/// Narrow one expert's 2D weight out of a packed `[num_experts, out, in]` tensor.
fn slice_packed_expert(packed: &Tensor, expert_idx: usize) -> Result<Tensor> {
    packed.narrow(0, expert_idx, 1)?.squeeze(0)?.contiguous()
}

impl Module for SparseMoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dims = xs.dims().to_vec();
        let Some(&hidden_size) = original_dims.last() else {
            candle_core::bail!("SparseMoeBlock input must have at least one dimension");
        };
        let original_dtype = xs.dtype();
        let xs_flat = xs.reshape(((), hidden_size))?;
        let xs_f32 = xs_flat.to_dtype(DType::F32)?;

        let (topk_ids, topk_weights) =
            prof::timed(Span::MoeRouter, || -> Result<(Tensor, Tensor)> {
                let logits = self.gate.forward(&xs_f32)?;
                let probs = candle_nn::ops::softmax_last_dim(&logits)?;
                let topk_ids = probs
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                    .contiguous()?;
                let mut topk_weights = probs.gather(&topk_ids, D::Minus1)?;
                if self.norm_topk_prob {
                    let sum = topk_weights.sum_keepdim(D::Minus1)?;
                    topk_weights = topk_weights.broadcast_div(&sum)?;
                }
                Ok((topk_ids, topk_weights))
            })?;

        // Routing dispatch is CPU-side: topk indices and weights are pulled to
        // the host each forward call, and per-expert token/weight lists below
        // are heap-allocated fresh each call. A fused GPU MoE kernel (as in
        // candle's moe_gemm_gguf) would eliminate both the sync and the
        // allocations, at the cost of losing the packed/per-expert layout
        // flexibility this dispatch loop gets for free.
        let topk_ids = topk_ids.to_vec2::<u32>()?;
        let topk_weights = topk_weights.to_vec2::<f32>()?;

        let mut token_lists: Vec<Vec<u32>> = vec![Vec::new(); self.experts.len()];
        let mut weight_lists: Vec<Vec<f32>> = vec![Vec::new(); self.experts.len()];
        for (token_idx, (ids, weights)) in topk_ids.iter().zip(topk_weights.iter()).enumerate() {
            // Token counts (batch * seq_len) never approach u32::MAX.
            #[allow(clippy::cast_possible_truncation)]
            let token_idx = token_idx as u32;
            for (&expert_idx, &weight) in ids.iter().zip(weights.iter()) {
                token_lists[expert_idx as usize].push(token_idx);
                weight_lists[expert_idx as usize].push(weight);
            }
        }

        // `Device::Cpu` is a unit variant, so this cross-device branch is only
        // exercised (and only exercisable in tests) on multi-device hardware.
        // All unit tests here run router and experts on the same CPU device.
        let same_device = xs_flat.device().location() == self.expert_device.location();
        let xs_dispatch = prof::timed(Span::MoeToDevice, || -> Result<Tensor> {
            if same_device {
                Ok(xs_flat.clone())
            } else {
                xs_flat.to_device(&self.expert_device)
            }
        })?;

        let output = prof::timed(Span::MoeExpert, || -> Result<Tensor> {
            let mut output =
                Tensor::zeros(xs_dispatch.dims(), xs_dispatch.dtype(), &self.expert_device)?;
            for (expert_idx, expert) in self.experts.iter().enumerate() {
                let tokens = &token_lists[expert_idx];
                if tokens.is_empty() {
                    continue;
                }
                let token_ids = Tensor::new(tokens.as_slice(), &self.expert_device)?;
                let selected = xs_dispatch.index_select(&token_ids, 0)?;
                let expert_out = expert.forward(&selected)?;
                let weights =
                    Tensor::new(weight_lists[expert_idx].as_slice(), &self.expert_device)?
                        .reshape((tokens.len(), 1))?
                        .to_dtype(expert_out.dtype())?;
                let scaled = expert_out.broadcast_mul(&weights)?;
                output = output.index_add(&token_ids, &scaled, 0)?;
            }
            Ok(output)
        })?;

        let output = prof::timed(Span::MoeToDevice, || -> Result<Tensor> {
            if same_device {
                Ok(output)
            } else {
                output.to_device(xs_flat.device())
            }
        })?;
        if output.dtype() != original_dtype {
            candle_core::bail!(
                "MoE output dtype {:?} differs from input dtype {:?}",
                output.dtype(),
                original_dtype,
            );
        }
        output.reshape(original_dims)
    }
}

/// Dense MLP or Mixture-of-Experts feed-forward layer.
///
/// Generic over the model's own dense MLP type `M`, so each model plugs in
/// its existing MLP struct unchanged (e.g. Qwen3's `Mlp`, Qwen3.5's `Mlp`).
/// `DecoderLayer` holds `MlpOrMoe<Mlp>` and calls `.forward()` without
/// branching on the layer type.
pub enum MlpOrMoe<M: Module> {
    /// Standard dense MLP (non-MoE layer).
    Dense(M),
    /// Mixture-of-Experts routing + dispatch block.
    Moe(SparseMoeBlock),
}

impl<M: Module> Module for MlpOrMoe<M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use candle_core::{DType, Device, Tensor};

    // Keys use dot-separated format ("gate_proj.weight") to match VarBuilder::pp("gate_proj").
    fn make_vb(
        hidden: usize,
        intermediate: usize,
        gate_data: Vec<f32>,
        up_data: Vec<f32>,
        down_data: Vec<f32>,
    ) -> candle_nn::VarBuilder<'static> {
        let device = &Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "gate_proj.weight".to_string(),
            Tensor::from_vec(gate_data, (intermediate, hidden), device).expect("gate weight"),
        );
        tensors.insert(
            "up_proj.weight".to_string(),
            Tensor::from_vec(up_data, (intermediate, hidden), device).expect("up weight"),
        );
        tensors.insert(
            "down_proj.weight".to_string(),
            Tensor::from_vec(down_data, (hidden, intermediate), device).expect("down weight"),
        );
        candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device)
    }

    fn zeros_expert(hidden: usize, intermediate: usize) -> MoeExpert {
        let gate = vec![0.0f32; hidden * intermediate];
        let up = vec![0.0f32; hidden * intermediate];
        let down = vec![0.0f32; intermediate * hidden];
        let vb = make_vb(hidden, intermediate, gate, up, down);
        MoeExpert::new(hidden, intermediate, vb).expect("MoeExpert::new")
    }

    fn identity_vb(hidden: usize) -> candle_nn::VarBuilder<'static> {
        let eye: Vec<f32> = (0..hidden * hidden)
            .map(|idx| {
                if idx / hidden == idx % hidden {
                    1.0_f32
                } else {
                    0.0
                }
            })
            .collect();
        make_vb(hidden, hidden, eye.clone(), eye.clone(), eye)
    }

    #[test]
    fn test_output_shape_2d() {
        let expert = zeros_expert(8, 16);
        let x = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = expert.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[4, 8]);
    }

    #[test]
    fn test_output_shape_3d() {
        let expert = zeros_expert(16, 32);
        let x = Tensor::zeros((2, 5, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = expert.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn test_zero_weights_give_zero_output() {
        let expert = zeros_expert(8, 16);
        let x = Tensor::ones((3, 8), DType::F32, &Device::Cpu).expect("ones");
        let y = expert.forward(&x).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero weights, got max={max_val}"
        );
    }

    #[test]
    fn test_zero_input_gives_zero_output() {
        let vb = identity_vb(8);
        let expert = MoeExpert::new(8, 8, vb).expect("new");
        let x = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = expert.forward(&x).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero input, got max={max_val}"
        );
    }

    #[test]
    fn test_intermediate_size_larger_than_hidden() {
        let expert = zeros_expert(32, 128);
        let x = Tensor::zeros((1, 32), DType::F32, &Device::Cpu).expect("zeros");
        let y = expert.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[1, 32]);
    }

    #[test]
    fn test_formula_manual_verification() {
        // hidden=2, intermediate=2
        // gate_proj.weight = [[1, 0], [0, 1]]  (I, row-major so output[i] = x[i])
        // up_proj.weight   = [[2, 0], [0, 2]]  (2*I)
        // down_proj.weight = [[1, 0], [0, 1]]  (I)
        // x = [1.0, 0.5]
        //
        // Linear stores weight as [out, in] and computes x @ weight.T:
        //   gate = x @ I.T = [1.0, 0.5]
        //   gate_activated = silu([1.0, 0.5])
        //   up = x @ (2*I).T = [2.0, 1.0]
        //   product = gate_activated * up
        //   output = product @ I.T = product
        let device = &Device::Cpu;
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let doubled = vec![2.0f32, 0.0, 0.0, 2.0];
        let vb = make_vb(2, 2, identity.clone(), doubled, identity);
        let expert = MoeExpert::new(2, 2, vb).expect("new");

        let x = Tensor::new(&[1.0f32, 0.5], device)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = expert.forward(&x).expect("forward");
        let got = y
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let silu = |v: f32| v / (1.0 + (-v).exp());
        let expected = [silu(1.0_f32) * 2.0, silu(0.5_f32) * 1.0];

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "output[{i}]: got {g}, expected {e}");
        }
    }

    #[test]
    fn test_batch_consistency() {
        // Identical rows in a batch should produce identical output rows.
        let device = &Device::Cpu;
        let hidden = 8usize;
        let intermediate = 16usize;
        let gate: Vec<f32> = (0..hidden * intermediate)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let up: Vec<f32> = gate.iter().map(|v| v * 0.5).collect();
        let down: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let vb = make_vb(hidden, intermediate, gate, up, down);
        let expert = MoeExpert::new(hidden, intermediate, vb).expect("new");

        let row: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let single = Tensor::from_vec(row, (1, hidden), device).expect("single row");
        let batch = Tensor::cat(&[&single, &single, &single], 0).expect("cat");

        let out_batch = expert.forward(&batch).expect("batch forward");
        let out_single = expert.forward(&single).expect("single forward");

        for b in 0..3 {
            let row_b = out_batch.narrow(0, b, 1).expect("narrow");
            let diff: f32 = (&row_b - &out_single)
                .expect("sub")
                .abs()
                .expect("abs")
                .max_all()
                .expect("max_all")
                .to_scalar()
                .expect("scalar");
            assert!(
                diff < 1e-6,
                "batch row {b} differs from single-row output, diff={diff}"
            );
        }
    }

    #[test]
    fn test_slice_packed_expert() {
        // [num_experts=3, out=2, in=4]; expert i's slice is filled with (i+1).
        let device = &Device::Cpu;
        let data: Vec<f32> = (0..3 * 2 * 4)
            .map(|idx| ((idx / (2 * 4)) + 1) as f32)
            .collect();
        let packed = Tensor::from_vec(data, (3, 2, 4), device).expect("packed tensor");

        for expert_idx in 0..3 {
            let sliced = slice_packed_expert(&packed, expert_idx).expect("slice");
            assert_eq!(sliced.dims(), &[2, 4]);
            let vals = sliced
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("to_vec1");
            let expected_val = (expert_idx + 1) as f32;
            assert!(
                vals.iter().all(|&v| (v - expected_val).abs() < 1e-6),
                "expert {expert_idx}: got {vals:?}, expected all {expected_val}"
            );
        }
    }

    // Router weight key is "gate.weight" (VarBuilder::pp("gate")); expert keys
    // are "experts.{i}.{gate,up,down}_proj.weight" (VarBuilder::pp("experts.{i}")).
    fn make_sparse_moe_vb(
        hidden: usize,
        num_experts: usize,
        gate_data: Vec<f32>,
        expert_data: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> candle_nn::VarBuilder<'static> {
        let device = &Device::Cpu;
        let intermediate = expert_data[0].0.len() / hidden;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "gate.weight".to_string(),
            Tensor::from_vec(gate_data, (num_experts, hidden), device).expect("gate weight"),
        );
        for (i, (gate, up, down)) in expert_data.iter().enumerate() {
            tensors.insert(
                format!("experts.{i}.gate_proj.weight"),
                Tensor::from_vec(gate.clone(), (intermediate, hidden), device)
                    .expect("expert gate weight"),
            );
            tensors.insert(
                format!("experts.{i}.up_proj.weight"),
                Tensor::from_vec(up.clone(), (intermediate, hidden), device)
                    .expect("expert up weight"),
            );
            tensors.insert(
                format!("experts.{i}.down_proj.weight"),
                Tensor::from_vec(down.clone(), (hidden, intermediate), device)
                    .expect("expert down weight"),
            );
        }
        candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device)
    }

    fn make_sparse_moe(
        hidden: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        gate_data: Vec<f32>,
        expert_data: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> SparseMoeBlock {
        let vb = make_sparse_moe_vb(hidden, num_experts, gate_data, expert_data);
        let config = MoeConfig {
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob,
            decoder_sparse_step: None,
        };
        SparseMoeBlock::new(&config, hidden, vb, &Device::Cpu).expect("SparseMoeBlock::new")
    }

    fn zeros_sparse_moe(
        hidden: usize,
        intermediate: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
    ) -> SparseMoeBlock {
        let gate_data = vec![0.0f32; num_experts * hidden];
        let expert_data: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..num_experts)
            .map(|_| {
                (
                    vec![0.0f32; hidden * intermediate],
                    vec![0.0f32; hidden * intermediate],
                    vec![0.0f32; intermediate * hidden],
                )
            })
            .collect();
        make_sparse_moe(
            hidden,
            intermediate,
            num_experts,
            num_experts_per_tok,
            false,
            gate_data,
            &expert_data,
        )
    }

    #[test]
    fn test_sparse_moe_output_shape_2d() {
        let moe = zeros_sparse_moe(8, 4, 4, 2);
        let x = Tensor::zeros((3, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = moe.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[3, 8]);
    }

    #[test]
    fn test_sparse_moe_output_shape_3d() {
        let moe = zeros_sparse_moe(8, 4, 4, 2);
        let x = Tensor::zeros((2, 3, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = moe.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[2, 3, 8]);
    }

    #[test]
    fn test_single_token_routing() {
        let moe = zeros_sparse_moe(8, 4, 4, 2);
        let x = Tensor::zeros((1, 8), DType::F32, &Device::Cpu).expect("zeros");
        let y = moe.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[1, 8]);
    }

    #[test]
    fn test_all_experts_zero_weights_gives_zero_output() {
        let moe = zeros_sparse_moe(8, 4, 4, 2);
        let x = Tensor::ones((3, 8), DType::F32, &Device::Cpu).expect("ones");
        let y = moe.forward(&x).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero weights, got max={max_val}"
        );
    }

    // 4 experts, top-2. Router logits (from x=[1,0] and gate rows
    // [2,0],[-2,0],[1.5,0],[-1.5,0]) are [2,-2,1.5,-1.5] — close enough
    // together that normalization measurably changes the combined weight,
    // while still unambiguously ranking experts 0 and 2 highest. Each expert
    // e has gate_proj=I, up_proj=c_e*I, down_proj=I, so
    // expert_e(x) = c_e * x * silu(x) elementwise. Experts 1 and 3 (not
    // selected) use a huge c=100 so any routing bug that includes them is
    // trivially detectable.
    fn routing_test_setup(norm_topk_prob: bool) -> (SparseMoeBlock, f32, f32) {
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let scaled = |c: f32| vec![c, 0.0, 0.0, c];
        let expert_data = vec![
            (identity.clone(), scaled(1.0), identity.clone()),
            (identity.clone(), scaled(100.0), identity.clone()),
            (identity.clone(), scaled(2.0), identity.clone()),
            (identity.clone(), scaled(100.0), identity.clone()),
        ];
        let gate_data = vec![
            2.0, 0.0, //
            -2.0, 0.0, //
            1.5, 0.0, //
            -1.5, 0.0,
        ];
        let moe = make_sparse_moe(2, 2, 4, 2, norm_topk_prob, gate_data, &expert_data);

        let logits = [2.0f32, -2.0, 1.5, -1.5];
        let exps: Vec<f32> = logits.iter().map(|l| l.exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        (moe, probs[0], probs[2])
    }

    #[test]
    fn test_routing_selects_top_k_experts() {
        let (moe, weight0, weight2) = routing_test_setup(false);
        let x = Tensor::new(&[1.0f32, 0.0], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = moe.forward(&x).expect("forward");
        let got = y
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let expected0 = (weight0 * 1.0 + weight2 * 2.0) * silu_1;
        assert!(
            (got[0] - expected0).abs() < 1e-4,
            "got {got:?}, expected [{expected0}, 0.0] (only experts 0 and 2 should contribute)"
        );
        assert!(
            got[1].abs() < 1e-5,
            "got {got:?}, expected second element 0"
        );
    }

    #[test]
    fn test_norm_topk_prob_normalizes_weights() {
        let (moe, weight0, weight2) = routing_test_setup(true);
        let x = Tensor::new(&[1.0f32, 0.0], &Device::Cpu)
            .expect("tensor")
            .reshape((1, 2))
            .expect("reshape");
        let y = moe.forward(&x).expect("forward");
        let got = y
            .squeeze(0)
            .expect("squeeze")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let norm0 = weight0 / (weight0 + weight2);
        let norm2 = weight2 / (weight0 + weight2);
        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let expected0 = (norm0 * 1.0 + norm2 * 2.0) * silu_1;
        assert!(
            (got[0] - expected0).abs() < 1e-4,
            "got {got:?}, expected [{expected0}, 0.0] with normalized top-k weights"
        );

        // Normalized weights must sum to 1, so this must differ from the
        // un-normalized (raw softmax) result.
        let raw_expected0 = (weight0 * 1.0 + weight2 * 2.0) * silu_1;
        assert!(
            (expected0 - raw_expected0).abs() > 1e-4,
            "normalized and raw-weighted outputs should differ"
        );
    }

    // Regression test: a checkpoint whose weights load in a dtype other than
    // F32 (e.g. F16/BF16 in production) must not have its gate weight
    // forwarded against `forward`'s F32-upcast router input, which used to
    // hard-error on a dtype mismatch in the router matmul. Uses F16 rather
    // than BF16 because candle's CPU matmul backend (no mkl/accelerate)
    // only supports F16/F32/F64. That is an unrelated candle limitation,
    // not something this test is meant to exercise.
    #[test]
    fn test_sparse_moe_non_f32_checkpoint_does_not_crash() {
        let device = &Device::Cpu;
        let hidden = 8;
        let num_experts = 4;
        let intermediate = 4;
        let gate_data = vec![0.0f32; num_experts * hidden];
        let expert_data: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..num_experts)
            .map(|_| {
                (
                    vec![0.0f32; hidden * intermediate],
                    vec![0.0f32; hidden * intermediate],
                    vec![0.0f32; intermediate * hidden],
                )
            })
            .collect();

        let to_f16 = |data: Vec<f32>, shape: (usize, usize)| {
            Tensor::from_vec(data, shape, device)
                .expect("tensor")
                .to_dtype(DType::F16)
                .expect("to f16")
        };
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "gate.weight".to_string(),
            to_f16(gate_data, (num_experts, hidden)),
        );
        for (i, (gate, up, down)) in expert_data.into_iter().enumerate() {
            tensors.insert(
                format!("experts.{i}.gate_proj.weight"),
                to_f16(gate, (intermediate, hidden)),
            );
            tensors.insert(
                format!("experts.{i}.up_proj.weight"),
                to_f16(up, (intermediate, hidden)),
            );
            tensors.insert(
                format!("experts.{i}.down_proj.weight"),
                to_f16(down, (hidden, intermediate)),
            );
        }
        let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F16, device);
        let config = MoeConfig {
            num_experts,
            num_experts_per_tok: 2,
            moe_intermediate_size: intermediate,
            norm_topk_prob: false,
            decoder_sparse_step: None,
        };
        let moe = SparseMoeBlock::new(&config, hidden, vb, device).expect("SparseMoeBlock::new");

        let x = Tensor::zeros((3, hidden), DType::F16, device).expect("zeros");
        let y = moe.forward(&x).expect("forward");
        assert_eq!(y.dtype(), DType::F16);
        assert_eq!(y.dims(), &[3, hidden]);
    }

    #[test]
    fn test_mlp_or_moe_dense_passthrough() {
        let expert = zeros_expert(8, 16);
        let x = Tensor::ones((3, 8), DType::F32, &Device::Cpu).expect("ones");
        let direct = expert.forward(&x).expect("direct forward");

        let wrapped: MlpOrMoe<MoeExpert> = MlpOrMoe::Dense(zeros_expert(8, 16));
        let via_enum = wrapped.forward(&x).expect("enum forward");

        let diff: f32 = (&direct - &via_enum)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            diff < 1e-8,
            "MlpOrMoe::Dense output should match direct expert forward, diff={diff}"
        );
    }
}
