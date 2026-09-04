// SPDX-License-Identifier: MIT

use crate::models::hunyuan_dense::modeling::Gguf;
use crate::ops::linear::LinearLayer;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder, linear_no_bias};
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
}

impl Module for MoeExpert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let gate = Activation::Silu.forward(&gate)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
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
}
