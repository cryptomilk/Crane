// SPDX-License-Identifier: MIT

//! Device assignment and VRAM budgeting for models with offloadable
//! sub-components (e.g. MoE experts).

use candle_core::Device;

/// Bundles the primary inference device with the device MoE expert weights
/// load onto.
///
/// Two adjacent same-typed `&Device` parameters invite an unchecked swap at
/// call sites; bundling them into named fields makes that swap a compile-time
/// impossibility instead of a silent bug.
#[derive(Debug, Clone)]
pub struct DeviceAssignment {
    /// Device for model weights and inference (everything but MoE experts).
    pub main: Device,
    /// Device for MoE expert weights. Same as `main` when expert offloading
    /// is not needed; ignored by models and formats without MoE experts.
    pub expert: Device,
}

impl DeviceAssignment {
    /// All weights, including MoE experts, on the same device.
    pub fn uniform(device: &Device) -> Self {
        Self {
            main: device.clone(),
            expert: device.clone(),
        }
    }
}

/// VRAM budget available for model weights, used to decide per-layer
/// expert placement during model loading.
///
/// Constructed from `--gpu-memory-limit` and `--offload-experts` in
/// `crane-serve` and threaded through to `Qwen3Backend`. Not yet consumed:
/// that lands when `Qwen3Model::from_gguf()`'s loading path uses it to
/// decide which `MoE` layers load expert weights to GPU vs CPU.
#[derive(Debug, Clone, Default)]
pub struct GpuBudget {
    /// VRAM ceiling for model weights.
    pub weight_budget: WeightBudget,
    /// When `true`, force all `MoE` expert weights to CPU regardless of
    /// `weight_budget` (`--offload-experts`).
    pub offload_all_experts: bool,
}

impl GpuBudget {
    /// No GPU available (`--cpu` mode): all weights, including experts,
    /// must load to CPU.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            weight_budget: WeightBudget::NoGpu,
            offload_all_experts: false,
        }
    }

    /// Derives a budget from a device for callers with no CLI-configured
    /// VRAM ceiling: [`Self::cpu()`] when `device` is a CPU device,
    /// otherwise an unlimited GPU budget.
    #[must_use]
    pub fn for_device(device: &Device) -> Self {
        if device.is_cpu() {
            Self::cpu()
        } else {
            Self::default()
        }
    }
}

/// VRAM ceiling for model weights.
///
/// Distinguishes "no GPU device" from "GPU present with no configured
/// limit" so callers can't conflate the two by reading a bare `None`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WeightBudget {
    /// No GPU device available (`--cpu` mode): all weights, including
    /// experts, must load to CPU.
    NoGpu,
    /// GPU present, no configured VRAM ceiling: prefer GPU for everything.
    #[default]
    Unlimited,
    /// GPU present with a VRAM budget, in bytes, for model weights (total
    /// VRAM minus estimated runtime needs like KV cache and activations).
    Limited(u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies `uniform` assigns the same device to both fields.
    #[test]
    fn uniform_assigns_same_device_to_both_fields() {
        let device = Device::Cpu;
        let assignment = DeviceAssignment::uniform(&device);
        assert!(assignment.main.same_device(&assignment.expert));
    }

    // Verifies `cpu()` reports no GPU device and does not force offload.
    #[test]
    fn gpu_budget_cpu_mode() {
        let budget = GpuBudget::cpu();
        assert_eq!(budget.weight_budget, WeightBudget::NoGpu);
        assert!(!budget.offload_all_experts);
    }

    // Verifies the default budget is unlimited and does not force offload.
    #[test]
    fn gpu_budget_default_no_limit() {
        let budget = GpuBudget::default();
        assert_eq!(budget.weight_budget, WeightBudget::Unlimited);
        assert!(!budget.offload_all_experts);
    }

    // Verifies `cpu()` and `default()` are distinct, unlike the old
    // `Option<usize>`-based encoding where both collapsed to `None`.
    #[test]
    fn gpu_budget_cpu_and_default_are_distinct() {
        assert_ne!(
            GpuBudget::cpu().weight_budget,
            GpuBudget::default().weight_budget
        );
    }

    // Verifies `for_device` picks `NoGpu` for a CPU device and `Unlimited`
    // otherwise, so a caller with no CLI budget doesn't silently claim GPU
    // headroom while actually running on CPU.
    #[test]
    fn gpu_budget_for_device_matches_device_kind() {
        assert_eq!(
            GpuBudget::for_device(&Device::Cpu).weight_budget,
            WeightBudget::NoGpu
        );
    }
}
