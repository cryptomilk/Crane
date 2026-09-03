// SPDX-License-Identifier: MIT

/// Configuration for Mixture-of-Experts feed-forward layers.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of experts per MoE layer.
    pub num_experts: usize,
    /// Number of experts activated per token (top-K).
    pub num_experts_per_tok: usize,
    /// Hidden dimension of each expert's feed-forward network.
    pub moe_intermediate_size: usize,
    /// Whether to renormalize the top-K routing weights to sum to 1.
    pub norm_topk_prob: bool,
    /// Every Nth layer is MoE; the rest stay dense MLP.
    /// `None` means dense-only (non-MoE checkpoints).
    /// Only used by the safetensors path; the GGUF path detects
    /// MoE-vs-dense per layer by tensor presence.
    pub decoder_sparse_step: Option<usize>,
}
