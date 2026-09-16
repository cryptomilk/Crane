// SPDX-License-Identifier: MIT
/**
 * Fused top-K MoE routing kernel.
 *
 * Fuses softmax -> top-K selection -> optional weight normalization into a
 * single kernel launch, replacing candle's softmax/arg_sort/narrow/gather
 * chain. One warp handles one token: each lane holds a strided slice of that
 * token's expert logits in registers, softmax is computed via
 * `__shfl_xor_sync` butterfly reductions (every lane ends up with the same
 * max/sum, unlike `__shfl_down_sync` which only delivers the result to lane
 * 0), and the top-K experts are selected by `top_k` rounds of warp-cooperative
 * iterative argmax, each round masking the previous winner to -infinity.
 *
 * Targets: sm_80+ (Ampere & newer), and AMD GPUs via HIP — the ROCm launcher
 * (`ops/fused_ops/topk_moe.rs`'s `rocm_impl`) hands this same source to hipcc
 * at runtime through candle's shim.
 */

#include <float.h>
#include <math.h>
#include <stdint.h>

static constexpr int WARP_SIZE = 32;

// Registers per lane for expert logits/probabilities. 16 * WARP_SIZE = 512,
// the largest expert count this kernel supports; larger counts fall back to
// the portable candle op chain (see `topk_moe.rs`'s `MAX_FUSED_EXPERTS`).
static constexpr int MAX_EPT = 16;

__device__ __forceinline__ float warp_reduce_max_xor(float val) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask, WARP_SIZE));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_xor(float val) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask, WARP_SIZE);
    }
    return val;
}

/**
 * logits: [n_tokens, n_experts], row-major, F32.
 * out_ids: [n_tokens, top_k], U32 expert indices, descending by weight.
 * out_weights: [n_tokens, top_k], F32 routing weights (post-softmax,
 *   optionally renormalized to sum to 1 across the selected experts).
 *
 * Launch: one block per token, one warp (32 threads) per block.
 */
extern "C" __global__ void topk_moe_routing_f32(
    const float *__restrict__ logits,
    uint32_t *__restrict__ out_ids,
    float *__restrict__ out_weights,
    int n_experts,
    int top_k,
    int norm_topk
) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    const int experts_per_thread = (n_experts + WARP_SIZE - 1) / WARP_SIZE;

    const float *row = logits + (size_t)token * n_experts;

    float wt[MAX_EPT];
#pragma unroll
    for (int i = 0; i < MAX_EPT; i++) {
        wt[i] = -INFINITY;
    }
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = lane + i * WARP_SIZE;
        wt[i] = (expert < n_experts) ? row[expert] : -INFINITY;
    }

    // Softmax over the full expert set, in registers.
    float max_val = -INFINITY;
    for (int i = 0; i < experts_per_thread; i++) {
        max_val = fmaxf(max_val, wt[i]);
    }
    max_val = warp_reduce_max_xor(max_val);

    float sum = 0.f;
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = lane + i * WARP_SIZE;
        if (expert < n_experts) {
            const float e = expf(wt[i] - max_val);
            wt[i] = e;
            sum += e;
        } else {
            wt[i] = 0.f;
        }
    }
    sum = warp_reduce_sum_xor(sum);
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] *= inv_sum;
    }

    // Sanitize NaN to -FLT_MAX: NaN comparisons always return false, which
    // would make the iterative argmax below re-select the same expert.
    for (int i = 0; i < experts_per_thread; i++) {
        if (isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    uint32_t *ids_out = out_ids + (size_t)token * top_k;
    float *weights_out = out_weights + (size_t)token * top_k;
    float wt_sum = 0.f;

    for (int k = 0; k < top_k; k++) {
        // wt[0] is read unguarded, unlike wt[i>0] below: when n_experts > 32
        // every lane's slot 0 is real, and when n_experts <= 32 an
        // out-of-range lane's local_expert (== lane >= n_experts) always
        // loses the smaller-id tie-break to a real expert, so this is safe
        // without an `expert < n_experts` check.
        float local_max = wt[0];
        int local_expert = lane;
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if (expert < n_experts && wt[i] > local_max) {
                local_max = wt[i];
                local_expert = expert;
            }
        }

        // Butterfly all-reduce: every lane ends up with the same winner,
        // ties broken by the smaller expert id for a reproducible order.
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float other_val = __shfl_xor_sync(0xffffffff, local_max, mask, WARP_SIZE);
            const int other_expert = __shfl_xor_sync(0xffffffff, local_expert, mask, WARP_SIZE);
            if (other_val > local_max || (other_val == local_max && other_expert < local_expert)) {
                local_max = other_val;
                local_expert = other_expert;
            }
        }

        if (lane == 0) {
            ids_out[k] = (uint32_t)local_expert;
            weights_out[k] = local_max;
            wt_sum += local_max;
        }

        // Every lane holds the same `local_expert` after the all-reduce, so
        // only the lane owning that slot clears it for the next round.
        if ((local_expert % WARP_SIZE) == lane) {
            wt[local_expert / WARP_SIZE] = -INFINITY;
        }
    }

    if (norm_topk && lane == 0) {
        const float denom = wt_sum > 0.f ? wt_sum : 1.0f;
        const float inv = 1.0f / denom;
        for (int k = 0; k < top_k; k++) {
            weights_out[k] *= inv;
        }
    }
}
