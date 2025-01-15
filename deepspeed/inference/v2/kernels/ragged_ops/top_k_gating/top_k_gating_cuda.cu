// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"
#include "top_k_gating.cuh"
#include "top_k_utils.h"

using ROp = reduce::ROpType;

template <typename T, int TOP_K>
__global__ void top_k_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    const T* logits,
                                    const RaggedBatchDescriptor* batch_metadata,
                                    const int32_t n_experts)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= batch_metadata->n_tokens) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                assignments[token_idx * TOP_K + i] = gating::unassigned;
                offsets[token_idx * TOP_K + i] = gating::unassigned;
            }
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_experts - expert_idx - 1;

    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val); }
    }

    const float max_logit = local_assigned_logits[0];
    float softmax_sum = __expf(logit_val - max_logit);
    reduce::block<ROp::Add>(tb, warp, softmax_sum);

    for (int i = 0; i < TOP_K; ++i) {
        const float softmax = __expf(local_assigned_logits[i] - max_logit) / softmax_sum;

        if (threadIdx.x == 0) {
            scores[token_idx * TOP_K + i] = softmax;
            assignments[token_idx * TOP_K + i] = 0; //local_assigned_experts[i];
            offsets[token_idx * TOP_K + i] =
                atomicAdd(expert_counts + local_assigned_experts[i], 1);
        }
    }
}

template <typename T>
void launch_top_k_gating(int32_t* expert_counts,
                         float* scores,
                         int32_t* assignments,
                         int32_t* offsets,
                         const T* logits,
                         const RaggedBatchDescriptor* batch_metadata,
                         const int32_t n_tokens,
                         const int32_t n_experts,
                         const int32_t n_top_k,
                         cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts + hw_warp_size - 1) / hw_warp_size) * hw_warp_size);

    TOP_K_SWITCH(n_top_k, [&] {
        top_k_gating_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
            expert_counts, scores, assignments, offsets, logits, batch_metadata, n_experts);
    });
}

#define INSTANTIATE_top_k_KERNEL(T)                                                   \
    template void launch_top_k_gating<T>(int32_t * expert_counts,                     \
                                         float* scores,                               \
                                         int32_t* assignments,                        \
                                         int32_t* offsets,                            \
                                         const T* logits,                             \
                                         const RaggedBatchDescriptor* batch_metadata, \
                                         const int32_t n_tokens,                      \
                                         const int32_t n_experts,                     \
                                         const int32_t n_top_k,                       \
                                         cudaStream_t stream);

INSTANTIATE_top_k_KERNEL(float) INSTANTIATE_top_k_KERNEL(__half)
#ifdef BF16_AVAILABLE
    INSTANTIATE_top_k_KERNEL(__nv_bfloat16)
#endif


template <typename T, int TOP_K>
__global__ void grouped_top_k_gating_kernel1(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    int32_t *mapped_expert_ids,
                                    T* logits,
                                    const int32_t n_tokens,
                                    const int32_t n_experts,
                                    const int32_t n_top_k_group,
                                    const int32_t num_local_experts)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t local_expert_idx = threadIdx.x;
    const int32_t expert_idx = threadIdx.x + threadIdx.y * num_local_experts;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= n_tokens) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
#pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                assignments[token_idx * TOP_K + i] = gating::unassigned;
                offsets[token_idx * TOP_K + i] = gating::unassigned;
            }
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts; 
    float logit_val;
    if (local_expert_idx < num_local_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;
    float reduce_val2;
    reduce::init<ROp::Max>(&reduce_val2);

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = num_local_experts - threadIdx.x - 1;

    for (int i = 0; i < n_top_k_group; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, 1>(tb, warp, reduce_val, inverted_expert);

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == num_local_experts - res.idx - 1) { 
            reduce_val2 = reduce_val;
            reduce::init<ROp::Max>(&reduce_val); 
        }
    }
    inverted_expert = n_experts - expert_idx - 1;
    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val2, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        local_assigned_experts[i] = mapped_expert_ids[local_assigned_experts[i]];
        // if (threadIdx.x == 0 && threadIdx.y == 0) {
        //     if (local_assigned_experts[i] < 0)
        //         printf("****************** [DEEPSPEED]: selecting %d expert with mapped id %d ***************** \n", n_experts - res.idx - 1, local_assigned_experts[i]);
        //     assert(local_assigned_experts[i] >= 0);
        // }
        // Set the max logit to -inf so that it is not selected again
        if (expert_idx == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val2); }
        if (local_assigned_experts[i] < 0) {
            i -= 1;
            continue;
        }
    }

    const float max_logit = local_assigned_logits[0];
    float softmax_sum = __expf(logit_val - max_logit);
    reduce::block<ROp::Add>(tb, warp, softmax_sum);

    for (int i = 0; i < TOP_K; ++i) {
        const float softmax = __expf(local_assigned_logits[i] - max_logit) / softmax_sum;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            scores[token_idx * TOP_K + i] = softmax;
            assignments[token_idx * TOP_K + i] = local_assigned_experts[i];
            offsets[token_idx * TOP_K + i] =
                atomicAdd(expert_counts + local_assigned_experts[i], 1);
        }
    }
}


template <typename T, int TOP_K>
__global__ void grouped_top_k_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    T* logits,
                                    const int32_t n_tokens,
                                    const int32_t n_experts,
                                    const int32_t n_top_k_group,
                                    const int32_t num_local_experts)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t local_expert_idx = threadIdx.x;
    const int32_t expert_idx = threadIdx.x + threadIdx.y * num_local_experts;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= n_tokens) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
#pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                assignments[token_idx * TOP_K + i] = gating::unassigned;
                offsets[token_idx * TOP_K + i] = gating::unassigned;
            }
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts;

    float logit_val;
    if (local_expert_idx < num_local_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;
    float reduce_val2;
    reduce::init<ROp::Max>(&reduce_val2);

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = num_local_experts - threadIdx.x - 1;

    for (int i = 0; i < n_top_k_group; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, 1>(tb, warp, reduce_val, inverted_expert);

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == num_local_experts - res.idx - 1) { 
            reduce_val2 = reduce_val;
            reduce::init<ROp::Max>(&reduce_val); 
        }
    }
    inverted_expert = n_experts - expert_idx - 1;
    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val2, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (expert_idx == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val2); }
    }

    const float max_logit = local_assigned_logits[0];
    float softmax_sum = __expf(logit_val - max_logit);
    reduce::block<ROp::Add>(tb, warp, softmax_sum);

    for (int i = 0; i < TOP_K; ++i) {
        const float softmax = __expf(local_assigned_logits[i] - max_logit) / softmax_sum;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            scores[token_idx * TOP_K + i] = softmax;
            assignments[token_idx * TOP_K + i] = local_assigned_experts[i];
            offsets[token_idx * TOP_K + i] =
                atomicAdd(expert_counts + local_assigned_experts[i], 1);
        }
    }
}

template <typename T>
void launch_grouped_top_k_gating(int32_t* expert_counts,
                         float* scores,
                         int32_t* assignments,
                         int32_t* offsets,
                         int32_t* expert_mapping,
                         T* logits,
                         const int32_t n_tokens,
                         const int32_t n_groups,
                         const int32_t n_experts,
                         const int32_t n_top_k,
                         const int32_t n_top_k_group,
                         cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    auto num_local_experts = n_experts / n_groups;
    const dim3 block(((num_local_experts + hw_warp_size - 1) / hw_warp_size) * hw_warp_size, n_groups);
    // printf("num_local_experts: %d, block: (%d, %d)!\n", num_local_experts, block.x, block.y);
    TOP_K_SWITCH(n_top_k, [&] {
        grouped_top_k_gating_kernel1<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
            expert_counts, scores, assignments, offsets, 
            expert_mapping,logits, n_tokens, n_experts, n_top_k_group, num_local_experts);
    });
}

#define INSTANTIATE_grouped_top_k_KERNEL(T)                                                   \
    template void launch_grouped_top_k_gating<T>(int32_t * expert_counts,                     \
                                         float* scores,                               \
                                         int32_t* assignments,                        \
                                         int32_t* offsets,                            \
                                         int32_t* expert_mapping,                     \
                                         T* logits,                             \
                                         const int32_t n_tokens,                      \
                                         const int32_t n_groups,                      \
                                         const int32_t n_experts,                     \
                                         const int32_t n_top_k,                       \
                                         const int32_t n_top_k_group,                       \
                                         cudaStream_t stream);

INSTANTIATE_grouped_top_k_KERNEL(float); 
INSTANTIATE_grouped_top_k_KERNEL(__half);
#ifdef BF16_AVAILABLE
    INSTANTIATE_grouped_top_k_KERNEL(__nv_bfloat16);
#endif
