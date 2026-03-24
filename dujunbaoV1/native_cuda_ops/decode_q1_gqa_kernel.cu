#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>
#include <vector>

namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxCacheLen = 1024;
constexpr int kDecodeThreads = 32;
constexpr int kDecodeVec = 4;
constexpr int kRmsThreads = 256;
constexpr int kPointwiseThreads = 256;
constexpr int kLmHeadThreads = 256;
constexpr int kLmHeadTile = 128;
constexpr int kDualLinearThreads = 256;
constexpr int kDualLinearWarpsPerBlock = kDualLinearThreads / 32;
constexpr int kQkvLinearThreads = 256;
constexpr int kQkvWarpsPerBlock = kQkvLinearThreads / 32;
constexpr int kQkNormThreads = 128;

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    const int warp_count = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < warp_count) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void decode_q1_gqa_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int visible_len,
    float softmax_scale) {
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int lane = threadIdx.x;
    const int group_size = num_heads / num_kv_heads;
    const int kv_head = head_idx / group_size;
    const int dim_base = lane * kDecodeVec;

    __shared__ float scores[kMaxCacheLen];
    __shared__ float max_score;
    __shared__ float denom;

    float q_reg[kDecodeVec];
    const int q_base = (batch_idx * num_heads + head_idx) * kHeadDim + dim_base;
#pragma unroll
    for (int i = 0; i < kDecodeVec; ++i) {
        q_reg[i] = __half2float(q[q_base + i]);
    }

    for (int token_idx = 0; token_idx < visible_len; ++token_idx) {
        const int kv_base = ((batch_idx * num_kv_heads + kv_head) * seq_len + token_idx) * kHeadDim + dim_base;
        float local_dot = 0.0f;
#pragma unroll
        for (int i = 0; i < kDecodeVec; ++i) {
            local_dot += q_reg[i] * __half2float(k[kv_base + i]);
        }
        const float dot = warp_reduce_sum(local_dot);
        if (lane == 0) {
            scores[token_idx] = dot * softmax_scale;
        }
    }
    __syncthreads();

    if (lane == 0) {
        max_score = -1.0e20f;
        for (int token_idx = 0; token_idx < visible_len; ++token_idx) {
            max_score = fmaxf(max_score, scores[token_idx]);
        }

        denom = 0.0f;
        for (int token_idx = 0; token_idx < visible_len; ++token_idx) {
            const float weight = __expf(scores[token_idx] - max_score);
            scores[token_idx] = weight;
            denom += weight;
        }
        denom = fmaxf(denom, 1e-6f);
    }
    __syncthreads();

    float acc[kDecodeVec] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int token_idx = 0; token_idx < visible_len; ++token_idx) {
        const int kv_base = ((batch_idx * num_kv_heads + kv_head) * seq_len + token_idx) * kHeadDim + dim_base;
        const float weight = scores[token_idx] / denom;
#pragma unroll
        for (int i = 0; i < kDecodeVec; ++i) {
            acc[i] += weight * __half2float(v[kv_base + i]);
        }
    }

#pragma unroll
    for (int i = 0; i < kDecodeVec; ++i) {
        out[q_base + i] = __float2half_rn(acc[i]);
    }
}

__global__ void rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_size,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const half* row_input = input + row * hidden_size;
    half* row_output = output + row * hidden_size;

    float sum_sq = 0.0f;
    for (int idx = tid; idx < hidden_size; idx += blockDim.x) {
        const float value = __half2float(row_input[idx]);
        sum_sq += value * value;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();

    for (int idx = tid; idx < hidden_size; idx += blockDim.x) {
        const float value = __half2float(row_input[idx]);
        const float gamma = __half2float(weight[idx]);
        row_output[idx] = __float2half_rn(value * inv_rms * gamma);
    }
}

__global__ void silu_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int64_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const float gate_value = __half2float(gate[idx]);
    const float up_value = __half2float(up[idx]);
    const float silu_value = gate_value / (1.0f + __expf(-gate_value));
    output[idx] = __float2half_rn(silu_value * up_value);
}

__global__ void gate_up_silu_matvec_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weight,
    const half* __restrict__ up_weight,
    half* __restrict__ output,
    int rows,
    int hidden_size,
    int out_features) {
    extern __shared__ half2 shared_input[];

    const int row = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_row = blockIdx.x * kDualLinearWarpsPerBlock + warp_id;
    const int hidden_half2 = hidden_size >> 1;

    const half* input_row = input + static_cast<int64_t>(row) * hidden_size;
    const half2* input_row_half2 = reinterpret_cast<const half2*>(input_row);
    for (int idx = threadIdx.x; idx < hidden_half2; idx += blockDim.x) {
        shared_input[idx] = input_row_half2[idx];
    }
    __syncthreads();

    if (row >= rows || out_row >= out_features) {
        return;
    }

    const half2* gate_weight_row = reinterpret_cast<const half2*>(gate_weight + static_cast<int64_t>(out_row) * hidden_size);
    const half2* up_weight_row = reinterpret_cast<const half2*>(up_weight + static_cast<int64_t>(out_row) * hidden_size);

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int idx = lane; idx < hidden_half2; idx += 32) {
        const float2 input_vec = __half22float2(shared_input[idx]);
        const float2 gate_weight_vec = __half22float2(gate_weight_row[idx]);
        const float2 up_weight_vec = __half22float2(up_weight_row[idx]);
        gate_acc += gate_weight_vec.x * input_vec.x + gate_weight_vec.y * input_vec.y;
        up_acc += up_weight_vec.x * input_vec.x + up_weight_vec.y * input_vec.y;
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc = warp_reduce_sum(up_acc);
    if (lane == 0) {
        const float silu_gate = gate_acc / (1.0f + __expf(-gate_acc));
        output[static_cast<int64_t>(row) * out_features + out_row] = __float2half_rn(silu_gate * up_acc);
    }
}

__global__ void dual_linear_matvec_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight0,
    const half* __restrict__ weight1,
    half* __restrict__ output0,
    half* __restrict__ output1,
    int rows,
    int hidden_size,
    int out_features) {
    extern __shared__ half2 shared_input[];

    const int row = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_row = blockIdx.x * kDualLinearWarpsPerBlock + warp_id;
    const int hidden_half2 = hidden_size >> 1;

    const half* input_row = input + static_cast<int64_t>(row) * hidden_size;
    const half2* input_row_half2 = reinterpret_cast<const half2*>(input_row);
    for (int idx = threadIdx.x; idx < hidden_half2; idx += blockDim.x) {
        shared_input[idx] = input_row_half2[idx];
    }
    __syncthreads();

    if (row >= rows || out_row >= out_features) {
        return;
    }

    const half2* weight0_row = reinterpret_cast<const half2*>(weight0 + static_cast<int64_t>(out_row) * hidden_size);
    const half2* weight1_row = reinterpret_cast<const half2*>(weight1 + static_cast<int64_t>(out_row) * hidden_size);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int idx = lane; idx < hidden_half2; idx += 32) {
        const float2 input_vec = __half22float2(shared_input[idx]);
        const float2 weight0_vec = __half22float2(weight0_row[idx]);
        const float2 weight1_vec = __half22float2(weight1_row[idx]);
        acc0 += weight0_vec.x * input_vec.x + weight0_vec.y * input_vec.y;
        acc1 += weight1_vec.x * input_vec.x + weight1_vec.y * input_vec.y;
    }

    acc0 = warp_reduce_sum(acc0);
    acc1 = warp_reduce_sum(acc1);
    if (lane == 0) {
        output0[static_cast<int64_t>(row) * out_features + out_row] = __float2half_rn(acc0);
        output1[static_cast<int64_t>(row) * out_features + out_row] = __float2half_rn(acc1);
    }
}

__global__ void qkv_linear_matvec_kernel(
    const half* __restrict__ input,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ v_weight,
    half* __restrict__ q_output,
    half* __restrict__ k_output,
    half* __restrict__ v_output,
    int rows,
    int hidden_size,
    int q_out_features,
    int k_out_features,
    int v_out_features) {
    extern __shared__ half2 shared_input[];

    const int row = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int total_out_features = q_out_features + k_out_features + v_out_features;
    const int out_row = blockIdx.x * kQkvWarpsPerBlock + warp_id;
    const int hidden_half2 = hidden_size >> 1;

    const half* input_row = input + static_cast<int64_t>(row) * hidden_size;
    const half2* input_row_half2 = reinterpret_cast<const half2*>(input_row);
    for (int idx = threadIdx.x; idx < hidden_half2; idx += blockDim.x) {
        shared_input[idx] = input_row_half2[idx];
    }
    __syncthreads();

    if (row >= rows || out_row >= total_out_features) {
        return;
    }

    const half2* weight_row = nullptr;
    half* output_row = nullptr;
    int output_idx = out_row;
    if (out_row < q_out_features) {
        weight_row = reinterpret_cast<const half2*>(q_weight + static_cast<int64_t>(out_row) * hidden_size);
        output_row = q_output + static_cast<int64_t>(row) * q_out_features;
    } else if (out_row < q_out_features + k_out_features) {
        output_idx = out_row - q_out_features;
        weight_row = reinterpret_cast<const half2*>(k_weight + static_cast<int64_t>(output_idx) * hidden_size);
        output_row = k_output + static_cast<int64_t>(row) * k_out_features;
    } else {
        output_idx = out_row - q_out_features - k_out_features;
        weight_row = reinterpret_cast<const half2*>(v_weight + static_cast<int64_t>(output_idx) * hidden_size);
        output_row = v_output + static_cast<int64_t>(row) * v_out_features;
    }

    float acc = 0.0f;
    for (int idx = lane; idx < hidden_half2; idx += 32) {
        const float2 input_vec = __half22float2(shared_input[idx]);
        const float2 weight_vec = __half22float2(weight_row[idx]);
        acc += weight_vec.x * input_vec.x + weight_vec.y * input_vec.y;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        output_row[output_idx] = __float2half_rn(acc);
    }
}

__global__ void qk_head_rmsnorm_inplace_kernel(
    half* __restrict__ q_output,
    half* __restrict__ k_output,
    const half* __restrict__ q_norm_weight,
    const half* __restrict__ k_norm_weight,
    int rows,
    int q_num_heads,
    int k_num_heads,
    int head_dim,
    float eps) {
    const int row = blockIdx.y;
    const int packed_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const bool is_q_head = packed_head_idx < q_num_heads;
    const int head_idx = is_q_head ? packed_head_idx : packed_head_idx - q_num_heads;
    const int num_heads = is_q_head ? q_num_heads : k_num_heads;
    half* output_ptr = is_q_head ? q_output : k_output;
    const half* norm_weight = is_q_head ? q_norm_weight : k_norm_weight;
    const int64_t row_offset = (static_cast<int64_t>(row) * num_heads + head_idx) * head_dim;

    float value = 0.0f;
    if (tid < head_dim) {
        value = __half2float(output_ptr[row_offset + tid]);
    }
    float sum_sq = (tid < head_dim) ? value * value : 0.0f;
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    if (tid < head_dim) {
        const float gamma = __half2float(norm_weight[tid]);
        output_ptr[row_offset + tid] = __float2half_rn(value * inv_rms * gamma);
    }
}

__global__ void packed_qkv_qk_norm_rope_layout_kernel(
    const half* __restrict__ packed_qkv,
    const half* __restrict__ q_norm_weight,
    const half* __restrict__ k_norm_weight,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    int rows,
    int batch_size,
    int seq_len,
    int q_num_heads,
    int k_num_heads,
    int v_num_heads,
    int head_dim,
    int total_out_features,
    float eps) {
    const int packed_head_idx = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int q_features = q_num_heads * head_dim;
    const int k_features = k_num_heads * head_dim;
    const int v_features = v_num_heads * head_dim;
    if (q_features + k_features + v_features != total_out_features) {
        return;
    }

    const int half_dim = head_dim >> 1;
    const int b = row / seq_len;
    const int s = row - b * seq_len;
    if (b >= batch_size) {
        return;
    }
    const half* row_ptr = packed_qkv + static_cast<int64_t>(row) * total_out_features;
    const half* cos_row = cos + static_cast<int64_t>(row) * head_dim;
    const half* sin_row = sin + static_cast<int64_t>(row) * head_dim;

    __shared__ float inv_rms;
    float value = 0.0f;
    float rotated = 0.0f;

    if (packed_head_idx < q_num_heads + k_num_heads) {
        const bool is_q_head = packed_head_idx < q_num_heads;
        const int head_idx = is_q_head ? packed_head_idx : packed_head_idx - q_num_heads;
        const int base = is_q_head ? head_idx * head_dim : q_features + head_idx * head_dim;
        const half* norm_weight = is_q_head ? q_norm_weight : k_norm_weight;

        if (tid < head_dim) {
            value = __half2float(row_ptr[base + tid]);
        }
        float sum_sq = (tid < head_dim) ? value * value : 0.0f;
        sum_sq = block_reduce_sum(sum_sq);

        if (threadIdx.x == 0) {
            inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + eps);
        }
        __syncthreads();

        if (tid < head_dim) {
            const float gamma = __half2float(norm_weight[tid]);
            const float norm_value = value * inv_rms * gamma;
            const float cos_v = __half2float(cos_row[tid]);
            const float sin_v = __half2float(sin_row[tid]);

            if (tid < half_dim) {
                const float pair_raw = __half2float(row_ptr[base + tid + half_dim]);
                const float pair_gamma = __half2float(norm_weight[tid + half_dim]);
                const float pair_norm = pair_raw * inv_rms * pair_gamma;
                rotated = norm_value * cos_v - pair_norm * sin_v;
            } else {
                const float pair_raw = __half2float(row_ptr[base + tid - half_dim]);
                const float pair_gamma = __half2float(norm_weight[tid - half_dim]);
                const float pair_norm = pair_raw * inv_rms * pair_gamma;
                rotated = norm_value * cos_v + pair_norm * sin_v;
            }

            if (is_q_head) {
                const int64_t out_offset =
                    ((static_cast<int64_t>(b) * q_num_heads + head_idx) * seq_len + s) * head_dim + tid;
                q_out[out_offset] = __float2half_rn(rotated);
            } else {
                const int64_t out_offset =
                    ((static_cast<int64_t>(b) * k_num_heads + head_idx) * seq_len + s) * head_dim + tid;
                k_out[out_offset] = __float2half_rn(rotated);
            }
        }
        return;
    }

    const int v_head_idx = packed_head_idx - q_num_heads - k_num_heads;
    if (v_head_idx >= v_num_heads) {
        return;
    }
    const int v_base = q_features + k_features + v_head_idx * head_dim;
    if (tid < head_dim) {
        const int64_t out_offset =
            ((static_cast<int64_t>(b) * v_num_heads + v_head_idx) * seq_len + s) * head_dim + tid;
        v_out[out_offset] = row_ptr[v_base + tid];
    }
}

__global__ void lm_head_argmax_stage1_kernel(
    const half* __restrict__ hidden,
    const half* __restrict__ weight,
    float* __restrict__ partial_values,
    int32_t* __restrict__ partial_indices,
    int vocab_size,
    int hidden_size) {
    __shared__ half hidden_tile[kLmHeadTile];
    __shared__ float reduce_values[kLmHeadThreads];
    __shared__ int reduce_indices[kLmHeadThreads];

    const int vocab_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = -std::numeric_limits<float>::infinity();
    if (vocab_idx < vocab_size) {
        acc = 0.0f;
        const half* weight_row = weight + static_cast<int64_t>(vocab_idx) * hidden_size;
        for (int tile_start = 0; tile_start < hidden_size; tile_start += kLmHeadTile) {
            if (threadIdx.x < kLmHeadTile) {
                hidden_tile[threadIdx.x] = hidden[tile_start + threadIdx.x];
            }
            __syncthreads();

#pragma unroll
            for (int tile_offset = 0; tile_offset < kLmHeadTile; ++tile_offset) {
                acc += __half2float(weight_row[tile_start + tile_offset]) * __half2float(hidden_tile[tile_offset]);
            }
            __syncthreads();
        }
    }

    reduce_values[threadIdx.x] = acc;
    reduce_indices[threadIdx.x] = vocab_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value = reduce_values[threadIdx.x + stride];
            const int other_index = reduce_indices[threadIdx.x + stride];
            if (other_value > reduce_values[threadIdx.x]) {
                reduce_values[threadIdx.x] = other_value;
                reduce_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_values[blockIdx.x] = reduce_values[0];
        partial_indices[blockIdx.x] = reduce_indices[0];
    }
}

__global__ void lm_head_argmax_stage2_kernel(
    const float* __restrict__ partial_values,
    const int32_t* __restrict__ partial_indices,
    int64_t* __restrict__ output_index,
    int partial_count) {
    __shared__ float reduce_values[kLmHeadThreads];
    __shared__ int reduce_indices[kLmHeadThreads];
    const int tid = threadIdx.x;

    float value = -std::numeric_limits<float>::infinity();
    int index = 0;
    for (int idx = tid; idx < partial_count; idx += blockDim.x) {
        const float candidate_value = partial_values[idx];
        const int candidate_index = partial_indices[idx];
        if (candidate_value > value) {
            value = candidate_value;
            index = candidate_index;
        }
    }

    reduce_values[tid] = value;
    reduce_indices[tid] = index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_value = reduce_values[tid + stride];
            const int other_index = reduce_indices[tid + stride];
            if (other_value > reduce_values[tid]) {
                reduce_values[tid] = other_value;
                reduce_indices[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_index[0] = static_cast<int64_t>(reduce_indices[0]);
    }
}

}  // namespace

torch::Tensor decode_q1_gqa_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t visible_len,
    double softmax_scale) {
    const auto batch_size = static_cast<int>(query.size(0));
    const auto num_heads = static_cast<int>(query.size(1));
    const auto head_dim = static_cast<int>(query.size(2));
    const auto num_kv_heads = static_cast<int>(key_cache.size(1));
    const auto seq_len = static_cast<int>(key_cache.size(2));

    TORCH_CHECK(head_dim == kHeadDim, "native decode attention currently only supports head_dim=128");
    TORCH_CHECK(seq_len <= kMaxCacheLen, "native decode attention cache is capped at 1024 tokens");
    TORCH_CHECK(visible_len <= seq_len, "visible_len exceeds allocated cache length");

    auto output = torch::empty_like(query);

    const dim3 grid(num_heads, batch_size);
    const dim3 block(kDecodeThreads);
    auto stream = at::cuda::getDefaultCUDAStream();
    decode_q1_gqa_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        num_heads,
        num_kv_heads,
        seq_len,
        static_cast<int>(visible_len),
        static_cast<float>(softmax_scale));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "decode_q1_gqa_kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
    const auto hidden_size = static_cast<int>(input.size(-1));
    const auto rows = static_cast<int>(input.numel() / hidden_size);
    auto output = torch::empty_like(input);

    const dim3 grid(rows);
    const dim3 block(kRmsThreads);
    auto stream = at::cuda::getDefaultCUDAStream();
    rmsnorm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        hidden_size,
        static_cast<float>(eps));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rmsnorm_kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor silu_mul_cuda(torch::Tensor gate, torch::Tensor up) {
    const auto numel = gate.numel();
    auto output = torch::empty_like(gate);

    const int blocks = static_cast<int>((numel + kPointwiseThreads - 1) / kPointwiseThreads);
    const dim3 grid(blocks);
    const dim3 block(kPointwiseThreads);
    auto stream = at::cuda::getDefaultCUDAStream();
    silu_mul_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int64_t>(numel));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "silu_mul_kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor gate_up_silu_cuda(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight) {
    const auto hidden_size = static_cast<int>(input.size(-1));
    const auto rows = static_cast<int>(input.numel() / hidden_size);
    const auto out_features = static_cast<int>(gate_weight.size(0));
    TORCH_CHECK(hidden_size % 2 == 0, "gate_up_silu_cuda requires an even hidden_size");

    auto input_2d = input.view({rows, hidden_size});
    auto output_sizes = input.sizes().vec();
    output_sizes.back() = out_features;
    auto output = torch::empty(output_sizes, input.options());

    const dim3 grid((out_features + kDualLinearWarpsPerBlock - 1) / kDualLinearWarpsPerBlock, rows);
    const dim3 block(kDualLinearThreads);
    const size_t shared_bytes = static_cast<size_t>(hidden_size / 2) * sizeof(half2);
    auto stream = at::cuda::getCurrentCUDAStream();
    gate_up_silu_matvec_kernel<<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<const half*>(input_2d.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gate_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up_weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        rows,
        hidden_size,
        out_features);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gate_up_silu_matvec_kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

std::vector<torch::Tensor> dual_linear_cuda(
    torch::Tensor input,
    torch::Tensor weight0,
    torch::Tensor weight1) {
    const auto hidden_size = static_cast<int>(input.size(-1));
    const auto rows = static_cast<int>(input.numel() / hidden_size);
    const auto out_features = static_cast<int>(weight0.size(0));
    TORCH_CHECK(hidden_size % 2 == 0, "dual_linear_cuda requires an even hidden_size");

    auto input_2d = input.view({rows, hidden_size});
    auto output_sizes = input.sizes().vec();
    output_sizes.back() = out_features;
    auto output0 = torch::empty(output_sizes, input.options());
    auto output1 = torch::empty(output_sizes, input.options());

    const dim3 grid((out_features + kDualLinearWarpsPerBlock - 1) / kDualLinearWarpsPerBlock, rows);
    const dim3 block(kDualLinearThreads);
    const size_t shared_bytes = static_cast<size_t>(hidden_size / 2) * sizeof(half2);
    auto stream = at::cuda::getCurrentCUDAStream();
    dual_linear_matvec_kernel<<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<const half*>(input_2d.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output0.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output1.data_ptr<at::Half>()),
        rows,
        hidden_size,
        out_features);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dual_linear_matvec_kernel launch failed: ", cudaGetErrorString(err));

    return {output0, output1};
}

std::vector<torch::Tensor> qkv_linear_cuda(
    torch::Tensor input,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight) {
    const auto hidden_size = static_cast<int>(input.size(-1));
    const auto rows = static_cast<int>(input.numel() / hidden_size);
    const auto q_out_features = static_cast<int>(q_weight.size(0));
    const auto k_out_features = static_cast<int>(k_weight.size(0));
    const auto v_out_features = static_cast<int>(v_weight.size(0));
    TORCH_CHECK(hidden_size % 2 == 0, "qkv_linear_cuda requires an even hidden_size");

    auto input_2d = input.view({rows, hidden_size});
    auto q_output_sizes = input.sizes().vec();
    q_output_sizes.back() = q_out_features;
    auto k_output_sizes = input.sizes().vec();
    k_output_sizes.back() = k_out_features;
    auto v_output_sizes = input.sizes().vec();
    v_output_sizes.back() = v_out_features;
    auto q_output = torch::empty(q_output_sizes, input.options());
    auto k_output = torch::empty(k_output_sizes, input.options());
    auto v_output = torch::empty(v_output_sizes, input.options());

    const int total_out_features = q_out_features + k_out_features + v_out_features;
    const dim3 grid((total_out_features + kQkvWarpsPerBlock - 1) / kQkvWarpsPerBlock, rows);
    const dim3 block(kQkvLinearThreads);
    const size_t shared_bytes = static_cast<size_t>(hidden_size / 2) * sizeof(half2);
    auto stream = at::cuda::getCurrentCUDAStream();
    qkv_linear_matvec_kernel<<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<const half*>(input_2d.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(q_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_output.data_ptr<at::Half>()),
        rows,
        hidden_size,
        q_out_features,
        k_out_features,
        v_out_features);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "qkv_linear_matvec_kernel launch failed: ", cudaGetErrorString(err));

    return {q_output, k_output, v_output};
}

std::vector<torch::Tensor> qkv_linear_qk_norm_cuda(
    torch::Tensor input,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    double eps,
    int64_t head_dim) {
    const auto hidden_size = static_cast<int>(input.size(-1));
    const auto rows = static_cast<int>(input.numel() / hidden_size);
    const auto q_out_features = static_cast<int>(q_weight.size(0));
    const auto k_out_features = static_cast<int>(k_weight.size(0));
    const auto v_out_features = static_cast<int>(v_weight.size(0));
    const auto head_dim_int = static_cast<int>(head_dim);
    TORCH_CHECK(hidden_size % 2 == 0, "qkv_linear_qk_norm_cuda requires an even hidden_size");
    TORCH_CHECK(head_dim_int == kHeadDim, "qkv_linear_qk_norm_cuda currently only supports head_dim=128");
    TORCH_CHECK(
        q_out_features % head_dim_int == 0 && k_out_features % head_dim_int == 0,
        "q/k out_features must be divisible by head_dim");
    TORCH_CHECK(
        q_norm_weight.size(0) == head_dim_int && k_norm_weight.size(0) == head_dim_int,
        "q/k norm weights must match head_dim");

    auto input_2d = input.view({rows, hidden_size});
    auto q_output_sizes = input.sizes().vec();
    q_output_sizes.back() = q_out_features;
    auto k_output_sizes = input.sizes().vec();
    k_output_sizes.back() = k_out_features;
    auto v_output_sizes = input.sizes().vec();
    v_output_sizes.back() = v_out_features;
    auto q_output = torch::empty(q_output_sizes, input.options());
    auto k_output = torch::empty(k_output_sizes, input.options());
    auto v_output = torch::empty(v_output_sizes, input.options());

    const int total_out_features = q_out_features + k_out_features + v_out_features;
    const dim3 linear_grid((total_out_features + kQkvWarpsPerBlock - 1) / kQkvWarpsPerBlock, rows);
    const dim3 linear_block(kQkvLinearThreads);
    const size_t shared_bytes = static_cast<size_t>(hidden_size / 2) * sizeof(half2);
    auto stream = at::cuda::getCurrentCUDAStream();
    qkv_linear_matvec_kernel<<<linear_grid, linear_block, shared_bytes, stream>>>(
        reinterpret_cast<const half*>(input_2d.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(q_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_output.data_ptr<at::Half>()),
        rows,
        hidden_size,
        q_out_features,
        k_out_features,
        v_out_features);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "qkv_linear_matvec_kernel launch failed: ", cudaGetErrorString(err));

    const int q_num_heads = q_out_features / head_dim_int;
    const int k_num_heads = k_out_features / head_dim_int;
    const dim3 norm_grid(q_num_heads + k_num_heads, rows);
    const dim3 norm_block(kQkNormThreads);
    qk_head_rmsnorm_inplace_kernel<<<norm_grid, norm_block, 0, stream>>>(
        reinterpret_cast<half*>(q_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_norm_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_norm_weight.data_ptr<at::Half>()),
        rows,
        q_num_heads,
        k_num_heads,
        head_dim_int,
        static_cast<float>(eps));
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "qk_head_rmsnorm_inplace_kernel launch failed: ", cudaGetErrorString(err));

    return {q_output, k_output, v_output};
}

std::vector<torch::Tensor> packed_qkv_qk_norm_rope_cuda(
    torch::Tensor packed_qkv,
    int64_t q_out_features,
    int64_t k_out_features,
    int64_t v_out_features,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    double eps,
    int64_t head_dim) {
    const auto batch_size = static_cast<int>(packed_qkv.size(0));
    const auto seq_len = static_cast<int>(packed_qkv.size(1));
    const auto rows = batch_size * seq_len;
    const auto total_out_features = static_cast<int>(packed_qkv.size(2));
    const auto q_out = static_cast<int>(q_out_features);
    const auto k_out = static_cast<int>(k_out_features);
    const auto v_out = static_cast<int>(v_out_features);
    const auto head_dim_int = static_cast<int>(head_dim);

    TORCH_CHECK(head_dim_int > 0, "head_dim must be positive");
    TORCH_CHECK(head_dim_int <= 256, "head_dim must be <= 256");
    TORCH_CHECK(head_dim_int % 2 == 0, "head_dim must be even for RoPE");
    TORCH_CHECK(q_out + k_out + v_out == total_out_features, "q/k/v out_features must sum to packed_qkv.size(-1)");
    TORCH_CHECK(k_out == v_out, "k/v out_features must match");
    TORCH_CHECK(q_out % head_dim_int == 0, "q_out_features must be divisible by head_dim");
    TORCH_CHECK(k_out % head_dim_int == 0, "k_out_features must be divisible by head_dim");
    TORCH_CHECK(v_out % head_dim_int == 0, "v_out_features must be divisible by head_dim");

    const int q_num_heads = q_out / head_dim_int;
    const int k_num_heads = k_out / head_dim_int;
    const int v_num_heads = v_out / head_dim_int;
    const int total_heads = q_num_heads + k_num_heads + v_num_heads;

    auto q_output = torch::empty({batch_size, q_num_heads, seq_len, head_dim_int}, packed_qkv.options());
    auto k_output = torch::empty({batch_size, k_num_heads, seq_len, head_dim_int}, packed_qkv.options());
    auto v_output = torch::empty({batch_size, v_num_heads, seq_len, head_dim_int}, packed_qkv.options());

    const int block_threads = 256;
    const dim3 grid(total_heads, rows);
    const dim3 block(block_threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    packed_qkv_qk_norm_rope_layout_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(packed_qkv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_norm_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_norm_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<half*>(q_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_output.data_ptr<at::Half>()),
        rows,
        batch_size,
        seq_len,
        q_num_heads,
        k_num_heads,
        v_num_heads,
        head_dim_int,
        total_out_features,
        static_cast<float>(eps));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "packed_qkv_qk_norm_rope_layout_kernel launch failed: ", cudaGetErrorString(err));

    return {q_output, k_output, v_output};
}

torch::Tensor lm_head_argmax_cuda(
    torch::Tensor hidden_states,
    torch::Tensor weight) {
    const auto vocab_size = static_cast<int>(weight.size(0));
    const auto hidden_size = static_cast<int>(weight.size(1));
    TORCH_CHECK(hidden_size % kLmHeadTile == 0, "lm_head_argmax currently requires hidden_size divisible by 128");

    const int blocks = (vocab_size + kLmHeadThreads - 1) / kLmHeadThreads;
    auto partial_values = torch::empty({blocks}, hidden_states.options().dtype(torch::kFloat32));
    auto partial_indices = torch::empty({blocks}, hidden_states.options().dtype(torch::kInt));
    auto output = torch::empty({1, 1}, hidden_states.options().dtype(torch::kLong));

    auto stream = at::cuda::getDefaultCUDAStream();
    lm_head_argmax_stage1_kernel<<<blocks, kLmHeadThreads, 0, stream>>>(
        reinterpret_cast<const half*>(hidden_states.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int32_t>(),
        vocab_size,
        hidden_size);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "lm_head_argmax_stage1_kernel launch failed: ", cudaGetErrorString(err));

    lm_head_argmax_stage2_kernel<<<1, kLmHeadThreads, 0, stream>>>(
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int32_t>(),
        reinterpret_cast<int64_t*>(output.data_ptr<int64_t>()),
        blocks);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "lm_head_argmax_stage2_kernel launch failed: ", cudaGetErrorString(err));

    return output;
}
