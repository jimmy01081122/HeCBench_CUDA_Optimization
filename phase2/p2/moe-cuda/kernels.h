#include <float.h>
#include <cub/cub.cuh>
#include <cub/util_type.cuh>

struct Sum
{
  template <typename T, typename U>
  __device__ __forceinline__ auto operator()(T &&t, U &&u) const
    -> decltype(std::forward<T>(t) + std::forward<U>(u))
  {
    return std::forward<T>(t) + std::forward<U>(u);
  }
};

struct Max
{
  template <typename T, typename U>
  __device__  __forceinline__
  typename std::common_type<T, U>::type
    operator()(T &&t, U &&u) const
  {
    return ((t) > (u)) ? (t) : (u);
  }
};

template <int TPB>
__global__ void moeSoftmax(
    const float* __restrict__ input,
    const bool* __restrict__ finished,
    float* output,
    const int num_cols)
{
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;

  const int thread_row_offset = blockIdx.x * num_cols;

  float threadData(-FLT_MAX);

  // Don't touch finished rows.
  if ((finished != nullptr) && finished[blockIdx.x]) {
    return;
  }

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData = fmaxf(static_cast<float>(input[idx]), threadData);
  }

  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, Max());

  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += expf((static_cast<float>(input[idx]) - float_max));
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, Sum());

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val = expf((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
    output[idx] = val;
  }
}


template <int TPB>
__global__ void moeTopK(
    const float* __restrict__ inputs_after_softmax,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert)
{
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int num_tokens = gridDim.x; // number of tokens
  const int token = blockIdx.x;
  const int tid = threadIdx.x;

  const bool row_is_active = finished ? !finished[token] : true;
  const int thread_read_offset = token * num_experts;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = -1.f;  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = tid; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * token + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (tid == 0) {
      // Ignore experts the node isn't responsible for with expert parallelism
      const int expert = result_kvp.key;
      const bool node_uses_expert = expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * token + k_idx;
      output[idx] = result_kvp.value;
      indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      source_rows[idx] = k_idx * num_tokens + token;
    }
    __syncthreads();
  }
}

template <int TPB>
__global__ void moeFusedTop1Softmax(
    const float* __restrict__ input,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_experts,
    const int start_expert,
    const int end_expert)
{
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduceFloat = cub::BlockReduce<float, TPB>;
  using BlockReduceKvp = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduceFloat::TempStorage floatStorage;
  __shared__ typename BlockReduceKvp::TempStorage kvpStorage;
  __shared__ float row_max;
  __shared__ float inv_sum;

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = token * num_experts;
  const bool row_is_active = finished ? !finished[token] : true;

  float thread_max = -FLT_MAX;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    thread_max = fmaxf(thread_max, input[row_offset + expert]);
  }

  const float max_val = BlockReduceFloat(floatStorage).Reduce(thread_max, Max());
  if (tid == 0) {
    row_max = max_val;
  }
  __syncthreads();

  float thread_sum = 0.0f;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    thread_sum += expf(input[row_offset + expert] - row_max);
  }

  const float sum_val = BlockReduceFloat(floatStorage).Reduce(thread_sum, Sum());
  if (tid == 0) {
    inv_sum = 1.0f / sum_val;
  }
  __syncthreads();

  cub_kvp thread_kvp;
  thread_kvp.key = 0;
  thread_kvp.value = -1.0f;
  cub::ArgMax arg_max;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    cub_kvp inp_kvp;
    inp_kvp.key = expert;
    inp_kvp.value = expf(input[row_offset + expert] - row_max) * inv_sum;
    thread_kvp = arg_max(inp_kvp, thread_kvp);
  }

  const cub_kvp result_kvp = BlockReduceKvp(kvpStorage).Reduce(thread_kvp, arg_max);
  if (tid == 0) {
    const int expert = result_kvp.key;
    const bool node_uses_expert = expert >= start_expert && expert < end_expert;
    const bool should_process_row = row_is_active && node_uses_expert;
    output[token] = result_kvp.value;
    indices[token] = should_process_row ? (expert - start_expert) : num_experts;
    assert(indices[token] >= 0);
    source_rows[token] = token;
  }
}

template <int TPB, int MAX_K>
__global__ void moeFusedTopKSoftmax(
    const float* __restrict__ input,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert)
{
  using BlockReduceFloat = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduceFloat::TempStorage floatStorage;
  __shared__ float shared_vals[TPB * MAX_K];
  __shared__ int shared_keys[TPB * MAX_K];
  __shared__ float row_max;
  __shared__ float inv_sum;

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = token * num_experts;
  const bool row_is_active = finished ? !finished[token] : true;

  float thread_max = -FLT_MAX;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    thread_max = fmaxf(thread_max, input[row_offset + expert]);
  }

  const float max_val = BlockReduceFloat(floatStorage).Reduce(thread_max, Max());
  if (tid == 0) {
    row_max = max_val;
  }
  __syncthreads();

  float thread_sum = 0.0f;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    const float val = input[row_offset + expert];
    thread_sum += expf(val - row_max);
  }

  const float sum_val = BlockReduceFloat(floatStorage).Reduce(thread_sum, Sum());
  if (tid == 0) {
    inv_sum = 1.0f / sum_val;
  }
  __syncthreads();

  float local_vals[MAX_K];
  int local_keys[MAX_K];
#pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    local_vals[i] = -1.0f;
    local_keys[i] = 0;
  }

  for (int expert = tid; expert < num_experts; expert += TPB) {
    const float val = expf(input[row_offset + expert] - row_max) * inv_sum;
#pragma unroll
    for (int slot = 0; slot < MAX_K; ++slot) {
      if (slot >= k) {
        break;
      }
      if (val > local_vals[slot] || (val == local_vals[slot] && expert < local_keys[slot])) {
#pragma unroll
        for (int move = MAX_K - 1; move > slot; --move) {
          if (move < k) {
            local_vals[move] = local_vals[move - 1];
            local_keys[move] = local_keys[move - 1];
          }
        }
        local_vals[slot] = val;
        local_keys[slot] = expert;
        break;
      }
    }
  }

#pragma unroll
  for (int slot = 0; slot < MAX_K; ++slot) {
    if (slot < k) {
      const int idx = tid * MAX_K + slot;
      shared_vals[idx] = local_vals[slot];
      shared_keys[idx] = local_keys[slot];
    }
  }
  __syncthreads();

  if (tid == 0) {
    float best_vals[MAX_K];
    int best_keys[MAX_K];
#pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
      best_vals[i] = -FLT_MAX;
      best_keys[i] = 0;
    }

    for (int item = 0; item < TPB * MAX_K; ++item) {
      const int slot_limit = item % MAX_K;
      if (slot_limit >= k) {
        continue;
      }
      const float val = shared_vals[item];
      const int expert = shared_keys[item];
#pragma unroll
      for (int slot = 0; slot < MAX_K; ++slot) {
        if (slot >= k) {
          break;
        }
        if (val > best_vals[slot] || (val == best_vals[slot] && expert < best_keys[slot])) {
#pragma unroll
          for (int move = MAX_K - 1; move > slot; --move) {
            if (move < k) {
              best_vals[move] = best_vals[move - 1];
              best_keys[move] = best_keys[move - 1];
            }
          }
          best_vals[slot] = val;
          best_keys[slot] = expert;
          break;
        }
      }
    }

#pragma unroll
    for (int k_idx = 0; k_idx < MAX_K; ++k_idx) {
      if (k_idx < k) {
        const int expert = best_keys[k_idx];
        const bool node_uses_expert = expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;
        const int idx = token * k + k_idx;
        output[idx] = best_vals[k_idx];
        indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
        assert(indices[idx] >= 0);
        source_rows[idx] = k_idx * gridDim.x + token;
      }
    }
  }
}
