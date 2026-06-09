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

struct ExpertScore
{
  int key;
  float value;
};

struct ArgMaxExpert
{
  __device__ __forceinline__ ExpertScore operator()(const ExpertScore &a, const ExpertScore &b) const
  {
    if (a.value > b.value) {
      return a;
    }
    if (a.value < b.value) {
      return b;
    }
    return (a.key <= b.key) ? a : b;
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
__global__ void moeSoftmaxTop1(
    const float* __restrict__ input,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_experts,
    const int start_expert,
    const int end_expert)
{
  using FloatReduce = cub::BlockReduce<float, TPB>;
  using ScoreReduce = cub::BlockReduce<ExpertScore, TPB>;
  __shared__ union {
    typename FloatReduce::TempStorage float_storage;
    typename ScoreReduce::TempStorage score_storage;
  } tmpStorage;

  __shared__ float row_max;
  __shared__ float inv_sum;
  __shared__ int best_expert;

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = token * num_experts;
  const bool row_is_active = finished ? !finished[token] : true;

  ArgMaxExpert arg_max;
  ExpertScore local_best;
  local_best.key = num_experts;
  local_best.value = -FLT_MAX;

  for (int expert = tid; expert < num_experts; expert += TPB) {
    ExpertScore candidate;
    candidate.key = expert;
    candidate.value = input[row_offset + expert];
    local_best = arg_max(candidate, local_best);
  }

  const ExpertScore best = ScoreReduce(tmpStorage.score_storage).Reduce(local_best, arg_max);
  if (tid == 0) {
    row_max = best.value;
    best_expert = best.key;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    local_sum += expf(input[row_offset + expert] - row_max);
  }

  const float sum_value = FloatReduce(tmpStorage.float_storage).Reduce(local_sum, Sum());
  if (tid == 0) {
    inv_sum = 1.0f / sum_value;
  }
  __syncthreads();

  if (tid == 0) {
    const bool node_uses_expert = best_expert >= start_expert && best_expert < end_expert;
    const bool should_process_row = row_is_active && node_uses_expert;
    output[token] = inv_sum;
    indices[token] = should_process_row ? (best_expert - start_expert) : num_experts;
    assert(indices[token] >= 0);
    source_rows[token] = token;
  }
}

// V2 Optimized: Specialized topk=2 kernel with reduced register pressure
template <int TPB>
__global__ void moeSoftmaxTopK2(
    const float* __restrict__ input,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_experts,
    const int start_expert,
    const int end_expert)
{
  using FloatReduce = cub::BlockReduce<float, TPB>;
  using ScoreReduce = cub::BlockReduce<ExpertScore, TPB>;
  __shared__ union {
    typename FloatReduce::TempStorage float_storage;
    typename ScoreReduce::TempStorage score_storage;
  } tmpStorage;

  __shared__ float row_max;
  __shared__ float inv_sum;
  __shared__ float softmax[384];
  __shared__ int selected[2];

  const int num_tokens = gridDim.x;
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = token * num_experts;
  const bool row_is_active = finished ? !finished[token] : true;

  float local_max = -FLT_MAX;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    local_max = fmaxf(input[row_offset + expert], local_max);
  }

  const float max_value = FloatReduce(tmpStorage.float_storage).Reduce(local_max, Max());
  if (tid == 0) {
    row_max = max_value;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    local_sum += expf(input[row_offset + expert] - row_max);
  }

  const float sum_value = FloatReduce(tmpStorage.float_storage).Reduce(local_sum, Sum());
  if (tid == 0) {
    inv_sum = 1.0f / sum_value;
  }
  __syncthreads();

  for (int expert = tid; expert < num_experts; expert += TPB) {
    softmax[expert] = expf(input[row_offset + expert] - row_max) * inv_sum;
  }
  __syncthreads();

  ArgMaxExpert arg_max;
  
  // First expert selection
  {
    ExpertScore local_best;
    local_best.key = num_experts;
    local_best.value = -1.0f;

    for (int expert = tid; expert < num_experts; expert += TPB) {
      ExpertScore candidate;
      candidate.key = expert;
      candidate.value = softmax[expert];
      local_best = arg_max(candidate, local_best);
    }

    const ExpertScore best = ScoreReduce(tmpStorage.score_storage).Reduce(local_best, arg_max);
    if (tid == 0) {
      selected[0] = best.key;
      const bool node_uses_expert = best.key >= start_expert && best.key < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;
      output[token * 2] = best.value;
      indices[token * 2] = should_process_row ? (best.key - start_expert) : num_experts;
      assert(indices[token * 2] >= 0);
      source_rows[token * 2] = token;
    }
    __syncthreads();
  }

  // Second expert selection (skip first selected)
  {
    ExpertScore local_best;
    local_best.key = num_experts;
    local_best.value = -1.0f;

    const int first_selected = selected[0];
    for (int expert = tid; expert < num_experts; expert += TPB) {
      if (expert != first_selected) {
        ExpertScore candidate;
        candidate.key = expert;
        candidate.value = softmax[expert];
        local_best = arg_max(candidate, local_best);
      }
    }

    const ExpertScore best = ScoreReduce(tmpStorage.score_storage).Reduce(local_best, arg_max);
    if (tid == 0) {
      selected[1] = best.key;
      const bool node_uses_expert = best.key >= start_expert && best.key < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;
      const int idx = token * 2 + 1;
      output[idx] = best.value;
      indices[idx] = should_process_row ? (best.key - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      source_rows[idx] = token;
    }
    __syncthreads();
  }
}

template <int TPB, int TOPK>
__global__ void moeSoftmaxTopK(
    const float* __restrict__ input,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_experts,
    const int start_expert,
    const int end_expert)
{
  using FloatReduce = cub::BlockReduce<float, TPB>;
  using ScoreReduce = cub::BlockReduce<ExpertScore, TPB>;
  __shared__ union {
    typename FloatReduce::TempStorage float_storage;
    typename ScoreReduce::TempStorage score_storage;
  } tmpStorage;

  __shared__ float row_max;
  __shared__ float inv_sum;
  __shared__ float softmax[384];
  __shared__ int selected[TOPK];

  const int num_tokens = gridDim.x;
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = token * num_experts;
  const bool row_is_active = finished ? !finished[token] : true;

  float local_max = -FLT_MAX;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    local_max = fmaxf(input[row_offset + expert], local_max);
  }

  const float max_value = FloatReduce(tmpStorage.float_storage).Reduce(local_max, Max());
  if (tid == 0) {
    row_max = max_value;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int expert = tid; expert < num_experts; expert += TPB) {
    local_sum += expf(input[row_offset + expert] - row_max);
  }

  const float sum_value = FloatReduce(tmpStorage.float_storage).Reduce(local_sum, Sum());
  if (tid == 0) {
    inv_sum = 1.0f / sum_value;
  }
  __syncthreads();

  for (int expert = tid; expert < num_experts; expert += TPB) {
    softmax[expert] = expf(input[row_offset + expert] - row_max) * inv_sum;
  }
  __syncthreads();

  ArgMaxExpert arg_max;
  for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
    ExpertScore local_best;
    local_best.key = num_experts;
    local_best.value = -1.0f;

    for (int expert = tid; expert < num_experts; expert += TPB) {
      bool already_selected = false;
      // V2 Optimization: Unroll loop only to k_idx iterations instead of full TOPK
      // This reduces register pressure and improves performance for small topk
      #pragma unroll
      for (int prior = 0; prior < TOPK; ++prior) {
        if (prior < k_idx && selected[prior] == expert) {
          already_selected = true;
          break;  // Early exit optimization
        }
      }

      if (!already_selected) {
        ExpertScore candidate;
        candidate.key = expert;
        candidate.value = softmax[expert];
        local_best = arg_max(candidate, local_best);
      }
    }

    const ExpertScore best = ScoreReduce(tmpStorage.score_storage).Reduce(local_best, arg_max);
    if (tid == 0) {
      selected[k_idx] = best.key;
      const bool node_uses_expert = best.key >= start_expert && best.key < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;
      const int idx = token * TOPK + k_idx;
      output[idx] = best.value;
      indices[idx] = should_process_row ? (best.key - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      source_rows[idx] = k_idx * num_tokens + token;
    }
    __syncthreads();
  }
}
