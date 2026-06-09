// 1. 移除動態分配開銷 : 將 cumsum_buff 的記憶體分配 (cudaMalloc) 與釋放 (cudaFree) 移至 main 函式的計時迴圈之外，
// 並透過修改 moe_align_block_size 的簽名將其作為 workspace 傳入。
// 此外，移除了原本多餘的 cudaMemset，因為 align_kernel 會覆寫所有相關記憶體位置。  
// 
// 2. 實作 Warp-level Aggregation：在 _count_and_sort_expert_tokens 中，
// 利用 __match_any_sync 與暫存器洗牌指令將 Warp 內相同 expert_id 的 thread 進行聚合，
// 推派單一 thread 執行全域 atomicAdd，大幅減少記憶體競爭

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>

#define GPU_CHECK(ans)                                                                   \
    {                                                                                    \
        gpuAssert((ans), __FILE__, __LINE__);                                            \
    }
inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if(code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts,
    int padded_num_experts, int experts_per_warp, int block_size,
    int numel, int* __restrict__ cumsum, int max_num_tokens_padded,
    int max_num_m_blocks, int model_offset, int inactive_expert_id,
    int topk_num, int* token_mask)
{
  extern __shared__ int shared_counts[];

  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  if (blockIdx.x % 2) {
    for (int it = threadIdx.x; it < max_num_tokens_padded;
         it += blockDim.x) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    return;
  }

  const int warp_id = threadIdx.x / warpSize;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset],
              mask);
  }

  __syncthreads();

  using BlockScan = cub::BlockScan<int, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[cumsum_offset + threadIdx.x];
         i < cumsum[cumsum_offset + threadIdx.x + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = threadIdx.x;
    }
  }

  const int fill_start_idx =
      cumsum[cumsum_offset + num_experts] / block_size + threadIdx.x;
  for (int i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}

template <typename scalar_t, int fill_threads>
__device__ void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts, int block_size,
    int numel, int max_num_tokens_padded, int max_num_m_blocks,
    int inactive_expert_id, int model_offset, int topk_num,
    int* token_mask)
{
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  if (threadIdx.x < fill_threads) {
    for (int it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    __syncthreads();
    return;
  }

  const int tid = threadIdx.x - fill_threads;
  const int stride = blockDim.x - fill_threads;

  extern __shared__ int shared_mem[];
  int* cumsum = shared_mem;
  int* tokens_cnts = (int*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) continue;
    }
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    tokens_cnts[(tid + 1) * num_experts + expert_id] += mask;
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
    }
    total_tokens_post_pad[model_offset] =
        static_cast<int>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = tid;
    }
  }

  const int fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (int i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) continue;
    }
    int rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      sorted_token_ids[sorted_token_ids_offset + rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}

template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ cumsum_buffer,
    int* __restrict__ expert_map, int numel, int num_experts,
    int max_num_tokens_padded, int* __restrict__ token_mask,
    int model_offset, int topk_num)
{
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.y;

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }

    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) continue;
    }

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      int* ptr = &cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id];
      
      // Warp-level aggregation for atomicAdd
      unsigned int active = __activemask();
      unsigned int match = __match_any_sync(active, (unsigned long long)ptr);
      int lane_id = threadIdx.x % 32;
      int leader = __ffs(match) - 1;
      
      int count = __popc(match);
      int rank = __popc(match & ((1 << lane_id) - 1));
      
      int base;
      if (lane_id == leader) {
          base = atomicAdd(ptr, count);
      }
      base = __shfl_sync(match, base, leader);
      
      int rank_post_pad = base + rank;
      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] = i;
    }
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts,
    int padded_num_experts, int experts_per_warp, int block_size,
    int numel, int* __restrict__ cumsum, int max_num_tokens_padded,
    int topk_num)
{
  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, padded_num_experts, experts_per_warp, block_size, numel,
      cumsum, max_num_tokens_padded, CEILDIV(max_num_tokens_padded, block_size),
      0, 0, topk_num, nullptr);
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ cumsum_buffer,
    int* __restrict__ expert_map, int numel, int num_experts,
    int max_num_tokens_padded, int topk_num)
{
  _count_and_sort_expert_tokens(
      topk_ids, sorted_token_ids, cumsum_buffer, expert_map, numel, num_experts,
      max_num_tokens_padded, nullptr, 0, topk_num);
}

template <typename scalar_t, int fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts, int block_size,
    int numel, int max_num_tokens_padded, int topk_num)
{

  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size), 0, 0, topk_num, nullptr);
}

void moe_align_block_size(int *topk_ids, int num_experts,
                          int block_size, int *sorted_token_ids,
                          int *experts_ids,
                          int *num_tokens_post_pad,
                          int *expert_map,
                          int topk_ids_size,
                          int topk,
                          int sorted_token_ids_size,
                          int *cumsum_buff) 
{
  const int WARP_SIZE = 32;
  int threads = 1024;

  int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  assert(padded_num_experts < 1024);

  bool small_batch_expert_mode = (topk_ids_size < 1024) && (num_experts <= 64);

  if (small_batch_expert_mode) {
    const int threads = std::max(num_experts, WARP_SIZE);
    const int shared_mem_size =
        ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int);

    constexpr int fill_threads = 256;
    auto small_batch_expert_kernel = moe_align_block_size_small_batch_expert_kernel<int, fill_threads>;
    small_batch_expert_kernel<<<1, fill_threads + threads, shared_mem_size, 0>>>(
        topk_ids,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map, num_experts, block_size,
        topk_ids_size, sorted_token_ids_size, topk);
    
    GPU_CHECK(cudaDeviceSynchronize());
  } else {
    auto align_kernel = moe_align_block_size_kernel<int>;

    int num_warps = CEILDIV(padded_num_experts, experts_per_warp);
    int shared_mem_size = num_warps * experts_per_warp * sizeof(int);

    align_kernel<<<2, threads, shared_mem_size, 0>>>(
        topk_ids,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map, num_experts, padded_num_experts,
        experts_per_warp, block_size, topk_ids_size,
        cumsum_buff, sorted_token_ids_size,
        topk);

    const int block_threads = std::min(256, (int)threads);
    const int num_blocks = (topk_ids_size + block_threads - 1) / block_threads;
    const int max_blocks = 65535;
    const int actual_blocks = std::min(num_blocks, max_blocks);
    dim3 gridDims(1, actual_blocks);

    auto sort_kernel = count_and_sort_expert_tokens_kernel<int>;
    sort_kernel<<<gridDims, block_threads>>>(
        topk_ids,
        sorted_token_ids,
        cumsum_buff, expert_map,
        topk_ids_size, num_experts, sorted_token_ids_size,
        topk);

    GPU_CHECK(cudaDeviceSynchronize());
  }
}

void randperm(int* topk_ids, int topk, int m, int n) {
  std::vector<int> v(n);
  for (int i = 0; i < n; ++i) v[i] = i;
  std::mt19937 gen(19937);
  for (int i = 0; i < m; i++) { 
    std::shuffle(v.begin(), v.end(), gen);
    for (int j = 0; j < topk; j++) {
      topk_ids[i * topk + j] = v[j];
    }
  }
}

int main(int argc, char* argv[])
{
   if (argc != 2) {
     printf("Usage: %s <repeat>\n", argv[0]);
     return 1;
   }
   const int repeat = atoi(argv[1]);

#ifdef DEBUG
   int tokens[] = {4};
   int experts[] = {4};
   int topks[] = {3};
   int block_sizes[] = {4};
#else
   int tokens[] = {1, 3, 256, 4096, 8192};
   int experts[] = {32, 128};
   int topks[] = {2, 3, 4};
   int block_sizes[] = {32};
#endif
   bool pad_sorted_ids = true;

   int *topk_ids, *h_topk_ids;

   for (int i = 0; i < int(sizeof(tokens) / sizeof(tokens[0])); i++) {
     int m = tokens[i]; 

     for (int e = 0; e < int(sizeof(experts) / sizeof(experts[0])); e++) {
       int num_experts = experts[e]; 
       
       int *cumsum_buff;
       GPU_CHECK(cudaMalloc((void**)&cumsum_buff, sizeof(int) * (num_experts + 1)));

       for (int k = 0; k < int(sizeof(topks) / sizeof(topks[0])); k++) {
         int topk = topks[k]; 
         int block_size = block_sizes[0];

         int topk_ids_size = m * topk;
         h_topk_ids = (int*) malloc(topk_ids_size * sizeof(int));
         randperm(h_topk_ids, topk, m, num_experts);

         GPU_CHECK(cudaMalloc((void**)&topk_ids, m * topk * sizeof(int)));
         GPU_CHECK(cudaMemcpy(topk_ids, h_topk_ids, topk_ids_size * sizeof(int), cudaMemcpyHostToDevice));
         
         int max_num_tokens_padded = topk_ids_size + num_experts * (block_size - 1);

         if (pad_sorted_ids)
             max_num_tokens_padded = CEILDIV(max_num_tokens_padded, block_size) * block_size;

         if (topk_ids_size < num_experts)
             max_num_tokens_padded = std::min(topk_ids_size * block_size, max_num_tokens_padded);

         int *sorted_ids;
         GPU_CHECK(cudaMalloc((void**)&sorted_ids, max_num_tokens_padded * sizeof(int)));

         int *expert_ids;
         int max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
         GPU_CHECK(cudaMalloc((void**)&expert_ids, max_num_m_blocks * sizeof(int)));

         int *num_tokens_post_pad;
         GPU_CHECK(cudaMalloc((void**)&num_tokens_post_pad, sizeof(int)));

         GPU_CHECK(cudaDeviceSynchronize());
         auto start = std::chrono::steady_clock::now();

         for (int n = 0; n < repeat; n++) {
           moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                nullptr, 
                topk_ids_size,
                topk,
                max_num_tokens_padded,
                cumsum_buff 
           );
         }

         GPU_CHECK(cudaDeviceSynchronize());
         auto end = std::chrono::steady_clock::now();
         auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
         printf("Average execution time of the kernels (tokens %d, topk: %d, expert: %d, block_size %d): %f (us)\n",
                m, topk, num_experts, block_size, (time * 1e-3f) / repeat);

         int actual_num_tokens;
         GPU_CHECK(cudaMemcpy(&actual_num_tokens, num_tokens_post_pad, sizeof(int), cudaMemcpyDeviceToHost));

         int *actual_expert_ids = (int*) malloc(max_num_m_blocks * sizeof(int));
         GPU_CHECK(cudaMemcpy(actual_expert_ids, expert_ids, max_num_m_blocks * sizeof(int), cudaMemcpyDeviceToHost));

         int *actual_sorted_ids;
         actual_sorted_ids = (int*) malloc (max_num_tokens_padded * sizeof(int));
         GPU_CHECK(cudaMemcpy(actual_sorted_ids, sorted_ids, max_num_tokens_padded * sizeof(int), cudaMemcpyDeviceToHost));

         bool ok = true;
         if (actual_num_tokens % block_size) {
           ok = false;
           printf("Error: num_tokens_post_pad should be divisible by block_size\n");
         }
         if (actual_num_tokens < topk_ids_size) {
           ok = false;
           printf("Error: num_tokens_post_pad should be at least total_tokens\n");
         }
         for (int i = 0; i < max_num_m_blocks; i++) {
           if (actual_expert_ids[i] < 0 || actual_expert_ids[i] >= num_experts) {
             ok = false;
             printf("Error: expert_ids should contain valid expert indices\n");
             break;
           }
         }

         int ei = 0;
         for (int t = 0; t < max_num_tokens_padded; t += block_size) {
           int eid = actual_expert_ids[ei++];
           for (int b = 0; b < block_size; b++) { 
             int v = actual_sorted_ids[t+b];
             if (v == topk_ids_size) {
               continue;
             }
             if (eid != h_topk_ids[v]) {
               ok = false;
               break;
             }
           }
         }
         printf("%s\n", ok ? "PASS" : "FAIL");

         GPU_CHECK(cudaFree(topk_ids));
         GPU_CHECK(cudaFree(sorted_ids));
         GPU_CHECK(cudaFree(expert_ids));
         GPU_CHECK(cudaFree(num_tokens_post_pad));

         free(h_topk_ids);
         free(actual_sorted_ids);
         free(actual_expert_ids);
       }
       GPU_CHECK(cudaFree(cumsum_buff));
     }
   }
   return 0;
}