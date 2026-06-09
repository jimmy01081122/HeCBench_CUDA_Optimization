#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>

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

// ---------------------------------------------------------
// 1. 初始化 Kernel (大批量使用)
// ---------------------------------------------------------
__global__ void init_sorted_tokens_kernel(int* __restrict__ sorted_token_ids, int max_num_tokens_padded, int numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < max_num_tokens_padded) {
        sorted_token_ids[tid] = numel;
    }
}

// ---------------------------------------------------------
// 2. 高效能大批量對齊與前綴和 Kernel
// ---------------------------------------------------------
template <typename scalar_t>
__global__ void moe_align_large_count_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map,
    int num_experts, int block_size,
    int numel, int* __restrict__ cumsum, int max_num_tokens_padded,
    int inactive_expert_id)
{
    __shared__ int smem_counts[256];
    int tid = threadIdx.x;
    
    if (tid < num_experts) smem_counts[tid] = 0;
    __syncthreads();

    // 統計每個 Expert 的 Token 數量
    for (int i = tid; i < numel; i += blockDim.x) {
        int expert_id = topk_ids[i];
        if (expert_map) expert_id = expert_map[expert_id];
        if (expert_id != -1 && expert_id < num_experts) {
            atomicAdd(&smem_counts[expert_id], 1);
        }
    }
    __syncthreads();

    // 計算 Prefix sum 並直接覆寫 cumsum
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < num_experts; ++i) {
            cumsum[i] = sum;
            sum += CEILDIV(smem_counts[i], block_size) * block_size;
        }
        cumsum[num_experts] = sum;
        total_tokens_post_pad[0] = sum;
    }
    __syncthreads();

    // 填充 expert_ids
    for (int e = tid; e < num_experts; e += blockDim.x) {
        int start = cumsum[e];
        int end = cumsum[e + 1];
        for (int i = start; i < end; i += block_size) {
            expert_ids[i / block_size] = e;
        }
    }

    int total_blocks = cumsum[num_experts] / block_size;
    int max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
    for (int i = total_blocks + tid; i < max_num_m_blocks; i += blockDim.x) {
        expert_ids[i] = inactive_expert_id;
    }
}

// ---------------------------------------------------------
// 3. 高效能大批量排序 Kernel (Block-level Atomics 消除競爭)
// ---------------------------------------------------------
__global__ void sort_kernel(
    const int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ cumsum,
    int* __restrict__ expert_map,
    int numel, int num_experts)
{
    __shared__ int smem_counts[256];
    __shared__ int smem_base[256];

    int tid = threadIdx.x;
    if (tid < num_experts) {
        smem_counts[tid] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int expert_id = -1;
    int local_rank = 0;
    bool valid = false;

    // 階段 1: 區域內統計 (Local Atomics)
    if (i < numel) {
        expert_id = topk_ids[i];
        if (expert_map) expert_id = expert_map[expert_id];
        if (expert_id != -1 && expert_id < num_experts) {
            valid = true;
            local_rank = atomicAdd(&smem_counts[expert_id], 1);
        }
    }
    __syncthreads();

    // 階段 2: 代表 Thread 執行全域原子加法 (每 Block 每 Expert 僅 1 次全域存取)
    if (tid < num_experts) {
        int c = smem_counts[tid];
        if (c > 0) {
            smem_base[tid] = atomicAdd(&cumsum[tid], c);
        }
    }
    __syncthreads();

    // 階段 3: 寫入全域位址
    if (valid) {
        sorted_token_ids[smem_base[expert_id] + local_rank] = i;
    }
}

// ---------------------------------------------------------
// 4. 小批量單一 Block 核心 (消除多次啟動延遲)
// ---------------------------------------------------------
template <typename scalar_t>
__global__ void moe_align_single_block_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map,
    int num_experts, int block_size,
    int numel, int max_num_tokens_padded, int inactive_expert_id)
{
    extern __shared__ int smem[];
    int* counts = smem;                         
    int* cumsum = smem + num_experts;           

    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    for (int i = tid; i < max_num_tokens_padded; i += bdim) {
        sorted_token_ids[i] = numel;
    }
    for (int i = tid; i <= num_experts; i += bdim) {
        if (i < num_experts) counts[i] = 0;
        cumsum[i] = 0;
    }
    __syncthreads();

    for (int i = tid; i < numel; i += bdim) {
        int expert_id = topk_ids[i];
        if (expert_map) expert_id = expert_map[expert_id];
        if (expert_id != -1 && expert_id < num_experts) {
            atomicAdd(&counts[expert_id], 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < num_experts; ++i) {
            cumsum[i] = sum;
            sum += CEILDIV(counts[i], block_size) * block_size;
        }
        cumsum[num_experts] = sum;
        total_tokens_post_pad[0] = sum;
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += bdim) {
        int start = cumsum[e];
        int end = cumsum[e + 1];
        for (int i = start; i < end; i += block_size) {
            expert_ids[i / block_size] = e;
        }
    }

    int total_blocks = cumsum[num_experts] / block_size;
    int max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
    for (int i = total_blocks + tid; i < max_num_m_blocks; i += bdim) {
        expert_ids[i] = inactive_expert_id;
    }
    __syncthreads();

    for (int i = tid; i < numel; i += bdim) {
        int expert_id = topk_ids[i];
        if (expert_map) expert_id = expert_map[expert_id];
        if (expert_id != -1 && expert_id < num_experts) {
            int rank = atomicAdd(&cumsum[expert_id], 1);
            sorted_token_ids[rank] = i;
        }
    }
}

// ---------------------------------------------------------
// 主派發函數
// ---------------------------------------------------------
void moe_align_block_size(int *topk_ids, int num_experts,
                          int block_size, int *sorted_token_ids,
                          int *experts_ids,
                          int *num_tokens_post_pad,
                          int *expert_map,
                          int topk_ids_size,
                          int topk,
                          int max_num_tokens_padded,
                          int *cumsum_buff)
{
  // 嚴格的分流閾值：1024。避免大批量使用單一 Block 造成序列化瓶頸。
  if (topk_ids_size <= 1024) {
    int threads = 1024;
    int shared_mem_size = (2 * num_experts + 1) * sizeof(int);
    moe_align_single_block_kernel<int><<<1, threads, shared_mem_size, 0>>>(
        topk_ids, sorted_token_ids, experts_ids, num_tokens_post_pad,
        expert_map, num_experts, block_size, topk_ids_size, max_num_tokens_padded, 0);
  } else {
    int init_threads = 1024;
    int init_blocks = CEILDIV(max_num_tokens_padded, init_threads);
    init_sorted_tokens_kernel<<<init_blocks, init_threads, 0, 0>>>(
        sorted_token_ids, max_num_tokens_padded, topk_ids_size);

    int align_threads = 1024;
    moe_align_large_count_kernel<int><<<1, align_threads, 0, 0>>>(
        topk_ids, experts_ids, num_tokens_post_pad, expert_map, 
        num_experts, block_size, topk_ids_size, cumsum_buff, max_num_tokens_padded, 0);

    int sort_threads = 256;
    int sort_blocks = CEILDIV(topk_ids_size, sort_threads);
    sort_kernel<<<sort_blocks, sort_threads, 0, 0>>>(
        topk_ids, sorted_token_ids, cumsum_buff, expert_map, 
        topk_ids_size, num_experts);
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
      #ifdef DEBUG
      printf("%d ", v[j]);    
      #endif
    }
    #ifdef DEBUG
    printf("\n");
    #endif
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

         #ifdef DEBUG
         printf("max_num_tokens_padded: %zu\n",  max_num_tokens_padded);
         #endif

         int *sorted_ids;
         GPU_CHECK(cudaMalloc((void**)&sorted_ids, max_num_tokens_padded * sizeof(int)));

         int *expert_ids;
         int max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
         GPU_CHECK(cudaMalloc((void**)&expert_ids, max_num_m_blocks * sizeof(int)));

         int *num_tokens_post_pad;
         GPU_CHECK(cudaMalloc((void**)&num_tokens_post_pad, sizeof(int)));

         int *cumsum_buff;
         GPU_CHECK(cudaMalloc((void**)&cumsum_buff, sizeof(int) * (num_experts + 1)));

         GPU_CHECK(cudaDeviceSynchronize());
         auto start = std::chrono::steady_clock::now();

         for (int n = 0; n < repeat; n++) {
           moe_align_block_size(
                topk_ids, num_experts, block_size, sorted_ids, expert_ids,
                num_tokens_post_pad, nullptr, topk_ids_size, topk,
                max_num_tokens_padded, cumsum_buff
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

         #ifdef DEBUG
         printf("actual sorted ids: ");
         for (int i = 0; i < max_num_tokens_padded; i++) {
           printf("%d ", actual_sorted_ids[i]);
         }
         printf("\nactual expert ids: ");
         for (int i = 0; i < max_num_m_blocks; i++) {
           printf("%d ", actual_expert_ids[i]);
         }
         printf("\nactual number of tokens: %d\n", actual_num_tokens);
         #endif

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
             if (v == topk_ids_size) continue;
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
         GPU_CHECK(cudaFree(cumsum_buff));

         free(h_topk_ids);
         free(actual_sorted_ids);
         free(actual_expert_ids);
       }
     }
   }
   return 0;
}