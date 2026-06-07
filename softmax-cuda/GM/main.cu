#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define BLOCK_SIZE 256

// A C model derived from the OpenCL kernel
void softMax_cpu(const int numSlice, const int sliceSize, const float* src, float* dest) {
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 0; j < sliceSize; j++) {
      max_ = (max_ < src[i * sliceSize + j]) ? src[i * sliceSize + j] : max_;
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      float e = expf(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

__global__
void softMax (const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = 0; j < sliceSize; j++) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    sum += expf(src[i * sliceSize + j] - max_);
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
  }
}

__global__
void softMax2 (const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int i = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  max_ = cg::reduce(warp, max_, cg::greater<float>{});
  float sum = 0;
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    sum += expf(src[i * sliceSize + j] - max_);
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
  }
}

__global__
void softMax3 (const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  extern __shared__ float s_mem[];
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int i = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (i >= numSlice) return;

  float* warp_s_mem = s_mem + warp.meta_group_rank() * sliceSize;

  float max_ = src[i * sliceSize];
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  max_ = cg::reduce(warp, max_, cg::greater<float>{});

  float sum = 0;
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    float e = expf(src[i * sliceSize + j] - max_);
    warp_s_mem[j] = e;
    sum += e;
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});

  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    dest[i * sliceSize + j] = warp_s_mem[j] / sum;
  }
}

// Helper functions for block reduction
__inline__ __device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

__inline__ __device__ float blockReduceMax(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceMax(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : -1e20f;

  if (wid == 0) val = warpReduceMax(val);

  return val;
}

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__inline__ __device__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;

  if (wid == 0) val = warpReduceSum(val);

  return val;
}

__global__
void softMax4 (const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  extern __shared__ float s_mem[];
  int i = blockIdx.x; // one block per row
  if (i >= numSlice) return;

  float max_ = -1e20f;
  for (int j = threadIdx.x; j < sliceSize; j += blockDim.x) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  max_ = blockReduceMax(max_);
  
  __shared__ float s_max;
  if (threadIdx.x == 0) s_max = max_;
  __syncthreads();
  max_ = s_max;

  float sum = 0.0f;
  for (int j = threadIdx.x; j < sliceSize; j += blockDim.x) {
    float e = expf(src[i * sliceSize + j] - max_);
    s_mem[j] = e;
    sum += e;
  }
  sum = blockReduceSum(sum);
  
  __shared__ float s_sum;
  if (threadIdx.x == 0) s_sum = sum;
  __syncthreads();
  sum = s_sum;

  for (int j = threadIdx.x; j < sliceSize; j += blockDim.x) {
    dest[i * sliceSize + j] = s_mem[j] / sum;
  }
}


int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of slices> <slice size> <implementations> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: optimized (warp-level)\n");
    printf("implementation 2: optimized (warp-level + shared memory cached expf)\n");
    printf("implementation 3: optimized (block-level + shared memory cached expf)\n");
    return 1;
  }

  int numSlice = atoi(argv[1]);
  int sliceSize = atoi(argv[2]);
  int kernel = atoi(argv[3]);
  int repeat = atoi(argv[4]);
  int numElem = numSlice * sliceSize;

  float* input = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_gpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_cpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);

  srand(2);
  for (int i = 0; i < numSlice; i++)
    for (int j = 0; j < sliceSize; j++)
      input[i*sliceSize+j] = rand() % 13;

  float *d_input, *d_output;
  cudaMalloc((void**)&d_input, sizeof(float) * numElem);
  cudaMalloc((void**)&d_output, sizeof(float) * numElem);
  cudaMemcpy(d_input, input, sizeof(float) * numElem, cudaMemcpyHostToDevice);

  cudaEvent_t start_ev, stop_ev;
  cudaEventCreate(&start_ev);
  cudaEventCreate(&stop_ev);

  float avg_ms = 0.0f;

  if (kernel == 3) {
    dim3 grids (numSlice);
    dim3 blocks (BLOCK_SIZE);
    int shared_mem_bytes = sliceSize * sizeof(float);
    cudaFuncSetAttribute(softMax4, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes);

    // Warmup
    for (int n = 0; n < 10; n++) {
      softMax4<<<grids, blocks, shared_mem_bytes>>>(numSlice, sliceSize, d_input, d_output);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start_ev);
    for (int n = 0; n < repeat; n++) {
      softMax4<<<grids, blocks, shared_mem_bytes>>>(numSlice, sliceSize, d_input, d_output);
    }
    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    avg_ms = elapsed_ms / repeat;
    printf("Average kernel execution time: %f (ms)\n", avg_ms);
  }
  else if (kernel == 2) {
    dim3 grids ((numSlice+BLOCK_SIZE/32-1)/(BLOCK_SIZE/32));
    dim3 blocks (BLOCK_SIZE);
    int shared_mem_bytes = (BLOCK_SIZE / 32) * sliceSize * sizeof(float);
    cudaFuncSetAttribute(softMax3, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes);

    // Warmup
    for (int n = 0; n < 10; n++) {
      softMax3<<<grids, blocks, shared_mem_bytes>>>(numSlice, sliceSize, d_input, d_output);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start_ev);
    for (int n = 0; n < repeat; n++) {
      softMax3<<<grids, blocks, shared_mem_bytes>>>(numSlice, sliceSize, d_input, d_output);
    }
    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    avg_ms = elapsed_ms / repeat;
    printf("Average kernel execution time: %f (ms)\n", avg_ms);
  }
  else if (kernel == 1) {
    dim3 grids ((numSlice+BLOCK_SIZE/32-1)/(BLOCK_SIZE/32));
    dim3 blocks (BLOCK_SIZE);

    // Warmup
    for (int n = 0; n < 10; n++) {
      softMax2<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start_ev);
    for (int n = 0; n < repeat; n++) {
      softMax2<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
    }
    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    avg_ms = elapsed_ms / repeat;
    printf("Average kernel execution time: %f (ms)\n", avg_ms);
  }
  else {
    dim3 grids ((numSlice+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 blocks (BLOCK_SIZE);

    // Warmup
    for (int n = 0; n < 10; n++) {
      softMax<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start_ev);
    for (int n = 0; n < repeat; n++) {
      softMax<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
    }
    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    avg_ms = elapsed_ms / repeat;
    printf("Average kernel execution time: %f (ms)\n", avg_ms);
  }

  cudaMemcpy(output_gpu, d_output, sizeof(float) * numElem, cudaMemcpyDeviceToHost);

  // verification
  bool ok = true;
  softMax_cpu(numSlice, sliceSize, input, output_cpu);
  for (int i = 0; i < numElem; i++) {
    if (fabsf(output_cpu[i] - output_gpu[i]) > 1e-3) {
      printf("@index %d host: %f device: %f\n", i, output_cpu[i], output_gpu[i]);
      ok = false;
      break;
    }
  }
  
  const char* status = ok ? "PASS" : "FAIL";
  printf("%s\n", status);
  printf("RESULT,impl=%d,batch=%d,slice=%d,repeat=%d,avg_ms=%f,status=%s\n",
         kernel, numSlice, sliceSize, repeat, avg_ms, status);

  free(input);
  free(output_cpu);
  free(output_gpu);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaEventDestroy(start_ev);
  cudaEventDestroy(stop_ev);
  return 0;
}
