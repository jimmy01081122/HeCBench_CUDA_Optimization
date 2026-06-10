#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define BLOCK_SIZE 256

__device__ __forceinline__
float warpReduceMax(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__
float warpReduceSum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

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
              const float* __restrict__ src, float* __restrict__ dest)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numSlice) return;
  const int base = i * sliceSize;
  float max_ = src[base];
  for (int j = 0; j < sliceSize; j++) {
    max_ = fmaxf(max_, src[base + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    sum += expf(src[base + j] - max_);
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[base + j] = expf(src[base + j] - max_) / sum;
  }
}

__global__
void softMax2Cg(const int numSlice, const int sliceSize,
                const float* __restrict__ src, float* __restrict__ dest)
{
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int i = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (i >= numSlice) return;
  const int base = i * sliceSize;
  float max_ = src[base];
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    max_ = fmaxf(max_, src[base + j]);
  }
  max_ = cg::reduce(warp, max_, cg::greater<float>{});
  float sum = 0;
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    sum += expf(src[base + j] - max_);
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    dest[base + j] = expf(src[base + j] - max_) / sum;
  }
}

__global__
void softMax2Shuffle(const int numSlice, const int sliceSize,
                     const float* __restrict__ src, float* __restrict__ dest)
{
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int i = blockIdx.x * warps_per_block + warp_id;
  if (i >= numSlice) return;
  const int base = i * sliceSize;
  float max_ = src[base];
  for (int j = lane; j < sliceSize; j += 32) {
    max_ = fmaxf(max_, src[base + j]);
  }
  max_ = warpReduceMax(max_);
  max_ = __shfl_sync(0xffffffff, max_, 0);

  float sum = 0;
  for (int j = lane; j < sliceSize; j += 32) {
    sum += expf(src[base + j] - max_);
  }
  sum = warpReduceSum(sum);
  sum = __shfl_sync(0xffffffff, sum, 0);

  for (int j = lane; j < sliceSize; j += 32) {
    dest[base + j] = expf(src[base + j] - max_) / sum;
  }
}


int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of slices> <slice size> <implementations> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: optimized\n");
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

  if (kernel == 1) {
    dim3 grids ((numSlice+BLOCK_SIZE/32-1)/(BLOCK_SIZE/32));
    dim3 blocks (BLOCK_SIZE);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      if (sliceSize >= 768) {
        softMax2Shuffle<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
      }
      else {
        softMax2Cg<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
      }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
  }
  else {
    dim3 grids ((numSlice+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 blocks (BLOCK_SIZE);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      softMax<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
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
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input);
  free(output_cpu);
  free(output_gpu);
  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}
