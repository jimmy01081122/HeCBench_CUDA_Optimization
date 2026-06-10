#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>

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
__launch_bounds__(BLOCK_SIZE)
void softMax2 (const int numSlice, const int sliceSize,
              const float* __restrict__ src, float* __restrict__ dest)
{
  constexpr unsigned FULL_MASK = 0xffffffffu;
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int i = blockIdx.x * warps_per_block + warp_id;
  if (i >= numSlice) return;

  int base = i * sliceSize;
  float max_ = -3.402823466e+38f;
  for (int j = lane; j < sliceSize; j += 32) {
    max_ = max(max_, src[base + j]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_ = max(max_, __shfl_down_sync(FULL_MASK, max_, offset));
  }
  max_ = __shfl_sync(FULL_MASK, max_, 0);

  float sum = 0;
  for (int j = lane; j < sliceSize; j += 32) {
    sum += expf(src[base + j] - max_);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }
  sum = __shfl_sync(FULL_MASK, sum, 0);

  float inv_sum = __fdividef(1.0f, sum);
  for (int j = lane; j < sliceSize; j += 32) {
    dest[base + j] = expf(src[base + j] - max_) * inv_sum;
  }
}

__global__
__launch_bounds__(BLOCK_SIZE)
void softMax784 (const int numSlice,
                 const float* __restrict__ src, float* __restrict__ dest)
{
  constexpr unsigned FULL_MASK = 0xffffffffu;
  constexpr int SLICE_SIZE = 784;
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int i = blockIdx.x * (BLOCK_SIZE / 32) + warp_id;
  if (i >= numSlice) return;

  int base = i * SLICE_SIZE;
  float max_ = -3.402823466e+38f;
#pragma unroll
  for (int j = 0; j < 768; j += 32) {
    max_ = max(max_, src[base + j + lane]);
  }
  if (lane < 16) {
    max_ = max(max_, src[base + 768 + lane]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_ = max(max_, __shfl_down_sync(FULL_MASK, max_, offset));
  }
  max_ = __shfl_sync(FULL_MASK, max_, 0);

  float sum = 0;
#pragma unroll
  for (int j = 0; j < 768; j += 32) {
    sum += expf(src[base + j + lane] - max_);
  }
  if (lane < 16) {
    sum += expf(src[base + 768 + lane] - max_);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }
  sum = __shfl_sync(FULL_MASK, sum, 0);

  float inv_sum = __fdividef(1.0f, sum);
#pragma unroll
  for (int j = 0; j < 768; j += 32) {
    dest[base + j + lane] = expf(src[base + j + lane] - max_) * inv_sum;
  }
  if (lane < 16) {
    dest[base + 768 + lane] = expf(src[base + 768 + lane] - max_) * inv_sum;
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
      if (sliceSize == 784) {
        softMax784<<<grids, blocks>>>(numSlice, d_input, d_output);
      } else {
        softMax2<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
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
