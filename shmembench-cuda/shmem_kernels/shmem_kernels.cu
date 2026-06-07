/**
 * shmem_kernels.cu: This file is part of the gpumembench suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TOTAL_ITERATIONS (1024)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define CUDA_CHECK(call) do {                                                     \
  cudaError_t err__ = (call);                                                     \
  if (err__ != cudaSuccess) {                                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                \
            cudaGetErrorString(err__));                                           \
    exit(EXIT_FAILURE);                                                           \
  }                                                                               \
} while (0)

// shared memory swap operation
__device__ void shmem_swap(float4 *v1, float4 *v2){
  float4 tmp;
  tmp = *v2;
  *v2 = *v1;
  *v1 = tmp;
}

__device__ float4 init_val(int i){
  return make_float4(i, i+11, i+19, i+23);
}

__device__ float4 reduce_vector(float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6){
  return make_float4(v1.x + v2.x + v3.x + v4.x + v5.x + v6.x, 
                     v1.y + v2.y + v3.y + v4.y + v5.y + v6.y,
                     v1.z + v2.z + v3.z + v4.z + v5.z + v6.z,
                     v1.w + v2.w + v3.w + v4.w + v5.w + v6.w);
}

__device__ void set_vector(float4 *target, int offset, float4 v){
  target[offset].x = v.x;
  target[offset].y = v.y;
  target[offset].z = v.z;
  target[offset].w = v.w;
}


__global__ void benchmark_shmem(float4 *g_data){

  __shared__ float4 shm_buffer[BLOCK_SIZE*6];

  int tid = threadIdx.x; 
  int globaltid = blockIdx.x*blockDim.x + tid;
  set_vector(shm_buffer, tid+0*blockDim.x, init_val(tid));
  set_vector(shm_buffer, tid+1*blockDim.x, init_val(tid+1));
  set_vector(shm_buffer, tid+2*blockDim.x, init_val(tid+3));
  set_vector(shm_buffer, tid+3*blockDim.x, init_val(tid+7));
  set_vector(shm_buffer, tid+4*blockDim.x, init_val(tid+13));
  set_vector(shm_buffer, tid+5*blockDim.x, init_val(tid+17));

  __syncthreads();  // __threadfence_block() is faster though

  #pragma unroll 32
  for(int j=0; j<TOTAL_ITERATIONS; j++){
    shmem_swap(shm_buffer+tid+0*blockDim.x, shm_buffer+tid+1*blockDim.x);
    shmem_swap(shm_buffer+tid+2*blockDim.x, shm_buffer+tid+3*blockDim.x);
    shmem_swap(shm_buffer+tid+4*blockDim.x, shm_buffer+tid+5*blockDim.x);

    __syncthreads();

    shmem_swap(shm_buffer+tid+1*blockDim.x, shm_buffer+tid+2*blockDim.x);
    shmem_swap(shm_buffer+tid+3*blockDim.x, shm_buffer+tid+4*blockDim.x);

    __syncthreads();
  }

  g_data[globaltid] = reduce_vector(shm_buffer[tid+0*blockDim.x], 
                                    shm_buffer[tid+1*blockDim.x],
                                    shm_buffer[tid+2*blockDim.x],
                                    shm_buffer[tid+3*blockDim.x],
                                    shm_buffer[tid+4*blockDim.x],
                                    shm_buffer[tid+5*blockDim.x]);
}

void shmembenchGPU(double *c, const long size, const int repeat) {
  const int TOTAL_BLOCKS = size/(BLOCK_SIZE);

  double *cd;
  CUDA_CHECK(cudaMalloc((void**)&cd, size*sizeof(double)));

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid_f4(TOTAL_BLOCKS/4, 1, 1);
  const int shared_bytes = BLOCK_SIZE * 6 * sizeof(float4);
  const long long operations_bytes  = (6LL+4*5*TOTAL_ITERATIONS+6)*size*sizeof(float);
  const long long operations_128bit = (6LL+4*5*TOTAL_ITERATIONS+6)*size/4;
  const int warmup = 5;
  const int trials = 5;
  CUDA_CHECK(cudaFuncSetCacheConfig(benchmark_shmem, cudaFuncCachePreferShared));

  for (int i = 0; i < warmup; i++)
    benchmark_shmem<<< dimGrid_f4, dimBlock >>>((float4*)cd);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float min_ms = FLT_MAX;
  float max_ms = 0.0f;
  double sum_ms = 0.0;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int trial = 0; trial < trials; trial++) {
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++)
      benchmark_shmem<<< dimGrid_f4, dimBlock >>>((float4*)cd);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float per_kernel_ms = elapsed_ms / (float)repeat;
    sum_ms += per_kernel_ms;
    if (per_kernel_ms < min_ms) min_ms = per_kernel_ms;
    if (per_kernel_ms > max_ms) max_ms = per_kernel_ms;
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  double time_shmem_128b = (sum_ms / (double)trials) * 1.0e6;
  double min_ns = (double)min_ms * 1.0e6;
  double max_ns = (double)max_ms * 1.0e6;
  printf("Average kernel execution time : %f (ms)\n", time_shmem_128b * 1e-6);
  printf("Kernel execution min/max      : %f / %f (ms)\n", min_ns * 1e-6, max_ns * 1e-6);
  printf("Timing config: repeat=%d, warmup=%d, trials=%d, block=%d, grid=%d, shared_bytes=%d, access_pattern=barriered_float4_swaps\n",
         repeat, warmup, trials, BLOCK_SIZE, dimGrid_f4.x, shared_bytes);


  // Copy results back to host memory
  CUDA_CHECK(cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(cd));

  // simple checksum
  double sum = 0;
  for (long i = 0; i < size; i++) sum += c[i];
  const double expected_sum = 21256458760384741137729978368.00;
  const char *correctness = (sum == expected_sum) ? "PASS" : "FAIL";
  if (sum != expected_sum) {
    printf("checksum failed\n");
    printf("CHECKSUM,sum=%.17e,expected=%.17e,status=FAIL\n", sum, expected_sum);
    exit(EXIT_FAILURE);
  }
  printf("CHECKSUM,sum=%.17e,expected=%.17e,status=PASS\n", sum, expected_sum);
  
  printf("Memory throughput\n");

  printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n", 
    (double)operations_bytes / time_shmem_128b,
    (double)operations_128bit / time_shmem_128b);
  printf("RESULT,test=barriered_float4_swaps,repeat=%d,block=%d,grid=%d,shared_bytes=%d,avg_us=%.6f,min_us=%.6f,max_us=%.6f,bandwidth_GBps=%.6f,correctness=%s,status=PASS\n",
         repeat, BLOCK_SIZE, dimGrid_f4.x, shared_bytes,
         time_shmem_128b * 1e-3, min_ns * 1e-3, max_ns * 1e-3,
         (double)operations_bytes / time_shmem_128b, correctness);
}
