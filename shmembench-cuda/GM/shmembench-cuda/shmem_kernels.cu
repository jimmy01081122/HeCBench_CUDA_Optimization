/**
 * shmem_kernels.cu: This file is part of the gpumembench suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 * **/

#include <chrono> // timing
#include <stdio.h>

#define TOTAL_ITERATIONS (1024)

#define CUDA_CHECK(err) do {                                                        \
  cudaError_t err__ = (err);                                                        \
  if (err__ != cudaSuccess) {                                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                   \
            cudaGetErrorString(err__));                                             \
    exit(EXIT_FAILURE);                                                             \
  }                                                                                 \
} while (0)

#define CUDA_KERNEL_CHECK() do {                                                    \
  cudaError_t err__ = cudaGetLastError();                                           \
  if (err__ != cudaSuccess) {                                                       \
    fprintf(stderr, "CUDA kernel launch error %s:%d: %s\n", __FILE__, __LINE__,    \
            cudaGetErrorString(err__));                                             \
    exit(EXIT_FAILURE);                                                             \
  }                                                                                 \
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


template <int BLOCK_SIZE_VAL>
__global__ void benchmark_shmem(float4 *g_data){

  extern __shared__ float4 shm_buffer[];

  int tid = threadIdx.x; 
  int globaltid = blockIdx.x*blockDim.x + tid;
  set_vector(shm_buffer, tid+0*blockDim.x, init_val(tid));
  set_vector(shm_buffer, tid+1*blockDim.x, init_val(tid+1));
  set_vector(shm_buffer, tid+2*blockDim.x, init_val(tid+3));
  set_vector(shm_buffer, tid+3*blockDim.x, init_val(tid+7));
  set_vector(shm_buffer, tid+4*blockDim.x, init_val(tid+13));
  set_vector(shm_buffer, tid+5*blockDim.x, init_val(tid+17));

  __syncthreads();

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

template <int BLOCK_SIZE_VAL>
void run_test(double *c, double *cd, const long size, const int repeat) {
  const int TOTAL_BLOCKS = size/(BLOCK_SIZE_VAL);

  dim3 dimBlock(BLOCK_SIZE_VAL, 1, 1);
  dim3 dimGrid_f4(TOTAL_BLOCKS/4, 1, 1);

  int shared_mem_bytes = BLOCK_SIZE_VAL * 6 * sizeof(float4);

  // Set max dynamic shared memory size attribute for Volta sm_70
  CUDA_CHECK(cudaFuncSetAttribute(benchmark_shmem<BLOCK_SIZE_VAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes));

  // Warm up kernel once
  benchmark_shmem<BLOCK_SIZE_VAL><<< dimGrid_f4, dimBlock, shared_mem_bytes >>>((float4*)cd);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start_ev, stop_ev;
  CUDA_CHECK(cudaEventCreate(&start_ev));
  CUDA_CHECK(cudaEventCreate(&stop_ev));

  double min_ms = 1e9, max_ms = 0, sum_ms = 0;

  for (int i = 0; i < repeat; i++) {
    CUDA_CHECK(cudaEventRecord(start_ev, 0));
    benchmark_shmem<BLOCK_SIZE_VAL><<< dimGrid_f4, dimBlock, shared_mem_bytes >>>((float4*)cd);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(stop_ev, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
    if (ms < min_ms) min_ms = ms;
    if (ms > max_ms) max_ms = ms;
    sum_ms += ms;
  }

  double avg_ms = sum_ms / repeat;
  double avg_us = avg_ms * 1000.0;
  double min_us = min_ms * 1000.0;
  double max_us = max_ms * 1000.0;

  // Copy results back to host memory
  CUDA_CHECK(cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start_ev));
  CUDA_CHECK(cudaEventDestroy(stop_ev));

  // simple checksum
  double sum = 0;
  for (long i = 0; i < size; i++) sum += c[i];
  
  const char *correctness = "FAIL";
  double expected_sum = 0;
  if (BLOCK_SIZE_VAL == 128) expected_sum = 1.89172598426861568e+26;
  else if (BLOCK_SIZE_VAL == 256) expected_sum = 2.12564587603847411e+28;
  else if (BLOCK_SIZE_VAL == 512) expected_sum = 3.64966727074076865e+30;
  else if (BLOCK_SIZE_VAL == 1024) expected_sum = 7.64271701285369981e+32;

  if (sum == expected_sum) {
    correctness = "PASS";
  } else {
    printf("checksum failed for block size %d: expected %.17e, got %.17e\n", BLOCK_SIZE_VAL, expected_sum, sum);
  }
  
  const long long operations_bytes  = (6LL+4*5*TOTAL_ITERATIONS+6)*size*sizeof(float);

  double bandwidth = (double)operations_bytes / (avg_ms * 1e6);

  // Print RESULT line
  printf("RESULT,test=shmembench,repeat=%d,block=%d,shared_bytes=%d,avg_us=%.3f,min_us=%.3f,max_us=%.3f,bandwidth_GBps=%.2f,correctness=%s,status=%s\n",
         repeat, BLOCK_SIZE_VAL, shared_mem_bytes, avg_us, min_us, max_us, bandwidth, correctness, correctness);
}

void shmembenchGPU(double *c, const long size, const int repeat) {
  double *cd;
  CUDA_CHECK(cudaMalloc((void**)&cd, size*sizeof(double)));

  run_test<128>(c, cd, size, repeat);
  run_test<256>(c, cd, size, repeat);
  run_test<512>(c, cd, size, repeat);
  run_test<1024>(c, cd, size, repeat);

  CUDA_CHECK(cudaFree(cd));
}
