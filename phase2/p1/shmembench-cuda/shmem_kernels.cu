/**
 * shmem_kernels.cu: This file is part of the gpumembench suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <float.h>

#define TOTAL_ITERATIONS (1024)
#define BLOCK_SIZE 64

static double packed_float_pair_as_double(float x, float y) {
  union {
    float f[2];
    double d;
  } u;
  u.f[0] = x;
  u.f[1] = y;
  return u.d;
}

static double expected_checksum(const long size) {
  const int total_blocks = size / BLOCK_SIZE;
  const int grid_blocks = total_blocks / 4;
  double sum = 0.0;

  for (int block = 0; block < grid_blocks; block++) {
    for (int tid = 0; tid < BLOCK_SIZE; tid++) {
      const float x = (float)(6 * tid + 41);
      const float y = (float)(6 * tid + 107);
      const float z = (float)(6 * tid + 155);
      const float w = (float)(6 * tid + 179);
      sum += packed_float_pair_as_double(x, y);
      sum += packed_float_pair_as_double(z, w);
    }
  }

  return sum;
}

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

void shmembenchGPU(double *c, const long size, const int repeat) {
  const int TOTAL_BLOCKS = size/(BLOCK_SIZE);

  double *cd;
  cudaMalloc((void**)&cd, size*sizeof(double));

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid_f4(TOTAL_BLOCKS/4, 1, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  benchmark_shmem<<< dimGrid_f4, dimBlock >>>((float4*)cd);
  cudaDeviceSynchronize();

  double total_us = 0.0;
  double min_us = DBL_MAX;
  double max_us = 0.0;

  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start);
    benchmark_shmem<<< dimGrid_f4, dimBlock >>>((float4*)cd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    const double elapsed_us = elapsed_ms * 1000.0;
    total_us += elapsed_us;
    if (elapsed_us < min_us) min_us = elapsed_us;
    if (elapsed_us > max_us) max_us = elapsed_us;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const double avg_us = total_us / repeat;
  const double time_shmem_128b = avg_us * 1000.0;
  printf("Average kernel execution time : %f (ms)\n", avg_us * 1e-3);
  printf("Kernel execution time us avg/min/max : %.3f / %.3f / %.3f\n", avg_us, min_us, max_us);

  // Copy results back to host memory
  cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(cd);

  // simple checksum
  double sum = 0;
  for (long i = 0; i < size; i++) sum += c[i];
  if (sum != expected_checksum(size))
    printf("checksum failed\n");
  
  printf("Memory throughput\n");
  const long long operations_bytes  = (6LL+4*5*TOTAL_ITERATIONS+6)*size*sizeof(float);
  const long long operations_128bit = (6LL+4*5*TOTAL_ITERATIONS+6)*size/4;

  printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n", 
    (double)operations_bytes / time_shmem_128b,
    (double)operations_128bit / time_shmem_128b);
}
