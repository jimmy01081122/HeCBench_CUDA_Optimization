/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs
 */

#include <math.h>
#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#endif

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

#include "simpleMultiDevice.h"

// Data configuration
#ifndef MAX_GPU_COUNT 
  #define MAX_GPU_COUNT 8
#endif

const int DATA_N = 1048576 * 32;

// Simple reduction kernel.
// Refer to the 'reduction' CUDA Sample describing
// reduction optimization strategies
__global__
void reduceKernel(float *d_Result, const float *d_Input, int N)
{
  const int     tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadN = gridDim.x * blockDim.x;
  float sum = 0;

  for (int pos = tid; pos < N; pos += threadN)
    sum += d_Input[pos];

  d_Result[tid] = sum;
}

// Program main
int main(int argc, char **argv)
{
  if (argc != 2)
  {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  //Solver config
  TGPUplan plan[MAX_GPU_COUNT];

  //GPU reduction results
  float h_SumGPU[MAX_GPU_COUNT];

  double sumGPU, sumCPU, diff;

  int i, j, k, GPU_N;

  const int  BLOCK_N = 32;
  const int THREAD_N = 256;
  const int  ACCUM_N = BLOCK_N * THREAD_N;
  const double TOLERANCE = 1e-5;

  printf("Starting simpleMultiDevice\n");
  CUDA_CHECK(cudaGetDeviceCount(&GPU_N));

  GPU_N = MIN(GPU_N, MAX_GPU_COUNT);

  printf("GPU device count: %i\n", GPU_N);

  printf("Generating input data of size %d ...\n\n", DATA_N);
  srand(12345);

  //Subdividing input data across GPUs
  //Get data sizes for each GPU
  for (i = 0; i < GPU_N; i++)
  {
    plan[i].dataN = DATA_N / GPU_N;
  }

  //Take into account "odd" data sizes
  for (i = 0; i < DATA_N % GPU_N; i++)
  {
    plan[i].dataN++;
  }

  //Assign data ranges to GPUs
  for (i = 0; i < GPU_N; i++)
  {
    plan[i].h_Sum = h_SumGPU + i;
  }

  auto init_start = std::chrono::steady_clock::now();

  //Create streams for issuing GPU command asynchronously
  //allocate memory (GPU and System page-locked)
  for (i = 0; i < GPU_N; i++)
  {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaStreamCreate(&plan[i].stream));
    //Allocate memory
    CUDA_CHECK(cudaMalloc((void **)&plan[i].d_Data, plan[i].dataN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&plan[i].d_Sum, ACCUM_N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void **)&plan[i].h_Sum_from_device, ACCUM_N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void **)&plan[i].h_Data, plan[i].dataN * sizeof(float)));

    for (j = 0; j < plan[i].dataN; j++)
    {
      plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
    }
  }

  auto init_end = std::chrono::steady_clock::now();
  auto init_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(init_end - init_start).count();
  double init_us = (double)init_ns * 1e-3;

  //Start timing and compute on GPU(s)
  printf("Computing with %d GPUs...\n", GPU_N);

  // Warm up once so context setup and first launch overhead do not dominate timing.
  for (i = 0; i < GPU_N; i++)
  {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMemcpyAsync(plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof(float),
        cudaMemcpyHostToDevice, plan[i].stream));
    reduceKernel<<<BLOCK_N, THREAD_N, 0, plan[i].stream>>>(
        plan[i].d_Sum, plan[i].d_Data, plan[i].dataN);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpyAsync(plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N * sizeof(float),
        cudaMemcpyDeviceToHost, plan[i].stream));
  }

  for (i = 0; i < GPU_N; i++)
  {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaStreamSynchronize(plan[i].stream));
  }

  long long h2d_ns = 0;
  long long kernel_ns = 0;
  long long d2h_ns = 0;
  auto total_start = std::chrono::steady_clock::now();

  for (k = 0; k < repeat; k++)
  {
    auto h2d_start = std::chrono::steady_clock::now();
    for (i = 0; i < GPU_N; i++)
    {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaMemcpyAsync(plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof(float),
          cudaMemcpyHostToDevice, plan[i].stream));
    }
    for (i = 0; i < GPU_N; i++)
    {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaStreamSynchronize(plan[i].stream));
    }
    auto h2d_end = std::chrono::steady_clock::now();
    h2d_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(h2d_end - h2d_start).count();

    auto kernel_start = std::chrono::steady_clock::now();
    for (i = 0; i < GPU_N; i++)
    {
      CUDA_CHECK(cudaSetDevice(i));
      reduceKernel<<<BLOCK_N, THREAD_N, 0, plan[i].stream>>>(
          plan[i].d_Sum, plan[i].d_Data, plan[i].dataN);
      CUDA_KERNEL_CHECK();
    }
    for (i = 0; i < GPU_N; i++)
    {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaStreamSynchronize(plan[i].stream));
    }
    auto kernel_end = std::chrono::steady_clock::now();
    kernel_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count();

    auto d2h_start = std::chrono::steady_clock::now();
    for (i = 0; i < GPU_N; i++)
    {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaMemcpyAsync(plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N * sizeof(float),
          cudaMemcpyDeviceToHost, plan[i].stream));
    }
    for (i = 0; i < GPU_N; i++)
    {
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaStreamSynchronize(plan[i].stream));
    }
    auto d2h_end = std::chrono::steady_clock::now();
    d2h_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(d2h_end - d2h_start).count();
  }

  auto total_end = std::chrono::steady_clock::now();
  auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start).count();

  double total_us = (double)total_ns * 1e-3 / (double)repeat;
  double h2d_us = (double)h2d_ns * 1e-3 / (double)repeat;
  double kernel_us = (double)kernel_ns * 1e-3 / (double)repeat;
  double d2h_us = (double)d2h_ns * 1e-3 / (double)repeat;

  printf("  Initialization/Allocation time: %f (us)\n", init_us);
  printf("  Average GPU Processing time: %f (us)\n", total_us);
  printf("  Average H2D copy time: %f (us)\n", h2d_us);
  printf("  Average kernel time: %f (us)\n", kernel_us);
  printf("  Average D2H partial copy time: %f (us)\n\n", d2h_us);

  for (i = 0; i < GPU_N; i++)
  {
    //Finalize GPU reduction for current subvector
    float sum = 0;

    for (j = 0; j < ACCUM_N; j++)
    {
      sum += plan[i].h_Sum_from_device[j];
    }

    *(plan[i].h_Sum) = sum;

    //Shut down this GPU
    CUDA_CHECK(cudaFreeHost(plan[i].h_Sum_from_device));
    CUDA_CHECK(cudaFree(plan[i].d_Sum));
    CUDA_CHECK(cudaFree(plan[i].d_Data));
    CUDA_CHECK(cudaStreamDestroy(plan[i].stream));
  }

  sumGPU = 0;

  for (i = 0; i < GPU_N; i++)
  {
    sumGPU += (double)h_SumGPU[i];
  }

  // Compute on Host CPU
  printf("Computing with Host CPU...\n\n");

  sumCPU = 0;

  for (i = 0; i < GPU_N; i++)
  {
    for (j = 0; j < plan[i].dataN; j++)
    {
      sumCPU += plan[i].h_Data[j];
    }
  }

  // Compare GPU and CPU results
  printf("Comparing GPU and Host CPU results...\n");
  diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
  printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
  printf("  Relative difference: %E \n\n", diff);

  // Cleanup and shutdown
  for (i = 0; i < GPU_N; i++)
  {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaFreeHost(plan[i].h_Data));
  }

  const char *status = (diff < TOLERANCE) ? "PASS" : "FAIL";
  printf("Status: %s\n", status);
  printf("CORRECTNESS,num_gpus=%d,gpu_sum=%.9f,cpu_sum=%.9f,relative_diff=%.9e,tolerance=%.1e,status=%s\n",
         GPU_N, sumGPU, sumCPU, diff, TOLERANCE, status);
  printf("RESULT,num_gpus=%d,repeat=%d,total_us=%.6f,h2d_us=%.6f,kernel_us=%.6f,d2h_us=%.6f,diff=%.9e,status=%s\n",
         GPU_N, repeat, total_us, h2d_us, kernel_us, d2h_us, diff, status);

  exit((diff < TOLERANCE) ? EXIT_SUCCESS : EXIT_FAILURE);
}
