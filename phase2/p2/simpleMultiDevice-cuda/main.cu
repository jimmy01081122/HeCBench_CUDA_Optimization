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

#include "simpleMultiDevice.h"

// Data configuration
#ifndef MAX_GPU_COUNT 
  #define MAX_GPU_COUNT 8
#endif

const int DATA_N = 1048576 * 32;

__global__
void reduceKernel(float *d_Result, const float *d_Input, int N)
{
  __shared__ float partial[256];

  const int tid = threadIdx.x;
  const int globalTid = blockIdx.x * blockDim.x + tid;
  const int threadN = gridDim.x * blockDim.x;
  float sum = 0;

  for (int pos = globalTid; pos < N; pos += threadN)
    sum += d_Input[pos];

  partial[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
  {
    if (tid < stride)
      partial[tid] += partial[tid + stride];
    __syncthreads();
  }

  if (tid < 32)
  {
    volatile float *warpPartial = partial;
    warpPartial[tid] += warpPartial[tid + 32];
    warpPartial[tid] += warpPartial[tid + 16];
    warpPartial[tid] += warpPartial[tid + 8];
    warpPartial[tid] += warpPartial[tid + 4];
    warpPartial[tid] += warpPartial[tid + 2];
    warpPartial[tid] += warpPartial[tid + 1];
  }

  if (tid == 0)
    d_Result[blockIdx.x] = partial[0];
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

  float sumGPU;
  double sumCPU, diff;

  int i, j, k, GPU_N;

  const int  BLOCK_N = 1024;
  const int THREAD_N = 256;
  const int  ACCUM_N = BLOCK_N;

  printf("Starting simpleMultiDevice\n");
  cudaGetDeviceCount(&GPU_N);

  GPU_N = MIN(GPU_N, MAX_GPU_COUNT);

  printf("GPU device count: %i\n", GPU_N);

  printf("Generating input data of size %d ...\n\n", DATA_N);

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

  //Create streams for issuing GPU command asynchronously
  //allocate memory (GPU and System page-locked)
  for (i = 0; i < GPU_N; i++)
  {
    cudaSetDevice(i);
    cudaStreamCreate(&plan[i].stream);
    cudaEventCreate(&plan[i].h2dStart);
    cudaEventCreate(&plan[i].h2dStop);
    cudaEventCreate(&plan[i].kernelStop);
    cudaEventCreate(&plan[i].d2hStop);
    //Allocate memory
    cudaMalloc((void **)&plan[i].d_Data, plan[i].dataN * sizeof(float));
    cudaMalloc((void **)&plan[i].d_Sum, ACCUM_N * sizeof(float));
    cudaMallocHost((void **)&plan[i].h_Sum_from_device, ACCUM_N * sizeof(float));
    cudaMallocHost((void **)&plan[i].h_Data, plan[i].dataN * sizeof(float));

    for (j = 0; j < plan[i].dataN; j++)
    {
      plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
    }
  }

  //Start timing and compute on GPU(s)
  printf("Computing with %d GPUs...\n", GPU_N);

  auto start = std::chrono::steady_clock::now();
  float h2dMsTotal = 0.0f;
  float kernelMsTotal = 0.0f;
  float d2hMsTotal = 0.0f;

  for (k = 0; k < repeat; k++)
  {
    //Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (i = 0; i < GPU_N; i++)
    {
      //Set device
      cudaSetDevice(i);

      //Copy input data from CPU
      cudaEventRecord(plan[i].h2dStart, plan[i].stream);
      cudaMemcpyAsync(plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof(float),
          cudaMemcpyHostToDevice, plan[i].stream);
      cudaEventRecord(plan[i].h2dStop, plan[i].stream);

      //Perform GPU computations
      reduceKernel<<<BLOCK_N, THREAD_N, 0, plan[i].stream>>>(
          plan[i].d_Sum, plan[i].d_Data, plan[i].dataN);
      cudaEventRecord(plan[i].kernelStop, plan[i].stream);

      //Read back GPU results
      cudaMemcpyAsync(plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N *sizeof(float),
          cudaMemcpyDeviceToHost, plan[i].stream);
      cudaEventRecord(plan[i].d2hStop, plan[i].stream);
    }

    //Process GPU results
    float h2dMsIter = 0.0f;
    float kernelMsIter = 0.0f;
    float d2hMsIter = 0.0f;

    for (i = 0; i < GPU_N; i++)
    {
      //Set device
      cudaSetDevice(i);

      //Wait for all operations to finish
      cudaStreamSynchronize(plan[i].stream);

      float h2dMs = 0.0f, kernelMs = 0.0f, d2hMs = 0.0f;
      cudaEventElapsedTime(&h2dMs, plan[i].h2dStart, plan[i].h2dStop);
      cudaEventElapsedTime(&kernelMs, plan[i].h2dStop, plan[i].kernelStop);
      cudaEventElapsedTime(&d2hMs, plan[i].kernelStop, plan[i].d2hStop);
      h2dMsIter = fmaxf(h2dMsIter, h2dMs);
      kernelMsIter = fmaxf(kernelMsIter, kernelMs);
      d2hMsIter = fmaxf(d2hMsIter, d2hMs);
    }

    h2dMsTotal += h2dMsIter;
    kernelMsTotal += kernelMsIter;
    d2hMsTotal += d2hMsIter;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("  Average GPU Processing time: %f (us)\n\n", time * 1e-3f / repeat); 
  printf("  Average H2D time: %f (us)\n", h2dMsTotal * 1000.0f / repeat);
  printf("  Average kernel time: %f (us)\n", kernelMsTotal * 1000.0f / repeat);
  printf("  Average D2H time: %f (us)\n\n", d2hMsTotal * 1000.0f / repeat);

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
    cudaFreeHost(plan[i].h_Sum_from_device);
    cudaFree(plan[i].d_Sum);
    cudaFree(plan[i].d_Data);
    cudaEventDestroy(plan[i].d2hStop);
    cudaEventDestroy(plan[i].kernelStop);
    cudaEventDestroy(plan[i].h2dStop);
    cudaEventDestroy(plan[i].h2dStart);
    cudaStreamDestroy(plan[i].stream);
  }

  sumGPU = 0;

  for (i = 0; i < GPU_N; i++)
  {
    sumGPU += h_SumGPU[i];
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
    cudaSetDevice(i);
    cudaFreeHost(plan[i].h_Data);
  }

  exit((diff < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE);
}
