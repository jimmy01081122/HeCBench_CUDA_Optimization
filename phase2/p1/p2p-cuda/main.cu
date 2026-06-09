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
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define GPU_CHECK(x) do {                                    \
    cudaError_t err = x;                                     \
    if (err != cudaSuccess) {                                \
        printf("CUDA error %s:%d: %s\n",                     \
               __FILE__, __LINE__, cudaGetErrorString(err)); \
    }                                                        \
} while (0)

inline bool IsAppBuiltAs64()
{
  return sizeof(void*) == 8;
}

bool pair_access(int src_dev, int dst_dev, int repeat)
{
  bool src_peer_enabled = false;
  bool dst_peer_enabled = false;

  // Enable peer access
#ifdef DEBUG
  printf("Enabling peer access from GPU%d to GPU%d...\n", src_dev, dst_dev);
#endif
  GPU_CHECK(cudaSetDevice(dst_dev));
  cudaError_t peer_err = cudaDeviceEnablePeerAccess(src_dev, 0);
  if (peer_err != cudaSuccess && peer_err != cudaErrorPeerAccessAlreadyEnabled)
  {
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(peer_err));
  }
  else
  {
    dst_peer_enabled = true;
  }

  int reverse_access = 0;
  GPU_CHECK(cudaDeviceCanAccessPeer(&reverse_access, src_dev, dst_dev));
  if (reverse_access)
  {
    GPU_CHECK(cudaSetDevice(src_dev));
    peer_err = cudaDeviceEnablePeerAccess(dst_dev, 0);
    if (peer_err != cudaSuccess && peer_err != cudaErrorPeerAccessAlreadyEnabled)
    {
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(peer_err));
    }
    else
    {
      src_peer_enabled = true;
    }
  }

  // Allocate buffers
  const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
  printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n",
         int(buf_size / 1024 / 1024), src_dev, dst_dev);

  GPU_CHECK(cudaSetDevice(src_dev));
  float *src;
  GPU_CHECK(cudaMalloc(&src, buf_size));

  float *host;
  GPU_CHECK(cudaMallocHost(&host, buf_size)); // Automatically portable with UVA

  GPU_CHECK(cudaSetDevice(dst_dev));
  float *dst;
  GPU_CHECK(cudaMalloc(&dst, buf_size));

  const int buf_len = buf_size / sizeof(float);
  for (int i=0; i<buf_len; i++)
  {
    host[i] = float(i % 4096);
  }

  GPU_CHECK(cudaSetDevice(src_dev));
  GPU_CHECK(cudaMemcpy(src, host, buf_size, cudaMemcpyHostToDevice));

  GPU_CHECK(cudaSetDevice(src_dev));
  cudaStream_t stream;
  cudaEvent_t start, stop;
  GPU_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  GPU_CHECK(cudaEventCreate(&start));
  GPU_CHECK(cudaEventCreate(&stop));

  GPU_CHECK(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, buf_size, stream));
  GPU_CHECK(cudaStreamSynchronize(stream));

  GPU_CHECK(cudaEventRecord(start, stream));
  for (int i=0; i<repeat; i++)
  {
    GPU_CHECK(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, buf_size, stream));
  }
  GPU_CHECK(cudaEventRecord(stop, stream));
  GPU_CHECK(cudaEventSynchronize(stop));

  float time_ms = 0.0f;
  GPU_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
  const double bandwidth = (double(repeat) * double(buf_size)) / (double(time_ms) * 1.0e6);

  printf("Peer-to-peer copy GPU%d -> GPU%d: %.2f GB/s\n", src_dev, dst_dev, bandwidth);

  // Copy data back to host and verify the measured direction.
#ifdef DEBUG
  printf("Copy data back to host from GPU%d and verify results...\n", dst_dev);
#endif
  GPU_CHECK(cudaMemcpy(host, dst, buf_size, cudaMemcpyDeviceToHost));

  int error_count = 0;

  for (int i=0; i<buf_len; i++)
  {
    if (host[i] != float(i % 4096))
    {
      printf("Verification error @ element %i: val = %f, ref = %f\n",
             i, host[i], float(i % 4096));

      if (error_count++ > 10)
      {
        break;
      }
    }
  }

  // Disable peer access (also unregisters memory for non-UVA cases)
#ifdef DEBUG
  printf("Disabling peer access...\n");
#endif
  if (dst_peer_enabled)
  {
    GPU_CHECK(cudaSetDevice(dst_dev));
    GPU_CHECK(cudaDeviceDisablePeerAccess(src_dev));
  }
  if (src_peer_enabled)
  {
    GPU_CHECK(cudaSetDevice(src_dev));
    GPU_CHECK(cudaDeviceDisablePeerAccess(dst_dev));
  }

  // Cleanup and shutdown
#ifdef DEBUG
  printf("Shutting down...\n");
#endif
  GPU_CHECK(cudaEventDestroy(start));
  GPU_CHECK(cudaEventDestroy(stop));
  GPU_CHECK(cudaStreamDestroy(stream));

  GPU_CHECK(cudaSetDevice(src_dev));
  GPU_CHECK(cudaFree(src));
  GPU_CHECK(cudaFreeHost(host));

  GPU_CHECK(cudaSetDevice(dst_dev));
  GPU_CHECK(cudaFree(dst));

  if (error_count != 0)
  {
    printf("FAIL\n");
    return false;
  }
  else
  {
    printf("PASS\n");
    return true;
  }
}

int main(int argc, char **argv)
{
  printf("[%s] - Starting...\n", argv[0]);
  const int repeat = atoi(argv[1]);

  if (!IsAppBuiltAs64())
  {
    printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target.  Test is being waived.\n", argv[0]);
    exit(0);
  }

  // Number of GPUs
  printf("Checking for multiple GPUs...\n");
  int gpu_n;
  GPU_CHECK(cudaGetDeviceCount(&gpu_n));
  printf("There are %d GPUs\n", gpu_n);

  if (gpu_n < 2)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
    printf("Waiving test.\n");
    exit(0);
  }

  // Query device properties
  cudaDeviceProp prop[64];

  for (int i=0; i < gpu_n; i++)
  {
    GPU_CHECK(cudaGetDeviceProperties(&prop[i], i));
  }
  // Check possibility for peer access
  printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

  int can_access_peer;
  int p2pCapableGPUs[2] = {-1, -1}; // We take only 1 pair of P2P capable GPUs

  // Show all the combinations of supported P2P GPUs
  for (int i = 0; i < gpu_n; i++)
  {
    for (int j = 0; j < gpu_n; j++)
    {
      if (i == j)
      {
        continue;
      }
      GPU_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, i, j));
      printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[i].name, i,
             prop[j].name, j, can_access_peer ? "Yes" : "No");
      if (can_access_peer && p2pCapableGPUs[0] == -1)
      {
        p2pCapableGPUs[0] = i;
        p2pCapableGPUs[1] = j;
      }
    }
  }

  if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
    printf("Peer to Peer access is not available amongst GPUs in the system, waiving test.\n");
    exit(0);
  }

  printf("\nMeasuring the bandwidth of peer to peer memory access...\n");
  int failures = 0;
  for (int i = 0; i < gpu_n; i++)
  {
    for (int j = 0; j < gpu_n; j++)
    {
      if (i == j)
      {
        continue;
      }
      GPU_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, i, j));
      if (can_access_peer)
      {
        printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : \n", prop[i].name, i,
               prop[j].name, j);
        if (!pair_access(i, j, repeat))
        {
          failures++;
        }
      }
    }
  }

  return failures == 0 ? 0 : 1;
}
