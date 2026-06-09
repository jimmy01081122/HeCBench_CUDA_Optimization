#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>
#include <nccl.h>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
  do{                                                                                     \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
  }while(0)


__global__
void test(double *d, const long int n) {
  for (long i = blockDim.x * blockIdx.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) {
    d[i] = d[i] + 1;
  }
}


int main(int argc, char *argv[])
{
  /* -------------------------------------------------------------------------------------------
     MPI Initialization 
     --------------------------------------------------------------------------------------------*/
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(size != 2){
    if(rank == 0){
      printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
    }
    MPI_Finalize();
    exit(0);
  }

  // Map MPI ranks to GPUs
  int num_devices = 0;
  cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
  cudaErrorCheck( cudaSetDevice(rank % num_devices) );

  //initialize NCCL
  ncclComm_t comm;
  ncclUniqueId id;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
  cudaStream_t stream;
  cudaErrorCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  //   Loop from 65536 B to 1 GB
  for(int i=16; i<=27; i++){

    long int N = 1 << i;

    double *h_A, *d_A;
    h_A = (double*) malloc (N*sizeof(double)); 
    cudaErrorCheck( cudaMalloc((void**)&d_A, N*sizeof(double)) );
    cudaErrorCheck( cudaMemset(d_A, 0, N*sizeof(double)) );
    cudaErrorCheck( cudaDeviceSynchronize() );

    int loop_count = 50;

    // Warm-up and validate NCCL pingpong
    for(int i=1; i<=5; i++){
      if(rank == 0){
        NCCLCHECK(ncclSend(d_A, N, ncclFloat64, 1, comm, stream));
        NCCLCHECK(ncclRecv(d_A, N, ncclFloat64, 1, comm, stream));
      }
      else if(rank == 1){
        NCCLCHECK(ncclRecv(d_A, N, ncclFloat64, 0, comm, stream));
        test<<<1024, 256, 0, stream>>>(d_A, N);
        cudaErrorCheck( cudaStreamSynchronize(stream) );
        NCCLCHECK(ncclSend(d_A, N, ncclFloat64, 0, comm, stream));
      }
    }
    cudaErrorCheck(cudaStreamSynchronize(stream));
    int correct = 1;
    if(rank == 0) {
      cudaErrorCheck(cudaMemcpy(h_A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost));
      for (long int i = 0; i < N; i++) {
        if(h_A[i] != 5) {
          printf("ERROR: NCCL pingpong test failed: %lf\n", h_A[i]);
          correct = 0;
          break;
        }
      }
    }
    MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(!correct) {
      free(h_A);
      cudaErrorCheck( cudaFree(d_A) );
      NCCLCHECK(ncclCommFinalize(comm));
      NCCLCHECK(ncclCommDestroy(comm));
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    free(h_A);

    // Time loop_count iterations of data transfer size 8*N bytes
    long int num_B = 8*N;
    double num_GB = (double)num_B / 1.0e9;
    const int trials = 3;
    double trial_times[trials];

    for(int trial=1; trial<=trials; trial++){
      double start_time, stop_time, elapsed_time;
      cudaErrorCheck(cudaStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);
      start_time = MPI_Wtime();

      for(int i=1; i<=loop_count; i++){
        if(rank == 0){
          NCCLCHECK(ncclSend(d_A, N, ncclFloat64, 1, comm, stream));
          NCCLCHECK(ncclRecv(d_A, N, ncclFloat64, 1, comm, stream));
        }
        else if(rank == 1){
          NCCLCHECK(ncclRecv(d_A, N, ncclFloat64, 0, comm, stream));
          NCCLCHECK(ncclSend(d_A, N, ncclFloat64, 0, comm, stream));
        }
      }
      cudaErrorCheck(cudaStreamSynchronize(stream));

      stop_time = MPI_Wtime();
      elapsed_time = stop_time - start_time;

      double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);
      trial_times[trial-1] = avg_time_per_transfer;

      if(rank == 0)
        printf("RESULT,backend=NCCL,size_bytes=%li,loop_count=%d,trial=%d,avg_time_s=%.9f,gbps=%.9f\n",
               num_B, loop_count, trial, avg_time_per_transfer, num_GB/avg_time_per_transfer );
    }

    if(rank == 0) {
      double sum_time = 0.0;
      double min_time = trial_times[0];
      double max_time = trial_times[0];
      for(int t=0; t<trials; t++){
        sum_time += trial_times[t];
        if(trial_times[t] < min_time) min_time = trial_times[t];
        if(trial_times[t] > max_time) max_time = trial_times[t];
      }
      double avg_time = sum_time / (double)trials;
      printf("SUMMARY,backend=NCCL,size_bytes=%li,loop_count=%d,trials=%d,avg_time_s=%.9f,min_time_s=%.9f,max_time_s=%.9f,gbps=%.9f\n",
             num_B, loop_count, trials, avg_time, min_time, max_time, num_GB/avg_time );
    }

    cudaErrorCheck( cudaFree(d_A) );
  }

  cudaErrorCheck(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPI_Finalize();

  return 0;
}
