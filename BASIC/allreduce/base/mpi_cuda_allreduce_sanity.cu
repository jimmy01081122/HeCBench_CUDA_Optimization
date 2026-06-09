#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do {                                      \
    cudaError_t err = (x);                                      \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error " << __FILE__ << ":"           \
                  << __LINE__ << ": "                           \
                  << cudaGetErrorString(err) << std::endl;      \
        MPI_Abort(MPI_COMM_WORLD, 1);                           \
    }                                                           \
} while (0)

int local_rank_from_env()
{
    const char *env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (env == nullptr) {
        env = std::getenv("SLURM_LOCALID");
    }
    if (env == nullptr) {
        throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK or SLURM_LOCALID");
    }
    return std::stoi(std::string(env));
}

int main(int argc, char **argv)
{
    const int local_rank = local_rank_from_env();
    CUDA_CHECK(cudaSetDevice(local_rank));

    MPI_Init(&argc, &argv);

    int rank = 0;
    int world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const size_t count = 1024;
    std::vector<float> host(count, 1.0f);
    std::vector<float> result(count, 0.0f);

    float *send = nullptr;
    float *recv = nullptr;
    CUDA_CHECK(cudaMalloc(&send, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(send, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recv, 0, count * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    int mpi_rc = MPI_Allreduce(send, recv, int(count), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (mpi_rc != MPI_SUCCESS) {
        std::cerr << "CUDA-aware MPI_Allreduce sanity FAIL: MPI_Allreduce returned "
                  << mpi_rc << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), recv, count * sizeof(float), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (size_t i = 0; i < count; i++) {
        if (result[i] != float(world)) {
            std::cerr << "CUDA-aware MPI_Allreduce sanity mismatch rank=" << rank
                      << " index=" << i
                      << " value=" << result[i]
                      << " expected=" << float(world) << std::endl;
            pass = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(send));
    CUDA_CHECK(cudaFree(recv));

    if (rank == 0) {
        std::cout << "CUDA-aware MPI_Allreduce sanity "
                  << (pass ? "PASS" : "FAIL")
                  << " count=" << count
                  << " world_size=" << world << std::endl;
    }

    MPI_Finalize();
    return pass ? 0 : 3;
}
