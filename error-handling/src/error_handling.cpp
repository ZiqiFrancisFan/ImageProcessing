#include "error_handling.h"
#include <cuda_runtime.h>

void checkLast(const char* const file, const int line)
{
#ifdef DEBUG
    cudaDeviceSynchronize();
#endif

    cudaError_t err{cudaGetLastError()};

    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;

        std::cerr << cudaGetErrorString(err) << std::endl;

        std::exit(EXIT_FAILURE);
    }
}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void throwOnCudaError(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::stringstream ss;
        ss << file << "(" << line << ")";
        std::string file_and_line;
        ss >> file_and_line;
        throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }
}

void throwOnCurandError(curandStatus_t code, const char* file, int line)
{
    if (code != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << file << "(" << ")";
        std::string file_and_line;
        ss >> file_and_line;
        throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }
}