#ifndef ERROR_HANDLING
#define ERROR_HANDLING

#include <cuda_runtime.h>
#include <iostream>
#include <curand.h>
#include <sstream>

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

#ifndef CHECK_LAST_CUDA_ERROR
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
#endif

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
#endif

#ifndef CHECK_CURAND_ERROR
#define CHECK_CURAND_ERROR(val) checkcurand((val), #val, __FILE__, __LINE__)
#endif

#ifndef CUDA_ABORT_ASSERT
#define CUDA_ABORT_ASSERT(x) { gpuAssert((x), __FILE__, __LINE__); }
#endif

#ifndef CUDA_THROW
#define CUDA_THROW(x) throwOnCudaError((x), __FILE__, __LINE__)
#endif

#ifndef CURAND_THROW
#define CURAND_THROW(x) throwOnCurandError((x), __FILE__, __LINE__)
#endif

#ifndef THRUST_CATCH
#define THRUST_CATCH() \
catch (thrust::system_error& e) \
{ \
    std::cerr << e.what() << std::endl; \
    exit(1); \
}
#endif

#ifndef CUDA_ERROR_HANDLING
#define CUDA_ERROR_HANDLING(x) \
try \
{ \
    CUDA_THROW(x); \
} \
THRUST_CATCH();
#endif

#ifndef CURAND_ERROR_HANDLING
#define CURAND_ERROR_HANDLING(x) \
try \
{ \
    CURAND_THROW(x); \
} \
THRUST_CATCH();
#endif

template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;

        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;

        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
void checkcurand(T err, const char* const func, const char* const file, const int line)
{
    if (err != CURAND_STATUS_SUCCESS)
    {
        std::cerr << "CURAND Runtime Error at: " << file << ":" << line << std::endl;

        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;

        std::exit(EXIT_FAILURE);
    }
}

void checkLast(const char* const file, const int line);

void gpuAssert(cudaError_t code, const char* file, int line, bool abort);

void throwOnCudaError(cudaError_t code, const char* file, int line);

void throwOnCurandError(curandStatus_t code, const char* file, int line);

#endif