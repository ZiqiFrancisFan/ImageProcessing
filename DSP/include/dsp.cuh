#ifndef DSP_CUH
#define DSP_CUH

#ifndef DEFAULT_SIGNAL_LENGTH
#define DEFAULT_SIGNAL_LENGTH 1024
#endif

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include "dsp_kernels.cuh"

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(x) \
do \
{ \
    x; \
    cudaDeviceSynchronize(); \
    try \
    { \
        cudaError_t e = cudaGetLastError(); \
        if (e != cudaSuccess) \
        { \
            throw e; \
        } \
    } \
    catch (const cudaError_t& e) \
    { \
        printf("CUDA error %s identified at line %s in file %s\n", cudaGetErrorString(e), __LINE__, __FILE__); \
        exit(1); \
    } \
} \
while (0)
#endif

template <typename DataType>
class DspGpuImpl1D
{
private:
    DataType* inputSignal_d_ = nullptr; // input signal on device
    DataType* outputSignal_d_ = nullptr; // output signal on device
    int signalStride_ = 0; // length of the signal array

    int inputLowerBound_ = 0; // lower bound of signal
    int inputUpperBound_ = 0; // upper bound of a signal

    int outputLowerBound_ = 0;
    int outputUpperBound_ = 0;

    int* lowerBound_d_ = nullptr; // device buffer for domain of signal, lower bound
    int* upperBound_d_ = nullptr; // device buffer for domain of signal, upper bound

    cudaStream_t stream_ = nullptr;

public:
    DspGpuImpl1D();
    DspGpuImpl1D(DspGpuImpl1D&&);
    DspGpuImpl1D(const DspGpuImpl1D&);
    DspGpuImpl1D(const int stride, const int lb, const int ub);

    ~DspGpuImpl1D();

    void ApplyFilterToSignal(const DataType* filter, const int filterSize);
    void UpsampleSignal(const int factor);
    void ResampleSignal(const int M, const int L);
    void ShiftSignal(const float dist);
};

/* Default constructor uses default signal size and domain is an empty set. */
template <typename DataType>
DspGpuImpl1D<DataType>::DspGpuImpl1D()
{
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));

    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&inputSignal_d_), DEFAULT_SIGNAL_LENGTH * sizeof(DataType), stream_));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&outputSignal_d_), DEFAULT_SIGNAL_LENGTH * sizeof(DataType), stream_));

    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&lowerBound_d_), sizeof(int), stream_));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&upperBound_d_), sizeof(int), stream_));

    inputLowerBound_ = 0;
    inputUpperBound_ = 0;
}

/* Allcoate memory for input and output signals and set up lower and upper bounds of the domain of input. */
template <typename DataType>
DspGpuImpl1D<DataType>::DspGpuImpl1D(const int stride, const int lb, const int ub)
{
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));

    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&inputSignal_d_), stride * sizeof(DataType), stream_));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&outputSignal_d_), stride * sizeof(DataType), stream_));

    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&lowerBound_d_), sizeof(int), stream_));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&upperBound_d_), sizeof(int), stream_));

    inputLowerBound_ = lb;
    inputUpperBound_ = ub;
}

template <typename DataType>
DspGpuImpl1D<DataType>::DspGpuImpl1D(DspGpuImpl1D&& impl)
{
    inputSignal_d_ = impl.inputSignal_d_;
    outputSignal_d_ = impl.outputSignal_d_;
    lowerBound_d_ = impl.lowerBound_d_;
    upperBound_d_ = impl.upperBound_d_;

    impl.inputSignal_d_ = nullptr;
    impl.outputSignal_d_ = nullptr;
    impl.lowerBound_d_ = nullptr;
    impl.upperBound_d_ = nullptr;

    inputLowerBound_ = impl.inputLowerBound_;
    inputUpperBound_ = impl.inputUpperBound_;
}

template <typename DataType>
DspGpuImpl1D<DataType>::DspGpuImpl1D(const DspGpuImpl1D& impl)
{
    stream_ = impl.stream_;
    signalStride_ = impl.signalStride_;

    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&inputSignal_d_), signalStride_ * sizeof(DataType), stream_));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&outputSignal_d_), signalStride_ * sizeof(DataType), stream_));

    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&lowerBound_d_), sizeof(int), stream_));
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)(&upperBound_d_), sizeof(int), stream_));

    inputLowerBound_ = impl.inputLowerBound_;
    inputUpperBound_ = impl.inputUpperBound_;
}

template <typename DataType>
DspGpuImpl1D<DataType>::~DspGpuImpl1D()
{
    if (inputSignal_d_ != nullptr) CHECK_CUDA_ERROR(cudaFreeAsync(inputSignal_d_, stream_));
    if (outputSignal_d_ != nullptr) CHECK_CUDA_ERROR(cudaFreeAsync(outputSignal_d_, stream_));
    if (lowerBound_d_ != nullptr) CHECK_CUDA_ERROR(cudaFreeAsync(lowerBound_d_, stream_));
    if (upperBound_d_ != nullptr) CHECK_CUDA_ERROR(cudaFreeAsync(upperBound_d_, stream_));

    if (stream_ != nullptr) CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

/* Signal filtering using an FIR. */
template <typename DataType>
void DspGpuImpl1D<DataType>::ApplyFilterToSignal(const DataType* filter, const int filterSize)
{
    
}

#endif
