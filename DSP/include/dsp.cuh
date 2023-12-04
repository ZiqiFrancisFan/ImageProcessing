#ifndef DSP_CUH
#define DSP_CUH

#ifndef DEFAULT_SIGNAL_LENGTH
#define DEFAULT_SIGNAL_LENGTH 1024
#endif

#ifndef DEFAULT_OUTPUT_CHUNK_SIZE
#define DEFAULT_OUTPUT_CHUNK_SIZE 128
#endif

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>

#include <curand.h>
#include "dsp_kernels.cuh"

#include "error_handling.h"

#include <thrust/device_vector.h>
#include <cub/cub.cuh>
// #include <cuda/std/atomic>

#ifndef CUDA_CALL
#define CUDA_CALL(x) \
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

    void InitInputSignal();

    ~DspGpuImpl1D();

    void ApplyFilter(const DataType* filter, const int filterSize);
    void ApplyFilterRef(const DataType* filter_h, const int filterSize);
    void UpsampleSignal(const int factor);
    void ResampleSignal(const int M, const int L);
    void ShiftSignal(const float dist);

    void DumpInput(const std::string filename);
    void DumpOutput(const std::string filename);
};

/* Default constructor uses default signal size and domain is an empty set. */
template <typename DataType>
DspGpuImpl1D<DataType>::DspGpuImpl1D()
{
    CUDA_CALL(cudaStreamCreate(&stream_));

    CUDA_CALL(cudaMallocAsync((void**)(&inputSignal_d_), DEFAULT_SIGNAL_LENGTH * sizeof(DataType), stream_));
    CUDA_CALL(cudaMallocAsync((void**)(&outputSignal_d_), DEFAULT_SIGNAL_LENGTH * sizeof(DataType), stream_));

    CUDA_CALL(cudaMallocAsync((void**)(&lowerBound_d_), sizeof(int), stream_));
    CUDA_CALL(cudaMallocAsync((void**)(&upperBound_d_), sizeof(int), stream_));

    inputLowerBound_ = 0;
    inputUpperBound_ = 0;
}

/* Allcoate memory for input and output signals and set up lower and upper bounds of the domain of input. */
template <typename DataType>
DspGpuImpl1D<DataType>::DspGpuImpl1D(const int stride, const int lb, const int ub)
{
    cudaStreamCreate(&stream_);

    cudaMallocAsync((void**)(&inputSignal_d_), stride * sizeof(DataType), stream_);
    cudaMallocAsync((void**)(&outputSignal_d_), stride * sizeof(DataType), stream_);

    cudaMallocAsync((void**)(&lowerBound_d_), sizeof(int), stream_);
    cudaMallocAsync((void**)(&upperBound_d_), sizeof(int), stream_);

    inputLowerBound_ = lb;
    inputUpperBound_ = ub;
    signalStride_ = stride;
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

    CUDA_CALL(cudaMallocAsync((void**)(&inputSignal_d_), signalStride_ * sizeof(DataType), stream_));
    CUDA_CALL(cudaMallocAsync((void**)(&outputSignal_d_), signalStride_ * sizeof(DataType), stream_));

    CUDA_CALL(cudaMallocAsync((void**)(&lowerBound_d_), sizeof(int), stream_));
    CUDA_CALL(cudaMallocAsync((void**)(&upperBound_d_), sizeof(int), stream_));

    inputLowerBound_ = impl.inputLowerBound_;
    inputUpperBound_ = impl.inputUpperBound_;
}

template <typename DataType>
DspGpuImpl1D<DataType>::~DspGpuImpl1D()
{
    if (inputSignal_d_ != nullptr) cudaFreeAsync(inputSignal_d_, stream_);
    if (outputSignal_d_ != nullptr) cudaFreeAsync(outputSignal_d_, stream_);
    if (lowerBound_d_ != nullptr) cudaFreeAsync(lowerBound_d_, stream_);
    if (upperBound_d_ != nullptr) cudaFreeAsync(upperBound_d_, stream_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);

#ifdef DEBUG
    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());
#endif
}

template <>
void DspGpuImpl1D<float>::InitInputSignal()
{
    curandGenerator_t gen;
    CURAND_ERROR_HANDLING(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_ERROR_HANDLING(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_ERROR_HANDLING(curandGenerateUniform(gen, inputSignal_d_, signalStride_));

#ifdef DEBUG
    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());
#endif

    CURAND_ERROR_HANDLING(curandDestroyGenerator(gen));
}

template <>
void DspGpuImpl1D<double>::InitInputSignal()
{
    curandGenerator_t gen;
    CURAND_ERROR_HANDLING(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_ERROR_HANDLING(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_ERROR_HANDLING(curandGenerateUniformDouble(gen, inputSignal_d_, signalStride_));

#ifdef DEBUG
    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());
#endif

    CURAND_ERROR_HANDLING(curandDestroyGenerator(gen));
}

/* Signal filtering using an FIR. */
template <typename DataType>
void DspGpuImpl1D<DataType>::ApplyFilter(const DataType* filter, const int filterSize)
{
    /* Domain of input signal is empty. No need to proceed. */
    if (inputUpperBound_ <= inputLowerBound_) return;

    size_t cacheSize = (DEFAULT_OUTPUT_CHUNK_SIZE + filterSize / 2 * 2 + filterSize) * sizeof(DataType);

    int numBlock = (inputUpperBound_ - inputLowerBound_ - filterSize / 2 * 2 + DEFAULT_OUTPUT_CHUNK_SIZE - 1) / DEFAULT_OUTPUT_CHUNK_SIZE;

    conv1d<DataType><<<numBlock, 32, cacheSize, stream_>>>(inputSignal_d_, signalStride_, inputLowerBound_, inputUpperBound_, 
        DEFAULT_OUTPUT_CHUNK_SIZE, filter, filterSize, outputSignal_d_, lowerBound_d_, upperBound_d_);

#ifdef DEBUG
    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());
#endif
    
    CUDA_ERROR_HANDLING(cudaMemcpyAsync(&outputLowerBound_, lowerBound_d_, sizeof(int), cudaMemcpyDeviceToHost, stream_));

#ifdef DEBUG
    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());
#endif

    cudaMemcpyAsync(&outputUpperBound_, upperBound_d_, sizeof(int), cudaMemcpyDeviceToHost, stream_);

#ifdef DEBUG
    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());
#endif
}

template <typename DataType>
void DspGpuImpl1D<DataType>::ApplyFilterRef(const DataType* filter_h, const int filterSize)
{
    std::unique_ptr<DataType[]> inputSignal_h = std::make_unique<DataType[]>(signalStride_);
    std::unique_ptr<DataType[]> outputSignal_h = std::make_unique<DataType[]>(signalStride_);

    CUDA_ERROR_HANDLING(cudaMemcpyAsync(inputSignal_h.get(), inputSignal_d_, 
        signalStride_ * sizeof(DataType), cudaMemcpyDeviceToHost, stream_));

#ifdef DEBUG
    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());
#endif

    int radius = filterSize / 2;
    int lowerBound = inputLowerBound_ + radius, upperBound = inputUpperBound_ - radius;

    outputLowerBound_ = lowerBound;
    outputUpperBound_ = upperBound;

    if (lowerBound < upperBound) // domain of output not empty
    {
        for (int i{lowerBound}; i < upperBound; i++)
        {
            DataType val = 0;

            for (int j{0}; j < filterSize; j++)
            {
                val += inputSignal_h[i - radius + j] * filter_h[j];
            }

            outputSignal_h[i] = val;
        }

        CUDA_ERROR_HANDLING(cudaMemcpyAsync(outputSignal_d_, outputSignal_h.get(), 
            signalStride_ * sizeof(DataType), cudaMemcpyHostToDevice, stream_));
    }
    else
    {
        return;
    }
}

template <typename DataType>
void DspGpuImpl1D<DataType>::DumpInput(const std::string filename)
{
    std::ofstream file;
    file.open(filename.c_str());
    file << "domain of signal: [" << inputLowerBound_ << ", " << inputUpperBound_ << ")" << std::endl;

    std::unique_ptr<DataType[]> signal_h = std::make_unique<DataType[]>(signalStride_);

    CUDA_ERROR_HANDLING(
        cudaMemcpyAsync(signal_h.get(), 
            inputSignal_d_, 
            signalStride_ * sizeof(DataType), 
            cudaMemcpyDeviceToHost, 
            stream_)
    );

    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());

    for (auto i{0}; i < signalStride_; i++) file << signal_h[i] << std::endl;
}

template <typename DataType>
void DspGpuImpl1D<DataType>::DumpOutput(const std::string filename)
{
    std::ofstream file;
    file.open(filename.c_str());
    file << "domain of signal: [" << outputLowerBound_ << ", " << outputUpperBound_ << ")" << std::endl;

    std::unique_ptr<DataType[]> signal_h = std::make_unique<DataType[]>(signalStride_);

    CUDA_ERROR_HANDLING(
        cudaMemcpyAsync(signal_h.get(),
            outputSignal_d_,
            signalStride_ * sizeof(DataType), 
            cudaMemcpyDeviceToHost,
            stream_)
    );

    CUDA_ERROR_HANDLING(cudaDeviceSynchronize());

    for (auto i{0}; i < signalStride_; i++) file << signal_h[i] << std::endl;
}

#endif
