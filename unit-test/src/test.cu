#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "SignalProcessing.h"
#include "dsp.cuh"
#include <iostream>

TEST_CASE("trial")
{
    std::cout << "Hello!" << std::endl;
}

TEST_CASE("Signal Processing")
{
    Signal1D<float> signal(128, {0, 127});
}

TEST_CASE("DSP")
{
    std::cout << "Testing DSP." << std::endl;
    DspGpuImpl1D<float> dsp(1024, 0, 1024);
    dsp.InitInputSignal();

    dsp.DumpInput("input.txt");

    float* filter_d = nullptr;
    int stride = 7;
    cudaError_t e = cudaMalloc((void**)&filter_d, stride * sizeof(float));

    curandGenerator_t gen;
    CURAND_ERROR_HANDLING(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_ERROR_HANDLING(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_ERROR_HANDLING(curandGenerateUniform(gen, filter_d, stride));

    std::unique_ptr<float[]> filter_h = std::make_unique<float[]>(stride);
    CUDA_ERROR_HANDLING(cudaMemcpy(filter_h.get(), filter_d, stride * sizeof(float), cudaMemcpyDeviceToHost));

    dsp.ApplyFilter(filter_d, stride);

    dsp.DumpOutput("gpu_output.txt");

    dsp.ApplyFilterRef(filter_h.get(), stride);

    dsp.DumpOutput("cpu_output.txt");

    e = cudaFree(filter_d);
}


