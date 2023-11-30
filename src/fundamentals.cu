#include "fundamentals.cuh"
#include "signal.h"
#include <iostream>

__global__ void addVec()
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("test in %s\n", __FUNCTION__);
    }
}

int main(int argc, char* argv[])
{
#ifdef DEBUG
    std::cout << "This is a debug version project." << std::endl;
#else
    std::cout << "This is a release version project." << std::endl;
#endif
    std::cout << "This is a project for image processing by Ziqi." << std::endl;
    addVec<<<1,32,0,0>>>();
    cudaDeviceSynchronize();
}