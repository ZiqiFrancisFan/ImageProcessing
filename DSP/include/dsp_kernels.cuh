#ifndef DSP_KERNELS_CUH
#define DSP_KERNELS_CUH

template <typename T>
__device__ T* shared_memory_proxy()
{
    // do we need an __align__() here? I don't think so...
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

/* This is a template that implements filtering of a signal by a filter. In this implementation,
    both the input and the filter are first loaded into shared memory and then filtering
    happens at the level of shared memory. In the end, output is written to global. */

template <typename DataType>
__global__ void conv1d(const DataType* input, const int signalStride, // input signal and its stride
    const int lb, const int ub, // domain of input signal
    const int outputChunkSize, // chunk size of output signal
    const DataType* filter, const int filterStride, // filter and its stride
    DataType* output, int* outLowerBound, int* outUpperBound)
{
    /* Input signal and filter are cached in shared memory. */
    DataType* cachedData = shared_memory_proxy<DataType>();

    /* Get radius of filter. */
    int filterRadius = filterStride / 2;

    /* Determine domain of output signal from input domain and filter radius. */
    auto outputLowerBound = lb + filterRadius;
    auto outputUpperBound = ub - filterRadius;

    /* Pass out domain of output sigal. */
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *outLowerBound = outputLowerBound;
        *outUpperBound = outputUpperBound;
    }

    /* If domain is empty set, then there is no need to proceed. */
    if (outputUpperBound <= outputLowerBound) return;

    /* Get chunks of cache for data of different purposes. */
    DataType* cachedInput = cachedData; // input data is stored at the start
    DataType* cachedFilter = cachedInput + (outputChunkSize + 2 * filterRadius); // filter comes next

    /* Load filter into shared memory. */
    for (auto i = threadIdx.x; i < filterStride; i += blockDim.x)
    {
        cachedFilter[i] = filter[i];
    }
    __syncthreads(); // block synchronization

    /* Calculate the total number of chunks. Note that domain of ouput cannot be empty here. */
    int numChunk = (outputUpperBound - outputLowerBound + outputChunkSize - 1) / outputChunkSize;

    /* Each thread first navigate to a particular chunk. */
    for (auto chunkIndex = blockIdx.x; chunkIndex < numChunk; chunkIndex += gridDim.x)
    {
        /* Initialize shared memory of input to zero. */
        for (auto i = threadIdx.x; i < outputChunkSize + 2 * filterRadius; i += blockDim.x)
        {
            cachedInput[i] = static_cast<DataType>(0);
        }
        __syncthreads();

        /* Calculate the domain of output signal for the current block of threads. */
        int chunkOutputStartIndex = (outputLowerBound + chunkIndex * outputChunkSize <= outputUpperBound) ? 
            outputLowerBound + chunkIndex * outputChunkSize : outputUpperBound;

        int chunkOutputEndIndex = (chunkOutputStartIndex + outputChunkSize <= outputUpperBound) ? 
            chunkOutputStartIndex + outputChunkSize : outputUpperBound;

        if (chunkOutputStartIndex >= chunkOutputEndIndex) continue;

        /* Load input signal into shared memory. */
        for (auto inputSignalIndex = chunkOutputStartIndex - filterRadius + threadIdx.x; 
            inputSignalIndex < chunkOutputEndIndex + filterRadius; inputSignalIndex += blockDim.x)
        {
            cachedInput[inputSignalIndex - (chunkOutputStartIndex - filterRadius)] = input[inputSignalIndex];
        }
        __syncthreads();
        
        /* Calculate filtering and write output to global memory. */
        for (auto outputSignalIndex = chunkOutputStartIndex + threadIdx.x;
            outputSignalIndex < chunkOutputEndIndex; outputSignalIndex += blockDim.x)
        {
            DataType filteredValue = static_cast<DataType>(0);

            for (auto filterIndex = 0; filterIndex < filterStride; filterIndex++)
            {
                filteredValue += 
                    cachedFilter[filterIndex] * cachedInput[outputSignalIndex - chunkOutputStartIndex + filterIndex];
            }

            output[outputSignalIndex] = filteredValue;
        }
        __syncthreads();
    }
}

#endif