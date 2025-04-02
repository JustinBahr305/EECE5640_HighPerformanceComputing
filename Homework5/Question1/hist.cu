// hist.cu
// Created by Justin Bahr on 4/2/2025.
// EECE 5640 - High Performance Computing
// Histogramming Kernel on a GPU

#include <cuda_runtime.h>

// defines the block size
const int B = 256;

// defines the maximum value and number of bins
const int NUM_BINS = 32;
const int MAX = 100000;

__global__ void histKernel(int *d_data, int *d_hist, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // creates a shared memory histogram
    __shared__ int sharedHist[NUM_BINS];

    // initialize sets shared histogram bins to zero
    if (threadIdx.x < NUM_BINS)
        sharedHist[threadIdx.x] = 0;
    __syncthreads(); // synchronizes threads

    // processes data in each thread
    if (idx < size)
    {
        // calculates the proper bins and updates atomically
        int bin = (d_data[idx] * NUM_BINS) / (MAX + 1);
        atomicAdd(&sharedHist[bin], 1);
    }
    __syncthreads(); // synchronizes threads

    // copies shared histogram to global memory
    if (threadIdx.x < NUM_BINS)
        atomicAdd(&d_hist[threadIdx.x], sharedHist[threadIdx.x]);
}

void gpuHistogram(int *h_data, int *h_hist, int size)
{
    // size, in bytes, of the data and histogram
    int dataBytes = size*sizeof(int);
    int histBytes = NUM_BINS*sizeof(int);

    // creates pointers for the device data and histogram
    int *d_data, *d_hist;

    // allocates memory on the GPU
    cudaMalloc((void**)&d_data, dataBytes);
    cudaMalloc((void**)&d_hist, histBytes);

    // copys data from the CPU host to the GPU device
    cudaMemcpy(d_data, h_data, dataBytes, cudaMemcpyHostToDevice);

    // initializes an empty device histogram
    cudaMemset(d_hist, 0, histBytes);

    // calculates the block and grid dimensions
    dim3 blockDim(B);
    dim3 gridDim((size+B-1)/B);

    // launches the histogramming kernel
    histKernel<<<gridDim, blockDim>>>(d_data, d_hist, size);

    // copys b tensor from  GPU device to the CPU host
    cudaMemcpy(h_hist, d_hist, histBytes, cudaMemcpyDeviceToHost);

    // free GPU device memory
    cudaFree(d_data);
    cudaFree(d_hist);
}