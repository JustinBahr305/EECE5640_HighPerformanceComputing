// hist.cu
// Created by Justin Bahr on 4/2/2025.
// EECE 5640 - High Performance Computing
// Histogramming Kernel on a GPU

#include <cuda_runtime.h>

// defines the block size
const int B = 512;

// defines the maximum value and number of bins
const int NUM_BINS = 32;
const int MAX = 100000;

// defines the warp size and number of warps
const int WARP_SIZE = 32;
const int NUM_WARPS = B / WARP_SIZE;

__global__ void histKernel(int *d_data, int *d_hist, int size)
{
    // defines the thread ID and stride size
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // calculates IDs for each warp and each lane within each warp
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    // each warp gets a histogram, so 2D shared histogram
    __shared__ int sharedHist[NUM_WARPS][NUM_BINS];

    // sets each warp's histogram to zero
    for (int i = laneId; i < NUM_BINS; i += WARP_SIZE)
        sharedHist[warpId][i] = 0;
    __syncthreads();

    // strided data processing per thread to streamline
    for (int i = idx; i < size; i += stride)
    {
        int bin = (d_data[i] * NUM_BINS) / (MAX + 1);
        atomicAdd(&sharedHist[warpId][bin], 1);
    }
    __syncthreads();

    // private histograms are reduced and saved in the global histogram
    for (int i = tid; i < NUM_BINS; i += blockDim.x)
    {
        int total = 0;
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            total += sharedHist[w][i];
        }
        atomicAdd(&d_hist[i], total);
    }
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