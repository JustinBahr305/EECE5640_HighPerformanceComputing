// tiled_stencil.cu
// Created by Justin Bahr on 3/28/2025.
// EECE 5640 - High Performance Computing
// Tiled Stencil CUDA Kernel

#include <cuda_runtime.h>

using namespace std;

// defines the block/tile size
const int B = 8;

__global__ void stencilKernel(float *d_a, float *d_b, int n)
{
    // creates a block/tile in shared memory
	__shared__ float tile[B + 2][B + 2][B + 2];

    // stores the global tensor indices based on block and thread IDs
    int i = blockIdx.x * B + threadIdx.x;
    int j = blockIdx.y * B + threadIdx.y;
    int k = blockIdx.z * B + threadIdx.z;

    // deals with outer ring around each tile (halo)
    if (i < n && j < n && k < n)
    {
        tile[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 1] = d_b[i * n * n + j * n + k];

        if (threadIdx.x == 0 && i > 0)
            tile[0][threadIdx.y + 1][threadIdx.z + 1] = d_b[(i - 1) * n * n + j * n + k];
        if (threadIdx.x == B - 1 && i < n - 1)
            tile[B + 1][threadIdx.y + 1][threadIdx.z + 1] = d_b[(i + 1) * n * n + j * n + k];

        if (threadIdx.y == 0 && j > 0)
            tile[threadIdx.x + 1][0][threadIdx.z + 1] = d_b[i * n * n + (j - 1) * n + k];
        if (threadIdx.y == B - 1 && j < n - 1)
            tile[threadIdx.x + 1][B + 1][threadIdx.z + 1] = d_b[i * n * n + (j + 1) * n + k];

        if (threadIdx.z == 0 && k > 0)
            tile[threadIdx.x + 1][threadIdx.y + 1][0] = d_b[i * n * n + j * n + (k - 1)];
        if (threadIdx.z == B - 1 && k < n - 1)
            tile[threadIdx.x + 1][threadIdx.y + 1][B + 1] = d_b[i * n * n + j * n + (k + 1)];

        // synchronizes threads
        __syncthreads();

        // combines the tiled results
        if (i > 0 && i < n - 1 && j > 0 && j < n - 1 && k > 0 && k < n - 1)
        {
            d_a[i * n * n + j * n + k] = 0.75f * (
                tile[threadIdx.x][threadIdx.y + 1][threadIdx.z + 1] +
                tile[threadIdx.x + 2][threadIdx.y + 1][threadIdx.z + 1] +
                tile[threadIdx.x + 1][threadIdx.y][threadIdx.z + 1] +
                tile[threadIdx.x + 1][threadIdx.y + 2][threadIdx.z + 1] +
                tile[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z] +
                tile[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 2]
            );
        }
    }
}

void stencil(float *h_a, float *h_b, int n)
{
	// size, in bytes, of the tensor
	size_t bytes = n*n*n*sizeof(float);

    // creates pointers for device tensors
    float *d_a, *d_b;

    // allocates memory on the GPU
	cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);

    // copys b tensor from the CPU host to the GPU device
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // calculates the grid and block dimensions
    dim3 blockDim(B, B, B);
    dim3 gridDim((n + B - 1) / B, (n + B - 1) / B, (n + B - 1) / B);

    // launches the stenciling kernel
    stencilKernel<<<gridDim, blockDim>>>(d_a, d_b, n);

    // copys b tensor from  GPU device to the CPU host
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    // free GPU device memory
    cudaFree(d_a);
    cudaFree(d_b);
}