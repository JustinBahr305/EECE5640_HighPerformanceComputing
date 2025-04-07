// tiled_stencil.cu
// Created by Justin Bahr on 4/3/2025.
// EECE 5640 - High Performance Computing
// Optimized Stencil CUDA Kernel

#include <cuda_runtime.h>

// defines the block and tile size
const int B = 8;
const int TILE_SIZE = B + 2;

__global__ void stencilKernel(float * __restrict__ d_a, const float * __restrict__ d_b, int n) {
    // precomputes n^2
    int n2 = n * n;

    // shared memory tiles with padding
    __shared__ float tile[TILE_SIZE][TILE_SIZE][TILE_SIZE+1];

	// saves thread IDs
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // stores the global tensor indices based on block and thread IDs
    int i = blockIdx.x * B + tx;
    int j = blockIdx.y * B + ty;
    int k = blockIdx.z * B + tz;
    int gidx = i * n2 + j * n + k;

    // deals with outer ring around each tile (halo)
    if (i < n && j < n && k < n)
        tile[tx + 1][ty + 1][tz + 1] = __ldg(&d_b[gidx]);

    // Load halo regions in x, y, and z directions (ensure correct offsets)
    if (tx == 0 && i > 0 && j < n && k < n)
        tile[0][ty + 1][tz + 1] = __ldg(&d_b[gidx - n2]);
    if (tx == B - 1 && i < n - 1 && j < n && k < n)
        tile[B + 1][ty + 1][tz + 1] = __ldg(&d_b[gidx + n2]);

    if (ty == 0 && j > 0 && i < n && k < n)
        tile[tx + 1][0][tz + 1] = __ldg(&d_b[gidx - n]);
    if (ty == B - 1 && j < n - 1 && i < n && k < n)
        tile[tx + 1][B + 1][tz + 1] = __ldg(&d_b[gidx + n]);

    if (tz == 0 && k > 0 && i < n && j < n)
        tile[tx + 1][ty + 1][0] = __ldg(&d_b[gidx - 1]);
    if (tz == B - 1 && k < n - 1 && i < n && j < n)
        tile[tx + 1][ty + 1][B + 1] = __ldg(&d_b[gidx + 1]);

    __syncthreads();

    // Compute stencil only if within the valid range.
    if (i > 0 && i < n - 1 && j > 0 && j < n - 1 && k > 0 && k < n - 1) {
        float sum = tile[tx][ty + 1][tz + 1] +
                    tile[tx + 2][ty + 1][tz + 1] +
                    tile[tx + 1][ty][tz + 1] +
                    tile[tx + 1][ty + 2][tz + 1] +
                    tile[tx + 1][ty + 1][tz] +
                    tile[tx + 1][ty + 1][tz + 2];
        d_a[gidx] = 0.75f * sum;
    }
}

void stencil(float *h_a, float *h_b, int n)
{
	// size, in bytes, of the tensor
	int bytes = n*n*n*sizeof(float);

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