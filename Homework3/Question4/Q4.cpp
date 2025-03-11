// Q4
// Created by Justin Bahr on 2/24/2025.
// EECE 5640 - High Performance Computing
// Matrix Multiplication using OpenBLAS

#include <iostream>
#include <chrono>
#include <x86intrin.h>
#include <cblas.h>

using namespace std;

const int B = 64;

// function to replicate single-threaded matrix-matrix multiplication from question 3
void stdMultiply(const float a[], const float b[], float e[], int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                e[i * N + j] += a[i * N + k] * b[k * N + j];
}

// function to replicate dense matrix-matrix multiplication from question 3
void denseMultiply(const float a[], const float b[], float d[], int N)
{
    // declares temporary variable sum for loop blocking
    float sum;

    // declares variables for iterations
    int kk, jj, i, j, k;

    // creates a transposed version of b for cache efficiency
    float bT[N*N];

    // transposes matrix b
    #pragma omp parallel for collapse(2) private(i,j)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            bT[j*N+i] = b[i*N+j];

    // matrix multiplication (multi-thread, with loop blocking)
    #pragma omp parallel private(i, j, jj, k, kk, sum)
    {
        #pragma omp for collapse(2)
        for (kk=0; kk<N; kk+=B)
        {
            for (jj=0; jj<N; jj+=B)
                for (i=0; i<N; i++)
                    for (j = jj; j<jj + B; j++)
                    {
                        sum = 0.0;
                        for (k=kk; k<kk + B; k++)
                            sum += a[i*N+k] * bT[j*N+k];
                        d[i*N+j] += sum;
                    }
        }
    } // end parallel region

}

int main()
{
    // defines dimension N
    const int N = 256;

    // creates variable for iteration
    int i,j,k,l;

    // creates matrices A, B, C, and D
    float A[N*N];
    float B[N*N];
    float C[N*N];
    float D[N*N] = {0};
    float E[N*N] = {0};

    /* initialize a dense matrix */
    for(i=0; i<N*N; i++)
    {
            A[i] = (float)(i+1);
            B[i] = (float)(i+1);
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // performs matrix-matrix multiplication using cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto runtime1 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // starts the clock
    start_time = clock::now();

    // performs matrix-matrix multiplication using dense approach from question 3
    denseMultiply(A, B, D, N);

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    auto runtime2 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // starts the clock
    start_time = clock::now();

    // performs matrix-matrix multiplication using dense approach from question 3
    stdMultiply(A, B, E, N);

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    auto runtime3 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // verifies result
    float verify = 0.0;
    for(i=0; i<N; i++)
        verify += A[i]*B[N*i+56];

    // outputs a result
    cout << "An OpenBLAS result: " << C[56] << endl;
    cout << "A dense result: " << D[56] << endl;
    cout << "Standard Result: " << E[56] << endl;
    cout << "Verified result: " << verify << endl << endl;
    cout << "The total time for OpenBLAS matrix multiplication: " << runtime1 << " nanoseconds" << endl;
    cout << "The total time for dense matrix multiplication: " << runtime2 << " nanoseconds" << endl;
    cout << "The total time for standard matrix multiplication: " << runtime3 << " nanoseconds" << endl;
    cout << "OpenBLAS Speedup: " << (float)(runtime3)/runtime1 << endl;
    cout << "Parallel Speedup: " << (float)(runtime3)/runtime2 << endl;

    return 0;
}
