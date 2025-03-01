// Q4
// Created by Justin Bahr on 2/24/2025.
// EECE 5640 - High Performance Computing
// Matrix Multiplication using OpenBLAS

#include <iostream>
#include <chrono>
#include <cblas.h>

using namespace std;

const int N = 256;

int main()
{
    int i,j,k,l;

    // creates matrices a, b, and c
    float A[N*N];
    float B[N*N];
    float C[N*N];

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

    // Perform matrix multiplication using cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto runtime = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // outputs dense results
    cout << "A result: " << C[56] << endl; /* prevent dead code elimination */
    cout << "The total time for matrix multiplication: " << runtime << " nanoseconds" << endl;

    return 0;
}
