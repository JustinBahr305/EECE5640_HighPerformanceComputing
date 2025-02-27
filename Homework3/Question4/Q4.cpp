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
    double a[N][N];
    double b[N][N];
    double c[N][N];

    /* initialize a dense matrix */
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            a[i][j] = (double)(i+j);
            b[i][j] = (double)(i-j);
        }
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // Perform matrix multiplication using cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto runtime = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // outputs dense results
    cout << "A result: " << c[7][8] << endl; /* prevent dead code elimination */
    cout << "The total time for matrix multiplication: " << runtime << " nanoseconds" << endl;

    return 0;
}
