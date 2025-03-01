// Q3
// Created by Justin Bahr on 2/23/2025.
// EECE 5640 - High Performance Computing
// Matrix Multiplication Optimization

#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;

const int N = 512;
const int LOOPS = 10;
const int B = 32;

int main()
{
    int i,j,k,l,kk,jj, num_zeros;

    // creates NxN matrices a, b, and c
    double a[N][N]; /* input matrix */
    double b[N][N]; /* input matrix */
    double c[N][N]; /* result matrix */

    /* initialize a dense matrix */
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            a[i][j] = (double)(i+j);
            b[i][j] = (double)(i-j);
        }
    }

    cout << "Starting dense matrix multiply" << endl;

    // declares temporary variable sum for loop blocking
    int sum;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    for (l=0; l<LOOPS; l++)
    {
        // large matrix multiplication (multi-thread, with loop blocking)
        #pragma omp parallel for private(i, j, jj, k, kk, sum)
        for (kk=0; kk<N; kk+=B)
            for (jj=0; jj<N; jj+=B)
                for (i=0; i< N; i++)
                    for (j = jj; j< jj + B; j++)
                    {
                        sum = c[i][j];
                        for (k=kk; k< kk + B; k++)
                            sum += a[i][k] * b[k][j];
                        c[i][j] = sum;
                    }
    }

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto runtime = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // outputs dense results
    cout << "A result: " << c[7][8] << endl; /* prevent dead code elimination */
    cout << "The total time for matrix multiplication with dense matrices = " << runtime << " nanoseconds" << endl;

    /* initialize a sparse matrix */
    num_zeros = 0;
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            if ((i<j)&&(i%2>0))
            {
                a[i][j] = (double)(i+j);
                b[i][j] = (double)(i-j);
            }
            else
            {
                num_zeros++;
                a[i][j] = 0.0;
                b[i][j] = 0.0;
            }
        }
    }

    cout << "Starting sparse matrix multiply" << endl;

    // starts the clock
    start_time = clock::now();

    for (l=0; l<LOOPS; l++)
    {
        for(i=0; i<N; i++)
        {
            for(j=0; j<N; j++)
            {
                c[i][j] = 0.0;
                for(k=0; k<N; k++)
                    c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    runtime = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // outputs dense results
    cout << "A result: " << c[7][8] << endl; /* prevent dead code elimination */
    cout << "The total time for matrix multiplication with sparse matrices = " << runtime << endl;;
    cout << "The sparsity of the a and b matrices = " << (float)num_zeros/(float)(N*N) << endl;

    return 0;
}