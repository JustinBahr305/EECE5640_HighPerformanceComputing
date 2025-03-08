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
    double** a = new double*[N];
    double** b = new double*[N];
    double** c = new double*[N];

    for(i=0;i<N;i++)
    {
        a[i] = new double[N];
        b[i] = new double[N];
        c[i] = new double[N];
    }

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
    double sum;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // performs 10 iterations of matrix-matrix multiplication
    for (l=0; l<10; l++)
    {
        // clears matrix c
        #pragma omp parallel for private(i, j, jj, kk)
        for (kk=0; kk<N; kk+=B)
        {
            for (jj=0; jj<N; jj+=B)
                for (i=kk; i<N; i++)
                    for (j = jj; j<jj + B; j++)
                    {
                        c[i][j] = 0.0;
                    }
        } // end parallel region

        // large matrix multiplication (multi-thread, with loop blocking)
        #pragma omp parallel for private(i, j, jj, k, kk, sum)
        for (kk=0; kk<N; kk+=B)
        {
            for (jj=0; jj<N; jj+=B)
                for (i=0; i<N; i++)
                    for (j = jj; j<jj + B; j++)
                    {
                        sum = 0.0;
                        for (k=kk; k<kk + B; k++)
                            sum += a[i][k] * b[k][j];
                        #pragma omp atomic
                        c[i][j] += sum;
                    }
        } // end parallel region
    }

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto runtime = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // verifies result
    double verify = 0.0;
    for(i=0; i<N; i++)
        verify += a[7][i] * b[i][8];

    // outputs dense results
    cout << "A dense result: " << c[7][8] << endl;
    cout << "Verified result: " << verify << endl;
    cout << "The total time for matrix multiplication with dense matrices = " << runtime << " nanoseconds" << endl << endl;

    // initializes a sparse matrix
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

    // stores the number of non-zero values
    int num_values = N*N - num_zeros;

    // creates arrays for the non-zero values
    double *a_values = new double[num_values];
    double *b_values = new double[num_values];

    // creates arrays for the row and column indices
    int *a_rows = new int[N+1];
    int *a_cols = new int[num_values];
    int *b_rows = new int[N+1];
    int *b_cols = new int[num_values];

    // sets first values to zero
    a_rows[0] = 0;
    b_rows[0] = 0;

    // creates variables to track the indices of non-zero elements
    int a_itr = 0;
    int b_itr = 0;

    // puts matrices a and b into CSR format
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            if (a[i][j] != 0.0)
            {
                a_values[a_itr] = a[i][j];
                a_cols[a_itr] = j;
                a_itr++;
            }

            if (b[i][j] != 0.0)
            {
                b_values[b_itr] = b[i][j];
                b_cols[b_itr] = j;
                b_itr++;
            }
        }

        a_rows[i+1] = a_itr;
        b_rows[i+1] = b_itr;
    }

    // performs 10 iterations of matrix-matrix multiplication
    for (l=0; l<LOOPS; l++)
    {
        // multiplies the matrices in parallel
        #pragma omp parallel for private(i, j, k)
        for(i=0; i<N; i++)
        {
            // clears row of matrix c
            for(j=0; j<N; j++)
                c[i][j] = 0.0;

            // multiplies nonzeros
            for (j=a_rows[i]; j<a_rows[i+1]; j++)
            {
                int a_col = a_cols[j];
                double a_val = a_values[j];
                for (k=b_rows[a_col]; k<b_rows[a_col+1]; k++)
                {
                    int b_col = b_cols[k];
                    double b_val = b_values[k];
                    c[i][b_col] += a_val * b_val;
                }
            }
        } // end parallel region
    }

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    runtime = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // verifies result
    verify = 0.0;
    for(i=0; i<N; i++)
        verify += a[1][i] * b[i][207];

    // outputs dense results
    cout << "A sparse result: " << c[1][207] << endl;
    cout << "Verified result: " << verify << endl;
    cout << "The total time for matrix multiplication with sparse matrices = " << runtime << " nanoseconds" << endl;
    cout << "The sparsity of the a and b matrices = " << (float)num_zeros/(float)(N*N) << endl;

    // free memory
    for(i=0;i<N;i++)
    {
        a[i] = nullptr;
        b[i] = nullptr;
        c[i] = nullptr;
    }
    a = nullptr;
    b = nullptr;
    c = nullptr;
    a_values = nullptr;
    b_values = nullptr;
    a_rows = nullptr;
    a_cols = nullptr;
    b_rows = nullptr;
    b_cols = nullptr;

    return 0;
}