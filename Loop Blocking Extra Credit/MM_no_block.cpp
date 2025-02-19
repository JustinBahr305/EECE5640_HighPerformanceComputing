// MM_block
// Created by Justin Bahr on 2/18/2025.
// EECE 5640 - High Performance Computing
// Matrix Multiplication without Loop Blocking

#include <iostream>
#include <chrono>

using namespace std;

const int M = 2048;
const int M1 = 32;
const int B = 64;
const int B1 = 8;

int main()
{
    // declares iterators
    int i,j,k,jj,kk,en;

    // declares temporary variable sum for loop blocking
    int sum;

    // creates large matrices
    int** a = new int*[M];
    int** b = new int*[M];
    int** c = new int*[M];

    for(i=0;i<M;i++)
    {
        a[i] = new int[M];
        b[i] = new int[M];
        c[i] = new int[M];
    }

    // fills large matrices a, b, and c
    for (i=0; i<M; i++)
        for (j=0; j<M; j++)
            a[i][j] = 1;

    for (i=0; i<M; i++)
        for (j=0; j<M; j++)
            b[i][j] = 2;

    for (i=0; i<M; i++)
        for (j=0; j<M; j++)
            c[i][j] = 0;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // matrix multiplication without loop blocking
    for (i=0; i<M; i++)
        for (j=0; j<M; j++)
            for (k=0; k<M; k++)
                c[i][j] += + a[i][k] * b[k][j];

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Check c[511][511] = " << c[M-1][M-1] << endl;

    // prints runtimes
    cout << "Time to execute without loop blocking: " << run_time << " ns" << endl;

    // free memory
    for(i=0;i<M;i++)
    {
        a[i] = nullptr;
        b[i] = nullptr;
        c[i] = nullptr;
    }
    a = nullptr;
    b = nullptr;
    c = nullptr;

    return 0;
}
