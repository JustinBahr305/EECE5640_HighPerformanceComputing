// MM_openMP
// Created by Justin Bahr on 2/18/2025.
// EECE 5640 - High Performance Computing
// Matrix Multiplication and Loop Blocking using OpenMP

#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;

const int M = 512;
const int M1 = 64;
const int B = 64;
const int B1 = 16;

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

    // large matrix multiplication (single-thread, no loop blocking)
    for (i=0; i<M; i++)
        for (j=0; j<M; j++)
            for (k=0; k<M; k++)
                c[i][j] += + a[i][k] * b[k][j];

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time1 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Check c[511][511] = " << c[M-1][M-1] << endl;

    // emptys array c
    for (i=0; i<M; i++)
        for (j=0; j<M; j++)
            c[i][j] = 0;

    // starts the clock
    start_time = clock::now();

    // large matrix multiplication (multi-thread, no loop blocking)
    #pragma omp parallel for private (i, j, k)
    for (i=0; i<M; i++)
        for (j=0; j<M; j++)
            for (k=0; k<M; k++)
                c[i][j] += + a[i][k] * b[k][j];

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time2 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Check c[511][511] = " << c[M-1][M-1] << endl;

    // emptys array c
    for (i=0; i<M; i++)
        for (j=0; j<M; j++)
            c[i][j] = 0;

    // sets stride size
    en = B * 8;

    // starts the clock
    start_time = clock::now();

    // large matrix multiplication (multi-thread, with loop blocking)
    #pragma omp parallel for private(i, j, jj, k, kk, sum)
    for (kk=0; kk<en; kk+=B)
        for (jj=0; jj<en; jj+=B)
            for (i=0; i< M; i++)
                for (j = jj; j< jj + B; j++)
                {
                    sum = c[i][j];
                    for (k=kk; k< kk + B; k++)
                        sum += a[i][k] * b[k][j];
                    c[i][j] = sum;
                }

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time3 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Check c[511][511] = " << c[M-1][M-1] << endl;

    // creates small matrices
    int a1[M1][M1];
    int b1[M1][M1];
    int c1[M1][M1];

    // fills small matrices a1, b1, and c1
    for (i=0; i<M1; i++)
        for (j=0; j<M1; j++)
            a1[i][j] = 1;

    for (i=0; i<M1; i++)
        for (j=0; j<M1; j++)
            b1[i][j] = 2;

    for (i=0; i<M1; i++)
        for (j=0; j<M1; j++)
            c1[i][j] = 0;

    // starts the clock
    start_time = clock::now();

    // small matrix multiplication (single-thread, no loop blocking)
    for (i=0; i<M1; i++)
        for (j=0; j<M1; j++)
            for (k=0; k<M1; k++)
                c1[i][j] += + a1[i][k] * b1[k][j];

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time4 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Check c1[15][15] = " << c1[M1-1][M1-1] << endl;

    // emptys array c1
    for (i=0; i<M1; i++)
        for (j=0; j<M1; j++)
            c1[i][j] = 0;

    // starts the clock
    start_time = clock::now();

    // small matrix multiplication (multi-thread, no loop blocking)
    #pragma omp parallel for private (i, j, k)
    for (i=0; i<M1; i++)
        for (j=0; j<M1; j++)
            for (k=0; k<M1; k++)
                c1[i][j] += + a1[i][k] * b1[k][j];

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time5 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Check c1[15][15] = " << c1[M1-1][M1-1] << endl;

    // emptys array c1
    for (i=0; i<M1; i++)
        for (j=0; j<M1; j++)
            c1[i][j] = 0;

    // sets stride size
    en = B1 * 4;

    // starts the clock
    start_time = clock::now();

    // large matrix multiplication (multi-thread, with loop blocking)
    #pragma omp parallel for private(i, j, jj, k, kk, sum)
    for (kk=0; kk<en; kk+=B1)
        for (jj=0; jj<en; jj+=B1)
            for (i=0; i< M1; i++)
                for (j = jj; j< jj + B1; j++)
                {
                    sum = c1[i][j];
                    for (k=kk; k< kk + B1; k++)
                        sum += a1[i][k] * b1[k][j];
                    c1[i][j] = sum;
                }

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time6 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Check c1[15][15] = " << c1[M1-1][M1-1] << endl << endl;

    // prints runtimes
    cout << "Time to execute large, single-threaded: " << run_time1 << " ns" << endl;
    cout << "Time to execute large, multi-threaded: " << run_time2 << " ns" << endl;
    cout << "Time to execute large, multi-threaded, with loop blocking: " << run_time3 << " ns" << endl;
    cout << "Large matrix parallel speedup: " << (double)run_time1/run_time2 << endl;
    cout << "Large matrix combined speedup: " << (double)run_time1/run_time3 << endl << endl;
    cout << "Time to execute small, single-threaded: " << run_time4 << " ns" << endl;
    cout << "Time to execute small, multi-threaded: " << run_time5 << " ns" << endl;
    cout << "Time to execute small, multi-threaded, with loop blocking: " << run_time6 << " ns" << endl;
    cout << "Small matrix parallel speedup: " << (double)run_time4/run_time5 << endl;
    cout << "Small matrix combined speedup: " << (double)run_time4/run_time6 << endl;

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

