// Q2
// Created by Justin Bahr on 2/22/2025.
// EECE 5640 - High Performance Computing
// Matrix Vector Operations with AVX

#include <iostream>
#include <chrono>
#include <x86intrin.h>

using namespace std;

// defines the dimension N
const int N = 1024;

// function to perform matrix vector multiplication without AVX
void matrix_vector(const float (*A)[N], const float *x, float *y, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            y[i] += A[i][j] * x[j];
}

// function to perform matrix vector multiplication with AVX
void matrix_vector_avx512f(const float (*A)[N], const float *x, float *y, int N)
{
    for (int i = 0; i < N; i++)
    {
        // creates a variable for iteration
        size_t j = 0;

        // declares vector datatypes
        __m512 y_vec, A_vec, x_vec;
        y_vec = _mm512_setzero_ps();

        // loads and multiplies vectors, each with 16 floats
        for (; j + 16 <= N; j += 16)
        {
            A_vec = _mm512_loadu_ps(&A[i][j]);
            x_vec = _mm512_loadu_ps(&x[j]);
            y_vec = _mm512_fmadd_ps(A_vec, x_vec, y_vec);
        }

        // adds the terms of the resultant vector
        float y_sum = _mm512_reduce_add_ps(y_vec);

        // calculates remaining terms
        for (; j < N; j++)
            y_sum += A[i][j] * x[j];

        // stores the dot product of a row in the respective index
        y[i] = y_sum;
    }
}

int main()
{
    // creates NxN matrix A and N length vectors x and y
    float A[N][N];
    float x[N];
    float y1[N] = {0};
    float y2[N] = {0};

    // fills matrix A
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = 0.1*(i+j);

    // fills vector x
    for (int i = 0; i < N; i++)
        x[i] = 0.1*i;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // calls the matrix vector multiplication function
    matrix_vector(A,x,y1,N);

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time1 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // starts the clock
    start_time = clock::now();

    // calls the matrix vector multiplication with AVX function
    matrix_vector_avx512f(A,x,y2,N);

    // stops the clock
    end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time2 = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // prints the results without AVX
    cout << "Without AVX512:" << endl;
    cout << "y[" << N-1 << "] = " << y1[N-1] << endl;
    cout << "Runtime in nanoseconds: " << run_time1 << endl << endl;

    // prints the results with AVX
    cout << "With AVX512:" << endl;
    cout << "y[" << N-1 << "] = " << y2[N-1] << endl;
    cout << "Runtime in nanoseconds: " << run_time2 << endl << endl;

    // prints the AVX speedup
    cout << "AVX512 Speedup: " << float(run_time1) / run_time2 << endl;

    return 0;
}
