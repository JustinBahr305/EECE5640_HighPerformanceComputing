// Q2
// Created by Justin Bahr on 3/28/2025.
// EECE 5640 - High Performance Computing
// Stencil Computation on a GPU

#include <iostream>
#include <chrono>

using namespace std;

// defines the tensor dimension
const int N = 32;

// function to fill a tensor
void fill(float *b, int n)
{
	for (int i = 0; i < n; i++)
    	for (int j = 0; j < n; j++)
        	for (int k = 0; k < n; k++)
            	b[i*n*n + j*n +k] = k;
}

void stencil(float *h_a, float *h_b, int n);

int main()
{
	// number of floats per tensor
	int num_floats = N*N*N;

    // allocates the input and output tensors
	float *h_a = new float[num_floats];
	float *h_b = new float[num_floats];

    // fills the input tensor
	fill(h_b, N);

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;

    // starts the clock
    auto start_time = clock::now();

	stencil(h_a, h_b, N);

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Runtime in nanoseconds: " << run_time << endl;

    cout << "a[3][3][3] = " << h_a[3*N*N + 3*N + 3] << endl;

    // frees allocated memory
    delete[] h_a;
    delete[] h_b;
    h_a = nullptr;
    h_b = nullptr;

    return 0;
}