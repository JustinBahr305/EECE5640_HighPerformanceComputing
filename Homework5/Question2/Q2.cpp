// Q2
// Created by Justin Bahr on 3/28/2025.
// EECE 5640 - High Performance Computing
// Stencil Computation on a GPU

#include <iostream>
#include <chrono>

using namespace std;

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
	// iteraties with N = 8,16,32,64
	int N = 8;
	size_t num_floats;

    // creates pointers for the input and output tensors
	float *h_a;
	float *h_b;

	// initialize the high resolution clock
	typedef chrono::high_resolution_clock clock;

	for (; N <= 64; N*=2)
    {
		// number of floats per tensor
		num_floats = N*N*N;

	    // allocates the input and output tensors
		h_a = new float[num_floats];
		h_b = new float[num_floats];

	    // fills the input tensor
		fill(h_b, N);

	    // starts the clock
	    auto start_time = clock::now();

	    // performs stenciling on a GPU
		stencil(h_a, h_b, N);

	    // stops the clock
	    auto end_time = clock::now();

	    // casts run_time in nanoseconds
	    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

	    // prints the runtime
	    cout << "Runtime in nanoseconds for " << N << "^3 dimension: " << run_time << endl;
    }

    // prints an example result
    cout << "Computed a[3][3][3] = " << h_a[3*N*N + 3*N + 3] << endl;

    // stores the confirmed result
    float check = 0.75 * (h_b[2*N*N+3*N+3] + h_b[4*N*N+3*N+3] + h_b[3*N*N+2*N+3]
                          + h_b[3*N*N+4*N+3] + h_b[3*N*N+3*N+2] + h_b[3*N*N+3*N+4]);

    // prints the confirmed result
    cout << "Confirmed a[3][3][3] = " << check << endl << endl;

    // frees allocated memory
    delete[] h_a;
    delete[] h_b;
    h_a = nullptr;
    h_b = nullptr;

    return 0;
}