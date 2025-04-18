// Q1_gpu
// Created by Justin Bahr on 3/28/2025.
// EECE 5640 - High Performance Computing
// Histogramming on a CPU using OpenMP

#include <iostream>
#include <chrono>
#include <random>
#include <ctime>
#include <omp.h>

using namespace std;

// set the number of threads to 16
const int NUM_THREADS = 16;

// defines the maximum value and number of bins
const int MAX = 100000;
const int NUM_BINS = 32;

// function to perform histogramming in parallel
void parallelHistogram(const int data[], int globalHistogram[], int size)
{
    // creates a matrix for the local histograms to avoid race conditions
    int localHistograms[NUM_THREADS][NUM_BINS] = {0};

    // performs histogramming in parallel
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < size; i++)
    {
        localHistograms[omp_get_thread_num()][(data[i] - 1) * NUM_BINS / MAX]++;
    } // end parallel section

    // combines the local histograms
    for (int i = 0; i < NUM_BINS; i++)
        for (int j = 0; j < NUM_THREADS; j++)
            globalHistogram[i] += localHistograms[j][i];
}

int main()
{
    // creates a variable for the number of terms
    int size = 2048;

    // creates a variable for the number of threads
    int numThreads;

    // creates a pointer to the data array
    int *data;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    for (; size <= 8388608; size*=2)
    {
        // creates random number generator
        random_device randD;

        // seeds randD distribution based on the time
        mt19937 gen(randD() ^ time(0));
        uniform_int_distribution<int> dist(1, MAX);

        // initializes full data array
        data = new int[size];

        // initializes empty bins for the global histogram
        int globalHistogram[NUM_BINS] = {0};

        // fills data array with random numbers 1-100,000
        for (int i = 0; i < size; i++)
        {
            data[i] = dist(gen);
        }

        // starts the clock
        auto start_time = clock::now();

        parallelHistogram(data, globalHistogram, size);

        // stops the clock
        auto end_time = clock::now();

        // casts run_time in nanoseconds
        auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

        // creates an array to store the first element encountered in each bin
        int first[NUM_BINS] = {0};

        // claculates the width of each class
        int classWidth = MAX / NUM_BINS;

        // finds the first elements in each class
        for (int i = 0; i < NUM_BINS; i++)
        {
            for (int j = 0; j < size; j++)
            {
                int min = i*classWidth;
                int max = min + classWidth;
                if (data[j] > min && data[j] <= max)
                {
                    first[i] = data[j];
                    break;
                }
            }
        }

        for (int i = 0; i < NUM_BINS; i++)
        {
            cout << "Bin " << i+1 << " has " << globalHistogram[i] << " elements, one is " << first[i] << endl;
        }

        // prints the runtime
        cout << "Runtime in nanoseconds for " << size << " integers: " << run_time << endl << endl;
    }

    // frees allocated memory
    delete[] data;
    data = nullptr;

    return 0;
}