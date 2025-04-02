// Q1_gpu
// Created by Justin Bahr on 3/28/2025.
// EECE 5640 - High Performance Computing
// Histogramming on a GPU

#include <iostream>
#include <chrono>
#include <random>
#include <ctime>

using namespace std;

// defines the maximum value and number of bins
const int NUM_BINS = 32;
const int MAX = 100000;

void gpuHistogram(int h_data[], int h_hist[], int size);

int main()
{
    // creates a variable for the number of terms
    int size = 2048;

    // creates a pointer to the data array
    int *data;

    // creates bins for the host histogram
    int *hist = new int [NUM_BINS];

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

        // fills data array with random numbers 1-100,000
        for (int i = 0; i < size; i++)
        {
            data[i] = dist(gen);
        }

        // starts the clock
        auto start_time = clock::now();

        gpuHistogram(data, hist, size);

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
            cout << "Bin " << i+1 << " has " << hist[i] << " elements, one is " << first[i] << endl;
        }

        // prints the runtime
        cout << "Runtime in nanoseconds for " << size << " integers: " << run_time << endl << endl;
    }

    // frees allocated memory
    delete[] data;
    data = nullptr;

    return 0;
}
