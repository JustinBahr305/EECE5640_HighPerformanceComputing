// Q2
// Created by Justin Bahr on 3/11/2025.
// EECE 5640 - High Performance Computing
// MPI Parallel Histogramming

#include <iostream>
#include <chrono>
#include <random>
#include <ctime>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[])
{
    const int MAX = 100000;
    const int DATA_SIZE = 8000000;

    // initializes MPI
    MPI_Init(&argc, &argv);

    // creates variables for the rank of each process and the total number of processes
    int rank, size;

    // stores the rank of each process and the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // creates a pointer to store the full data array
    int *data = nullptr;

    // creates an array for the global histogram
    int globalHist[size] = {0};

    // computes the slice size for each process and creates an array to store the processes slice
    int processSlice = DATA_SIZE / size;
    int *localData = new int[processSlice];

    // creates an array for the local histogram
    int localHist[size] = {0};

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // the first process generates the random numbers and distributes them to all processes
    if (rank == 0)
    {
        // creates random number generator
        random_device randD;

        // seeds randD distribution based on the thread number
        mt19937 gen(randD() ^ time(0));
        uniform_int_distribution<int> dist(1, MAX);

        // initializes full data array
        data = new int[DATA_SIZE];

        // fills data array with random numbers 1-100,000
        for (int i = 0; i < DATA_SIZE; i++)
        {
            data[i] = dist(gen);
        }
    }

    // starts the clock
    auto start_time = clock::now();

    // scatters the data to all processes
    MPI_Scatter(data, processSlice, MPI_INT, localData, processSlice, MPI_INT, 0, MPI_COMM_WORLD);

    // each process histograms its slice
    for (int i = 0; i < processSlice; i++)
    {
        localHist[(localData[i] - 1) * size / MAX]++;
    }

    // reduces local histograms into global histogram in the first process
    MPI_Reduce(localHist, globalHist, size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // first process prints the resulting histogram
    if (rank == 0)
    {
        // stops the clock
        auto end_time = clock::now();

        // casts run_time in nanoseconds
        auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

        cout << endl << "Resulting Histogram:" << endl;
        for (int i = 0; i < size; i++)
        {
            cout << "Bin " << i + 1 << ": " << globalHist[i] << endl;
        }

        // prints the runtime
        cout << endl << "Runtime in nanoseconds: " << run_time << endl;

        delete[] data;
    }

    // free allocated memory
    delete[] localData;
    localData = nullptr;

    // finalizes MPI
    MPI_Finalize();

    return 0;
}