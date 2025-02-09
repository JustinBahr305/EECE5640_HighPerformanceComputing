// Q3
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Graph Coloring Algorithm using OpenMP

#include <iostream>
#include <chrono>
#include <omp.h>
#include "Graph.h"

using namespace std;

int main()
{
    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // main code

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // prints the number of colors used and the runtime
    // cout << "Colors used: " << numColors << endl;
    cout << "Time to execute: " << run_time << " ns" << endl;

    return 0;
}
