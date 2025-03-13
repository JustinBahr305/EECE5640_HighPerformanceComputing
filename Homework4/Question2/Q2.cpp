// Q2
// Created by Justin Bahr on 3/11/2025.
// EECE 5640 - High Performance Computing
// MPI Parallel Histogramming

#include <iostream>
#include <chrono>
#include <random>
#include <math.h>
#include <ctime>
#include <mpi.h>

using namespace std;

const int MAX = 100000;

int main()
{
    // creates random number generator
    random_device randD;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // seeds randDom distribution based on the thread number
    mt19937 gen(randD() ^ );
    uniform_int_distribution<int> dist(1, MAX);

    return 0;
}