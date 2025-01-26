// Q1_intbench
// Created by Justin Bahr on 1/20/2025.
// EECE 5640 - High Performance Computing
// Benchmark for integer computations

#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;

int main()
{
    // defines the array size
    const int SIZE = 100000;

    // creates input arrays a, b, c, and d
    int a[SIZE];
    int b[SIZE];
    int c[SIZE];
    int d[SIZE];

    // seeds the random generator
    srand(time(0));

    // fills arrays a, b, c, and d with random integers 0 to 99
    for (int i = 0; i < SIZE; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
        c[i] = rand() % 100;
        d[i] = rand() % 100;
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // executes integer operations
    for (int i = 0; i < SIZE; i++)
    {
        a[i] = a[i] * b[i] + c[i] - d[i];
    }

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // creates a boolean
    bool isDone = true;

    // loops through every element in array a
    // if array a is never used after the operations, optimizations may delete "redundant" code
    for (int i = 0; i < SIZE; i++)
    {
        if (a[i] >= 10000) { isDone = false; }
    }

    // print success message
    if (isDone)
    {
        cout << "Success" << endl;
    }


    // outputs the execution time
    cout << "Execution time for " << 3*SIZE << " integer operations: " << run_time << " nanoseconds." << endl;

    return 0;
}