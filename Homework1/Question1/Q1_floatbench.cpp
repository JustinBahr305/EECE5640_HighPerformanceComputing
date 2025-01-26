// Q1_floatbench
// Created by Justin Bahr on 1/20/2025.
// EECE 5640 - High Performance Computing
// Benchmark for 100,000 floating point multiplications

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>

using namespace std;

int main()
{
    // defines the array size
    const int SIZE = 100000;

    // creates input arrays a & b, and output array c
    float a[SIZE];
    float b[SIZE];
    float c[SIZE];

    // seeds the random generator
    srand(time(0));

    // fills arrays a & b with random floats
    for (int i = 0; i < SIZE; i++)
    {
        a[i] = (float)(rand()) / (float)(rand());
        b[i] = (float)(rand()) / (float)(rand());
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // executes floating point multiplications
    for (int i = 0; i < SIZE; i++)
    {
        c[i] = a[i] * b[i];
    }

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // creates a boolean for success terms
    bool isDone = true;

    // loops through every element in array c
    // if array c is never used after the operations, optimizations may delete "redundant" code
    for (int i = 0; i < SIZE; i++)
    {
        if (isnan(c[i]))
        {
            isDone = false;
        }
    }

    // print success message
    if (isDone)
    {
        cout << "Success" << endl;
    }

    // outputs the execution time
    cout << "Execution time for " << SIZE << " float multiplications: " << run_time << " nanoseconds." << endl;

    return 0;
}
