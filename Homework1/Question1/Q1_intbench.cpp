// Q1_intbench
// Created by justb on 1/20/2025.
// Benchmark for integer computations

#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;

int main()
{
    // defines the vector size
    const int SIZE = 100000;

    // creates input vectors a, b, c, and d
    int a[SIZE];
    int b[SIZE];
    int c[SIZE];
    int d[SIZE];

    // seeds the random generator
    srand(time(0));

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

    for (int i = 0; i < SIZE; i++)
    {
        a[i] *= b[i];
        a[i] += c[i];
        a[i] -= d[i];
    }

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // outputs the execution time
    cout << "Execution time for " << 3*SIZE << " integer operations: " << run_time << " nanoseconds." << endl;

    return 0;
}