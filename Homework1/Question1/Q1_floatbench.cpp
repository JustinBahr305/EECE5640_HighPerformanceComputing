// Q1_floatbench
// Created by Justin Bahr on 1/20/2025.
// EECE 5640 - High Performance Computing
// Benchmark for 100,000 floating point multiplications

#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;

int main()
{
    // defines the vector size
    const int SIZE = 100000;

    // creates input vectors a & b, and output vector c
    float a[SIZE] = {0};
    float b[SIZE] = {0};
    float c[SIZE] = {0};

    // seeds the random geenrator
    srand(time(0));

    // auto start_time;
    // auto end_time;
    // auto run_time;

    // fills vectors a & b with random floats
    for (int i = 0; i < SIZE; i++)
    {
        a[i] = (float)(rand()) / (float)(rand());
        b[i] = (float)(rand()) / (float)(rand());
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    for (int i = 0; i < SIZE; i++)
    {
        c[i] = a[i] * b[i];
    }

    // stops the clock
    auto end_time = clock::now();

    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // outputs the execution time
    cout << "Execution time for " << SIZE << " float multiplications: " << run_time << " nanoseconds." << endl;

    return 0;
}
