// Q1_leibniz_omp
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Leibniz Estimation for pi using OpenMP

#include <iostream>
#include <chrono>
#include <math.h>
#include <omp.h>

using namespace std;

double piByLeibniz(int numTerms, int numThreads)
{
    // creates a variable to store the total sum
    double piSum = 0.0;

    // creates a variable to store the local sums
    double localSum = 0.0;

    // creates a parallel process for each thread to execute
    #pragma omp parallel num_threads(numThreads) private(localSum)
    {
        #pragma omp for nowait
        for (int i = 0; i < numTerms; i++)
        {
            localSum += (i % 2 == 0 ? 1.0 : -1.0)/(2*i+1);
        }

        // atomic call to add the local sums to the total process sum
        #pragma omp atomic
            piSum += localSum;
    } // end parallel section

    // returns the estimated value of pi
    return 4*piSum;
}

int main()
{
    // creates a variable for the number of terms and threads
    int numTerms;
    int numThreads;

    // creates the placeholder for the calculated value of pi
    double calc_pi;

    // allows the user to choose the number of threads
    do {
        cout << "How many threads would you like to use?" << endl;
        cin >> numThreads;
    } while (numThreads < 1);

    // allows the user to choose the number of terms
    cout << "How many terms would you like to add?" << endl;
    cin >> numTerms;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    calc_pi = piByLeibniz(numTerms, numThreads);

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // prints the calculated value of pi and the runtime
    cout << "Time to execute: " << run_time << " ns" << endl;
    cout << "Calculated value of pi: " << calc_pi << endl;
    cout << "Absolute error: " << abs(calc_pi - M_PI) << endl;

    return 0;
}
