// Q1_omp
// Created by justb on 1/31/2025.
// EECE 5640 - High Performance Computing
// Monte Carlo Estimation for pi using OpenMP

#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <math.h>

using namespace std;

// defines the dimensions of the dartboard
// x coordinates from -DIM to +DIM
// y coordinates from -DIM to +DIM
const int DIM = 500;

bool inCircle(int x, int y)
{
    return (x * x + y * y <= DIM * DIM);
}

// function to estimate pi by throwing darts
double piByDarts(int numThreads, int numDarts)
{
    //creates arrays for the x and y values of dart throws
    int x;
    int y;

    // creates a variable to accumulate the number of darts thrown inside the circle
    int sumInCircle = 0;

    // creates a parallel process for each thread to execute
    #pragma omp parallel num_threads(numThreads)
    {
        // each thread given a unique random number generator
        random_device randD;

        // seeds randDom distribution based on the thread number
        mt19937 gen(randD() ^ omp_get_thread_num());
        uniform_int_distribution<int> dist(-DIM, DIM);

        // creates a local sum variable privately owned by an individual thread
        int localSum = 0;

        // parallelizes the loop throwing darts using nowait since throws are independent
        #pragma omp for nowait
        for (int i = 0; i < numDarts; i++)
        {
            int x = dist(gen);
            int y = dist(gen);
            if (inCircle(x,y))
            {
                localSum++;
            }
        }

        // atomic call to add the local sums to the total process sum
        #pragma omp atomic
        sumInCircle += localSum;
    }

    // returns the estimated value of pi
    return 4.0 * sumInCircle / numDarts;
}

int main()
{
    // creates variables for the numbers of threads and darts
    int numThreads;
    int numDarts;

    // creates the placeholder for the calculated value of pi
    double calc_pi;

    // allows the user to choose the number of threads
    cout << "How many threads would you like to use?" << endl;
    cin >> numThreads;

    // allows the user to choose the number of darts
    cout << "How many darts would you like to throw?" << endl;
    cin >> numDarts;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    calc_pi = piByDarts(numThreads, numDarts);

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
