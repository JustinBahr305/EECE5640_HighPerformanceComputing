// Q1_omp
// Created by justb on 1/31/2025.
// EECE 5640 - High Performance Computing
// Monte Carlo Estimation for pi using OpenMP

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <omp.h>

using namespace std;

// defines the dimensions of the dartboard
// x coordinates from -DIM to +DIM
// y coordinates from -DIM to +DIM
const int DIM = 500;

bool inCircle(int x, int y)
{
    if (x*x + y*y <= DIM*DIM)
        return true;
    else
        return false;
}

// function to estimate pi by throwing darts
double piByDarts(int numThreads, int numDarts)
{
    //creates arrays for the x and y values of dart throws
    int x;
    int y;

    // creates a variable to accumulate the number of darts thrown inside the circle
    int sumInCircle = 0;
    int localSum = 0;

    // creates a parallel process with numThreads threads
    #pragma omp parallel for num_threads(numThreads)
    {
        for (int i = 0; i < numDarts; i++)
        {
            x = rand() % (2*DIM+1) - DIM;
            y = rand() % (2*DIM+1) - DIM;
            if (inCircle(x, y))
            {
                localSum += 1;
            }
        }
        #pragma omp atomic
        sumInCircle += localSum;
    }
    return 4*(double)sumInCircle/(double)numDarts;
}

int main()
{
    // seeds the random generator
    srand(time(0));

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
