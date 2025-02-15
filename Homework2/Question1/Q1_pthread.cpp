// Q1_pthread
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Monte Carlo Estimation for pi using Pthreads

#include <iostream>
#include <chrono>
#include <random>
#include <pthread.h>
#include <math.h>

using namespace std;

// defines the dimensions of the dartboard
// x coordinates from -DIM to +DIM
// y coordinates from -DIM to +DIM
const int DIM = 500;

// global array for local sums
int *localSums;

// structure for thread arguments
struct ThreadArgs
{
    int numDarts;
    int threadNum;
};

bool inCircle(int x, int y)
{
    return (x * x + y * y <= DIM * DIM);
}

void* threadDart(void* numDartsArg)
{
    // converts the numDarts arguments to ints
    ThreadArgs* numDartsStruct = (ThreadArgs*)numDartsArg;
    int numDarts = numDartsStruct->numDarts;
    int threadNum = numDartsStruct->threadNum;

    //creates variables for the x and y values of dart throws
    int x;
    int y;

    // each thread given a unique random number generator
    random_device randD;

    // seeds randDom distribution based on the thread number
    mt19937 gen(randD() ^ pthread_self());
    uniform_int_distribution<int> dist(-DIM, DIM);

    // creates a temporary variable to store the local sum
    int localSum = 0;

    for (int i = 0; i < numDarts; i++)
    {
        x = dist(gen);
        y = dist(gen);
        if (inCircle(x,y))
        {
            localSum++;
        }
    }

    localSums[threadNum] = localSum;

    return nullptr;
}

// function to estimate pi by throwing darts
double piByDarts(int numThreads, int numDarts)
{
    // creates an array to store each thread's local sum, setting it to zero
    localSums = new int[numThreads];
    for (int i = 0; i < numThreads; i++)
    {
        localSums[i] = 0;
    }

    // creates and array of pthreads and thread arguments
    pthread_t threads[numThreads];
    ThreadArgs args[numThreads];

    // defines the size for slicing the array to run in parallel
    int sliceSize = numDarts / numThreads;

    for (int i = 0; i < numThreads; i++)
    {
        args[i] = {sliceSize, i};
    }
    args[numThreads-1] = {numDarts - (numThreads-1)*sliceSize, numThreads-1};

    // slices into as many pieces as there are threads and creates each thread process
    for (int i = 0; i < numThreads - 1; i++)
    {
        pthread_create(&threads[i], nullptr, threadDart, &args[i]);
    }

    pthread_create(&threads[numThreads - 1], nullptr, threadDart, &args[numThreads-1]);

    // joins the Pthreads once they have completed
    for (int i = 0; i < numThreads; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    int sumInCircle = 0;

    for (int i = 0; i < numThreads; i++)
    {
        sumInCircle += localSums[i];
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
    do {
        cout << "How many threads would you like to use?" << endl;
        cin >> numThreads;
    } while (numThreads < 1);

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
