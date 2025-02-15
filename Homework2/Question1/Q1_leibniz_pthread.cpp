// Q1_leibniz_pthread
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Leibniz Estimation for pi using Pthreads

#include <iostream>
#include <chrono>
#include <random>
#include <pthread.h>
#include <math.h>

using namespace std;

// global array for local sums
double *localSums;

// structure for thread arguments
struct ThreadArgs
{
    int numTerms;
    int threadNum;
    int sliceSize;
};

void* threadLeibniz(void* numTermsArg)
{
    // converts the numDarts arguments to ints
    ThreadArgs* numTermsStruct = (ThreadArgs*)numTermsArg;
    int numTerms = numTermsStruct->numTerms;
    int threadNum = numTermsStruct->threadNum;
    int sliceSize = numTermsStruct->sliceSize;

    // stores the iteration starting and end points based on thread number
    int start = threadNum*sliceSize;
    int end = start + numTerms;

    // creates a temporary variable to store the local sum
    double localSum = 0.0;

    for (int i = start; i < end; i++)
    {
        localSum += (i % 2 == 0 ? 1.0 : -1.0)/(2*i+1);
    }

    localSums[threadNum] = localSum;

    return nullptr;
}

double piByLeibniz(int numTerms, int numThreads)
{
    // creates an array to store each thread's local sum, setting it to zero
    localSums = new double[numThreads];
    for (int i = 0; i < numThreads; i++)
    {
        localSums[i] = 0;
    }

    // creates and array of pthreads and thread arguments
    pthread_t threads[numThreads];
    ThreadArgs args[numThreads];

    // defines the size for slicing the array to run in parallel
    int sliceSize = numTerms / numThreads;

    for (int i = 0; i < numThreads; i++)
    {
        args[i] = {sliceSize, i, sliceSize};
    }
    args[numThreads-1] = {numTerms - (numThreads-1)*sliceSize, numThreads-1, sliceSize};

    // slices into as many pieces as there are threads and creates each thread process
    for (int i = 0; i < numThreads - 1; i++)
    {
        pthread_create(&threads[i], nullptr, threadLeibniz, &args[i]);
    }

    pthread_create(&threads[numThreads - 1], nullptr, threadLeibniz, &args[numThreads-1]);

    // joins the Pthreads once they have completed
    for (int i = 0; i < numThreads; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    double totalSum = 0.0;

    for (int i = 0; i < numThreads; i++)
    {
        totalSum += localSums[i];
    }

    // returns the estimated value of pi
    return 4.0 * totalSum;;
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