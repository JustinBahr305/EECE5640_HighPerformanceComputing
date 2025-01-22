// Q2_PthreadSort
// Created by justb on 1/21/2025.
// Sorting 10,000 Integers using Pthreads

#include <iostream>
#include <pthread.h>
#include <chrono>
#include <ctime>
#include <set>

using namespace std;

/*
void merge(vector<int>& arr, int left, int mid, int right)
{

}
*/

int main()
{
    // creates a variable to store the number of threads used
    int numThreads = 0;

    // set that stores the valid numbers of threads
    set<int> threadSet = {1, 2, 4, 8, 32};

    while (threadSet.find(numThreads) == threadSet.end())
    {
        cout << "How many threads would you like to use (1, 2, 4, 8, or 32)?" << endl;
        cin >> numThreads;
        cout << endl;
    }

    // seeds the random generator
    srand(time(0));

    // creates an array for 10,000 integers
    int randArr[10000];

    // fills the array with random integers 1-10000
    for (int i = 0; i < 10000; i++)
    {
        randArr[i] = rand() % 10000 + 1;
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();



    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    cout << "Run time to sort 10,000 random integers using " << numThreads << " threads: "
    << run_time << " nanoseconds." << endl;

    return 0;
}
