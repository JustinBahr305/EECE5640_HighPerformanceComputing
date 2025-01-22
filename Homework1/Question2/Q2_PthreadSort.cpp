// Q2_PthreadSort
// Created by Justin Bahr on 1/21/2025.
// Sorting 10,000 Integers using Pthreads

#include <iostream>
#include <pthread.h>
#include <chrono>
#include <ctime>
#include <set>

using namespace std;

// function to check whether a list is sorted
void checkSorted(int* arr, int size)
{
    bool isSorted = true;

    for (int i = 1; i < size; i++)
    {
        if (arr[i] < arr[i - 1])
        {
            isSorted = false;
            break;
        }
    }

    if (isSorted) { cout << "Sorted Correctly" << endl; }
    else { cout << "Sorted Incorrectly" << endl; }
}

// structure for thread arguments
struct ThreadArgs
{
    int* arr;
    int left;
    int right;
};

// merge function for mergeSort
void merge(int* arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* leftArr = new int[n1];
    int* rightArr = new int[n2];

    for (int i = 0; i < n1; ++i)
    {
        leftArr[i] = arr[left + i];
    }

    for (int i = 0; i < n2; ++i)
    {
        rightArr[i] = arr[mid + 1 + i];
    }

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (leftArr[i] <= rightArr[j]) { arr[k++] = leftArr[i++]; }
        else { arr[k++] = rightArr[j++]; }
    }

    while (i < n1) { arr[k++] = leftArr[i++]; }
    while (j < n2) { arr[k++] = rightArr[j++]; }

    delete[] leftArr;
    delete[] rightArr;
}

// mergeSort function that returns void* because the Pthreads API,
// which requires that inputs have a specific signature
void* mergeSort(void* args)
{
    ThreadArgs* threadArgs = (ThreadArgs*)args;
    int* array = threadArgs->arr;
    int left = threadArgs->left;
    int right = threadArgs->right;

    if (left < right)
    {
        int mid = left + (right - left) / 2;
        mergeSort(new ThreadArgs{array, left, mid});
        mergeSort(new ThreadArgs{array, mid + 1, right});
        merge(array, left, mid, right);
    }

    return nullptr;
}

// function that allows mergeSorts to be done in parallel using the specified number of threads
void parallelMergeSort(int* arr, int size, int numThreads)
{
    // checks for single-threaded case
    if (numThreads <= 1)
    {
        mergeSort(new ThreadArgs{arr, 0,size - 1});
    }

    // creates a pthread array of size numThreads
    pthread_t threads[numThreads];

    ThreadArgs args[numThreads];

    // defines the size for slicing the array to run in parallel
    int sliceSize = size / numThreads;

    // slices the array into as many pieces as there are threads and creates each thread process
    for (int i = 0; i < numThreads - 1; i++)
    {
        int left = i * sliceSize;
        int right = left + sliceSize - 1;
        args[i] = {arr, left, right};
        pthread_create(&threads[i], nullptr, mergeSort, &args[i]);
    }

    // handles the last slice to ensure no indices are left out
    int left = (numThreads - 1) * sliceSize;
    int right = size - 1;
    args[numThreads - 1] = {arr, left, right};
    pthread_create(&threads[numThreads - 1], nullptr, mergeSort, &args[numThreads - 1]);

    // joins the threads once they have completed
    for (int i = 0; i < numThreads; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    int step = sliceSize;

    // merges each sorted slice
    while (step < size)
    {
        for (int i = 0; i < size; i += 2 * step)
        {
            int mid = min(i + step - 1, size - 1);
            int right = min(i + 2 * step - 1, size - 1);
            merge(arr, i, mid, right);
        }
        step*= 2;
    }

}

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

    // defines the array size
    const int SIZE = 10000;

    // creates an array for 10,000 integers
    int randArr[SIZE];

    // fills the array with random integers 1-10000
    for (int i = 0; i < SIZE; i++)
    {
        randArr[i] = rand() % SIZE + 1;
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // calls parallelMergeSort() to sort randArr
    parallelMergeSort(randArr, SIZE, numThreads);

    // stops the clock
    auto end_time = clock::now();

    /* prints the contents of the sorted array
    for (int i = 0; i < SIZE; i++)
    {
        cout << randArr[i] << ", ";
    } */

    // calls checkSorted to check if randArr is sorted correctly
    checkSorted(randArr, SIZE);

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // outputs the execution time
    cout << "Run time to sort 10,000 random integers using " << numThreads << " threads: "
    << run_time << " nanoseconds." << endl;

    return 0;
}
