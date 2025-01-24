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

    // loops through checking if any element is less than it predecessor
    for (int i = 1; i < size; i++)
    {
        if (arr[i] < arr[i - 1])
        {
            isSorted = false;
            break;
        }
    }

    // prints whether the array is sorted correctly
    cout << "Status: ";
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
    // computes and defines the lengths of the sub arrays being merged
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // dynamically allocates arrays for the left and right sub arrays being merged
    int* leftArr = new int[n1];
    int* rightArr = new int[n2];

    // fills the left array with the left sub array within arr
    for (int i = 0; i < n1; ++i)
    {
        leftArr[i] = arr[left + i];
    }

    // fills the right array with the right sub array within arr
    for (int i = 0; i < n2; ++i)
    {
        rightArr[i] = arr[mid + 1 + i];
    }

    // defines iterative variables i, j, and k to move through the left, right, and main arrays respectively
    int i = 0, j = 0, k = left;

    // checks which sub array has the lower value and merges it to the next slot in the main array
    while (i < n1 && j < n2)
    {
        if (leftArr[i] <= rightArr[j]) { arr[k++] = leftArr[i++]; }
        else { arr[k++] = rightArr[j++]; }
    }

    // fills the remaining values in this slice of the main array
    while (i < n1) { arr[k++] = leftArr[i++]; }
    while (j < n2) { arr[k++] = rightArr[j++]; }

    // deallocates memory from the left and right sub arrays
    delete[] leftArr;
    delete[] rightArr;
}

// mergeSort function that returns void* because the Pthreads API,
// which requires that inputs have a specific signature
void* mergeSort(void* args)
{
    // creates and fills ThreadArgs structures to recursively pass to mergeSort
    ThreadArgs* threadArgs = (ThreadArgs*)args;
    int* array = threadArgs->arr;
    int left = threadArgs->left;
    int right = threadArgs->right;

    // recursively calls mergeSort on smaller arrays until it gets to individual elements
    if (left < right)
    {
        int mid = left + (right - left) / 2;
        mergeSort(new ThreadArgs{array, left, mid});
        mergeSort(new ThreadArgs{array, mid + 1, right});
        merge(array, left, mid, right);
    }

    // returns a null pointer to comply with Pthreads API for start_routine
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

    // creates an array to store each Pthread address
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

    // joins the Pthreads once they have completed their independent mergeSorts
    for (int i = 0; i < numThreads; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    // defines and initial value for this iterative variable
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

    // allows the user to input a valid number of threads
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
