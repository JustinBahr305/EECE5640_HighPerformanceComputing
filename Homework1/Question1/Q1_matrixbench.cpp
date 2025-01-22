// Q1_matrixbench
// Created by Justin Bahr on 1/20/2025.
// Benchmark for spare matrix multiplication

#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;

int main()
{
    // defines the matrix dimensions
    const int DIM = 100;

    // creates matrices
    int mat1[DIM][DIM] = {0};
    int mat2[DIM][DIM] = {0};
    int mat3[DIM][DIM];

    // creates temporary int
    int temp;

    // seeds the random generator
    srand(time(0));

    // fills mat1 sparsely
    for (int i = 0; i < DIM; i += 2)
    {
        for (int j = 0; j < DIM; j += 2)
        {
            mat1[i][j] = rand() % 100;
        }
    }

    // fills mat2 completely
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            mat2[i][j] = rand() % 100;
        }
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // performs mat3 = mat1 * mat2
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            temp = 0;
            for (int k = 0; k < DIM; k++)
            {
                temp += mat1[i][k] * mat2[k][j];
            }
            mat3[i][j] = temp;
        }
    }

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // outputs the execution time
    cout << "Execution time for sparse matrix multiplication of dimension " << DIM
    << ": "<< run_time << " nanoseconds." << endl;

    /* code to test correctness of the matrix multiplication
    cout << endl;
    cout << "Mat1:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            cout << mat1[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "Mat2:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            cout << mat2[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "Mat3:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            cout << mat3[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    */

    return 0;
}
