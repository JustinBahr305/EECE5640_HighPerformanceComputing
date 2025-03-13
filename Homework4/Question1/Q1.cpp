// Q1
// Created by Justin Bahr on 3/11/2025.
// EECE 5640 - High Performance Computing
// MPI Integer Incrementing

#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    // initializes MPI
    MPI_Init(&argc, &argv);

    // creates variables for the rank of each process, total number of processes, and an interger to increment
    int rank, size, i;

    // stores the rank of each process and the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // prints, increments, and passes i to the next process
    if (rank == 0)
    {
        i = 1;
        cout << "Incrementing ..." << endl;
        cout << "Process " << rank + 1 << ": " << i << endl; // rank + 1 since the first process is rank 0
        i++;
        MPI_Send(&i, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(&i, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Process " << rank + 1 << ": " << i << endl; // rank + 1 since the first process is rank 0
        if (rank < size-1)
        {
            i++;
            MPI_Send(&i, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
        else
        {
            cout << endl << "Decrementing ..." << endl;
            i-=2;
            MPI_Send(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    // prints, decrements, and passes i to the next process
    if (rank == 0)
    {
        MPI_Recv(&i, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Process " << rank + 1 << ": " << i << endl; // rank + 1 since the first process is rank 0
        i-=2;
        MPI_Send(&i, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
    else if (rank < size / 2)
    {
        MPI_Recv(&i, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Process " << rank + 1 << ": " << i << endl; // rank + 1 since the first process is rank 0
        if (rank < size/2 - 1)
        {
            i-=2;
            MPI_Send(&i, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    // finalizes MPI
    MPI_Finalize();
    return 0;
}
