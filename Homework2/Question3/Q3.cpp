// Q3
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Graph Coloring Algorithm using OpenMP

#include <iostream>
#include <chrono>
#include <ctime>
#include <omp.h>
#include "Graph.h"

using namespace std;

int main()
{
    int numThreads;
    int numVertices;

    // allows the user to choose the number of threads
    do {
        cout << "How many threads would you like to use?" << endl;
        cin >> numThreads;
    } while (numThreads < 1);

    // allows the user to choose the number of vertices on the graph
    do {
        cout << "How many vertices would you like on this graph?" << endl;
        cin >> numVertices;
    } while (numVertices < 1);

    // creates a graph with the specified number of vertices
    Graph graph(numVertices);

    graph.printGraph();

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // main code


    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // prints the number of colors used and the runtime
    // cout << "Colors used: " << numColors << endl;
    cout << "Time to execute: " << run_time << " ns" << endl;

    return 0;
}
