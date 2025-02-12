// Q3
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Graph Coloring Algorithm using OpenMP

#include <iostream>
#include <chrono>
#include <ctime>
#include <map>
#include <string>
#include <omp.h>
#include "Graph.h"

using namespace std;

// function to color a graph and return the number of colors used
int color(Graph g, int colors[], int numThreads)
{
    // stores the number of vertices
    int numVertices = g.getSize();

    // colors the first vertex
    colors[0] = 0;

    for (int i = 1; i < numVertices; i++)
    {
        // creates boolean array to trak unavailable colors
        bool unavailable[numVertices] = {0};

        #pragma omp parallel for num_threads(numThreads)
        for (int j = 0; j < numVertices; j++)
        {
            // marks a colors as unavailable if a neighboring vertex is that color
            if (g.isEdge(i,j) && colors[j] != -1)
                unavailable[colors[j]] = true;
        } // end parallel region

        // colors a vertex with the first available color
        for (int k = 0; k < numVertices; k++)
        {
            if (!unavailable[k])
            {
                colors[i] = k;
                break;
            }
        }
    }

    // creates a variable to store the number of colors used
    int numColors = 0;

    #pragma omp parallel num_threads(numThreads)
    {
        // creates a variable for the local maxes
        int localMax = 0;

        #pragma omp for nowait // finds local maximums in parallel
        for (int i = 1; i < numVertices; i++)
        {
            // updates the local maximum if a higher one is found
            if (colors[i] > localMax)
            {
                localMax = colors[i];
            }
        }

        // critical call to update numColors with the highest local max
        #pragma omp critical
        {
            if (localMax > numColors)
            {
                numColors = localMax;
            }
        }
    } // end parallel section

    return numColors + 1;
}

int main()
{
    srand(time(0));

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

    // creates random edges (ignores main diagonal since same vertex)
    for (int i = 0; i < numVertices; i++)
        for (int j = 0; j < numVertices; j++)
            if (i != j && rand() % 2 == 0)
                graph.addEdge(i, j);

    // prints the graph's adjacency matrix
    cout << endl << "Here is the graph:" << endl;
    graph.printGraph();
    cout << endl;

    // creates an array to store the color of each vertex
    int colors[numVertices] = {-1};

    // creates a map to translate color numbers to strings for output
    map<int, string> colorMap;
    colorMap[-1] = "None";
    colorMap[0] = "Red";
    colorMap[1] = "Blue";
    colorMap[2] = "Green";
    colorMap[3] = "Yellow";
    colorMap[4] = "Orange";
    colorMap[5] = "Purple";
    colorMap[6] = "White";
    colorMap[7] = "Black";
    colorMap[8] = "Pink";
    colorMap[9] = "Gray";
    for (int i = 10; i < numVertices; i++)
    {
        colorMap[i] = "Color" + to_string(i+1);
    }

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    // colors the graph and returns the number of colors used
    int numColors = color(graph, colors, numThreads);

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // prints the colors of all vertices
    cout << "Colors of each vertex: " << endl;
    for (int i = 0; i < numVertices; i++)
    {
        cout << "Vertex " << i  << ": " << colorMap[colors[i]] << endl;
    }
    cout << endl;

    // prints the number of colors used and the runtime
    cout << "Colors used: " << numColors << endl;
    cout << "Time to execute: " << run_time << " ns" << endl;

    return 0;
}
