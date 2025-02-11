// Q3
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Graph Coloring Algorithm using OpenMP

#include <iostream>
#include <chrono>
#include <ctime>
#include <map>
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

    // creates a boolean array outside the parallel section to stores improperly colored vertices
    bool defective[numVertices] = {true};
    bool inProgress = true;

    // continues till correctly colored
    while (inProgress)
    {
        #pragma omp parallel num_threads(numThreads)
        #pragma omp for
        for (int i = 1; i < numVertices; i++)
        {
            if (defective[i])
            {
                // creates boolean array to trak unavailable colors
                bool unavailable[numVertices] = {false};

                for (int j = 0; j < numVertices; j++)
                {
                    // marks a colors as unavailable if a neighboring vertex is that color
                    if (g.isEdge(i,j) && colors[j] != -1)
                        unavailable[colors[j]] = true;
                }

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
        }

        #pragma omp barrier
        bool def[numVertices] = {false};
        inProgress = false;

        #pragma omp parallel num_threads(numThreads)
        #pragma omp for
        for (int i = 1; i < numVertices; i++)
        {
            for (int j = i+1; j < numVertices; j++)
            {
                // marks a colors as unavailable if a neighboring vertex is that color
                if (g.isEdge(i,j) && colors[i] == colors[j])
                {
                    def[j] = true;
                }
            }
        }

        #pragma omp barrier
        for (int i = 1; i < numVertices; i++)
        {
            inProgress = true;
            break;
        }

        #pragma omp parallel num_threads(numThreads)
        #pragma omp for
        for (int i = 0; i < numVertices; i++)
            defective[i] = def[i];
    }

    // creates a variable to store the number of colors used
    int numColors;

    #pragma omp parallel num_threads(numThreads)
    {
        int localMax = 0;
        #pragma omp for nowait // finds local maximums in parallel
        for (int i = 1; i < numVertices; i++)
        {
            // updates the local maximum if a higher one is found
            if (colors[i] > localMax)
                localMax = colors[i];
        }

        // critical call to update numColors with the highest local max
        #pragma omp critical
        {
            if (localMax > numColors)
                numColors = localMax;
        }
    }

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
    colorMap[0] = "red";
    colorMap[1] = "blue";
    colorMap[2] = "green";
    colorMap[3] = "yellow";
    colorMap[4] = "orange";
    colorMap[5] = "purple";
    colorMap[6] = "white";
    colorMap[7] = "black";
    colorMap[8] = "pink";
    colorMap[9] = "gray";

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

    // prints the number of colors used and the runtime
    cout << "Colors used: " << numColors << endl;
    cout << "Time to execute: " << run_time << " ns" << endl << endl;;

    // prints the colors of all vertices
    cout << "Colors of each vertex: " << endl;
    for (int i = 0; i < numVertices; i++)
    {
        cout << "Vertex " << i  << ": " << colorMap[colors[i]] << endl;
    }

    return 0;
}
