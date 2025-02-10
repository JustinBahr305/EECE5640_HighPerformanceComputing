// Graph.h
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Graph Header File

#ifndef GRAPH_H
#define GRAPH_H

class Graph
{
    private:
        int V; // number of vertices
        bool **adj; // adjacency matrix

    public:
        // Graph constructor header
        Graph(int v);

        // Graph destructor
        ~Graph();

        // function to print Graph
        void printGraph();

        // function to add an edge
        void addEdge(int x, int y);

        // function to remove an edge
        void removeEdge(int x, int y);

        // function to detect an edge
        bool isEdge(int x, int y);

        // function to return the number of vertices
        int getSize();
};

#endif //GRAPH_H
