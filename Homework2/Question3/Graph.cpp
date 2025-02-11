// Graph
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Graph Class

#include<iostream>
#include "Graph.h"

using namespace std;

// Graph constructor
Graph::Graph(int v)
{
    V = v;
    adj = new bool*[V];

    // initializes the adjacency matrix with zeros
    for (int i = 0; i < V; i++)
    {
        adj[i] = new bool[V];
        for (int j = 0; j < V; j++)
        {
            adj[i][j] = false;
        }
    }
}

// Graph destructor
Graph::~Graph()
{
    for (int i = 0; i < V; i++)
    {
        delete[] adj[i];
    }
    delete[] adj;
}

// function to print Graph
void Graph::printGraph()
{
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            cout << adj[i][j] << " ";
        }
        cout << endl;
    }
}

// function to add an edge
void Graph::addEdge(int x, int y)
{
    if (x >= 0 && y >= 0 && x < V && y < V)
    {
        adj[x][y] = true;
        adj[y][x] = true;
    }
    else
        cout << "Not a valid edge - out of bounds" << endl;
}

// function to remove an edge
void Graph::removeEdge(int x, int y)
{
    if (x >= 0 && y >= 0 && x < V && y < V)
    {
        adj[x][y] = false;
        adj[y][x] = false;
    }
    else
        cout << "Not a valid edge - out of bounds" << endl;
}

// function to detect an edge
bool Graph::isEdge(int x, int y)
{
    if (x >= 0 && y >= 0 && x < V && y < V)
        return adj[x][y];
    return false;
}

// function to return the number of vertices
int Graph::getSize()
{
	return V;
}

