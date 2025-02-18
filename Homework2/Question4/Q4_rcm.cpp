// Q4_rcm
// Created by Justin Bahr on 2/17/2025.
// EECE 5640 - High Performance Computing
// Reverse Cuthill-McKee SpMV Reordering

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include "mmio.h"

using namespace std;

// Function to build the adjacency list of the graph from a symmetric sparse matrix A
vector<vector<int>> build_adjacency_list(double* A, int n)
{
    vector<vector<int>> adj_list(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (A[i * n + j] != 0 && i != j)
            {
                adj_list[i].push_back(j);
            }
        }
    }
    return adj_list;
}

// Function to find the starting node for BFS: the one with the smallest degree
int find_starting_node(const vector<vector<int>>& adj_list)
{
    int min_degree = adj_list[0].size();
    int start_node = 0;

    for (int i = 1; i < adj_list.size(); i++)
    {
        if (adj_list[i].size() < min_degree)
        {
            min_degree = adj_list[i].size();
            start_node = i;
        }
    }
    return start_node;
}

// Function to perform the Reverse Cuthill-McKee algorithm
int* reverse_cuthill_mckee(double* A, int n)
{
    vector<vector<int>>* adj_list;
    adj_list = new vector<vector<int>>(n);
    adj_list = build_adjacency_list(A, n);
    int perm[n];
    int permInd = 0;
    bool visited[n] = {0};

    int start_node = find_starting_node(adj_list);
    queue<int> q;
    q.push(start_node);
    visited[start_node] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        perm[permInd++] = node;

        vector<int> neighbors;
        for (int neighbor : adj_list[node])
        {
            if (!visited[neighbor])
            {
                neighbors.push_back(neighbor);
            }
        }

        sort(neighbors.begin(), neighbors.end(), [&](int a, int b)
        {
            return adj_list[a].size() < adj_list[b].size();
        });

        for (int neighbor : neighbors)
        {
            visited[neighbor] = true;
            q.push(neighbor);
        }
    }

    reverse(perm,perm+1); // Reverse the Cuthill-McKee order
    return perm;
}

int main()
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;

    if ((f = fopen("e40r0000.mtx", "r")) == nullptr)
    {
        cout << "File does not exist" << endl;
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
    {
        printf("Could not read mtx_crd_size\n");
        exit(1);
    }

    /* reserve memory for matrices */

    int* I = new int[nz];
    int* J = new int[nz];
    double* val = new double[nz];

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    fclose(f);

    // creates an empty MxN matrix
    double** matA = new double*[M];
    for (int i=0; i<M; i++)
    {
        matA[i] = new double[N];
    }

    // copy the matrix market matrix into matA
    for (int k = 0; k < nz; k++)
    {
        matA[I[k]][J[k]] = val[k];
    }

    // deletes dynamically allocated memory
    delete[] I;
    delete[] J;
    delete[] val;

    int* perm = new int[nz];
    perm = reverse_cuthill_mckee(*matA, nz);

    cout << "Permutation Order:" << endl;

    for (int i = 0; i < nz; i++)
    {
        cout << perm[i] << ", ";
    }
    cout << endl;

    cout << "Reordered Matrix:" << endl;
    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            cout << matA[r][c] << ", ";
        }
    }

    // deletes perm
    delete[] perm;

    // deletes matA from the heap
    for (int i = 0; i < M; i++)
    {
        matA[i] = nullptr;
    }
    matA = nullptr;

    return 0;
}
