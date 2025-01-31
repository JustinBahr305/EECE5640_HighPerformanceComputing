// Q2
// Created by justb on 1/31/2025.
// EECE 5640 - High Performance Computing
// Philospopher's Table Problem using Pthreads

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <pthread.h>

using namespace std;

void table_eat(int numPhilosophers)
{

}

int main()
{

    // seeds the random generator
    srand(time(0));

    int numPhilosophers = 0;

    while (numPhilosophers % 2 == 0 || numPhilosophers < 0)
    {
        cout << "Please enter an odd number of philosophers: " << endl;
        cin >> numPhilosophers;
    }

    table_eat(numPhilosophers);

    return 0;
}
