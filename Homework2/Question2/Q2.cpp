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

// creates global variables for the number of philosophers and iterations
int numPhilosophers = 0;
int numIterations;

struct philosopher
{
    int num;
    bool isEating;
    int timeToEat;
};

void printTable(int round, philosopher philosophers[])
{
    cout << "Round: " << round << endl;
    for (int i = 0; i < numPhilosophers; i++)
    {
        cout << "Philosopher " << i  << ": ";
        if (philosophers[i].isEating)
            cout << "Eating" << endl;
        else
            cout << "Thinking" << endl;
    }
    cout << endl;
}

void* sit(void* numArg)
{

}

void table_eat(int numPhilosophers, int numIterations)
{
    // creates an array to store the states of all philosophers
    philosopher philosophers[numPhilosophers];

    // creates arrays for the philosopher threads
    pthread_t threads[numPhilosophers];

    // creates mutexes for forks and threads for philosophers
    pthread_mutex_t *forks;
    forks = new pthread_mutex_t[numPhilosophers];
    for (int i = 0; i < numPhilosophers; i++) {
        pthread_mutex_init(&forks[i], nullptr);
        philosophers[i] = {i, false, rand() % 10};
        pthread_create(&threads[i], nullptr, sit, &(philosophers[i].num));
    }

    // joins the philosopher threads
    for (int i = 0; i < numPhilosophers; i++) {
        pthread_join(threads[i], nullptr);
    }

    // destroys the mutexes
    for (int i = 0; i < numPhilosophers; i++) {
        pthread_mutex_destroy(&forks[i]);
    }
}

int main()
{
    // seeds the random generator
    srand(time(0));

    while (numPhilosophers % 2 == 0 || numPhilosophers < 0)
    {
        cout << "Please enter an odd number of philosophers: " << endl;
        cin >> numPhilosophers;
    }

    cout << "Please enter the number of iterations: " << endl;
    cin >> numIterations;

    table_eat(numPhilosophers, numIterations);

    return 0;
}
