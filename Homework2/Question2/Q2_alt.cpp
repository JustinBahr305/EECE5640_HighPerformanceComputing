// Q2_alt
// Created by Justin Bahr on 2/17/2025.
// EECE 5640 - High Performance Computing
// Dining Philosopher's Problem using Pthreads  - Alternating Method

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <unistd.h>
#include <pthread.h>

using namespace std;

// creates global variables for the number of philosophers and iterations
int numPhilosophers;
int numIterations;

// creates a global variable for the current iteration
int currItr = 0;
pthread_mutex_t *currEat;

struct Philosopher
{
    int num;
    bool isEating;
    int timeToEat;
};

// creates global variable pointers for the philosophers and fork mutexes
Philosopher *philosophers;
pthread_mutex_t *forks;
bool *forkInUse;

void printTable(int round)
{

    // prints the round number
    cout << "Round: " << round+1 << endl;

    // prints the state of each philosopher
    for (int i = 0; i < numPhilosophers; i++)
    {
        cout << "Philosopher " << i+1 << ": ";
        if (philosophers[i].isEating)
            cout << "Eating" << endl;
        else
            cout << "Thinking" << endl;
    }
    cout << endl;

    // prints the state of each fork
    for (int i = 0; i < numPhilosophers; i++)
    {
        cout << "Fork " << i+1 << ": ";
        if (forkInUse[i])
        {
            cout << "in use" << endl;
        }
        else
        {
            cout << "not in use" << endl;
        }
    }
    cout << endl;
    cout << endl;
}

void* sit(void* numArg)
{
    int philosopherNum = *(int*)numArg;

    // assigns an alternating order to pick up forks
    int fork1;
    int fork2;
    if (philosopherNum % 2 == 1)
    {
        fork1 = philosopherNum + 1;
        fork2 = philosopherNum;
    }
    else
    {
        fork1 = philosopherNum;
        fork2 = (philosopherNum + 1) % numPhilosophers;
    }

    while (currItr < numIterations)
    {
        // thinking while waiting for first fork
        while (forkInUse[fork1])
        {
            // think
        }

        // pick up first fork
        pthread_mutex_lock(&forks[fork1]);
        forkInUse[fork1] = true;

        // thinking while waiting for second fork
        while (forkInUse[fork2])
        {
            // think
        }

        // pick up second fork
        pthread_mutex_lock(&forks[fork2]);
        forkInUse[fork2] = true;

        // eating with the forks
        philosophers[philosopherNum].isEating = true;
        usleep(philosophers[philosopherNum].timeToEat*10000); // Eat for assigned time

        if (pthread_mutex_trylock(currEat) == 0)
        {
            printTable(currItr++);
            pthread_mutex_unlock(currEat);
        }

        // put down forks
        pthread_mutex_unlock(&forks[fork1]);
        pthread_mutex_unlock(&forks[fork2]);
        forkInUse[fork1] = false;
        forkInUse[fork2] = false;
        philosophers[philosopherNum].isEating = false;

        // thinking
        usleep(1000);
    }
    return nullptr;

}

void table_eat(int numPhilosophers, int numIterations)
{
    // creates an array to store the states of all philosophers
    philosophers = new Philosopher[numPhilosophers];

    // creates arrays for the philosopher threads
    pthread_t threads[numPhilosophers];

    // initializes an array for the forks' mutexes and one for the forks' status
    forks = new pthread_mutex_t[numPhilosophers];
    forkInUse = new bool[numPhilosophers];

    // initializes mutex for thread eating
    currEat = new pthread_mutex_t;
    pthread_mutex_init(currEat, nullptr);

    for (int i = 0; i < numPhilosophers; i++)
    {
        // initializes mutexes for the forks
        pthread_mutex_init(&forks[i], nullptr);

        // makes sure no forks are in use
        forkInUse[i] = false;

        // assigns random eating times for each philosopher
        philosophers[i] = {i, false, rand() % 10};

        // creates a thread for each philosopher
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

    do {
        cout << "Please enter an odd number of philosophers: " << endl;
        cin >> numPhilosophers;
    } while (numPhilosophers % 2 == 0 || numPhilosophers < 0);

    cout << "Please enter the number of iterations: " << endl;
    cin >> numIterations;

    table_eat(numPhilosophers, numIterations);

    // deletes global variables
    delete[] philosophers;
    delete[] forks;
    delete[] forkInUse;
    philosophers = nullptr;
    forks = nullptr;
    forkInUse = nullptr;

    return 0;
}
