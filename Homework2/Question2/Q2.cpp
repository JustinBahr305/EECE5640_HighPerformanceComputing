// Q2
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Philospopher's Table Problem using Pthreads

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <unistd.h>
#include <pthread.h>

using namespace std;

// creates global variables for the number of philosophers and iterations
int numPhilosophers = 0;
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
        cout << "Fork " << i << ": ";
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
    int leftFork = philosopherNum;
    int rightFork = (philosopherNum + 1) % numPhilosophers;

    while (currItr < numIterations)
    {
        // thinking while waiting for forks
        while (forkInUse[leftFork] || forkInUse[rightFork])
        {
            // think
        }

        // pick up forks
        pthread_mutex_lock(&forks[leftFork]);
        pthread_mutex_lock(&forks[rightFork]);

        // eating with the forks
        forkInUse[leftFork] = true;
        forkInUse[rightFork] = true;
        philosophers[philosopherNum].isEating = true;
        usleep(philosophers[philosopherNum].timeToEat*10000); // Eat for assigned time

        if (pthread_mutex_trylock(currEat) == 0)
        {
            printTable(currItr++);
            pthread_mutex_unlock(currEat);
        }

        // put down forks
        pthread_mutex_unlock(&forks[rightFork]);
        pthread_mutex_unlock(&forks[leftFork]);
        forkInUse[leftFork] = false;
        forkInUse[rightFork] = false;
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

    while (numPhilosophers % 2 == 0 || numPhilosophers < 0)
    {
        cout << "Please enter an odd number of philosophers: " << endl;
        cin >> numPhilosophers;
    }

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
