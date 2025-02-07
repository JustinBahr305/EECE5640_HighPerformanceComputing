// Q1_leibniz
// Created by Justin Bahr on 1/31/2025.
// EECE 5640 - High Performance Computing
// Leibniz Estimation for pi using

#include <iostream>
#include <chrono>
#include <math.h>

using namespace std;

double piByLeibniz(int numTerms)
{
    double piSum = 0;

    for (int i = 0; i < numTerms; i = i+2)
    {
        piSum += (double)1/(2*i+1);
    }

    for (int i = 1; i < numTerms; i = i + 2)
    {
        piSum -= (double)1/(2*i+1);
    }

    return 4*piSum;
}

int main()
{
    // creates a variable for the number of terms
    int numTerms;

    // creates the placeholder for the calculated value of pi
    double calc_pi;

    // allows the user to choose the number of darts
    cout << "How many terms would you like to add?" << endl;
    cin >> numTerms;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the clock
    auto start_time = clock::now();

    calc_pi = piByLeibniz(numTerms);

    // stops the clock
    auto end_time = clock::now();

    // casts run_time in nanoseconds
    auto run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

    // prints the calculated value of pi and the runtime
    cout << "Time to execute: " << run_time << " ns" << endl;
    cout << "Calculated value of pi: " << calc_pi << endl;
    cout << "Absolute error: " << abs(calc_pi - M_PI) << endl;

    return 0;
}
