// Q1
// Created by Justin Bahr on 2/22/2025.
// EECE 5640 - High Performance Computing
// sin(x) Taylor Series using Floats and Doubles

#include<iostream>
#include <cmath>
#include <math.h>
#include <iomanip>

using namespace std;

// defines the number of terms
const int numTerms = 15;

// function to calculate factorial iteratively
int fact(int n)
{
    // initialize fact variable with 1
    int fact = 1;

    // loop calculating factorial
    for (int i = 1; i <= n; i++)
        fact *= i;

    return fact;
}

// function to compute sin(x) using floats
float sinTaylorFloat(float x)
{
    float sinx = 0.0;
    for (int n = 0; n < numTerms; n++)
        sinx += pow(x,2*n+1)*(n % 2 == 0 ? 1.0 : -1.0)/fact(2*n+1);
    return sinx;
}

// function to compute sin(x) using doubles
double sinTaylorDouble(double x)
{
    double sinx = 0.0;
    for (int n = 0; n < numTerms; n++)
        sinx += pow(x,2*n+1)*(n % 2 == 0 ? 1.0 : -1.0)/fact(2*n+1);
    return sinx;
}

int main()
{
    // calculates sin(x) values
    double sin_05 = sin(0.05);
    double sin_5 = sin(0.5);
    double sin1 = sin(1);
    double sin1_5 = sin(1.5);

    // calculates float taylor estimations
    float sin_05f = sinTaylorFloat(0.05);
    float sin_5f = sinTaylorFloat(0.5);
    float sin1f = sinTaylorFloat(1.0);
    float sin1_5f = sinTaylorFloat(1.5);

    // calculates double taylor estimations
    double sin_05d = sinTaylorDouble(0.05);
    double sin_5d = sinTaylorDouble(0.5);
    double sin1d = sinTaylorDouble(1.0);
    double sin1_5d = sinTaylorDouble(1.5);

    // prints single precision results
    cout << fixed << setprecision(27);
    cout << "Single Precision Floating Point Calculations:" << endl;
    cout << "sin(0.05) = " << sin_05f << endl;
    cout << "sin(0.5) = " << sin_5f << endl;
    cout << "sin(1) = " << sin1f << endl;
    cout << "sin(1.5) = " << sin1_5f << endl << endl;

    // prints double precision results
    cout << fixed << setprecision(57);
    cout << "Double Precision Floating Point Calculations:" << endl;
    cout << "sin(0.05) = " << sin_05d << endl;
    cout << "sin(0.5) = " << sin_5d << endl;
    cout << "sin(1) = " << sin1d << endl;
    cout << "sin(1.5) = " << sin1_5d << endl << endl;

    // prints single precision errors
    cout << scientific<< setprecision(3);
    cout << "|Single Precision Error|:" << endl;
    cout << "sin(0.05): " << abs(sin_05f - sin_05) << endl;
    cout << "sin(0.5): " << abs(sin_5f - sin_5) << endl;
    cout << "sin(1): " << abs(sin1f - sin1) << endl;
    cout << "sin(1.5): " << abs(sin1_5f - sin1_5) << endl << endl;

    // prints double precision errors
    cout << "|Double Precision Error|:" << endl;
    cout << "sin(0.05): " << abs(sin_05d - sin_05) << endl;
    cout << "sin(0.5): " << abs(sin_5d - sin_5) << endl;
    cout << "sin(1): " << abs(sin1d - sin1) << endl;
    cout << "sin(1.5): " << abs(sin1_5d - sin1_5) << endl;

    return 0;
}
