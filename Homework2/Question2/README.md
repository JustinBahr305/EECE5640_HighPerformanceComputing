This folder contains the following files:
1. README.md
2. makefile
3. Q2.cpp: requires both forks to be available before picking up either
4. Q2_alt.cpp: alternating fork pickup order
4. Q2.pdf

Run the command "make" to generate the following executables:
1. Q2
2. Q2_alt

Run the command "make clean" to delete the executables.

Both executables will prompt the user to enter the number of philosophers, then the number of iterations.
Note that each thread will print the state of the table whenever it's philosopher eats. 
Philosophers will eat until the specified number of iterations is reached, at which time, all philosophers currently
eating will print the state of the table (ie 12 iterations could result in ~15 rounds printed).