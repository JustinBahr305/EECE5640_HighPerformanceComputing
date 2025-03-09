This folder contains the following files:
1. README.md
2. makefile
3. Q4.cpp
4. Q4.pdf

Run the command "make" to generate the following executable:
1. Q4

Run the command "make clean" to delete the executable.

Q4 should only be run on the Explorer Cluster on a node with the OpenBLAS module loaded

Run the command following command to obtain and interactive CPU-based node:
"srun -p courses -N 1 --pty --time=03:00:00 /bin/bash"

Run the following command to load the OpenBLAS module:
"module load OpenBLAS/0.3.29"

Additionally, Q4 can be run using the command "sbatch Q4.script".
The output can be found in output.txt.
Any errors can be found in error.txt.