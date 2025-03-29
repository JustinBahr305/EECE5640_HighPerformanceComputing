This folder contains the following files:
1. README.md
2. makefile
3. Q1_gpu.cpp
4. hist.cu 
5. Q1_omp.cpp 
6. Q1_gpu.script 
7. Q1_omp.script
8. Q1.pdf

Run the command "make TARGET=Q1_gpu" to generate the following executable:
1. Q1_gpu

Run the command "make TARGET=Q1_omp" to generate the following executable:
1. Q1_omp

Run the command "make clean" to delete both executables.

To execute the program on a GPU node, run the command "sbatch Q1_gpu.script".
To execute the program on a CPU node, run the command "sbatch Q1_omp.script".