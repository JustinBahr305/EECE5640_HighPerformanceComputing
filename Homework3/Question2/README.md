This folder contains the following files:
1. README.md
2. makefile
3. Q2.cpp
4. Q2.s
5. Q2.script
4. Q2.pdf

Run the command "make" to generate the following executable and assembly listing:
1. Q2
2. Q2.s

Run the command "make clean" to delete the executable and assembly listing.

Q2 should only be run on the Explorer Cluster on a node with AVX512 support.

Run the command following command to obtain and interactive CPU-based node with AVX512 support: 
"srun -p courses -N 1 --constraint=cascadelake --pty --time=03:00:00 /bin/bash"

Additionally, Q2 can be run using the command "sbatch Q2.script".
The output can be found in output.txt.
Any errors can be found in error.txt.