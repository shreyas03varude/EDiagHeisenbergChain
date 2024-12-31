# Exact Diagonalization for 1D J1-J2 Heisenberg Chain
C++ program using Eigen/Spectra libraries to perform exact diagonalization and calculate the spin dynamical structure factor of the 1D Heisenberg model.
Accounts for nearest and next-nearest neighbor coupling, computing for up to 30 sites given sufficient memory allocation.
The program is intended to be run in a high-performance computing environment to take advantage of larger memory nodes and parallelization, but it can also be run on a PC.

## Compilation
### PC
To compile the program on your own computer without MKL, comment out line 2 of the source code and run the following command:
```
g++ -O2 -o xxz XXZ.cpp -I.
```
### HPC
On a Linux environment, the compilation looks something like:
```
module load intel
/./apps/compilers/intel/2020/0.166/mkl/bin/mklvars.sh intel64
icpc -I. -g -O2 -o xxz XXZ.cpp -qopenmp -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -I"${MKLROOT}/include" -msse3 -axcore-avx2,core-avx-i
```
--------------------
Example plot for 24 sites and J2/J1 = 0.5:

<img src="https://raw.githubusercontent.com/shreyas03varude/EDiagHeisenbergChain/main/example.png">
