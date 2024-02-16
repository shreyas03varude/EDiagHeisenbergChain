# Exact Diagonalization for 1D Heisenberg Chain
C++ program using Eigen/Spectra libraries to perform exact diagonalization and calculate spin dynamical structure factors of the 1D Heisenberg model.
Accounts for first-neighbor coupling and also second-neighbor coupling, and can compute for up to 30 sites given proper resources.
Intended to be parallelized using OpenMP and ran on a cluster.

NOTE: To compile the program on your own computer, you will need the "Eigen/" and the "Spectra/" source libraries in the same directory as main.cpp. Then, you can run the command

g++ -o main -O2 main.cpp

Example plot for 24 sites and J2/J1 = 0.5:

<img src="https://raw.githubusercontent.com/shreyas03varude/EDiagHeisenbergChain/main/example.png">
