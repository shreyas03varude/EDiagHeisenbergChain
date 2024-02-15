# Exact Diagonalization for 1D Heisenberg Chain
C++ program using Eigen/Spectra libraries to perform exact diagonalization and calculate spin dynamical structure factors of the 1D Heisenberg model.
Accounts for first-neighbor coupling and also second-neighbor coupling, and can compute for up to 30 sites given proper resources.
Intended to be parallelized using OpenMP and ran on a cluster.

NOTE: Currently, the program will not compile due to issues with the include statements that will be resolved soon.

Example plot for 24 sites and J2/J1 = 0.5:
<img src="https://raw.githubusercontent.com/shreyas03varude/EDiagHeisenbergChain/main/example%20plot%20for%20L%20%3D%2024%20and%20J2%5CJ1%20%3D%200.5.png">
