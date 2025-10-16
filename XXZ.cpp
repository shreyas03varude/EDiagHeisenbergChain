#define _USE_MATH_DEFINES
#define EIGEN_USE_MKL_ALL // comment if running without MKL
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "Spectra/SymEigsSolver.h"
#include "Spectra/MatOp/SparseSymMatProd.h"
#include <iostream>
#include <unordered_map>
#include <complex>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace Eigen;

typedef complex<double> dcomplex;
typedef Triplet<float> T;
typedef SparseMatrix<double, ColMajor, std::ptrdiff_t> Hamiltonian;
typedef Eigen::Matrix<unsigned int, Dynamic, 1> VectorXui;

uint64_t hSize(unsigned int L);
int countBits(int x);
VectorXui makeSz0Basis(unsigned int L);
Hamiltonian makeSparseH(unsigned int L, float Jz, float Jxy, float Jz2, float Jxy2, float h);
VectorXcd makeSzqket0(VectorXui& Sz0Basis, unsigned int L, double q, VectorXd& psi_0);
MatrixXd generateSqw(unsigned int L, float Jz, float Jxy, float Jz2, float Jxy2, float h, double w_0, double w_f, double dw, double gamma);
void writeData(MatrixXd& data, string filename);
VectorXd CmplxContFracExpan(VectorXcd& phi_n0, double E_0, Hamiltonian& sparseH, VectorXd& omegas, double gamma, double q);

// Given L, calculate L choose L/2, the dimension of the L-site Hilbert space. (L! / 2(L/2)!)
uint64_t hSize(unsigned int L) {
	double n = L;
	for (double i = L / 2; i > 1; i--) {
		n = n * --L / i;
	}
	return n + 0.5f;
}

// Count the number of 1s in the binary representation of x.
int countBits(int x) {
	// From Hacker's Delight, p. 66
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F;
	x = x + (x >> 8);
	x = x + (x >> 16);
	return x & 0x0000003F;
}

// Generate the basis of states with Sz = 0 for L sites.
VectorXui makeSz0Basis(unsigned int L) {
	if (L % 2 != 0) { // L must be even for states to have Sz = 0
		cout << "Please input even L." << endl;
		exit(-1);
	}
	unsigned int basis_size = hSize(L);
	VectorXui Sz0Basis(basis_size);
	uint64_t stateID = 0;
	for (uint64_t i = 0; i < (uint64_t) (pow(2, L) + 0.5); i++) {
		int Sz = 2 * countBits(i) - L;
		if (Sz == 0) {
			Sz0Basis[stateID] = i;
			stateID++;
		}
	}
	return Sz0Basis;
}

// Adapted from Ryan Levy, 2018: https://www.ryanlevy.science/physics/Heisenberg1D-ED/ 
Hamiltonian makeSparseH(VectorXui& Sz0Basis, unsigned int L, float Jz, float Jxy, float Jz2, float Jxy2, float h) {
	unordered_map<int, int> basisMap;
	unsigned int stateID = 0;
	for (auto& state : Sz0Basis) { // enumerate each state, assigning an index to each
		basisMap[state] = stateID;
		stateID++;
	}
	unsigned int hSize = stateID; // # of states
	Hamiltonian sparseH(hSize, hSize);
	vector<T> triplets; // 3-tuples storing row index, column index, and value corresponding to that matrix element
	uint64_t triplets_size = (uint64_t) L * (uint64_t) hSize * 2ull;
	// cout << "Reserving space for " << triplets_size << " triplets" << endl;
	triplets.reserve(triplets_size); // reserving space in advance saves time when size of this vector is known
	// cout << "Storing the nonzeros in a vector of triplets" << endl;
	for (auto& state : Sz0Basis) {
		int idxA = basisMap[state];
		for (int i = 0; i < L; i++) {
			float diag = 0; // diagonal element
			// nearest neighbor interaction (J1)
			int j = (i + 1) % L;
			if (j > 0) { // skipping interaction of last site with first site - closed BC
				if (((state >> i) & 1) == ((state >> j) & 1)) { // are sites i and j the same spin?
					diag += -Jz / 4; // diagonal term
				}
				else {
					diag += Jz / 4;
					int mask = (1 << i) + (1 << j);
					int stateB = state ^ mask; // flips spins at sites i and j
					int idxB = basisMap[stateB]; // finds column corresponding to this state
					triplets.push_back(T(idxA, idxB, -Jxy/2)); // add off-diagonal term to vector of 3-tuples
				}
			}
			// next-nearest neighbor interaction (J2)
			if (Jz2 != 0) {
				j = (i + 2) % L; // site j is now 2 sites removed from site i
				if (j > 1) { // closed BC
					if (((state >> i) & 1) == ((state >> j) & 1)) {
						diag += -Jz2 / 4;
					}
					else {
						diag += Jz2 / 4;
						int mask = (1 << i) + (1 << j);
						int stateB = state ^ mask;
						int idxB = basisMap[stateB];
						triplets.push_back(T(idxA, idxB, -Jxy2/2));
					}
				}
			}
		triplets.push_back(T(idxA, idxA, diag)); // add diagonal term to vector of 3-tuples
		}
	}
	// cout << "The size of sparseH is " << hSize << " x " << hSize << endl;
	// cout << "triplets.size() = " << triplets.size() << endl;
	// cout << "Setting the Hamiltonian matrix from triplets" << endl;
	sparseH.setFromTriplets(triplets.begin(), triplets.end()); // set the sparse matrix from the vector of 3-tuples
	// cout << "# nnz in the Hamiltonian = " << sparseH.nonZeros() << endl;
	return sparseH;
}

// Calculate S_z^q|0> using a discrete Fourier transform.
VectorXcd makeSzqket0(VectorXui& Sz0Basis, unsigned int L, double q, VectorXd& psi_0) {
	VectorXcd Szqket0(psi_0.size());
	dcomplex i(0, 1); // imaginary number i
	for (int j = 0; j < Sz0Basis.size(); j++) {
		dcomplex fac = psi_0[j]; // jth component of |0>
		dcomplex val = 0;
		for (int R = 0; R < L; R++) { // discrete FT of S^z_R
			if ((Sz0Basis[j] >> R) & 1) { // is site R spin up?
				val += std::exp(q * R * i);
			}
			else {
				val -= std::exp(q * R * i); // site R spin down
			}
		}
		Szqket0[j] = fac * val; // jth component of S^z_q * jth component of |0>
	}
	return Szqket0;
}

// Generate the dynamic structure factor S(q,w).
MatrixXd generateSqw(unsigned int L, float Jz, float Jxy, float Jz2, float Jxy2, float h, double w_0, double w_f, double dw, double gamma) {
	auto start = chrono::high_resolution_clock::now();
	VectorXui Sz0Basis = makeSz0Basis(L); // basis of Sz = 0 states
	Hamiltonian sparseH = makeSparseH(Sz0Basis, L, Jz, Jxy, Jz2, Jxy2, h); // sparse Hamiltonian matrix
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
	cout << "Filled the Hamiltonian in " << duration.count() << " seconds" << endl;
	start = chrono::high_resolution_clock::now();
	cout << "Diagonalizing the Hamiltonian matrix" << endl;
	VectorXd energies;
	MatrixXd eigenstates;
	Spectra::SparseSymMatProd<double, 1, 0, ptrdiff_t> obj(sparseH); // using upper triangular column-major matrix of type double and large storage index
	Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double, 1, 0, ptrdiff_t>> solver(obj, 2, 10); // 2 eigenvalues, dim of Krylov subspace = 10
	solver.init();
	solver.compute(Spectra::SortRule::SmallestAlge, 500, 1.0e-7, Spectra::SortRule::SmallestAlge); // selecting smallest algebraic EVs, 500 max iterations
	stop = chrono::high_resolution_clock::now();
	duration = chrono::duration_cast<chrono::seconds>(stop - start);
	cout << "Diagonalization finished in " << duration.count() << " seconds" << endl;
	if (solver.info() == Spectra::CompInfo::Successful) {
		energies = solver.eigenvalues();
		eigenstates = solver.eigenvectors();
	}
	else {
		cout << "Failed!" << endl;
		exit(-1);
	}
	double E_0 = energies[0]; // ground-state energy
	VectorXd psi_0 = eigenstates.col(0); // ground-state eigenvector

	cout << "Ground State Energy at total Sz = 0 sector: " << E_0 << endl;

	// computing S(q,w) with the ground state values
	start = chrono::high_resolution_clock::now();
	// frequency values are evenly spaced over an open interval [omega_start, omega_end)
	int num_points = ceil((w_f - w_0) / dw);
	VectorXd omegas(num_points); // all omega values
	for (int i = 0; i < num_points; i++) {
		omegas[i] = w_0 + i * dw;
	}
	// momentum values are evenly spaced over a closed interval [pi, 2pi] instead of [0, 2pi] to exploit the evenness of S(q,w) about q = pi
	VectorXd qs(L/2 + 1);
	for (int i = 0; i < L/2 + 1; i++) {
		qs[i] = (1 + i*2.0f/L) * M_PI;
	}
	MatrixXd data(L/2 + 1, num_points);
	for (int i = 0; i < L/2 + 1; i++) {
		VectorXcd Szqket0 = makeSzqket0(Sz0Basis, L, qs[i], psi_0); // S^z_q|0>
		VectorXd specs = CmplxContFracExpan(Szqket0, E_0, sparseH, omegas, gamma, qs[i]);
		data.row(i) = specs;		
	}
	stop = chrono::high_resolution_clock::now();
	duration = chrono::duration_cast<chrono::seconds>(stop - start);
	cout << "Computed S(q,w) in " << duration.count() << " seconds" << endl;
	return data;
}

// Adapted from Elbio Dagotto, Correlated Electrons in High-temperature 
// Superconductors, Rev. Mod. Phys. 66 (1994), p.774-777.
VectorXd CmplxContFracExpan(VectorXcd& phi_n0, double E_0, Hamiltonian& sparseH, VectorXd& omegas, double gamma, double q) {
	// casting double-type matrix in column major to complex-type matrix in row major
	SparseMatrix<dcomplex, RowMajor, ptrdiff_t> complexSparseH = sparseH.cast<dcomplex>(); 

	int iter_num = 100;
	int hSize = phi_n0.size();
	VectorXd specY(omegas.size());
	VectorXcd carray_1(hSize);
	VectorXcd phi_n1(hSize);
	VectorXcd phi_nm(VectorXcd::Zero(hSize));

	VectorXd work_an(iter_num);
	VectorXd work_bn(iter_num);

	double r0 = phi_n0.squaredNorm();
	dcomplex r1 = 0;
	dcomplex r2 = 0;
	dcomplex a_n = 0;
	dcomplex b_n = 0;
	dcomplex c1 = 0;

	for (int i = 0; i < iter_num; i++) {
		// auto start = chrono::high_resolution_clock::now();
		carray_1 = complexSparseH * phi_n0; // slow matrix multiplication
		// auto stop = chrono::high_resolution_clock::now();
		// auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		// if (iiter == 0) {
		// 	cout << "The first iteration of sparse matrix multiplication was done in " 
		// 		 << duration.count() / 1000000.0 << " seconds for q = " << q << endl;
		// }
		c1 = phi_n0.dot(carray_1);
		r1 = phi_n0.squaredNorm();
		a_n = c1 / r1;

		phi_n1 = carray_1 - a_n * phi_n0 - b_n * phi_nm; // recursively generating an orthogonal basis
														 // such that H is tridiagonal

		work_an[i] = a_n.real();
		work_bn[i] = b_n.real();

		r2 = phi_n1.squaredNorm();
		b_n = r2 / r1;

		phi_nm = phi_n0 / sqrt(r2); // normalization
		phi_n0 = phi_n1 / sqrt(r2);
	}

	dcomplex i(0, 1);
	dcomplex z = 0;
	dcomplex c2 = 0;
	dcomplex c3 = 0;
	dcomplex con_ct1 = 0;
	for (int j = 0; j < omegas.size(); j++) {
		z = omegas[j] + E_0 + i * gamma;
		c3 = 0;
		for (int k = iter_num - 1; k > 0; k--) { // complex continued fraction expansion
			c2 = work_bn[k] / (z - work_an[k] - c3);
			c3 = c2;
		}
		con_ct1 = r0 / (z - work_an[0] - c3);
		specY[j] = con_ct1.imag() / (-M_PI); // Cramer's rule
	}
	return specY;
}

// Write the S(q,w) data out to a .csv file.
void writeData(MatrixXd& data, string filename) {
	int nX = data.rows();
	ofstream file(filename);
	for (int i = 0; i < nX; i++) {
		auto row = data.row(i);
		for (int j = 0; j < 10; j++) {
			for (int k = 0; k < row.size(); k++) {
				file << row[k] << ',';
			}
			file << '\n';
		}
	}
}

int main(int argc, char** argv) {
	auto start = chrono::high_resolution_clock::now();
	unsigned int L; // # sites
	float Jz; // Jz for nearest-neighbor interaction
	float Jxy; // Jxy for nearest-neighbor interaction
	float Jz2; // Jz for next-nearest-neighbor interaction
	float Jxy2; // Jxy for next-nearest-neighbor interaction
	string outfile; // name of (.csv) output file to write data to
	string help = "Usage:\t./xxz <L> <Jz> <Jxy> <Jz2> <Jxy2> <output filename>\n";
	if (argc == 7) {
		try {
			L = stoi(argv[1]);
			Jz = stof(argv[2]);
			Jxy = stof(argv[3]);
			Jz2 = stof(argv[4]);
			Jxy2 = stof(argv[5]);
			outfile = argv[6];
		}
		catch (invalid_argument& e) { // if non-numeric value entered when numeric value expected
			cout << help;
			return -1;
		}
	}
	else {
		cout << help;
		return -1;
	}
	float h = 0; // external magnetic field
	float w_0 = 0; // omega start value
	float w_f = 3; // omega end value
	float dw = 0.02; // omega spacing
	float gamma = 0.1;
	MatrixXd data = generateSqw(L, Jz, Jxy, Jz2, Jxy2, h, w_0, w_f, dw, gamma);
	writeData(data, outfile);
	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::seconds>(end - start);
	cout << "The entire program took " << duration.count() << " seconds" << endl;
	return 0;
}
