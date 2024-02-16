#define _USE_MATH_DEFINES

#include "Eigen/Sparse"
#include "Eigen/Dense"
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
// #include <omp.h> // If OpenMP installed
using namespace std;
using namespace Eigen;
using namespace Spectra;
typedef complex<double> dcomplex;
typedef Eigen::Triplet<float> T;

VectorXd CmplxContFracExpan(VectorXcd& v, float E_0, SparseMatrix<float>& sparseH, VectorXd& specX, double epsilone_CFE, double q);

unsigned int hSize(unsigned int L) { // L choose L/2
	float n = L;
	for (float i = L / 2; i > 1; i--) {
		n = n * --L / i;
	}
	return n + 0.5f;
}

// Count the number of 1s in the binary representation of x
int countBits(int x) {
	// From Hacker's Delight, p. 66
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F;
	x = x + (x >> 8);
	x = x + (x >> 16);
	return x & 0x0000003F;
}


VectorXi makeSz0Basis(unsigned int L) {
	if (L % 2 != 0) {
		cout << "Please input even L." << endl;
		exit(-1);
	}
	VectorXi basisSz0List(hSize(L));
	int n = 0;
	for (int i = 0; i < pow(2, L); i++) {
		int Szi = 2 * countBits(i) - L;
		if (Szi == 0) {
			basisSz0List[n] = i;
			n++;
		}
	}
	return basisSz0List;
}

///Make a 1D Heisenberg chain of length L with Jxy,Jz and magnetic field h out of an SzList of states
SparseMatrix<float> makeSparseH(VectorXi& SzList, unsigned int L, float Jxy, float Jz, float Jxy2, float Jz2, float h) {
	unordered_map<int, int> basisMap;
	unsigned int stateID = 0;
	for (auto& state : SzList) {
		basisMap[state] = stateID;
		stateID++;
	}
	unsigned int nH = stateID;
	SparseMatrix<float> sparseH(nH, nH);
    vector<T> triplets;
	unsigned long long triplets_size = L * nH * 2;
    triplets.reserve(triplets_size);
	// cout << "Reserving space for nonzeros" << endl;
	// sparseH.reserve(VectorXi::Constant(nH, L));
	cout << "Storing the nonzeros in a vector of triplets" << endl;
	for (auto& state : SzList) {
		int idxA = basisMap[state];
		// sparseH.coeffRef(idxA, idxA) += -h * L / 2;
		for (int i = 0; i < L; i++) {
			int j = (i + 1) % L;
			float diag = 0;
			if (j != 0) {
                if (((state >> i) & 1) == ((state >> j) & 1)) {
                    diag += -Jz / 4;
                }
                else {
                    diag += Jz / 4;
                    int mask = (1 << i) + (1 << j);
                    int stateB = state ^ mask;
                    int idxB = basisMap[stateB];
                    triplets.push_back(T(idxA, idxB, -Jxy/2));
					if (idxA > nH || idxB > nH) {
						cout << "HEY! idxA is " << idxA << " and idxB is " << idxB << endl;
					}
                }
            }
            j = (i + 2) % L;
            if (j != 0 && Jz2 != 0) {
                if (((state >> i) & 1) == ((state >> j) & 1)) {
                    diag += -Jz2 / 4;
                }
                else {
					diag += Jz2 / 4;
                    int mask = (1 << i) + (1 << j);
                    int stateB = state ^ mask;
                    int idxB = basisMap[stateB];
                    triplets.push_back(T(idxA, idxB, -Jxy2/2));
					if (idxA > nH || idxB > nH) {
						cout << "HEY! idxA is " << idxA << " and idxB is " << idxB << endl;
					}
                }
            }
            triplets.push_back(T(idxA, idxA, diag));
			if (idxA > nH) {
				cout << "HEY! idxA is " << idxA << endl;
			}
		}
	}
	cout << "sparseH.size() = " << nH << "x" << nH << endl;
	cout << "triplets.size() = " << triplets.size() << endl;
	cout << "Setting the Hamiltonian matrix from triplets" << endl;
    sparseH.setFromTriplets(triplets.begin(), triplets.end());
	return sparseH;
}

VectorXcd getSzqket0(int L, double q, VectorXi& basisList, VectorXf& GSEigenvector) {
	int istate = 0;
	VectorXcd newVector(GSEigenvector.size());
	dcomplex i(0, 1);
	for (auto& state : basisList) {
		dcomplex fac = GSEigenvector[istate];
		dcomplex val = 0;
		for (int R = 0; R < L; R++) {
			if ((state >> R) & 1) {
				val += std::exp(q * R * i);
			}
			else {
				val -= std::exp(q * R * i);
			}
		}
		newVector[istate] = fac * val;
		istate++;
	}
	return newVector;
}

//Generate the dynamic structure factor
MatrixXd generateSqw(int L, float Jxy, float Jz, float Jxy2, float Jz2, float h, double omega_start, double omega_end, double omega_step, double gamma, VectorXd& qs) {
	auto start = chrono::high_resolution_clock::now();
	VectorXi basisList = makeSz0Basis(L);
    SparseMatrix<float> sparseH = makeSparseH(basisList, L, Jxy, Jz, Jxy2, Jz2, h);
	VectorXf e;
	MatrixXf v;
	cout << "Diagonalizing..." << endl;
    auto diag_start = chrono::high_resolution_clock::now();
  	SparseSymMatProd<float> obj(sparseH);
	SymEigsSolver<SparseSymMatProd<float>> solver(obj, 2, 10);
	solver.init();
	solver.compute(SortRule::SmallestAlge, 500, 1.0e-7, SortRule::SmallestAlge);
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - diag_start);
	cout << "Diagonalization finished in " << duration.count() / 1000000.0 << " seconds." << endl;
	if (solver.info() == CompInfo::Successful) {
		e = solver.eigenvalues();
		v = solver.eigenvectors();
	}
	else {
		cout << "Failed!" << endl;
		exit(-1);
	}
	float GSEnergy = e[0];
	VectorXf GSEigenvector = v.col(0);

	cout << "Ground State Energy at total Sz = 0 sector: " << GSEnergy << endl;

	double omega_range = omega_end - omega_start;
	int num_points = ceil(omega_range / omega_step);
	VectorXd omegas(num_points);
	for (int i = 0; i < num_points; i++) {
		omegas[i] = omega_start + i * omega_step;
	}
	MatrixXd data(qs.size(), num_points);
	int iq = 0;
	for (double& q : qs) {
		VectorXcd vector = getSzqket0(L, q, basisList, GSEigenvector);
		VectorXd specs = CmplxContFracExpan(vector, GSEnergy, sparseH, omegas, gamma, q);
		data.row(iq) = specs;
		iq++;
	}
	stop = chrono::high_resolution_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	cout << "Program finished in " << duration.count() / 1000000.0 << " seconds." << endl;
	return data;
}

VectorXd CmplxContFracExpan(VectorXcd& v, float E_0, SparseMatrix<float>& sparseH, VectorXd& specX, double epsilone_CFE, double q) {
	SparseMatrix<dcomplex, RowMajor> complexSparseH = sparseH.cast<dcomplex>();

	int iter_num = 100;
	int Hsize = v.size();
	VectorXd specY(specX.size());
	VectorXcd carray_1(Hsize);
	VectorXcd phi_n0 = v;
	VectorXcd phi_n1(Hsize);
	VectorXcd phi_nm(VectorXcd::Zero(Hsize));

	VectorXd work_an(iter_num);
	VectorXd work_bn(iter_num);

	dcomplex rtemp1 = 0;
	dcomplex rtemp2 = 0;
	dcomplex a_n = 0;
	dcomplex b_n = 0;
	dcomplex ctemp1 = 0;

	for (int iiter = 0; iiter < iter_num; iiter++) {
		auto start = chrono::high_resolution_clock::now();
		carray_1 = complexSparseH * phi_n0;
		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		if (iiter == 0) {
			cout << "The sparse matrix multiplication was done in " << duration.count() / 1000000.0 << " seconds for "
				<< "iiter = 0 and q = " << q << endl;
		}
		ctemp1 = phi_n0.dot(carray_1);
		rtemp1 = phi_n0.squaredNorm();
		a_n = ctemp1 / rtemp1;

		if (iiter == 0) {
			b_n = 0;
		}
		phi_n1 = carray_1 - a_n * phi_n0 - b_n * phi_nm;

		work_an[iiter] = a_n.real();
		work_bn[iiter] = b_n.real();

		rtemp2 = phi_n1.squaredNorm();
		b_n = rtemp2 / rtemp1;

		phi_nm = phi_n0 / sqrt(rtemp2);
		phi_n0 = phi_n1 / sqrt(rtemp2);
	}
	carray_1 = v;
	rtemp1 = carray_1.squaredNorm();

	dcomplex i(0, 1);
	dcomplex ctemp3 = 0;
	dcomplex ctemp4 = 0;
	dcomplex con_ct1 = 0;
	for (int Eind = 0; Eind < specX.size(); Eind++) {
		ctemp1 = specX[Eind] + E_0 + epsilone_CFE * i;
		ctemp4 = 0;
		for (int jj = 99; jj > 0; jj--) {
			ctemp3 = work_bn[jj] / (ctemp1 - work_an[jj] - ctemp4);
			ctemp4 = ctemp3;
		}
		con_ct1 = rtemp1 / (ctemp1 - work_an[0] - ctemp4);
		specY[Eind] = con_ct1.imag() / (-M_PI);
	}
	return specY;
}

void writeData(MatrixXd data, unsigned int L, float ratio) {
    int nX = data.rows();
    int nY = data.cols();
    string fileName = "datas/" + to_string(L) + "/L=" + to_string(L) + " and ratio = " + to_string(ratio).substr(0,3) + ".csv";
    ofstream file(fileName);
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
    // omp_set_num_threads(4);

    unsigned int L = 14;
    float Jxy = -1;
    float Jz = -1;
    float ratio = 0;
    if (argc > 1) {
        L = stoi(argv[1]);
    }
    if (argc == 3) {
        ratio = stof(argv[2]);
    }
    float Jxy2 = ratio * Jxy;
    float Jz2 = ratio * Jz;
    cout << "L = " << L << endl;
	cout << "J2/J1 = " << ratio << endl;
    float h = 0;
    VectorXd qs(L / 2 + 1);
    for (int i = L / 2 - 1; i < L; i++) {
        qs[i - L / 2 + 1] = (i + 1) * 2 * M_PI / L;
    }
    MatrixXd data = generateSqw(L, Jxy, Jz, Jxy2, Jz2, h, 0, 3, 0.02, 0.1, qs);
    writeData(data, L, ratio);
	return 0;
}
