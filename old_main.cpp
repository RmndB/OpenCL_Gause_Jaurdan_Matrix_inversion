//
//  main.cpp
//

#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>

// #include <mpi.h>

using namespace std;

double* castValarrayToArray(std::valarray<double> array) {
	double* castedArray = (double*)malloc(array.size() * sizeof(double));
	for (size_t i = 0; i < array.size(); i++)
		castedArray[i] = array[i];
	return castedArray;
}

// Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
void invertSequential(Matrix& iA) {

	// vérifier que la matrice est carrée
	assert(iA.rows() == iA.cols());
	// construire la matrice [A I]
	MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

	// traiter chaque rangée
	for (size_t k = 0; k < iA.rows(); ++k) {
		// trouver l'index p du plus grand pivot de la colonne k en valeur absolue
		// (pour une meilleure stabilité numérique).
		size_t p = k;
		double lMax = fabs(lAI(k, k));
		for (size_t i = k; i < lAI.rows(); ++i) {
			if (fabs(lAI(i, k)) > lMax) {
				lMax = fabs(lAI(i, k));
				p = i;
			}
		}

		// vérifier que la matrice n'est pas singulière
		if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

		// échanger la ligne courante avec celle du pivot
		if (p != k) lAI.swapRows(p, k);

		double lValue = lAI(k, k);
		for (size_t j = 0; j < lAI.cols(); ++j) {
			// On divise les éléments de la rangée k
			// par la valeur du pivot.
			// Ainsi, lAI(k,k) deviendra égal à 1.
			lAI(k, j) /= lValue;
		}

		// Pour chaque rangée...
		for (size_t i = 0; i < lAI.rows(); ++i) {
			if (i != k) { // ...différente de k
				// On soustrait la rangée k
				// multipliée par l'élément k de la rangée courante
				double lValue = lAI(i, k);
				lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
			}
		}
	}

	// On copie la partie droite de la matrice AI ainsi transformée
	// dans la matrice courante (this).
	for (unsigned int i = 0; i < iA.rows(); ++i) {
		iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols() + iA.cols(), iA.cols(), 1)];
	}
}

// Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
/*
void invertParallel(Matrix& iA) {
	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int localP[2];
	int globalP[2];

	assert(iA.rows() == iA.cols());
	MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

	for (size_t k = 0; k < iA.rows(); ++k) {
		// Compute pivot for assigned rows
		size_t p = k;
		double lMax = fabs(lAI(k, k));
		for (size_t i = k; i < lAI.rows(); ++i) {
			if (i%size == rank) {
				if (fabs(lAI(i, k)) > lMax) {
					lMax = fabs(lAI(i, k));
					p = i;
				}
			}
		}

		// Reduce MPI_MAXLOC operation to find the strongest pivot - needs 2d array with rank as input
		localP[0] = p;
		localP[1] = rank;
		MPI_Allreduce(localP, globalP, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
		p = globalP[0];

		if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");


		// Broadcast k & p row from k%size and p%size for update
		double* rowPivot = castValarrayToArray(lAI.getRowCopy(p));
		double* rowK = castValarrayToArray(lAI.getRowCopy(k));

		MPI_Bcast(rowPivot, lAI.cols(), MPI_DOUBLE, p%size, MPI_COMM_WORLD);
		MPI_Bcast(rowK, lAI.cols(), MPI_DOUBLE, k%size, MPI_COMM_WORLD);

		for (size_t i = 0; i < lAI.cols(); i++) {
			lAI(p, i) = rowPivot[i];
			lAI(k, i) = rowK[i];
		}
		if (p != k) lAI.swapRows(p, k);

		double lValue = lAI(k, k);
		for (size_t j = 0; j < lAI.cols(); ++j) {
			lAI(k, j) /= lValue;
		}

		// Compute assigned rows
		for (size_t i = 0; i < lAI.rows(); ++i) {
			if (i != k) {
				if (i%size == rank) {
					double lValue = lAI(i, k);
					lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
				}
			}
		}
	}

	for (unsigned int i = 0; i < iA.rows(); ++i) {
		iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols() + iA.cols(), iA.cols(), 1)];
	}

	// Rank == 0, gather scattered rows except itself
	if (rank == 0) {
		for (size_t i = 0; i < iA.rows(); ++i) {
			if (i%size != rank) {
				double* newRow = (double*)malloc(iA.rows() * sizeof(double));
				MPI_Recv(newRow, iA.rows(), MPI_DOUBLE, i%size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				valarray<double> castedArray(newRow, iA.rows());
				iA.getRowSlice(i) = castedArray;
			}
		}
	}
	// Rank != 0, send assigned rows to rank 0
	else {
		for (size_t i = 0; i < iA.rows(); ++i) {
			if (i%size == rank) {
				double* newRow = castValarrayToArray(iA.getRowCopy(i));
				MPI_Send(newRow, iA.rows(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
	}
}
*/

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {

	// vérifier la compatibilité des matrices
	assert(iMat1.cols() == iMat2.rows());
	// effectuer le produit matriciel
	Matrix lRes(iMat1.rows(), iMat2.cols());
	// traiter chaque rangée
	for (size_t i = 0; i < lRes.rows(); ++i) {
		// traiter chaque colonne
		for (size_t j = 0; j < lRes.cols(); ++j) {
			lRes(i, j) = (iMat1.getRowCopy(i)*iMat2.getColumnCopy(j)).sum();
		}
	}
	return lRes;
}

int main(int argc, char** argv) {
	bool seq = true;

	// MPI_Init(&argc, &argv);

	// int rank, size;
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// MPI_Comm_size(MPI_COMM_WORLD, &size);
	double startTime, endTime;

	srand((unsigned)time(NULL));

	unsigned int lS = 5;
	if (argc == 2) {
		lS = atoi(argv[1]);
	}

	MatrixRandom lA(lS, lS);
	// if (rank == 0) cout << "Matrice random:\n" << lA.str() << endl;

	Matrix lB(lA);

	// if (rank == 0) startTime = MPI_Wtime();

	/*
	if (size == 1)
		invertSequential(lB);
	else
		invertParallel(lB);
	*/

	if (seq)
		invertSequential(lB);
	else
		cout << "TODO" << endl;
	
	/*
	if (rank == 0) {
		endTime = MPI_Wtime();

		cout << "Matrice inverse:\n" << lB.str() << endl;

		Matrix lRes = multiplyMatrix(lA, lB);
		cout << "Produit des deux matrices:\n" << lRes.str() << endl;

		cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;

		cout << "Time: " << endTime - startTime << endl;
	}

	MPI_Finalize();
	*/

	cout << "Matrice inverse:\n" << lB.str() << endl;

	Matrix lRes = multiplyMatrix(lA, lB);
	cout << "Produit des deux matrices:\n" << lRes.str() << endl;

	cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;

	cout << "Time: " << endTime - startTime << endl;

	return 0;
}