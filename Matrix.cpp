//
//  Matrix.cpp
//

#include <string>
#include <valarray>
#include <cassert>
#include <iostream>

#include "Matrix.hpp"
#include <sstream>

using namespace std;

// Permuter deux rang�es de la matrice.
Matrix& Matrix::swapRows(size_t iR1, size_t iR2) {
	// v�rifier la validit� des indices de rang�e
	assert(iR1 < rows() && iR2 < rows());
	// tester la n�cessit� de permuter
	if (iR1 == iR2) return *this;
	// permuter les deux rang�es
	valarray<double> lTmp(mData[slice(iR1*cols(), cols(), 1)]);
	mData[slice(iR1*cols(), cols(), 1)] = mData[slice(iR2*cols(), cols(), 1)];
	mData[slice(iR2*cols(), cols(), 1)] = lTmp;
	return *this;
}

// Permuter deux colonnes de la matrice.
Matrix& Matrix::swapColumns(size_t iC1, size_t iC2) {
	// v�rifier la validit� des indices de colonne
	assert(iC1 < cols() && iC2 < cols());
	// tester la n�cessit� de permuter
	if (iC1 == iC2) return *this;
	// permuter les deux rang�es
	valarray<double> lTmp(mData[slice(iC1, rows(), cols())]);
	mData[slice(iC1, rows(), cols())] = mData[slice(iC2, rows(), cols())];
	mData[slice(iC2, rows(), cols())] = lTmp;
	return *this;
}

// Repr�senter la matrice sous la forme d'une cha�ne de caract�res.
// Pratique pour le d�buggage...
string Matrix::str(void) const {
	ostringstream oss;
	for (size_t i = 0; i < rows(); ++i) {
		if (i == 0) oss << "[[ ";
		else oss << " [ ";
		for (size_t j = 0; j < cols(); ++j) {
			oss << (*this)(i, j);
			if (j + 1 != cols()) oss << ", ";
		}
		if (i + 1 == this->rows()) oss << "]]";
		else oss << " ]," << endl;
	}
	return oss.str();
}

// Construire une matrice identit�.
MatrixIdentity::MatrixIdentity(size_t iSize) : Matrix(iSize, iSize) {
	for (size_t i = 0; i < iSize; ++i) {
		(*this)(i, i) = 1.0;
	}
}

// Construire une matrice al�atoire [0,1) iRows x iCols.
// Utiliser srand pour initialiser le g�n�rateur de nombres.
MatrixRandom::MatrixRandom(size_t iRows, size_t iCols) : Matrix(iRows, iCols) {
	for (size_t i = 0; i < mData.size(); ++i) {
		mData[i] = (double)rand() / RAND_MAX;;
	}
}

// Construire une matrice en concat�nant les colonnes de deux matrices de m�me hauteur.
MatrixConcatCols::MatrixConcatCols(const Matrix& iMat1, const Matrix& iMat2) : Matrix(iMat1.rows(), iMat1.cols() + iMat2.cols()) {
	// v�rifier la compatibilit� des matrices
	assert(iMat1.rows() == iMat2.rows());
	// Pour chaque rang�e
	for (size_t i = 0; i < rows(); ++i) {
		// rang�e i de la premi�re matrice
		mData[slice(i*cols(), iMat1.cols(), 1)] = iMat1.getRowSlice(i);
		// rang�e i de la seconde matrice
		mData[slice(i*cols() + iMat1.cols(), iMat2.cols(), 1)] = iMat2.getRowSlice(i);
	}
}

// Construire une matrice en concat�nant les rang�es de deux matrices de m�me largeur.
MatrixConcatRows::MatrixConcatRows(const Matrix& iMat1, const Matrix& iMat2) : Matrix(iMat1.rows() + iMat2.rows(), iMat1.cols()) {
	// v�rifier la compatibilit� des matrices
	assert(iMat1.cols() == iMat2.cols());
	// Pour chaque colonne
	for (size_t j = 0; j < cols(); ++j) {
		// colonne j de la premi�re matrice
		mData[slice(j, iMat1.rows(), cols())] = iMat1.getColumnSlice(j);
		// colonne j de la seconde matrice
		mData[slice(j + iMat1.rows(), iMat2.rows(), cols())] = iMat2.getColumnSlice(j);
	}
}

// Ins�rer une matrice dans un flot de sortie.
ostream& operator<<(ostream& oStream, const Matrix& iMat) {
	oStream << iMat.str();
	return oStream;
}