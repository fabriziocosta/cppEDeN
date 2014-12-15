/* -*- mode:c++ -*- */
#ifndef GRAM_MATRIX_MANAGER_H
#define GRAM_MATRIX_MANAGER_H

#include "BaseManager.h"

using namespace std;

/**
 * @class GramMatrixManager
 *
 * @brief Computes the pairwise kernel matrix.
 */
class GramMatrixManager: public BaseManager {
protected:
public:
	GramMatrixManager(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void ComputeAndOutput();
	void ComputeAndOutput(ostream& out);
};

#endif /* GRAM_MATRIX_MANAGER_H */
