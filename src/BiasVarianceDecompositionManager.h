/* -*- mode:c++ -*- */
#ifndef BIAS_VARIANCE_DECOMPOSITION_MANAGER_H
#define BIAS_VARIANCE_DECOMPOSITION_MANAGER_H

#include "BaseManager.h"
#include "CrossValidationManager.h"

using namespace std;

class BiasVarianceDecompositionManager: public BaseManager {
protected:
	CrossValidationManager mCrossValidationManager;
public:
	BiasVarianceDecompositionManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void Main();
	void BiasVarianceComputation();
};

#endif /* BIAS_VARIANCE_DECOMPOSITION_MANAGER_H */
