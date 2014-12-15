/* -*- mode:c++ -*- */
#ifndef CROSS_VALIDATION_MANAGER_H
#define CROSS_VALIDATION_MANAGER_H

#include "BaseManager.h"
#include "SGDSVMManager.h"

using namespace std;

class CrossValidationManager: public BaseManager {
protected:
	StochasticGradientDescentSupportVectorMachineManager mSGDSVMManager;
public:
	CrossValidationManager();
	CrossValidationManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void Main();
	double GetBalancedFMeasure();
	void ShuffleDataIndices(vector<unsigned>& oDataIdList);
	double CrossValidation(map<unsigned, vector<double> >& oTestResultMap);
	void OutputPredictions(map<unsigned, vector<double> >& oTestResultMap);
	double ComputePerformance(map<unsigned, vector<double> >& oTestResultMap);
};

#endif /* CROSS_VALIDATION_MANAGER_H */
