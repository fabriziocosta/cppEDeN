/* -*- mode:c++ -*- */
#ifndef PARAMETERS_OPTIMIZATION_MANAGER_H
#define PARAMETERS_OPTIMIZATION_MANAGER_H

#include "BaseManager.h"
#include "CrossValidationManager.h"

using namespace std;

class ParametersOptimizationManager: public BaseManager {
protected:
	CrossValidationManager mCrossValidationManager;

	double mCurrentBFMeasure;
	double mBestBFMeasure;

	double mLambdaLimit;
	unsigned mEpochsLimit;
	unsigned mRadiusLimit;
	unsigned mDistanceLimit;
	unsigned mTopologicalRegularizationNumNeighborsLimit;
	double mTopologicalRegularizationRateLimit;
	unsigned mSparsificationNumIterationsLimit;
	double mTreeLambdaLimit;
	double mRadiusTwoLimit;

public:
	ParametersOptimizationManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Exec();
	void OutputManager();
	void ParametersOptimization(ostream& ofs);
	void OptimizeTreeLambda();
	void OptimizeRadiusTwo();
	void SetDefaultParameters();
	void OptimizeLambda();
	void OptimizeEpochs();
	void OptimizeDistance();
	void OptimizeRadius();
	void OptimizeTopologicalRegularizationNumNeighbors();
	void OptimizeTopologicalRegularizationRate();
	void OptimizeSparsificationNumIterations();
	void OutputParameters(ostream& out);
};
#endif /* PARAMETERS_OPTIMIZATION_MANAGER_H */
