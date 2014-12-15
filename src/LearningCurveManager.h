/* -*- mode:c++ -*- */
#ifndef LEARNING_CURVE_MANAGER_H
#define LEARNING_CURVE_MANAGER_H

#include "BaseManager.h"
#include "SGDSVMManager.h"

using namespace std;

class LearningCurveManager: public BaseManager {
protected:
	StochasticGradientDescentSupportVectorMachineManager mSGDSVMManager;

public:
	LearningCurveManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void Main();
	void LearningCurve();
	void ComputePredictionsOnTest(unsigned aFoldID, vector<unsigned>& aTestIdList, vector<double>& aTestTargetList);
	void ComputePredictionsOnTrain(unsigned aFoldID, vector<unsigned>& aTrainIdList, vector<double>& aTrainTargetList);
};

#endif /* LEARNING_CURVE_MANAGER_H */
