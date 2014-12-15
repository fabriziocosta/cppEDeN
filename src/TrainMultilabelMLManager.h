/* -*- mode:c++ -*- */
#ifndef TRAIN_MULTILABEL_ML_MANAGER_H
#define TRAIN_MULTILABEL_ML_MANAGER_H

#include "SGDSVMManager.h"

using namespace std;

class TrainMultilabelMLManager: public BaseManager{
protected:
	vector<StochasticGradientDescentSupportVectorMachine> mSGDSVMList;
public:
	TrainMultilabelMLManager();
	TrainMultilabelMLManager(Parameters* apParameters, Data* apData);
	void Init();
	void Exec();
};

#endif /* TRAIN_MULTILABEL_ML_MANAGER_H */
