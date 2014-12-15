/* -*- mode:c++ -*- */
#ifndef SGDSVMMANAGER_H
#define SGDSVMMANAGER_H

#include "BaseManager.h"
#include "SGDSVM.h"

using namespace std;

class StochasticGradientDescentSupportVectorMachineManager: public BaseManager{
protected:
	StochasticGradientDescentSupportVectorMachine mSGDSVM;
public:
	StochasticGradientDescentSupportVectorMachineManager();
	StochasticGradientDescentSupportVectorMachineManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Exec();
	void LoadTarget();
	void LoadData();
	void LoadModel();
	void OutputPerformanceMeasures(ostream& out, const vector<double>& aMarginList, const vector<double>& aTargetList);
	double ComputeBalancedFMeasure(const vector<double>& aMarginList, const vector<double>& aTargetList);
	double Predict(const SVector& x);
	vector<double> Test(vector<unsigned> aTestSetIDList);
	double Test(GraphClass& aG);
	vector<double> TestPart(GraphClass& aG);
	void Train();
	void SaveModel(string aLocalSuffix = string());
	void Train(vector<double> aTargetList, vector<unsigned> aTrainSetIDList);
};

#endif /* SGDSVMMANAGER_H */
