/* -*- mode:c++ -*- */
#ifndef SGDSVM_H
#define SGDSVM_H

#include "Utility.h"
#include "Parameters.h"
#include "Kernel.h"
#include "Data.h"

using namespace std;


//------------------------------------------------------------------------------------------------------------------------
// Available losses
#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11

// Select loss: NOTE: the selection done in the makefile
//#define LOSS LOGLOSS
//#define LOSS HINGELOSS
//#define LOSS SMOOTHHINGELOSS

// Zero when no bias
// One when bias term
#define BIAS 1

inline
double loss(double z) {
#if LOSS == LOGLOSS
	if (z > 18)
	return exp(-z);
	if (z < -18)
	return -z;
	return log(1+exp(-z));
#elif LOSS == LOGLOSSMARGIN
	if (z > 18)
	return exp(1-z);
	if (z < -18)
	return 1-z;
	return log(1+exp(1-z));
#elif LOSS == SMOOTHHINGELOSS
	if (z < 0)
	return 0.5 - z;
	if (z < 1)
	return 0.5 * (1-z) * (1-z);
	return 0;
#elif LOSS == SQUAREDHINGELOSS
	if (z < 1)
	return 0.5 * (1 - z) * (1 - z);
	return 0;
#elif LOSS == HINGELOSS
	if (z < 1)
		return 1 - z;
	return 0;
#else
	return 0;
#endif
}

inline
double dloss(double z) {
#if LOSS == LOGLOSS
	if (z > 18)
	return exp(-z);
	if (z < -18)
	return 1;
	return 1 / (exp(z) + 1);
#elif LOSS == LOGLOSSMARGIN
	if (z > 18)
	return exp(1-z);
	if (z < -18)
	return 1;
	return 1 / (exp(z-1) + 1);
#elif LOSS == SMOOTHHINGELOSS
	if (z < 0)
	return 1;
	if (z < 1)
	return 1-z;
	return 0;
#elif LOSS == SQUAREDHINGELOSS
	if (z < 1)
	return (1 - z);
	return 0;
#else
	if (z < 1)
		return 1;
	return 0;
#endif
}

/**
 Encapsulates a linear SVM model trainable with stochastic gradient
 descent over graph instances explicitly mapped by the NSPDK kernel
 */
class StochasticGradientDescentSupportVectorMachine {
	/**
	 Data structure to: 1) facilitate rebalancing of dataset by copying
	 multiple times a reference to the instance; and 2) retrieve
	 prediction efficiently by overwriting a reference to the margin
	 list cell element.
	 */
	struct TrainItem {
		int mInstanceID;
		double mTarget;
		double* mpMargin;
		SVector* mpInstance;
	};
public:
	Parameters* mpParameters;
	Data* mpData;
protected:
	double mWScale;
	double mBias;
	SVector mW;

public:
	StochasticGradientDescentSupportVectorMachine();
	StochasticGradientDescentSupportVectorMachine(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Clear();
	void VectorElementwiseProductWithModel(SVector& oX);
	void OutputPerformanceMeasures(ostream& out, const vector<double>& aMarginList, const vector<double>& aTargetList);
	double ComputeBalancedFMeasure(const vector<double>& aMarginList, const vector<double>& aTargetList);
	void Save(ostream& out);
	void Save(string aLocalSuffix = string());
	void Load(istream& in);
	vector<double> Train(vector<double>& aTargetList, vector<unsigned>& aTrainsetIDList);
	void BalanceDataset(vector<unsigned>& aDatasetIDList, vector<double>& aTargetList, vector<double>& oMarginList, vector<SVector*>& aSVDataList, vector<TrainItem>& oDataset);
	void CoreTrain(vector<TrainItem>& aDataset);
	void FeatureTopologicalRegularization(double aGamma);
	vector<double> Test(vector<unsigned>& aTestSetIDList);
	double Predict(const SVector& x);
	void OutputTrainingInfo();
	void OutputModelInfo();
};


#endif /* SGDSVM_H */
