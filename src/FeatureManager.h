/* -*- mode:c++ -*- */
#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include "BaseManager.h"
#include "MinHashEncoder.h"
#include "SGDSVM.h"

using namespace std;

class FeatureManager: public BaseManager {
protected:
	MinHashEncoder mMinHashEncoder;

public:
	FeatureManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Exec();
	void InputOutputManager();
	void Main(istream& fin, ostream& ofs);
	void FeatureOutputManager();
	void Main(ostream& out);
	void ComputeSmoothedFeatureRepresentation(SVector& x);
	void ReHash(SVector& x, unsigned aReHashCode);
};

//------------------------------------------------------------------------------------------------------------------------
class FeaturePartManager: public BaseManager {
protected:
public:
	FeaturePartManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Exec();
	void InputOutputManager();
	void Main(istream& fin, ostream& ofs);
};

//------------------------------------------------------------------------------------------------------------------------
class FeatureScaledManager: public BaseManager {
protected:
	StochasticGradientDescentSupportVectorMachine mSGDSVM;

public:
	FeatureScaledManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void LoadModel();
	void Exec();
	void InputOutputManager();
	void Main(istream& fin, ostream& ofs);
};

#endif /* FEATURE_MANAGER_H */
