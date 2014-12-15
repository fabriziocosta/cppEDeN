/* -*- mode:c++ -*- */
#ifndef KERNEL_H
#define KERNEL_H

#include "Utility.h"
#include "Parameters.h"
#include "vectors.h"
#include "BaseGraphClass.h"
#include "GraphClass.h"
#include "NSPDK_FeatureGenerator.h"
#include "DDK_FeatureGenerator.h"

using namespace std;

class Kernel {
public:
	NSPDK_FeatureGenerator* mpFeatureGenerator; // use this if USEMULTITHREAD not defined
	vector<NSPDK_FeatureGenerator*> mpFeatureGeneratorList; // use this if USEMULTITHREAD defined
	Parameters* mpParameters;
public:
	void Init(Parameters* apParameters);
	void ParametersSetup();
	void OutputParameters(ostream& out);
	void GenerateFeatureVector(GraphClass& aG, SVector& oX);
	void GenerateVertexFeatureVector(GraphClass& aG, vector<SVector>& oXList);
	double ComputeKernel(GraphClass& aG, GraphClass& aM);
	double ComputeKernel(const SVector& aX, const SVector& aZ);
};

#endif /* KERNEL_H */
