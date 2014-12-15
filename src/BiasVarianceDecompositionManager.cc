#include "BiasVarianceDecompositionManager.h"

BiasVarianceDecompositionManager::BiasVarianceDecompositionManager(Parameters* apParameters, Data* apData) :BaseManager(apParameters, apData){
	Init(apParameters, apData);
}

void BiasVarianceDecompositionManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mCrossValidationManager.Init(apParameters, apData);
}

void BiasVarianceDecompositionManager::Load() {
	mCrossValidationManager.Load();
}

void BiasVarianceDecompositionManager::Exec() {
	Load();
	Main();
}

void BiasVarianceDecompositionManager::Main() {
	BiasVarianceComputation();
}

void BiasVarianceDecompositionManager::BiasVarianceComputation() {
	OutputManager om( "confidence", mpParameters->mDirectoryPath);

	ProgressBar pbt;
	pbt.Count();
	vector<map<unsigned, vector<double> > > multi_result_list;
	for (unsigned i = 0; i < mpParameters->mNumPoints; ++i) {
		mpParameters->mRandomSeed += i;
		map<unsigned, vector<double> > result_list;
		mCrossValidationManager.CrossValidation(result_list);
		multi_result_list.push_back(result_list);
	}

	unsigned size = mpData->Size();
	vector<double> accuracy_list(size);
	vector<VectorClass> margin_list(size);
	for (unsigned r = 0; r < mpParameters->mNumPoints; ++r) {
		for (unsigned i = 0; i < size; ++i) {
			double target = multi_result_list[r][i][0];
			//double average_prediction=margin_list[i].Mean();
			double prediction = multi_result_list[r][i][1];
			double margin = multi_result_list[r][i][2];
			margin_list[i].PushBack(margin);
			if (prediction == target)
				accuracy_list[i] += 1 / (double) mpParameters->mNumPoints;
		}
	}

	//squared bias and variance computation
	vector<double> squared_bias_list;
	vector<double> variance_list;
	for (unsigned i = 0; i < size; ++i) {
		double squared_bias = 0;
		double variance = 0;
		double average_prediction = margin_list[i].Mean();
		for (unsigned r = 0; r < mpParameters->mNumPoints; ++r) {
			double target = multi_result_list[r][i][0];
			squared_bias += (target - average_prediction) * (target - average_prediction);
			double prediction = multi_result_list[r][i][1];
			variance += (average_prediction - prediction) * (average_prediction - prediction);
		}
		squared_bias_list.push_back(squared_bias);
		variance_list.push_back(variance);
	}

	//write results to file
	for (unsigned i = 0; i < size; ++i) {
		double target = multi_result_list[0][i][0];
		double mean_margin = margin_list[i].Mean();
		double std_margin = margin_list[i].StandardDeviation();
		double squared_bias = squared_bias_list[i];
		double variance = variance_list[i];
		om.mOut << target << " " << accuracy_list[i] << " " << mean_margin << " " << std_margin << " " << squared_bias << " " << variance << endl;
	}
	cout << "Target, average accuracy, average margin, margin standard deviation, squared bias, variance saved in file " << om.GetFullPathFileName() << endl;
}
