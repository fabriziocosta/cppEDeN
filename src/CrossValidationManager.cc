#include "CrossValidationManager.h"

CrossValidationManager::CrossValidationManager() :
		BaseManager(0, 0) {
}

CrossValidationManager::CrossValidationManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void CrossValidationManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mSGDSVMManager.Init(apParameters, apData);
}

void CrossValidationManager::Load() {
	mSGDSVMManager.LoadTarget();
	mSGDSVMManager.LoadData();
}

void CrossValidationManager::Exec() {
	Load();
	Main();
}

void CrossValidationManager::Main() {
	map<unsigned, vector<double> > results;
	CrossValidation(results);
}

double CrossValidationManager::GetBalancedFMeasure() {
	map<unsigned, vector<double> > results;
	return CrossValidation(results);
}

//map<unsigned, vector<double> >& oTestResultMap has the following semantic: map<instance_id, <target,prediction,margin> >
double CrossValidationManager::CrossValidation(map<unsigned, vector<double> >& oTestResultMap) {
	oTestResultMap.clear();
	//since CrossValidation can be called in a procedure, we reset the seed to the writable mpParameters->mRandomSeed
	srand(mpParameters->mRandomSeed);
	ProgressBar pbt;
	pbt.Count();

	//randomly shuffle indices of data instances
	vector<unsigned> data_id_list;
	MakeShuffledDataIndicesList(data_id_list,mpData->Size());

	//loop to build train-test split in the cross validation way
	for (unsigned f = 0; f < mpParameters->mCrossValidationNumFolds; f++) {
		ProgressBar pbf;
		pbf.Count();
		if (!mpParameters->mMinimalOutput)
			cout << SEP << endl << TAB << TAB << "Fold: " << f + 1 << " of " << mpParameters->mCrossValidationNumFolds << endl;
		vector<unsigned> train_id_list;
		vector<unsigned> test_id_list;
		map<unsigned, unsigned> class_counter_map;
		for (unsigned i = 0; i < mpData->Size(); ++i) {
			unsigned id = data_id_list[i];
			double target = mpData->mTargetList[id];
			if (class_counter_map.count(target) == 0)
				class_counter_map[target] = 1;
			else
				class_counter_map[target]++;
			if (target != 0 && class_counter_map[target] % mpParameters->mCrossValidationNumFolds == f) //NOTE: exclude un supervised material from test element list
				test_id_list.push_back(id);
			else
				train_id_list.push_back(id);
		}

		//extract target list for training
		vector<double> train_target_list;
		for (unsigned i = 0; i < train_id_list.size(); i++) {
			unsigned id = train_id_list[i];
			train_target_list.push_back(mpData->mTargetList[id]);
		}
		//extract target list for testing
		vector<double> test_target_list;
		for (unsigned i = 0; i < test_id_list.size(); i++) {
			unsigned id = test_id_list[i];
			test_target_list.push_back(mpData->mTargetList[id]);
		}

		//perform training
		mSGDSVMManager.Train(train_target_list, train_id_list);
		string model_filename_prefix = "_" + stream_cast<string>(f + 1);
		mSGDSVMManager.SaveModel(model_filename_prefix);

		//perform testing
		vector<double> fold_margin_list = mSGDSVMManager.Test(test_id_list);
		//add to test_result_map
		assert(fold_margin_list.size() == test_id_list.size());
		for (unsigned i = 0; i < fold_margin_list.size(); i++) {
			unsigned id = test_id_list[i];
			//pack all the result fields sequentially in a vector
			double target = test_target_list[i];
			double margin = fold_margin_list[i];
			double prediction = margin > 0 ? 1 : -1;
			vector<double> res;
			res.push_back(target);
			res.push_back(prediction);
			res.push_back(margin);
			//memoize the result vector with the test id
			oTestResultMap[id] = res;
		}
		if (!mpParameters->mMinimalOutput)
			cout << "Fold phase concluded in:" << endl;
	}
	OutputPredictions(oTestResultMap);
	return ComputePerformance(oTestResultMap);
}

void CrossValidationManager::OutputPredictions(map<unsigned, vector<double> >& oTestResultMap) {
	OutputManager om("cv_predictions", mpParameters->mDirectoryPath);
	//for all test ids read in order
	for (map<unsigned, vector<double> >::iterator it = oTestResultMap.begin(); it != oTestResultMap.end(); ++it) {
		unsigned id = it->first;
		//unpack the result fields from the result vector memoized with the test id
		double target = it->second[0];
		double prediction = it->second[1];
		double margin = it->second[2];
		om.mOut << id << " " << target << " " << prediction << " " << margin << endl;
	}
	if (!mpParameters->mMinimalOutput) {
		cout << endl << "Instance id, true target, prediction and margin saved in file: " << om.GetFullPathFileName() << endl;
	}
}

double CrossValidationManager::ComputePerformance(map<unsigned, vector<double> >& oTestResultMap) {
	vector<double> cv_prediction_list;
	vector<double> cv_target_list;

	//for all test ids read in order
	for (map<unsigned, vector<double> >::iterator it = oTestResultMap.begin(); it != oTestResultMap.end(); ++it) {
		//unpack the result fields from the result vector memoized with the test id
		double target = it->second[0];
		double prediction = it->second[1];
		cv_prediction_list.push_back(prediction);
		cv_target_list.push_back(target);
	}
	if (!mpParameters->mMinimalOutput) {
		cout << SEP << endl << "Performance on data set in cross validation:" << endl;
		mSGDSVMManager.OutputPerformanceMeasures(cout, cv_prediction_list, cv_target_list);
		cout << "Crossvalidation concluded in:" << endl;
	}
	double bfmeasure = mSGDSVMManager.ComputeBalancedFMeasure(cv_prediction_list, cv_target_list);
	return bfmeasure;
}
