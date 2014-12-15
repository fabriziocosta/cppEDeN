#include "LearningCurveManager.h"

LearningCurveManager::LearningCurveManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void LearningCurveManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mSGDSVMManager.Init(apParameters, apData);
}

void LearningCurveManager::Load() {
	mSGDSVMManager.LoadTarget();
	mSGDSVMManager.LoadData();
}

void LearningCurveManager::Exec() {
	Load();
	Main();
}

void LearningCurveManager::Main() {
	LearningCurve();
}

void LearningCurveManager::LearningCurve() {
	cout << SEP << endl;
	cout << "Computing learning curve with " << mpParameters->mNumPoints << " folds." << endl;
	ProgressBar pbt;
	pbt.Count();
	vector<pair<double, double> > prediction_list;
	vector<pair<double, double> > margin_list;

	unsigned size = mpData->Size();
	vector<unsigned> data_id_list;
	//randomly shuffle indices
	MakeShuffledDataIndicesList(data_id_list, size);

	//build train-test split in the learning curve way
	vector<unsigned> test_id_list;
	//test data is the first fold
	for (unsigned i = 0; i < size / mpParameters->mNumPoints; ++i) {
		unsigned id = data_id_list[i];
		test_id_list.push_back(id);
	}
	//sort the indices in order to guarantee sequential file access
	sort(test_id_list.begin(), test_id_list.end());
	//extract target list for testing
	vector<double> test_target_list;
	for (unsigned i = 0; i < test_id_list.size(); i++) {
		unsigned id = test_id_list[i];
		test_target_list.push_back(mpData->mTargetList[id]);
	}
	//training data is built incrementally adding 1/mpParameters->mLearningCurveNumPoints * size instances
	for (unsigned f = 1; f < mpParameters->mNumPoints; f++) { //NOTE: start from 1 as the first fold is used for the test data
		ProgressBar pbf;
		pbf.Count();
		cout << SEP << endl;
		cout << TAB << TAB << "Fold: " << f << " of " << mpParameters->mNumPoints - 1 << endl;
		//generate the training set
		vector<unsigned> train_id_list;
		for (unsigned i = size / mpParameters->mNumPoints; i < size * (f + 1) / mpParameters->mNumPoints; ++i) {
			unsigned id = data_id_list[i];
			train_id_list.push_back(id);
		}
		//sort the indices in order to guarantee sequential file access
		sort(train_id_list.begin(), train_id_list.end());
		//extract target list for training
		vector<double> train_target_list;
		for (unsigned i = 0; i < train_id_list.size(); i++) {
			unsigned id = train_id_list[i];
			train_target_list.push_back(mpData->mTargetList[id]);
		}

		//perform training
		mSGDSVMManager.Train(train_target_list, train_id_list);

		ComputePredictionsOnTest(f, test_id_list, test_target_list);
		ComputePredictionsOnTrain(f, train_id_list, train_target_list);

		cout << "Fold phase concluded in:" << endl;
	}
	cout << "Learning curve concluded in:" << endl;
}

void LearningCurveManager::ComputePredictionsOnTest(unsigned aFoldID, vector<unsigned>& aTestIdList, vector<double>& aTestTargetList) {
	vector<double> fold_margin_list = mSGDSVMManager.Test(aTestIdList);
	assert(fold_margin_list.size() == aTestIdList.size());
	cout << SEP << endl << "Performance on test set:" << endl;
	mSGDSVMManager.OutputPerformanceMeasures(cout, fold_margin_list, aTestTargetList);

	//save results to file
	string ofs_name = "lc_predictions_test_fold_" + stream_cast<string>(aFoldID);
	OutputManager om(ofs_name, mpParameters->mDirectoryPath);

	for (unsigned i = 0; i < fold_margin_list.size(); i++) {
		unsigned id = aTestIdList[i];
		double target = aTestTargetList[i];
		double margin = fold_margin_list[i];
		double prediction = margin > 0 ? 1 : -1;
		om.mOut << id << " " << target << " " << prediction << " " << margin << endl;
	}
	cout << endl << "Instance id, true target, prediction and margin saved in file: " << om.GetFullPathFileName() << endl;
}

void LearningCurveManager::ComputePredictionsOnTrain(unsigned aFoldID, vector<unsigned>& aTrainIdList, vector<double>& aTrainTargetList) {
	vector<double> fold_margin_list = mSGDSVMManager.Test(aTrainIdList);
	assert(fold_margin_list.size() == aTrainIdList.size());
	cout << SEP << endl << "Performance on train set:" << endl;
	mSGDSVMManager.OutputPerformanceMeasures(cout, fold_margin_list, aTrainTargetList);

	//save results to file
	string ofs_name = "lc_predictions_train_fold_" + stream_cast<string>(aFoldID);
	OutputManager om(ofs_name, mpParameters->mDirectoryPath);

	for (unsigned i = 0; i < fold_margin_list.size(); i++) {
		unsigned id = aTrainIdList[i];
		double target = aTrainTargetList[i];
		double margin = fold_margin_list[i];
		double prediction = margin > 0 ? 1 : -1;
		om.mOut << id << " " << target << " " << prediction << " " << margin << endl;
	}
	cout << endl << "Instance id, true target, prediction and margin saved in file: " << om.GetFullPathFileName() << endl;

}
