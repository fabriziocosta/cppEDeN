#include "SGDSVMManager.h"

StochasticGradientDescentSupportVectorMachineManager::StochasticGradientDescentSupportVectorMachineManager():BaseManager(0,0){
}

StochasticGradientDescentSupportVectorMachineManager::StochasticGradientDescentSupportVectorMachineManager(Parameters* apParameters, Data* apData) :BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void StochasticGradientDescentSupportVectorMachineManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mSGDSVM.Init(apParameters, apData);
}

void StochasticGradientDescentSupportVectorMachineManager::Exec() {
	LoadTarget();
	LoadData();
	Train();
}

void StochasticGradientDescentSupportVectorMachineManager::LoadTarget() {
	mpData->LoadTarget();
}

void StochasticGradientDescentSupportVectorMachineManager::LoadData() {
	mpData->LoadData(false,false,false);
}

void StochasticGradientDescentSupportVectorMachineManager::LoadModel() {
	string filename = mpParameters->mModelFileName + mpParameters->mSuffix;
	ifstream ifs;
	ifs.open(filename.c_str());
	if (!ifs)
		throw range_error("ERROR StochasticGradientDescentSupportVectorMachineManager::LoadModel: Cannot open file:" + filename);
	if (!mpParameters->mMinimalOutput)
		cout << endl << "Loading model file: " << filename << endl;
	mSGDSVM.Load(ifs);
	if (!mpParameters->mMinimalOutput) {
		mSGDSVM.OutputModelInfo();
		cout << endl;
	}
}

void StochasticGradientDescentSupportVectorMachineManager::SaveModel(string aLocalSuffix) {
	string file_name=mpParameters->mModelFileName+aLocalSuffix;
	OutputManager om(file_name , mpParameters->mDirectoryPath);
	mSGDSVM.Save(om.mOut);
	if (!mpParameters->mMinimalOutput)
			cout << endl << "Model saved in file: "<< om.GetFullPathFileName() << endl;
}

void StochasticGradientDescentSupportVectorMachineManager::OutputPerformanceMeasures(ostream& out, const vector<double>& aMarginList, const vector<double>& aTargetList) {
	mSGDSVM.OutputPerformanceMeasures(out, aMarginList, aTargetList);
}

double StochasticGradientDescentSupportVectorMachineManager::ComputeBalancedFMeasure(const vector<double>& aMarginList, const vector<double>& aTargetList) {
	return mSGDSVM.ComputeBalancedFMeasure(aMarginList, aTargetList);
}

double StochasticGradientDescentSupportVectorMachineManager::Predict(const SVector& x) {
	return mSGDSVM.Predict(x);
}

vector<double> StochasticGradientDescentSupportVectorMachineManager::Test(vector<unsigned> aTestSetIDList) {
	vector<unsigned> testset_id_list = aTestSetIDList;
	vector<double> margin_list = mSGDSVM.Test(testset_id_list);
	return margin_list;
}

double StochasticGradientDescentSupportVectorMachineManager::Test(GraphClass& aG) {
	SVector x(pow(2, mpParameters->mHashBitSize));
	;
	mpData->mKernel.GenerateFeatureVector(aG, x);
	return Predict(x);
}

vector<double> StochasticGradientDescentSupportVectorMachineManager::TestPart(GraphClass& aG) {
	vector<double> margin_list;
	vector<SVector> graph_vertex_vector_list;
	mpData->mKernel.GenerateVertexFeatureVector(aG, graph_vertex_vector_list);
	//for each vertex, compute margin
	unsigned size = mpParameters->mGraphType == "DIRECTED" ? graph_vertex_vector_list.size() / 2 : graph_vertex_vector_list.size();
	for (unsigned vertex_id = 0; vertex_id < size; ++vertex_id) {
		double margin = Predict(graph_vertex_vector_list[vertex_id]);
		if (mpParameters->mGraphType == "DIRECTED")
			margin += Predict(graph_vertex_vector_list[vertex_id + size]);
		margin_list.push_back(margin);
	}
	return margin_list;
}

void StochasticGradientDescentSupportVectorMachineManager::Train() {
	if (!mpParameters->mMinimalOutput)
		cout << endl << SEP << endl << "Train phase" << endl << SEP << endl;
	ProgressBar pb;
	pb.Count();

	vector<unsigned> train_id_list;
	for (unsigned i = 0; i < mpData->mTargetList.size(); ++i)
		train_id_list.push_back(i);
	Train(mpData->mTargetList, train_id_list);
	SaveModel();

	if (!mpParameters->mMinimalOutput)
		cout << endl << "Train phase completed:";
}

void StochasticGradientDescentSupportVectorMachineManager::Train(vector<double> aTargetList, vector<unsigned> aTrainSetIDList) {
	assert(aTargetList.size() == aTrainSetIDList.size());
	//wrapper for semi-supervised case: self-training
	//assume unsupervised material receives 0 target
	//filter the unsupervised material and put it in separate lists
	//iterate:
	//train on supervised and test on unsupervised
	//replace 0 target with prediction
	vector<double> target_list(aTargetList);
	vector<unsigned> train_supervised_id_list;
	vector<double> train_supervised_target_list;
	vector<unsigned> train_unsupervised_id_list;
	for (unsigned i = 0; i < target_list.size(); ++i) {
		unsigned id = aTrainSetIDList[i];
		double target = target_list[i];
		if (target != 0) {
			train_supervised_id_list.push_back(id);
			train_supervised_target_list.push_back(target);
		} else {
			train_unsupervised_id_list.push_back(id);
		}
	}
	if (train_unsupervised_id_list.size() > 0) {
		//if unsupervised material is present then
		//train on supervised material
		if (!mpParameters->mMinimalOutput) {
			cout << endl << "Semisupervised training on " << target_list.size() << " instances" << endl;
			cout << TAB << "supervised instances: " << train_supervised_id_list.size() << " (" << 100 * train_supervised_id_list.size() / (double) target_list.size() << "%)" << endl;
			cout << TAB << "unsupervised instances: " << train_unsupervised_id_list.size() << " (" << 100 * train_unsupervised_id_list.size() / (double) target_list.size() << "%)" << endl;
		}
		vector<double> margin_list = mSGDSVM.Train(train_supervised_target_list, train_supervised_id_list);

		//repeat for a predefined number of iteration
		for (unsigned iteration = 0; iteration < mpParameters->mSemiSupervisedNumIterations; iteration++) {
			//test on unsupervised material
			if (!mpParameters->mMinimalOutput) {
				cout << endl << TAB << "Iteration " << iteration + 1 << "/" << mpParameters->mSemiSupervisedNumIterations << endl;
				cout << "Testing on unsupervised instances: " << train_unsupervised_id_list.size() << endl;
			}
			vector<double> margin_list = Test(train_unsupervised_id_list);
			//find high and low threshold for margin (i.e. high confidence predictions)
			vector<double> sorted_margin_list;
			vector<double> sorted_positive_margin_list;
			vector<double> sorted_negative_margin_list;
			for (unsigned i = 0; i < margin_list.size(); ++i) {
				sorted_margin_list.push_back(margin_list[i]);
				if (margin_list[i] > 0)
					sorted_positive_margin_list.push_back(margin_list[i]);
				else
					sorted_negative_margin_list.push_back(margin_list[i]);
			}

			unsigned high_threshold_id, low_threshold_id;
			double high_threshold, low_threshold;
			if (!mpParameters->mMinimalOutput)
				cout << "Predicted class distribution:  +:" << sorted_positive_margin_list.size() << " (" << 100 * sorted_positive_margin_list.size() / (double) train_unsupervised_id_list.size() << " %)" << " -:" << sorted_negative_margin_list.size() << " (" << 100 * sorted_negative_margin_list.size() / (double) train_unsupervised_id_list.size() << " %)" << endl;
			if (sorted_positive_margin_list.size() == 0 || sorted_negative_margin_list.size() == 0) {
				if (!mpParameters->mMinimalOutput)
					cout << "Warning: margins are one sided. Proceeding to use margin rank irrespectively of margin sign. Retaining " << mpParameters->mSemiSupervisedThreshold / 2 * 100 << "% of most reliable predictions" << endl;
				sort(sorted_margin_list.begin(), sorted_margin_list.end());
				high_threshold_id = (sorted_margin_list.size() - 1) * (1 - mpParameters->mSemiSupervisedThreshold / 2);
				low_threshold_id = (sorted_margin_list.size() - 1) * (mpParameters->mSemiSupervisedThreshold / 2);
				high_threshold = sorted_margin_list.size() > 0 ? sorted_margin_list[high_threshold_id] : sorted_margin_list[sorted_margin_list.size() - 1];
				low_threshold = sorted_margin_list.size() > 0 ? sorted_margin_list[low_threshold_id] : sorted_margin_list[0];
			} else {
				sort(sorted_positive_margin_list.begin(), sorted_positive_margin_list.end());
				sort(sorted_negative_margin_list.begin(), sorted_negative_margin_list.end());
				high_threshold_id = (sorted_positive_margin_list.size() - 1) * (1 - mpParameters->mSemiSupervisedThreshold);
				low_threshold_id = (sorted_negative_margin_list.size() - 1) * (mpParameters->mSemiSupervisedThreshold);
				high_threshold = sorted_positive_margin_list.size() > 0 ? sorted_positive_margin_list[high_threshold_id] : sorted_negative_margin_list[sorted_negative_margin_list.size() - 1];
				low_threshold = sorted_negative_margin_list.size() > 0 ? sorted_negative_margin_list[low_threshold_id] : sorted_positive_margin_list[0];
			}
			if (!mpParameters->mMinimalOutput)
				cout << "Low score threshold:" << low_threshold << " High score threshold:" << high_threshold << endl;
			//replace 0 target with predicted target only for high confidence predictions
			map<unsigned, double> semi_supervise_augmented_target_map;
			for (unsigned i = 0; i < target_list.size(); ++i) //copy target for supervised instances
				if (target_list[i] != 0) {
					unsigned id = aTrainSetIDList[i];
					semi_supervise_augmented_target_map[id] = target_list[i];
				}
			unsigned counter_p = 0;
			unsigned counter_n = 0;
			for (unsigned i = 0; i < train_unsupervised_id_list.size(); ++i) { //copy prediction for unsupervised instances
				unsigned id = train_unsupervised_id_list[i];
				double margin = margin_list[i];
				double predicted_target;
				bool margin_test = false;
				if (mpParameters->mSemiSupervisedInduceOnlyPositive) {
					margin_test = (margin >= high_threshold);
					predicted_target = 1;
				} else if (mpParameters->mSemiSupervisedInduceOnlyNegative) {
					margin_test = (margin <= low_threshold);
					predicted_target = -1;
				} else {
					margin_test = (margin <= low_threshold || margin >= high_threshold);
					predicted_target = margin <= low_threshold ? -1 : 1;
				}
				if (margin_test == true) {
					assert(semi_supervise_augmented_target_map.count(id) == 0);
					semi_supervise_augmented_target_map[id] = predicted_target;
					if (predicted_target > 0)
						counter_p++;
					else
						counter_n++;
				}
			}
			if (mpParameters->mSemiSupervisedInduceOnlyPositive)
				if (!mpParameters->mMinimalOutput)
					cout << "Adding only predicted positives" << endl;
			if (mpParameters->mSemiSupervisedInduceOnlyNegative)
				if (!mpParameters->mMinimalOutput)
					cout << "Adding only predicted negatives" << endl;
			if (!mpParameters->mMinimalOutput)
				cout << "Added +:" << counter_p << " and -:" << counter_n << " instances from unsupervised set to training set of size " << train_supervised_id_list.size() << endl;
			//compose indices vectors for training instances and target
			vector<unsigned> train_semi_supervise_augmented_id_list;
			vector<double> train_semi_supervise_augmented_target_list;
			for (map<unsigned, double>::iterator it = semi_supervise_augmented_target_map.begin(); it != semi_supervise_augmented_target_map.end(); ++it) {
				unsigned id = it->first;
				double target = it->second;
				train_semi_supervise_augmented_id_list.push_back(id);
				train_semi_supervise_augmented_target_list.push_back(target);
			}
			//retrain
			margin_list = mSGDSVM.Train(train_semi_supervise_augmented_target_list, train_semi_supervise_augmented_id_list);
		}

	} else { //if no unsupervised material is present then train directly
		vector<double> margin_list = mSGDSVM.Train(target_list, aTrainSetIDList);
	}
}
