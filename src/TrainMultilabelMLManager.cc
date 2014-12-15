#include "TrainMultilabelMLManager.h"
TrainMultilabelMLManager::TrainMultilabelMLManager() :
		BaseManager(0, 0) {
}

TrainMultilabelMLManager::TrainMultilabelMLManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
}

void TrainMultilabelMLManager::Init() {
	//load target
	mpData->LoadMultilabelTarget();
	//load data
	mpData->LoadData(false,false,false);
	//instantiate a vector of models
	for (unsigned i = 0; i < mpData->MultilabelTargetDimension(); ++i) {
		StochasticGradientDescentSupportVectorMachine sgdsvm(mpParameters, mpData);
		mSGDSVMList.push_back(sgdsvm);
	}
}

void TrainMultilabelMLManager::Exec() {
	//load data and targts
	Init();

	//train and save models
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
	for (unsigned i = 0; i < mpData->MultilabelTargetDimension(); ++i) {
		//extract only 1 and -1 targets for task i-th
		//TODO: have as a flag the possibility to do semi-supervised
		vector<unsigned> id_list;
		vector<double> target_list;
		for (unsigned j = 0; j < mpData->mMultilabelTargetList.size(); ++j) {
			double target = mpData->mMultilabelTargetList[j][i];
			if (target == 1 || target == -1) {
				id_list.push_back(j);
				target_list.push_back(target);
			}
		}
		mSGDSVMList[i].Train(target_list, id_list);
		string suffix = "_" + stream_cast<string>(i);
		mSGDSVMList[i].Save(suffix);
	}
}
