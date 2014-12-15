#include "TargetAlignmentManager.h"

TargetAlignmentManager::TargetAlignmentManager(Parameters* apParameters, Data* apData):BaseManager(apParameters, apData) {
}

void TargetAlignmentManager::Load() {
	mpData->LoadData(true,true,false);
}

void TargetAlignmentManager::Exec() {
	Load();
	Main();
}

void TargetAlignmentManager::Main() {
	cout << SEP << endl << "Target alignment phase" << endl << SEP << endl;
	double ta = 0;
	double ka = 0;
	{
		cout << "Computing target alignment for " << mpData->Size() << " instances." << endl;
		ProgressBar ppb;
		for (unsigned i = 0; i < mpData->Size(); ++i) {
			for (unsigned j = 0; j < mpData->Size(); ++j) {
				if (i != j) {
					double k_ij = mpData->ComputeKernel(i, j);
					double t_ij = mpData->mTargetList[i] * mpData->mTargetList[j];
					ta += k_ij * t_ij;
					ka += k_ij * k_ij;
				}
			}
			ppb.Count();
		}
	}
	double target_alignment = ta / sqrt(ka * (mpData->Size() * mpData->Size() - mpData->Size()));
	cout << "Target alignment= " << target_alignment << endl;
}
