#include "SemiSupervisedManager.h"

SemiSupervisedManager::SemiSupervisedManager(Parameters* apParameters, Data* apData):BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void SemiSupervisedManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mMinHashEncoder.Init(apParameters, apData);
}

void SemiSupervisedManager::Load() {
	mpData->LoadData(true,true,false);
}

void SemiSupervisedManager::Exec() {
	Load();
	SSOutputManager();
}

void SemiSupervisedManager::SSOutputManager() {
	cout << SEP << endl << "Semisupervised phase" << endl << SEP << endl;
	ProgressBar pb;
	pb.Count();

	//output
	OutputManager om("semisupervised", mpParameters->mDirectoryPath);

	Main(om.mOut);

	cout << "Target value list saved in file " << om.GetFullPathFileName() << endl;
}

void SemiSupervisedManager::Main(ostream& out) {
	if (mpParameters->mUseApproximate == true)
		mMinHashEncoder.ComputeInverseIndex();

	vector < SVector > neighbor_matrix;
	{
		cout << "Extracting nearest neighbor information" << endl;
		ProgressBar ppb;
		for (unsigned ii = 0; ii < mpData->Size(); ++ii) {
			vector < pair<double, unsigned> > neighbor_list;

			if (mpParameters->mUseApproximate == false) { //do full comparison of each row instance vs. each column instance and sort to find nearest
				vector < pair<double, unsigned> > init_neighbor_list(mpData->Size());
				neighbor_list = init_neighbor_list;
				for (unsigned jj = 0; jj < mpData->Size(); ++jj) {
					double k_ii_jj = mpData->ComputeKernel(ii, jj);
					neighbor_list[jj] = make_pair(-k_ii_jj, jj); //NOTE: use -k to sort in descending order of similarity
				}
				unsigned effective_size = min((unsigned) neighbor_list.size(), mpParameters->mNumNearestNeighbors);
				partial_sort(neighbor_list.begin(), neighbor_list.begin() + effective_size, neighbor_list.end());
			} else { // extract approximate neighbors
				vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(ii);
				vector<unsigned> approximate_neighborhood = mMinHashEncoder.ComputeApproximateNeighborhood(signature);
				vector < pair<double, unsigned> > rank_list;
				for (unsigned j = 0; j < approximate_neighborhood.size(); ++j) {
					unsigned jj = approximate_neighborhood[j];
					double k = mpData->ComputeKernel(jj, ii);
					rank_list.push_back(make_pair(-k, jj));
				}
				unsigned effective_size = min((unsigned) rank_list.size(), mpParameters->mNumNearestNeighbors);
				partial_sort(rank_list.begin(), rank_list.begin() + effective_size, rank_list.end());
				for (unsigned j = 0; j < effective_size; j++)
					neighbor_list.push_back(rank_list[j]);
			}
			ppb.Count();

			unsigned effective_size = min((unsigned) neighbor_list.size(), mpParameters->mNumNearestNeighbors);

			SVector effective_neighbor_list(pow(2,mpParameters->mHashBitSize));
			double sum = 0;
			for (unsigned t = 0; t < effective_size; ++t) {
				double neighbor_kernel = -neighbor_list[t].first; //NOTE:revert to positive kernel
				unsigned neighbor_id = neighbor_list[t].second;
				if (neighbor_id != ii) { //avoid self loops
					effective_neighbor_list.coeffRef((int) neighbor_id) = neighbor_kernel;
					sum += neighbor_kernel;
				}
			}
			//normalize
			effective_neighbor_list /= sum;
			neighbor_matrix.push_back(effective_neighbor_list);
		}
	}

	//iterate activation spreading
	FVector y(mpData->mTargetList.size());
	//init y with target
	for (unsigned i = 0; i < mpData->mTargetList.size(); ++i)
		y(i) = mpData->mTargetList[i];
	//init f
	FVector f(mpData->mTargetList.size());
	f = y;
	//iterate
	{
		cout << "Spreading phase." << endl;
		ProgressBar ppb(1);
		for (unsigned itera = 0; itera < mpParameters->mSemiSupervisedNumIterations; ++itera) {
			ppb.Count();
			//spread
			FVector fprime(mpData->mTargetList.size());
			for (unsigned i = 0; i < neighbor_matrix.size(); ++i) {
				double val = neighbor_matrix[i].dot(f);
				fprime(i) = val;
			}
			f = (fprime * mpParameters->mSemiSupervisedAlpha) + (y * (1 - mpParameters->mSemiSupervisedAlpha));
		}
	}
	//save
	for (int i = 0; i < f.size(); ++i) {
		out << f(i);
		out << endl;
	}
}
