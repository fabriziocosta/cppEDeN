#include "NearestNeighbor.h"

NearestNeighbor::NearestNeighbor(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void NearestNeighbor::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	if (mpData->IsDataLoaded() == false) {
		mpData->LoadData(true, true, false);
	}

	if (mpParameters->mUseApproximate) {
		mMinHashEncoder.Init(apParameters, apData);
		mMinHashEncoder.ComputeInverseIndex();
	}

	if (mpParameters->mNoNeighborhoodCache == false)
		CacheReset();

	if (mpParameters->mSharedNeighborhood == true)
		ComputeSharedNeighborhoods();
}

void NearestNeighbor::CacheReset() {
	cout << "... nearest neighbor chace reset ..." << endl;
	mNeighborhoodCache.clear();
	if (mpData->IsDataLoaded() == false)
		throw range_error("ERROR: Cannot clean cache if data is not loaded");
	mNeighborhoodCache.resize(mpData->Size());
}

vector<unsigned> NearestNeighbor::ComputeNeighborhood(unsigned aID) {
	//cache neighborhoods (if opted for)
	vector<unsigned> neighborhood_list;
	if (mpParameters->mNoNeighborhoodCache == true) {
		if (mpParameters->mUseApproximate)
			neighborhood_list = ComputeApproximateNeighborhood(aID);
		else
			neighborhood_list = ComputeTrueNeighborhood(aID);
	} else {
		if (mNeighborhoodCache[aID].size() != 0) {
			neighborhood_list = mNeighborhoodCache[aID];
		} else {
			if (mpParameters->mUseApproximate)
				neighborhood_list = ComputeApproximateNeighborhood(aID);
			else
				neighborhood_list = ComputeTrueNeighborhood(aID);
			mNeighborhoodCache[aID] = neighborhood_list;
		}
	}
	return neighborhood_list;
}

vector<unsigned> NearestNeighbor::ComputeTrueSubNeighborhood(unsigned aID, vector<unsigned>& aApproximateNeighborhoodList) {
//returns a subset of the approximate neighbors sorted by the true kernel function
	vector<unsigned> neighborhood_list;
	vector<pair<double, unsigned> > rank_list;
	for (unsigned i = 0; i < aApproximateNeighborhoodList.size(); ++i) {
		unsigned id_neighbor = aApproximateNeighborhoodList[i];
		double k = mpData->ComputeKernel(aID, id_neighbor);
		rank_list.push_back(make_pair(-k, id_neighbor));
	}
	unsigned effective_size = min((unsigned) rank_list.size(), mpParameters->mNumNearestNeighbors);
	partial_sort(rank_list.begin(), rank_list.begin() + effective_size, rank_list.end());

	for (unsigned j = 0; j < effective_size; j++) {
		neighborhood_list.push_back(rank_list[j].second);
	}
	return neighborhood_list;
}

vector<unsigned> NearestNeighbor::ComputeTrueNeighborhood(unsigned aID) {
	vector<pair<double, unsigned> > rank_list(mpData->mColIndexList.size());
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
	for (unsigned j = 0; j < mpData->mColIndexList.size(); ++j) {
		unsigned jj = mpData->mColIndexList[j];
		double k = mpData->ComputeKernel(aID, jj);
		rank_list[j] = make_pair(-k, jj); //NOTE: use -k to sort in descending order of similarity
	}
	unsigned effective_size = min((unsigned) rank_list.size(), mpParameters->mNumNearestNeighbors);
	partial_sort(rank_list.begin(), rank_list.begin() + effective_size, rank_list.end());
	vector<unsigned> neighborhood_list;
	for (unsigned j = 0; j < effective_size; j++)
		neighborhood_list.push_back(rank_list[j].second);
	return neighborhood_list;
}

vector<unsigned> NearestNeighbor::ComputeApproximateNeighborhood(unsigned aID) {
	vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(aID);
	vector<unsigned> approximate_neighborhood = mMinHashEncoder.ComputeApproximateNeighborhood(signature);
	vector<unsigned> approximate_true_neighborhood = ComputeTrueSubNeighborhood(aID, approximate_neighborhood);
	return approximate_true_neighborhood;
}

/**
 Computes the fraction of neighbors that are common between instance I and J
 */
double NearestNeighbor::ComputeSharedNeighborhoodSimilarity(unsigned aI, unsigned aJ) {
	vector<unsigned> neighborhood_i = ComputeNeighborhood(aI);
	vector<unsigned> neighborhood_j = ComputeNeighborhood(aJ);
	unsigned intersection_size = ComputeNeighborhoodIntersection(aI, aJ);
	double shared_neighborhood_value = (double) intersection_size / sqrt((double) neighborhood_i.size() * (double) neighborhood_j.size());
	return shared_neighborhood_value;
}

vector<unsigned> NearestNeighbor::ComputeSharedNeighborhood(unsigned aID) {
	vector<unsigned> shared_neighborhood;
	vector<unsigned> neighborhood = ComputeNeighborhood(aID);
	//for each element in the neighborhood consider their neighborhood and check if there is aID, if yes then the element can stay in the neighborhood otherwise not
	for (unsigned i = 0; i < neighborhood.size(); ++i) {
		unsigned nn_id = neighborhood[i];
		vector<unsigned> nn_neighborhood = ComputeNeighborhood(nn_id);
		bool is_present = false;
		for (unsigned j = 0; j < nn_neighborhood.size() && is_present == false; j++) {
			if (nn_neighborhood[j] == aID)
				is_present = true;
		}
		if (is_present == true)
			shared_neighborhood.push_back(nn_id);
	}
	return shared_neighborhood;
}

unsigned NearestNeighbor::ComputeNeighborhoodIntersection(unsigned aI, unsigned aJ) {
	vector<unsigned> neighborhood_i = ComputeNeighborhood(aI);
	set<unsigned> neighborhood_i_set;
	neighborhood_i_set.insert(neighborhood_i.begin(), neighborhood_i.end());
	vector<unsigned> neighborhood_j = ComputeNeighborhood(aJ);
	set<unsigned> neighborhood_j_set;
	neighborhood_j_set.insert(neighborhood_j.begin(), neighborhood_j.end());

	set<unsigned> intersection;
	set_intersection(neighborhood_i.begin(), neighborhood_i.end(), neighborhood_j_set.begin(), neighborhood_j_set.end(), inserter(intersection, intersection.begin()));
	return intersection.size();
}

void NearestNeighbor::ComputeSharedNeighborhoods() {
	cout << "Computing shared neighborhoods" << endl;
	ProgressBar pb;
	vector<vector<unsigned> > new_neighborhood_cache(mpData->Size());
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		vector<unsigned> neighborhood = ComputeSharedNeighborhood(i);
		new_neighborhood_cache[i] = neighborhood;
		pb.Count();
	}
	mNeighborhoodCache = new_neighborhood_cache;
}

vector<unsigned> NearestNeighbor::ComputeNeighborhood(SVector& aX) {
	vector<unsigned> neighborhood_list;
	if (mpParameters->mUseApproximate)
		neighborhood_list = ComputeApproximateNeighborhood(aX);
	else
		neighborhood_list = ComputeTrueNeighborhood(aX);
	return neighborhood_list;
}

vector<unsigned> NearestNeighbor::ComputeApproximateNeighborhood(SVector& aX) {
	vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(aX);
	vector<unsigned> approximate_neighborhood = mMinHashEncoder.ComputeApproximateNeighborhood(signature);
	vector<unsigned> approximate_true_neighborhood = ComputeTrueSubNeighborhood(aX, approximate_neighborhood);
	return approximate_true_neighborhood;
}

vector<unsigned> NearestNeighbor::ComputeTrueSubNeighborhood(SVector& aX, vector<unsigned>& aApproximateNeighborhoodList) {
//returns a subset of the approximate neighbors sorted by the true kernel function
	vector<unsigned> neighborhood_list;
	vector<pair<double, unsigned> > rank_list;
	for (unsigned i = 0; i < aApproximateNeighborhoodList.size(); ++i) {
		unsigned id_neighbor = aApproximateNeighborhoodList[i];
		SVector& z = mpData->mVectorList[id_neighbor];
		double k = mpData->ComputeKernel(aX, z);
		rank_list.push_back(make_pair(-k, id_neighbor));
	}
	unsigned effective_size = min((unsigned) rank_list.size(), mpParameters->mNumNearestNeighbors);
	partial_sort(rank_list.begin(), rank_list.begin() + effective_size, rank_list.end());

	for (unsigned j = 0; j < effective_size; j++) {
		neighborhood_list.push_back(rank_list[j].second);
	}
	return neighborhood_list;
}

vector<unsigned> NearestNeighbor::ComputeTrueNeighborhood(SVector& aX) {
	vector<pair<double, unsigned> > rank_list(mpData->mColIndexList.size());
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
	for (unsigned j = 0; j < mpData->mColIndexList.size(); ++j) {
		unsigned jj = mpData->mColIndexList[j];
		SVector& z = mpData->mVectorList[jj];
		double k = mpData->ComputeKernel(aX, z);
		rank_list[j] = make_pair(-k, jj); //NOTE: use -k to sort in descending order of similarity
	}
	unsigned effective_size = min((unsigned) rank_list.size(), mpParameters->mNumNearestNeighbors);
	partial_sort(rank_list.begin(), rank_list.begin() + effective_size, rank_list.end());
	vector<unsigned> neighborhood_list;
	for (unsigned j = 0; j < effective_size; j++)
		neighborhood_list.push_back(rank_list[j].second);
	return neighborhood_list;
}
