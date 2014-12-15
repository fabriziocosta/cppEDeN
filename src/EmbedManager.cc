#include "EmbedManager.h"
EmbedManager::EmbedManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData), mNearestNeighbor(apParameters, apData) {
	//Init(apParameters, apData);
	mTau = 0;
}

//void EmbedManager::Init(Parameters* apParameters, Data* apData) {
//	BaseManager::Init(apParameters, apData);
//	if (mpParameters->mUseApproximate) {
//		mMinHashEncoder.Init(mpParameters, mpData);
//		mMinHashEncoder.ComputeInverseIndex();
//	}
//}

void EmbedManager::Load() {
	mpData->LoadData(false,false,false);
}

void EmbedManager::Exec() {
	Load();
	Main();
}

void EmbedManager::Main() {
	cout << SEP << endl << "Local Multi Dimensional Scaling phase" << endl << SEP << endl;
	vector<FVector> x_list;
	ComputeLocalMultiDimensionalScaling(x_list);

	SaveEmbedding(x_list);
	SaveDistortion(x_list);
	SaveNeighborhoodList();
}

double EmbedManager::Norm(const FVector& aX) {
	return aX.norm();
}

double EmbedManager::Distance(const FVector& aX, const FVector& aZ) {
	return Norm(aX - aZ);
}

FVector EmbedManager::Versor(const FVector& aX, const FVector& aZ) {
	FVector diff = aX - aZ;
	diff.normalize();
	return diff;
}

void EmbedManager::ComputeLocalMultiDimensionalScaling(vector<FVector>& oXList) {
	const unsigned NUM_STEPS_IN_NEIGHBORHOOD_RANGE = 3;
	const double STEP_SIZE_POWER = 1;
	vector<FVector> best_x_list;
	double best_distortion = 1;
	double distortion = 1;
	unsigned best_neighborhood_size = 0;
	double best_log_counter = 0;
	bool has_succesfully_completed_flag = false;
	cout << "Computing low dimensional layout normalized distortion" << endl;
	unsigned step = (2 * mpParameters->mLMDSNeighborhoodSizeRange / NUM_STEPS_IN_NEIGHBORHOOD_RANGE);
	step = step < 1 ? 1 : step;
	for (unsigned neighborhood_size_modifier = 0; neighborhood_size_modifier <= 2 * mpParameters->mLMDSNeighborhoodSizeRange; neighborhood_size_modifier += step) {
		unsigned effective_neighborhood_size = mpParameters->mLMDSNeighborhoodSize + neighborhood_size_modifier - mpParameters->mLMDSNeighborhoodSizeRange;
		if (effective_neighborhood_size >= 3) {
			InitNeighborhoodList(effective_neighborhood_size, mpParameters->mLMDSNonNeighborhoodSize);
			for (double log_counter = 0; log_counter < mpParameters->mLMDSTauExponentRange; log_counter += STEP_SIZE_POWER) {
				double repulsive_force_tau = mTau * mpParameters->mLMDSTau * pow(10, log_counter);
				for (unsigned random_restart_counter = 0; random_restart_counter < mpParameters->mLMDSNumRandomRestarts; random_restart_counter++) {
					cout << "Neighborhood size: " << effective_neighborhood_size << " [" << mpParameters->mLMDSNeighborhoodSize - mpParameters->mLMDSNeighborhoodSizeRange << ".." << mpParameters->mLMDSNeighborhoodSize + mpParameters->mLMDSNeighborhoodSizeRange << "] " << endl << "Repulsive force: " << repulsive_force_tau << endl << "Restart num " << random_restart_counter + 1 << "/" << mpParameters->mLMDSNumRandomRestarts << " (max num iterations: " << mpParameters->mLMDSNumIterations << ")" << endl;
					//initialization phase: random coordinates, computation of long-short distance tradeoff
					vector<FVector> current_x_list;
					LowDimensionalCoordinateInitialization(current_x_list);

					LMDS(current_x_list, repulsive_force_tau);
					if (mpParameters->mLMDSRefineNumNearestNeighbor > 0)
						LMDS_OnlyNN(current_x_list, mpParameters->mLMDSRefineNumNearestNeighbor); //performs a contraction of nearest neighbors

					distortion = Distortion(current_x_list);
					//keep solution with lowest distortion
					if (distortion < best_distortion) {
						best_x_list = current_x_list;
						best_distortion = distortion;
						best_neighborhood_size = effective_neighborhood_size;
						best_log_counter = log_counter;
						cout << endl << "Saving solution: achieved new low distortion: " << best_distortion << endl;
						SaveEmbedding(best_x_list);
						SaveDistortion(best_x_list);
						SaveNeighborhoodList();
						has_succesfully_completed_flag = true;
					} else {
						cout << endl << "Current distortion: " << distortion << " [not better than current best: " << best_distortion << "]" << endl;
					}
				}
			}
		}
	}
	cout << endl;
	cout << "Best solution found at distortion level: " << best_distortion << " Neighborhood size: " << best_neighborhood_size << " repulsive force multiplicative factor: " << best_log_counter << endl;
	oXList = best_x_list;
	if (has_succesfully_completed_flag == false)
		throw range_error("ERROR: Something went wrong in the embedding. Nothing was done. Try increasing the number of iterations  --num_of_iterations");
}

void EmbedManager::LMDS(vector<FVector>& current_x_list, double repulsive_force_tau) {
	//iterative minimization loop
	ProgressBar pb;
	cout << "Multi-dimensional-scaling phase..." << endl;
	for (unsigned itera = 0; itera < mpParameters->mLMDSNumIterations; ++itera) {
		pb.Count();
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
		for (unsigned i = 0; i < mpData->Size(); ++i) {
			//for all instances in the neighborhood
			assert(mNeighborhoodList[i].size()>0);
			for (unsigned t = 0; t < mNeighborhoodList[i].size(); ++t) {
				unsigned j = mNeighborhoodList[i][t];
				if (i != j) { //NOTE:exclude the case of having in the neighborhood the instance itself
					FVector diff_versor = Versor(current_x_list[j], current_x_list[i]);
					double current_distance = Distance(current_x_list[j], current_x_list[i]);
					double desired_distance = (1 - mpData->ComputeKernel(i, j));
					//double stress = sqrt((current_distance - desired_distance) * (current_distance - desired_distance));
					double stress = fabs(current_distance - desired_distance);
					double correction = stress * mpParameters->mLMDSIterationEpsilon;
					current_x_list[i] += diff_versor * correction;
				}
			}
			//for all instances not in the neighborhood
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
			for (unsigned t = 0; t < mNonNeighborhoodList[i].size(); ++t) {
				unsigned j = mNonNeighborhoodList[i][t];
				FVector diff_versor = Versor(current_x_list[i], current_x_list[j]);
				//double long_distance_repulsive_force = mTau * mpParameters->mLMDSTau * (1 - (double) (itera + 1) / (double) mpParameters->mLMDSNumIterations);
				double long_distance_repulsive_force = repulsive_force_tau;
				double correction = long_distance_repulsive_force * mpParameters->mLMDSIterationEpsilon;
				current_x_list[i] += diff_versor * correction;
			}
		}
	}
}

void EmbedManager::LMDS_OnlyNN(vector<FVector>& current_x_list, unsigned aNumNeighbors) {
	//iterative minimization loop
	ProgressBar pb;
	cout << "Refinement of multi-dimensional-scaling for neighborhood of size " << aNumNeighbors << endl;
	for (unsigned itera = 0; itera < mpParameters->mLMDSNumIterations; ++itera) {
		pb.Count();
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
		for (unsigned i = 0; i < mpData->Size(); ++i) {
			unsigned effective_size = min((unsigned) mNeighborhoodList[i].size(), aNumNeighbors);
			//for all instances in the neighborhood
			for (unsigned t = 0; t < effective_size; ++t) {
				unsigned j = mNeighborhoodList[i][t];
				if (i != j) { //NOTE:exclude the case of having in the neighborhood the instance itself
					FVector diff_versor = Versor(current_x_list[j], current_x_list[i]);
					double current_distance = Distance(current_x_list[j], current_x_list[i]);
					double desired_distance = (1 - pow(mpData->ComputeKernel(i, j), .5));
					double stress = fabs(current_distance - desired_distance);
					//double stress = sqrt((current_distance - desired_distance) * (current_distance - desired_distance));
					double correction = stress * mpParameters->mLMDSIterationEpsilon;
					current_x_list[i] += diff_versor * correction;
				}
			}
		}
	}
}

void EmbedManager::SaveEmbedding(vector<FVector>& aXList) {
	OutputManager om("embedding", mpParameters->mDirectoryPath);

	for (unsigned t = 0; t < aXList.size(); t++) {
		for (unsigned i = 0; i < mpParameters->mLMDSDimensionality; i++) {
			double val = aXList[t](i);
			om.mOut << val << " ";
		}
		om.mOut << endl;
	}
	cout << "Embedding saved in file " << om.GetFullPathFileName() << endl;
}

void EmbedManager::SaveDistortion(vector<FVector>& aXList) {
	OutputManager om("distortion", mpParameters->mDirectoryPath);

	vector<set<unsigned> > low_dim_neighborhood_list;
	MakeNeighborhoodList(aXList, low_dim_neighborhood_list);
	assert(aXList.size() == mpData->Size());
	for (unsigned i = 0; i < aXList.size(); ++i) {
		set<unsigned> high_dim_neighborhood;
		high_dim_neighborhood.insert(mNeighborhoodList[i].begin(), mNeighborhoodList[i].end());
		//compute normalized size of the neighborhood intersection
		set<unsigned> intersection;
		set_intersection(high_dim_neighborhood.begin(), high_dim_neighborhood.end(), low_dim_neighborhood_list[i].begin(), low_dim_neighborhood_list[i].end(), inserter(intersection, intersection.begin()));
		double distortion = 1 - (double) intersection.size() / sqrt((double) high_dim_neighborhood.size() * (double) low_dim_neighborhood_list[i].size());
		om.mOut << distortion << endl;
	}

	cout << "Distortion saved in file " << om.GetFullPathFileName() << endl;
}

void EmbedManager::SaveNeighborhoodList() {
	OutputManager om("neighborhood", mpParameters->mDirectoryPath);
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		for (unsigned t = 0; t < mNeighborhoodList[i].size(); ++t) {
			om.mOut << mNeighborhoodList[i][t] << " ";
		}
		om.mOut << endl;
	}
	cout << "Neighbors identity saved in file " << om.GetFullPathFileName() << endl;

	//------------------------------TMP
	OutputManager omn("non_neighborhood", mpParameters->mDirectoryPath);
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		omn.mOut << i << "] ";
		for (unsigned t = 0; t < mNonNeighborhoodList[i].size(); ++t) {
			omn.mOut << mNonNeighborhoodList[i][t] << " ";
		}
		omn.mOut << endl;
	}
	cout << "Non Neighbors identity saved in file " << omn.GetFullPathFileName() << endl;

}

void EmbedManager::InitNeighborhoodList(unsigned aNeighborhoodSize, unsigned aNonNeighborhoodSize) {
	cout << "Neighborhood indicators computation" << endl;
	mNeighborhoodList.clear();
	mNonNeighborhoodList.clear();
	mNearestNeighbor.CacheReset();

	//swap current neighborhood size
	unsigned neighborhood_size = mpParameters->mNumNearestNeighbors;
	mpParameters->mNumNearestNeighbors = aNeighborhoodSize;

	//extract neighbors and a random sample of non_neighbors
	{
		ProgressBar pb;
		for (unsigned current_id = 0; current_id < mpData->Size(); ++current_id) {
			//compute neighbors
			vector<unsigned> neighbor_list = mNearestNeighbor.ComputeNeighborhood(current_id);
			mNeighborhoodList.push_back(neighbor_list);

			//compute non neighbors by random sampling
			unsigned furthest_neighbor = neighbor_list.back();
			double horizon_similarity = mpData->ComputeKernel(current_id, furthest_neighbor);
			set<unsigned> non_neighbor_list;
			for (unsigned j = 0; j < aNonNeighborhoodSize; j++) {
				unsigned non_neighbor_id = randomUnsigned(mpData->Size());
				double current_similarity = mpData->ComputeKernel(current_id, non_neighbor_id);
				if (current_similarity < horizon_similarity)
					non_neighbor_list.insert(non_neighbor_id);
			}
			vector<unsigned> non_neighbor_vec;
			for (set<unsigned>::const_iterator it = non_neighbor_list.begin(); it != non_neighbor_list.end(); ++it)
				non_neighbor_vec.push_back((*it));
			mNonNeighborhoodList.push_back(non_neighbor_vec);

			pb.Count();
		}
	}
	mpParameters->mNumNearestNeighbors = neighborhood_size;
	InitTau();
}

void EmbedManager::InitTau() {
	ProgressBar pb;
	vector<double> all_distance_list;
	cout << "Computing local-global tradeoff factor: " << endl;
	for (unsigned current_id = 0; current_id < mpData->Size(); ++current_id) {
		//for neighbors
		vector<unsigned>& neighbor_list = mNeighborhoodList[current_id];
		for (unsigned i = 0; i < neighbor_list.size(); ++i) {
			unsigned neighbor_id = neighbor_list[i];
			double k = mpData->ComputeKernel(current_id, neighbor_id);
			all_distance_list.push_back(k);
		}
		//for non-neighbors
		vector<unsigned>& non_neighbor_list = mNonNeighborhoodList[current_id];
		for (unsigned i = 0; i < non_neighbor_list.size(); ++i) {
			unsigned non_neighbor_id = non_neighbor_list[i];
			double k = mpData->ComputeKernel(current_id, non_neighbor_id);
			all_distance_list.push_back(k);
		}
		pb.Count();
	}
	sort(all_distance_list.begin(), all_distance_list.end());
	unsigned median_index = all_distance_list.size() / 2;
	double median_distance = all_distance_list[median_index];
	mTau = (double) mNeighborhoodList[0].size() / (double) mNonNeighborhoodList[0].size() * median_distance;
	cout << endl << "Local-global tradeoff factor: " << mTau << endl;
}

void EmbedManager::MakeNeighborhoodList(vector<FVector>& aXList, vector<set<unsigned> >& oNeighborhoodList) {
	for (unsigned i = 0; i < aXList.size(); ++i) {
		vector<pair<double, unsigned> > similarity_list(aXList.size());
		//determine all pairwise similarities
		for (unsigned j = 0; j < aXList.size(); ++j) {
			if (i != j) { //NOTE: exclude self
				FVector difference_x = aXList[i] - aXList[j];
				double distance = difference_x.dot(difference_x);
				similarity_list[j] = make_pair(distance, j);
			}
		}
		//sort and retain the closest k indices
		sort(similarity_list.begin(), similarity_list.end());
		set<unsigned> neighbor_list;
		unsigned effective_size = min(mpParameters->mLMDSNeighborhoodSize, (unsigned) similarity_list.size());
		for (unsigned k = 0; k < effective_size; ++k) {
			unsigned neighbor_id = similarity_list[k].second;
			neighbor_list.insert(neighbor_id);
		}
		oNeighborhoodList.push_back(neighbor_list);
	}
}

void EmbedManager::LowDimensionalCoordinateInitialization(vector<FVector>& oXList) {
	oXList.clear();

	//read from file
	if (mpParameters->mEmbedFileName != "") {
		ProgressBar pb;
		cout << "Reading initial coordinates from file: " << mpParameters->mEmbedFileName << endl;
		ifstream fin;
		fin.open(mpParameters->mEmbedFileName.c_str());
		if (!fin)
			throw range_error("ERROR2.236: Cannot open file:" + mpParameters->mEmbedFileName);
		while (!fin.eof()) {
			string line;
			getline(fin, line);
			if (line != "") {
				stringstream ss;
				ss << line << endl;
				FVector x(mpParameters->mLMDSDimensionality);
				unsigned dimension_counter = 0;
				while (!ss.eof() && ss.good()) {
					double value;
					ss >> value;
					if (dimension_counter < mpParameters->mLMDSDimensionality)
						x(dimension_counter) = value;
					dimension_counter++;
				}
				oXList.push_back(x);
				pb.Count();
			}
		}
		fin.close();
	}
	else {
		const double value_range = 2;
		for (unsigned i = 0; i < mpData->Size(); ++i) {
			FVector x(mpParameters->mLMDSDimensionality);
			for (unsigned j = 0; j < mpParameters->mLMDSDimensionality; ++j) {
				double value = value_range * random01() - value_range / 2;
				x(j) = value;
			}
			oXList.push_back(x);
		}
	}
}

double EmbedManager::Distortion(vector<FVector>& aXList) {
unsigned local_continuity = 0;
vector<set<unsigned> > low_dim_neighborhood_list;
MakeNeighborhoodList(aXList, low_dim_neighborhood_list);
assert(aXList.size() == mpData->Size());
for (unsigned i = 0; i < aXList.size(); ++i) {

	set<unsigned> high_dim_neighborhood;
	high_dim_neighborhood.insert(mNeighborhoodList[i].begin(), mNeighborhoodList[i].end());
	//compute size of the neighborhood intersection
	set<unsigned> intersection;
	set_intersection(high_dim_neighborhood.begin(), high_dim_neighborhood.end(), low_dim_neighborhood_list[i].begin(), low_dim_neighborhood_list[i].end(), inserter(intersection, intersection.begin()));
	local_continuity += intersection.size();
}
double average_local_continuity = ((double) local_continuity / (double) mpParameters->mLMDSNeighborhoodSize) / (double) mpData->Size();
double average_local_continuity_adjusted_for_chance = average_local_continuity - (double) mpParameters->mLMDSNeighborhoodSize / (double) mpData->Size();
return 1 - average_local_continuity_adjusted_for_chance;
}

double EmbedManager::Stress(vector<FVector>& aXList) {
assert(aXList.size() == mpData->Size());
double stress = 0;

//compute average difference of distances
unsigned counter = 0;
for (unsigned i = 0; i < aXList.size(); ++i) {
	for (unsigned j = 0; j < aXList.size(); ++j) {
		if (i != j) {
			double desired_distance = 1 - mpData->ComputeKernel(i, j);
			double current_distance = Distance(aXList[i], aXList[j]);
			stress += fabs(current_distance - desired_distance) / (desired_distance);
			//stress += sqrt((current_distance - desired_distance) * (current_distance - desired_distance))/(desired_distance) ;
			counter++;
		}
	}
}
double average_stress = stress / (double) counter;
return average_stress;
}
