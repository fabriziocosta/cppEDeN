#include "DensityManager.h"

DensityManager::DensityManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData), mNearestNeighbor(apParameters, apData) {
}

void DensityManager::Exec() {
	cout << SEP << endl << "Probability Estimation phase" << endl << SEP << endl;
	ProgressBar pb;
	pb.Count();

	//computing probability as normalized inverse of voronoi cell volume estimate
	OutputManager omp("prob", mpParameters->mDirectoryPath);
	OutputManager omlp("logprob", mpParameters->mDirectoryPath);
	ComputeVoronoiCellVolumeEstimate(omp.mOut, omlp.mOut);
	cout << "Probability estimate saved in file " << omp.GetFullPathFileName() << endl;
	cout << "Log of Probability estimate saved in file " << omlp.GetFullPathFileName() << endl;
}

void DensityManager::ComputeVoronoiCellVolumeEstimate(ofstream& aOut, ofstream& aOutLog) {
	vector<double> p=ComputeVoronoiCellVolumeEstimate();
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		aOut << p[i] << endl;
		aOutLog << log(p[i]) << endl;
	}
}

vector<double> DensityManager::ComputeVoronoiCellVolumeEstimate() {
	ProgressBar pb;
	cout << "Computing volume estimate for voronoi cells for " << mpData->Size() << " instances." << endl;
	mpParameters->mNoNeighborhoodCache = false; //force caching
	if (mpParameters->mSharedNeighborhood)
		mNearestNeighbor.ComputeSharedNeighborhoods();

	vector<double> voronoi_cell_volume(mpData->Size());
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		pb.Count();
		voronoi_cell_volume[i] = ComputeVoronoiCellVolumeEstimate(i);
	}
	double total_volume = 0;
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		total_volume += voronoi_cell_volume[i];
	}
	cout << "Total volume: " << total_volume << endl;
	vector<double> prob(mpData->Size());
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		prob[i] = 1 / (voronoi_cell_volume[i] / total_volume);
	}
	double mass = 0;
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		mass += prob[i];
	}
	vector<double> p(mpData->Size());
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		p[i] = prob[i] / mass;
	}
	return p;
}

double DensityManager::ComputeVoronoiCellVolumeEstimate(unsigned aID) {
	vector<unsigned> neighborhood = mNearestNeighbor.ComputeNeighborhood(aID);
	if (neighborhood.size() == 0)
		throw range_error("ERROR: DensityManager::ComputeVoronoiCellVolumeEstimate: empty neighborhood for instance id:" + stream_cast<string>(aID));
	double vol = ComputeVoronoiCellVolumeEstimate(aID, neighborhood);
	return vol;
}

double DensityManager::ComputeVoronoiCellVolumeEstimate(unsigned aID, vector<unsigned>& aNeighborhood) {
	VectorClass distance_list;
	//compute kernel pairs between i and all elements in Neighborhood
	for (unsigned i = 0; i < aNeighborhood.size(); i++) {
		VectorClass local_distance_list;
		for (unsigned j = 0; j < aNeighborhood.size(); j++) {
			unsigned u = aNeighborhood[i];
			unsigned v = aNeighborhood[j];
			if (u != v) {
				double d_uv = 1 - (mpData->ComputeKernel(u, v)); //assuming normalized kernel the distance is computed as 1-cos(u,v)
				if (d_uv != 0)
					local_distance_list.PushBack(d_uv);
			}
		}
		//median distance of each neighbor wrt all others is used so to simulate random sampling of neighbors
		double median = local_distance_list.Median();
		double guarded_median = median != 0 ? median : 1;
		distance_list.PushBack(guarded_median);
	}
	double vol = distance_list.Prod(); //product of median distances for each neighbor is dimensional spaces with dimension so high that the neighbos will never be collinear, can approximate the volume of the convex hull which in turns approximates the volume of the voronoi cell
	return vol;
}
