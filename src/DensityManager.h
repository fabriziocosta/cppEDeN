/* -*- mode:c++ -*- */
#ifndef DENSITYMANAGER_H
#define DENSITYMANAGER_H

#include "BaseManager.h"
#include "NearestNeighbor.h"

using namespace std;

class DensityManager: public BaseManager {
	protected:
		NearestNeighbor mNearestNeighbor;
		vector<double> mPDF;
	public:
		DensityManager(Parameters* apParameters, Data* apData);
		void Exec();
		void ComputeVoronoiCellVolumeEstimate(ofstream& aOut, ofstream& aOutLog);
		vector<double> ComputeVoronoiCellVolumeEstimate();
		double ComputeVoronoiCellVolumeEstimate(unsigned aID);
		double ComputeVoronoiCellVolumeEstimate(unsigned aID, vector<unsigned>& aNeighborhood);
};

#endif /* DENSITYMANAGER_H */
