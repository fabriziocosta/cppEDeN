/* -*- mode:c++ -*- */
#ifndef CLUSTER_MANAGER_H
#define CLUSTER_MANAGER_H

#include "NearestNeighbor.h"

using namespace std;

class ClusterManager: public BaseManager {
protected:
	NearestNeighbor mNearestNeighbor;
public:
	ClusterManager(Parameters* apParameters, Data* apData);
	void Exec();
protected:
	double ComputeDensity(unsigned aID);
	double ComputeDensity(unsigned aID, vector<unsigned>& aNeighborhood);
	double ComputeDensityStandardDeviation(unsigned aID, double aAverageDensity, vector<unsigned>& aNeighborhood);
	void ComputeDensity(vector<unsigned>& aSelectedIndexList, vector<pair<double, unsigned> >& oDensityList);
	pair<double, double> ComputeLocalCentrality(unsigned aID, vector<vector<unsigned> >& aApproximateNeighborhoodList);
};

class DensityClusterManager: public ClusterManager {
public:
	DensityClusterManager(Parameters* apParameters, Data* apData);
	void Exec();
protected:
	void DenseCluster(ostream& out, ostream& out_sim, ostream& out_ecc);
	vector<unsigned> ComputeMinimallyOverlappingHighDensityCenterList(unsigned aSampleSize, double aFractionCenterScan, unsigned aMaxIntersectionSize);
	void OutputDenseClusterNeighbors(ostream& out, vector<unsigned>& aNeighborhood);
	void OutputDenseClusterNeighborsSimilarity(ostream& out, vector<unsigned>& aNeighborhood);
	vector<unsigned> ComputeHighDensityCenterList(unsigned aSampleSize, double aFractionCenterScan);
	vector<unsigned> SampleID(double aFractionCenterScan);
};

class ConnectedDensityClusterManager: public DensityClusterManager {
public:
	ConnectedDensityClusterManager(Parameters* apParameters, Data* apData);
	void Exec();
protected:
	void DenseCluster(ostream& out, ostream& out_sim);
	vector<vector<unsigned> > ConnectedComponents(vector<vector<unsigned> >& aAdjacencyList, vector<unsigned>& aElementList);
};

class KQuickShiftClusterManager: public ClusterManager {
public:
	KQuickShiftClusterManager(Parameters* apParameters, Data* apData);
	void Exec();
protected:
	void KQuickShiftCluster(ostream& out, ostream& out_parent, ostream& out_ext, ostream& out_sim);
	void TreeVisit(unsigned aID, vector<vector<unsigned> >& aChildPointerList, vector<unsigned>& oDominatedTree);
};

class AccuracyClusterManager: public ClusterManager {
public:
	AccuracyClusterManager(Parameters* apParameters, Data* apData);
	void Exec();
protected:
	void Accuracy(ostream& out);
	void Density(ostream& out);
	void SignatureSize(ostream& out);
	void ApproximateNNStatistics(ostream& out);
};

#endif /* CLUSTER_MANAGER_H */
