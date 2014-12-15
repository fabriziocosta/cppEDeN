/* -*- mode:c++ -*- */
#ifndef NEAREST_NEIGHBOR_H
#define NEAREST_NEIGHBOR_H

#include "BaseManager.h"
#include "MinHashEncoder.h"

using namespace std;

class NearestNeighbor: public BaseManager {
public:
	MinHashEncoder mMinHashEncoder;
protected:
	vector<vector<unsigned> > mNeighborhoodCache;

public:
	NearestNeighbor(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void CacheReset();
	vector<unsigned> ComputeNeighborhood(unsigned aID);
	vector<unsigned> ComputeTrueSubNeighborhood(unsigned aID, vector<unsigned>& aApproximateNeighborhoodList);
	vector<unsigned> ComputeTrueNeighborhood(unsigned aID);
	vector<unsigned> ComputeApproximateNeighborhood(unsigned aID);
	vector<unsigned> ComputeSharedNeighborhood(unsigned aID);
	double ComputeSharedNeighborhoodSimilarity(unsigned aI, unsigned aJ);
	unsigned ComputeNeighborhoodIntersection(unsigned aI, unsigned aJ);
	void ComputeSharedNeighborhoods();

	vector<unsigned> ComputeNeighborhood(SVector& aX);
	vector<unsigned> ComputeApproximateNeighborhood(SVector& aX);
	vector<unsigned> ComputeTrueSubNeighborhood(SVector& aX, vector<unsigned>& aApproximateNeighborhoodList);
	vector<unsigned> ComputeTrueNeighborhood(SVector& aX);
};

#endif /* NEAREST_NEIGHBOR_H */
