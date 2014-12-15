/* -*- mode:c++ -*- */
#ifndef NEAREST_NEIGHBOR_MANAGER_H
#define NEAREST_NEIGHBOR_MANAGER_H

#include "NearestNeighbor.h"

using namespace std;

class NearestNeighborManager:public BaseManager {
protected:
	NearestNeighbor mNearestNeighbor;

public:
	NearestNeighborManager(Parameters* apParameters, Data* apData);
	void Exec();
	void Main(ostream& out_knn, ostream& out_k, ostream& out_t, ostream& out_f);
};

#endif /* NEAREST_NEIGHBOR_H */
