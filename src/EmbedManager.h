/* -*- mode:c++ -*- */
#ifndef EMBED_MANAGER_H
#define EMBED_MANAGER_H

#include "BaseManager.h"
#include "NearestNeighbor.h"

using namespace std;

class EmbedManager: public BaseManager{
protected:
protected:
	NearestNeighbor mNearestNeighbor;

	//MinHashEncoder mMinHashEncoder;
	double mTau;
	vector<vector<unsigned> > mNeighborhoodList;
	vector<vector<unsigned> > mNonNeighborhoodList;
public:
	EmbedManager(Parameters* apParameters, Data* apData);
	//void Init(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void Main();
	double Norm(const FVector& aX);
	double Distance(const FVector& aX, const FVector& aZ);
	FVector Versor(const FVector& aX, const FVector& aZ);
	void ComputeLocalMultiDimensionalScaling(vector<FVector>& oXList);
	void LMDS(vector<FVector>& current_x_list, double repulsive_force_tau);
	void LMDS_OnlyNN(vector<FVector>& current_x_list, unsigned aNumNeighbors);
	void SaveEmbedding(vector<FVector>& aXList);
	void SaveDistortion(vector<FVector>& aXList);
	void SaveNeighborhoodList();
	void InitNeighborhoodList(unsigned aNeighborhoodSize, unsigned aNonNeighborhoodSize);
	void InitTau();
	void MakeNeighborhoodList(vector<FVector>& aXList, vector<set<unsigned> >& oNeighborhoodList);
	void LowDimensionalCoordinateInitialization(vector<FVector>& oXList);
	double Distortion(vector<FVector>& aXList);
	double Stress(vector<FVector>& aXList);
};

#endif /* EMBED_MANAGER_H */
