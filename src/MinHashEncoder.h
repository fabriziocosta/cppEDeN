/* -*- mode:c++ -*- */
#ifndef MIN_HASH_ENCODER_H
#define MIN_HASH_ENCODER_H

#include "Utility.h"
#include "Parameters.h"
#include "Kernel.h"
#include "Data.h"

using namespace std;

class MinHashEncoder {
protected:
	Parameters* mpParameters;
	Data* mpData;
	vector<umap_uint_vec_uint> mInverseIndex;
	vector<vector<unsigned> > mMinHashCache;

public:
	MinHashEncoder();
	MinHashEncoder(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void CacheReset();
	void ComputeInverseIndex();
	void UpdateInverseIndex(vector<unsigned>& aSignature, unsigned aIndex);
	void CleanUpInverseIndex();
	vector<unsigned> ComputeHashSignature(unsigned aID);
	vector<unsigned> ComputeHashSignature(SVector& aX);
	vector<unsigned> ComputeHashSignatureSize(vector<unsigned>& aSignature);
	vector<unsigned> ComputeApproximateNeighborhood(const vector<unsigned>& aSignature, unsigned aNeighborhoodSize = 0);
	vector<unsigned> TrimNeighborhood(umap_uint_int& aNeighborhood, unsigned aNeighborhoodSize = 0);
};

#endif /* MIN_HASH_ENCODER_H */
