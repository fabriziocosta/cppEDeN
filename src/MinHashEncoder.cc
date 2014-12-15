#include "MinHashEncoder.h"
MinHashEncoder::MinHashEncoder() :
		mpParameters(0), mpData(0) {
}

MinHashEncoder::MinHashEncoder(Parameters* apParameters, Data* apData) {
	Init(apParameters, apData);
}

void MinHashEncoder::Init(Parameters* apParameters, Data* apData) {
	mpParameters = apParameters;
	mpData = apData;

	//init inverse index data structure
	mInverseIndex.clear();
	for (unsigned k = 0; k < mpParameters->mNumHashFunctions; ++k)
		mInverseIndex.push_back(umap_uint_vec_uint());
}

void MinHashEncoder::CacheReset() {
	mMinHashCache.clear();
	if (mpParameters->mNoMinHashCache == false)
		if (mpData->Size() > 0)
			mMinHashCache.resize(mpData->Size());

}

void MinHashEncoder::ComputeInverseIndex() {
	CacheReset();
	cout << "Computing Inverse Index on " << mpData->mColIndexList.size() << " instances." << endl;
	cout << "Using " << mpParameters->mNumHashFunctions << " hash functions (with factor " << mpParameters->mNumRepeatsHashFunction << " for single minhash)" << endl;

	ProgressBar progress_bar;
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
	for (unsigned ii = 0; ii < mpData->mColIndexList.size(); ++ii) { //for every instance
		unsigned i = mpData->mColIndexList[ii];
#ifdef USEMULTITHREAD
#pragma omp critical
#endif
		{
			vector<unsigned> signature = ComputeHashSignature(i); //compute the signature
			UpdateInverseIndex(signature, i);
		}
		progress_bar.Count();
	}
	CleanUpInverseIndex();
}

void MinHashEncoder::UpdateInverseIndex(vector<unsigned>& aSignature, unsigned aIndex) {
	for (unsigned k = 0; k < mpParameters->mNumHashFunctions; ++k) { //for every hash value
		unsigned key = aSignature[k];
		if (key != MAXUNSIGNED && key != 0) { //if key is equal to markers for empty bins then skip insertion instance in data structure
			if (mInverseIndex[k].count(key) == 0) { //if this is the first time that an instance exhibits that specific value for that hash function, then store for the first time the reference to that instance
				vector<unsigned> tmp;
				tmp.push_back(aIndex);
				mInverseIndex[k].insert(make_pair(key, tmp));
			} else if (mInverseIndex[k].count(key) > mpParameters->mMaxSizeBin) {
				//do not insert instance if bin is too full
			} else { //Otherwise just append the instance to the list
				mInverseIndex[k][key].push_back(aIndex);
			}
		}
	}
}

void MinHashEncoder::CleanUpInverseIndex() {
	for (unsigned k = 0; k < mpParameters->mNumHashFunctions; ++k) {
		for (umap_uint_vec_uint::const_iterator jt = mInverseIndex[k].begin(); jt != mInverseIndex[k].end(); ++jt) {
			unsigned hash_id = jt->first;
			if (hash_id != 0 && hash_id != MAXUNSIGNED) { //do not consider buckets corresponding to null bins
				unsigned collision_size = mInverseIndex[k][hash_id].size();

				if (collision_size < mpParameters->mMaxSizeBin) {
				} else {//remove bins that are too full from inverse index
					mInverseIndex[k].erase(hash_id);
				}
			}
		}
	}
}

vector<unsigned> MinHashEncoder::ComputeHashSignatureSize(vector<unsigned>& aSignature) {
	vector<unsigned> signature_size(mpParameters->mNumHashFunctions);
	assert(aSignature.size()==mpParameters->mNumHashFunctions);
	for (unsigned i = 0; i < aSignature.size(); ++i) {
		unsigned key = aSignature[i];
		signature_size[i] = mInverseIndex[i][key].size();
	}
	return signature_size;
}

vector<unsigned> MinHashEncoder::ComputeHashSignature(unsigned aID) {
	if (mpParameters->mNoMinHashCache == false) {
		if (mMinHashCache[aID].size() > 0)
			return mMinHashCache[aID];
		else {
			vector<unsigned> signature = ComputeHashSignature(mpData->mVectorList[aID]);
			mMinHashCache[aID] = signature;
			return signature;
		}
	} else
		return ComputeHashSignature(mpData->mVectorList[aID]);
}

vector<unsigned> MinHashEncoder::ComputeHashSignature(SVector& aX) {
	unsigned sub_hash_range = mpParameters->mNumHashFunctions / mpParameters->mNumRepeatsHashFunction;

	vector<unsigned> signature(mpParameters->mNumHashFunctions);
	//init with MAXUNSIGNED
	for (unsigned k = 0; k < mpParameters->mNumHashFunctions; ++k)
		signature[k] = MAXUNSIGNED;

	//prepare a vector containing the signature as the k min values
	//for each element of the sparse vector
	for (SVector::InnerIterator it(aX); it; ++it) {
		unsigned feature_id = it.index();
		//for each sub_hash
		for (unsigned l = 1; l <= mpParameters->mNumRepeatsHashFunction; ++l) {
			unsigned key = IntHash(feature_id, MAXUNSIGNED, l);
			for (unsigned kk = 0; kk < sub_hash_range; ++kk) { //for all k values
				unsigned lower_bound = MAXUNSIGNED / sub_hash_range * kk;
				unsigned upper_bound = MAXUNSIGNED / sub_hash_range * (kk + 1);

				if (key >= lower_bound && key < upper_bound) { //if we are in the k-th slot
					unsigned signature_feature = kk + (l - 1) * sub_hash_range;
					if (key < signature[signature_feature]) //keep the min hash within the slot
						signature[signature_feature] = key;
				}
			}
		}
	}

	return signature;
}

vector<unsigned> MinHashEncoder::ComputeApproximateNeighborhood(const vector<unsigned>& aSignature, unsigned aNeighborhoodSize) {
	umap_uint_int neighborhood;
	for (unsigned k = 0; k < mpParameters->mNumHashFunctions; ++k) {
		unsigned hash_id = aSignature[k];
		if (hash_id != 0 && hash_id != MAXUNSIGNED) { //do not consider buckets corresponding to null bins
			unsigned collision_size = mInverseIndex[k][hash_id].size();

			if (collision_size < mpParameters->mMaxSizeBin) {
				//fill neighborhood set counting number of occurrences
				for (vector<unsigned>::iterator it = mInverseIndex[k][hash_id].begin(); it != mInverseIndex[k][hash_id].end(); ++it) {
					unsigned instance_id = *it;
					if (neighborhood.count(instance_id) > 0)
						neighborhood[instance_id]++;
					else
						neighborhood[instance_id] = 1;
				}
			} else { //do nothing, as these over filled  bins are not representative
				//..or
				//remove bins that are too full from inverse index
				//mInverseIndex[k].erase(hash_id);
			}
		}
	}
	return TrimNeighborhood(neighborhood, aNeighborhoodSize);
}

vector<unsigned> MinHashEncoder::TrimNeighborhood(umap_uint_int& aNeighborhood, unsigned aNeighborhoodSize) {
	unsigned neighborhood_size;
	if (aNeighborhoodSize == 0)
		neighborhood_size = mpParameters->mNumNearestNeighbors;
	else
		neighborhood_size = aNeighborhoodSize;
	const int MIN_BINS_IN_COMMON = 2; //Minimum number of bins that two instances have to have in common in order to be considered similar
	//given a list of neighbors with an associated occurrences count, return only a fraction of the highest count ones
	vector<unsigned> neighborhood_list;
	if (mpParameters->mEccessNeighborSizeFactor > 0) {
		//sort by num occurrences
		vector<pair<int, unsigned> > count_list;
		for (umap_uint_int::const_iterator it = aNeighborhood.begin(); it != aNeighborhood.end(); ++it) {
			unsigned id = it->first;
			int count = it->second;
			if (count >= MIN_BINS_IN_COMMON) //NOTE: consider instances that have at least MIN_BINS_IN_COMMON
				count_list.push_back(make_pair(-count, id)); //NOTE:-count to sort from highest to lowest
		}
		sort(count_list.begin(), count_list.end());
		unsigned effective_size = min((unsigned) count_list.size(), (unsigned) (mpParameters->mEccessNeighborSizeFactor * neighborhood_size));
		for (unsigned i = 0; i < effective_size; ++i)
			neighborhood_list.push_back(count_list[i].second);
	} else { //if mEccessNeighborSizeFactor is negative then consider all instances in the approximate neighborhood that have a co-occurrence count higher than - mEccessNeighborSizeFactor
		for (umap_uint_int::const_iterator it = aNeighborhood.begin(); it != aNeighborhood.end(); ++it) {
			int count = it->second;
			if (count >= (int) (-mpParameters->mEccessNeighborSizeFactor))
				neighborhood_list.push_back(it->first);
		}
	}
	return neighborhood_list;
}
