/* -*- mode:c++ -*- */
#ifndef PRE_CLUSTER_FILTER_MANAGER_H
#define PRE_CLUSTER_FILTER_MANAGER_H

#include "BaseManager.h"
#include "MinHashEncoder.h"

using namespace std;

class PreClusterFilterManager: public BaseManager {
protected:
	MinHashEncoder mMinHashEncoder;
	unsigned mThreshold;
public:
	PreClusterFilterManager();
	PreClusterFilterManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Exec();
	void OnlineDataProcessing(istream& fin, ostream& out, ostream& outi, const string& aActionCode);
};

#endif /* PRE_CLUSTER_FILTER_MANAGER_H */
