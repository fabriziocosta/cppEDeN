/* -*- mode:c++ -*- */
#ifndef MIN_HASH_MANAGER_H
#define MIN_HASH_MANAGER_H

#include "BaseManager.h"
#include "MinHashEncoder.h"

using namespace std;

class MinHashManager: public BaseManager {
protected:
	MinHashEncoder mMinHashEncoder;

public:
	MinHashManager();
	MinHashManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Exec();
	void InputOutputManager();
	void Main(istream& fin, ostream& ofs);
};

#endif /* MIN_HASH_MANAGER_H */
