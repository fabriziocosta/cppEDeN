/* -*- mode:c++ -*- */
#ifndef SEMI_SUPERVISED_MANAGER_H
#define SEMI_SUPERVISED_MANAGER_H

#include "BaseManager.h"
#include "MinHashEncoder.h"

using namespace std;

class SemiSupervisedManager:public BaseManager {
protected:
	MinHashEncoder mMinHashEncoder;

public:
	SemiSupervisedManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void SSOutputManager();
	void Main(ostream& out);
};

#endif /* SEMI_SUPERVISED_MANAGER_H */
