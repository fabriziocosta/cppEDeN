/* -*- mode:c++ -*- */
#ifndef TEST_MANAGER_H
#define TEST_MANAGER_H

#include "BaseManager.h"
#include "SGDSVMManager.h"

using namespace std;

class TestManager : public BaseManager {
protected:
	StochasticGradientDescentSupportVectorMachineManager mSGDSVMManager;
public:
	TestManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void InputOutputManager();
	void Main(istream& fin, ostream& ofs_pred, ostream& ofs_marg);
};

//------------------------------------------------------------------------------------------------------------------------
class TestPartManager {
protected:
	Parameters* mpParameters;
	Data* mpData;
	StochasticGradientDescentSupportVectorMachineManager mSGDSVMManager;
public:
	TestPartManager(Parameters* apParameters, Data* apData);
	void Init(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void InputOutputManager();
	void Main(istream& fin, ostream& ofs);
};

#endif /* TEST_MANAGER_H */
