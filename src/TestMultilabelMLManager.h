/* -*- mode:c++ -*- */
#ifndef TEST_MULTILABEL_ML_MANAGER_H
#define TEST_MULTILABEL_ML_MANAGER_H

#include "SGDSVMManager.h"

using namespace std;

class TestMultilabelMLManager: public BaseManager {
	protected:
		vector<StochasticGradientDescentSupportVectorMachine> mSGDSVMList;
	public:
		TestMultilabelMLManager();
		TestMultilabelMLManager(Parameters* apParameters, Data* apData);
		void Init(Parameters* apParameters, Data* apData);
		void Exec();
		void LoadModels();
		void InputOutputManager();
		void Main(istream& fin, ostream& ofs, ostream& ofsm);
};

#endif /* TEST_MULTILABEL_ML_MANAGER_H */
