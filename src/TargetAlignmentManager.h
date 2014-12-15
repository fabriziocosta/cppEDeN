/* -*- mode:c++ -*- */
#ifndef TARGET_ALIGNMENT_MANAGER_H
#define TARGET_ALIGNMENT_MANAGER_H

#include "BaseManager.h"

using namespace std;

class TargetAlignmentManager: public BaseManager{
protected:
public:
	TargetAlignmentManager(Parameters* apParameters, Data* apData);
	void Load();
	void Exec();
	void Main();
};

#endif /* TARGET_ALIGNMENT_MANAGER_H */
