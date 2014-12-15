#include "MinHashManager.h"

MinHashManager::MinHashManager() : BaseManager(0,0){
}

MinHashManager::MinHashManager(Parameters* apParameters, Data* apData) : BaseManager(apParameters, apData){
	Init(apParameters, apData);
}

void MinHashManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mpData->mKernel.ParametersSetup();
	mMinHashEncoder.Init(apParameters, apData);
}

void MinHashManager::Exec() {
	InputOutputManager();
}

void MinHashManager::InputOutputManager() {
	//output
	OutputManager om("min_hash_feature", mpParameters->mDirectoryPath);
	OutputManager omfs("min_hash_feature_size", mpParameters->mDirectoryPath);

	//input
	igzstream fin;
	fin.open(mpParameters->mInputDataFileName.c_str());
	if (!fin)
		throw range_error("ERROR2.117: Cannot open file: " + mpParameters->mInputDataFileName);
	//perform online action
	if (!mpParameters->mMinimalOutput)
		cout << "Processing I/O file: " << mpParameters->mInputDataFileName << endl;
	Main(fin, om.mOut);
		if (!mpParameters->mMinimalOutput)
			cout << "Result saved in file " << om.GetFullPathFileName() << endl;
}

void MinHashManager::Main(istream& fin, ostream& ofs) {
	ProgressBar pb;
	bool valid_input = true;
	while (!fin.eof() && valid_input) {

		switch (mpParameters->mFileTypeCode) {
		case GRAPH:
#ifdef USEOBABEL
			case MOLECULAR_GRAPH:
#endif
		case SEQUENCE: {
			GraphClass g;
			mpData->SetGraphFromFile(fin, g);
			if (!g.IsEmpty()) {
				SVector x(pow(2,mpParameters->mHashBitSize));
				mpData->mKernel.GenerateFeatureVector(g, x);
				vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(x); //compute the signature
				for (unsigned k = 0; k < signature.size(); k++) {
					ofs << signature[k] << " ";
				}
				ofs << endl;
				pb.Count();
			} else
				valid_input = false;
		}
			break;
		default:
			throw range_error("ERROR2.45: file type not recognized: " + mpParameters->mFileType);
		}
	}
}
