#include "TestMultilabelMLManager.h"

TestMultilabelMLManager::TestMultilabelMLManager() :
		BaseManager(0, 0) {
}

TestMultilabelMLManager::TestMultilabelMLManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void TestMultilabelMLManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mpData->mKernel.ParametersSetup();
}

void TestMultilabelMLManager::LoadModels() {
	string dirname = mpParameters->mDirectoryPath != "" ? mpParameters->mDirectoryPath + "/" : "";
	string filename = dirname + mpParameters->mModelFileName + "_<number>";
	cout << "Loading model files with template: " << filename << endl;
	ProgressBar pb(1);
	bool valid_filename = false;
	unsigned model_counter = 0;
	do {
		string filename = dirname + mpParameters->mModelFileName + "_" + stream_cast<string>(model_counter);
		igzstream fin;
		fin.open(filename.c_str());
		if (!fin) {
			valid_filename = false;
		} else {
			pb.Count();
			valid_filename = true;
			StochasticGradientDescentSupportVectorMachine sgdsvm(mpParameters, mpData);
			sgdsvm.Load(fin);
			mSGDSVMList.push_back(sgdsvm);
			model_counter++;
		}
	} while (valid_filename == true);
	if (mSGDSVMList.size() == 0)
		throw range_error("ERROR:TestMultilabelMLManager::LoadModels: No model was loaded");
	else
		cout << "Read " << mSGDSVMList.size() << " models." << endl;
}

void TestMultilabelMLManager::Exec() {
	LoadModels();
	InputOutputManager();
}

void TestMultilabelMLManager::InputOutputManager() {
	//output
	OutputManager omp("prediction", mpParameters->mDirectoryPath);
	OutputManager omm("margin", mpParameters->mDirectoryPath);

	//input
	igzstream fin;
	fin.open(mpParameters->mInputDataFileName.c_str());
	if (!fin)
		throw range_error("ERROR:TestMultilabelMLManager::InputOutputManager: Cannot open file: " + mpParameters->mInputDataFileName);
	//perform online action
	if (!mpParameters->mMinimalOutput)
		cout << "Processing I/O file: " << mpParameters->mInputDataFileName << endl;
	Main(fin, omp.mOut, omm.mOut);
	if (!mpParameters->mMinimalOutput) {
		cout << "Predictions saved in file " << omp.GetFullPathFileName() << endl;
		cout << "Margins saved in file " << omm.GetFullPathFileName() << endl;
	}
}

void TestMultilabelMLManager::Main(istream& fin, ostream& ofs_pred, ostream& ofs_marg) {
	ProgressBar pb;
	unsigned instance_counter = 0;
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
					SVector x(pow(2, mpParameters->mHashBitSize));
					mpData->mKernel.GenerateFeatureVector(g, x);
					for (unsigned i = 0; i < mSGDSVMList.size(); ++i) {
						double margin = mSGDSVMList[i].Predict(x);
						int prediction = margin > 0 ? 1 : -1;
						ofs_pred << prediction << " ";
						ofs_marg << margin << " ";
					}
					ofs_pred << endl;
					ofs_marg << endl;
					if (!mpParameters->mMinimalOutput)
						pb.Count();
					instance_counter++;
				} else
					valid_input = false;
			}
				break;
			case SPARSE_VECTOR: {
				SVector x(pow(2, mpParameters->mHashBitSize));
				if (mpParameters->mBinaryFormat)
					mpData->SetVectorFromSparseVectorBinaryFile(fin, x);
				else
					mpData->SetVectorFromSparseVectorAsciiFile(fin, x);
				if (x.nonZeros() > 0) {
					for (unsigned i = 0; i < mSGDSVMList.size(); ++i) {
						double margin = mSGDSVMList[i].Predict(x);
						int prediction = margin > 0 ? 1 : -1;
						ofs_pred << prediction << " ";
						ofs_marg << margin << " ";
						if (!mpParameters->mMinimalOutput)
							pb.Count();
						instance_counter++;
					}
					ofs_pred << endl;
					ofs_marg << endl;
				} else
					valid_input = false;
			}
				break;
			default:
				throw range_error("ERROR:TestMultilabelMLManager::Main: file type not recognized: " + mpParameters->mFileType);
		}
	}
}
