#include "TestManager.h"

//------------------------------------------------------------------------------------------------------------------------
TestManager::TestManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void TestManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mpData->mKernel.ParametersSetup();
	mSGDSVMManager.Init(apParameters, apData);
}

void TestManager::Load() {
	mSGDSVMManager.LoadModel();
}

void TestManager::Exec() {
	Load();
	InputOutputManager();
}

void TestManager::InputOutputManager() {
	//output
	OutputManager omp("prediction", mpParameters->mDirectoryPath);
	OutputManager omm("margin", mpParameters->mDirectoryPath);
	//input
	igzstream fin;
	fin.open(mpParameters->mInputDataFileName.c_str());
	if (!fin)
		throw range_error("ERROR:TestManager::InputOutputManager: Cannot open file: " + mpParameters->mInputDataFileName);
	//perform online action
	if (!mpParameters->mMinimalOutput)
		cout << "Processing I/O file: " << mpParameters->mInputDataFileName << endl;
	mpData->mKernel.OutputParameters(cout);
	Main(fin, omp.mOut, omm.mOut);
	if (!mpParameters->mMinimalOutput) {
		cout << "Predictions saved in file " << omp.GetFullPathFileName() << endl;
		cout << "Margins saved in file " << omm.GetFullPathFileName() << endl;
	}
}

void TestManager::Main(istream& fin, ostream& ofs_pred, ostream& ofs_marg) {
	ProgressBar pb;
	unsigned instance_counter = 0;
	bool valid_input = true;
	while (!fin.eof() && valid_input) {

		switch (mpParameters->mFileTypeCode) {
			case GRAPH:
#ifdef USEOBABEL
				case MOLECULAR_GRAPH:
#endif
			case SEQUENCE:
			case STRINGSEQ: {
				GraphClass g;
				mpData->SetGraphFromFile(fin, g);
				if (!g.IsEmpty()) {
					SVector x(pow(2, mpParameters->mHashBitSize));

					mpData->mKernel.GenerateFeatureVector(g, x);
					double margin = mSGDSVMManager.Predict(x);
					int prediction = margin > 0 ? 1 : -1;
					ofs_pred << prediction << endl;
					ofs_marg << margin << endl;
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
					double margin = mSGDSVMManager.Predict(x);
					int prediction = margin > 0 ? 1 : -1;
					ofs_pred << prediction << endl;
					ofs_marg << margin << endl;
					if (!mpParameters->mMinimalOutput)
						pb.Count();
					instance_counter++;
				} else
					valid_input = false;
			}
				break;
			default:
				throw range_error("ERROR:TestManager::Main: file type not recognized: " + mpParameters->mFileType);
		}
	}
}

//------------------------------------------------------------------------------------------------------------------------
TestPartManager::TestPartManager(Parameters* apParameters, Data* apData) {
	Init(apParameters, apData);
}

void TestPartManager::Init(Parameters* apParameters, Data* apData) {
	mpParameters = apParameters;
	mpData = apData;
	mpData->mKernel.ParametersSetup();
	mSGDSVMManager.Init(apParameters, apData);
}

void TestPartManager::Load() {
	mSGDSVMManager.LoadModel();
}

void TestPartManager::Exec() {
	Load();
	InputOutputManager();
}

void TestPartManager::InputOutputManager() {
	//output
	OutputManager om("prediction_part", mpParameters->mDirectoryPath);

	//input
	igzstream fin;
	fin.open(mpParameters->mInputDataFileName.c_str());
	if (!fin)
		throw range_error("ERROR TestPartManager::InputOutputManager: Cannot open file: " + mpParameters->mInputDataFileName);
	//perform online action
	if (!mpParameters->mMinimalOutput)
		cout << "Processing I/O file: " << mpParameters->mInputDataFileName << endl;
	Main(fin, om.mOut);
	if (!mpParameters->mMinimalOutput)
		cout << "Result saved in file " << om.GetFullPathFileName() << endl;
}

void TestPartManager::Main(istream& fin, ostream& ofs) {
	ProgressBar pb;

	unsigned instance_counter = 0;
	bool valid_input = true;
	while (!fin.eof() && valid_input) {

		switch (mpParameters->mFileTypeCode) {
			case GRAPH:
#ifdef USEOBABEL
				case MOLECULAR_GRAPH:
#endif
			case STRINGSEQ:
			case SEQUENCE: {
				GraphClass g;
				mpData->SetGraphFromFile(fin, g);
				if (!g.IsEmpty()) {
					vector<SVector> graph_vertex_vector_list;
					mpData->mKernel.GenerateVertexFeatureVector(g, graph_vertex_vector_list);
					//for each vertex, compute margin
					unsigned size = mpParameters->mGraphType == "DIRECTED" ? graph_vertex_vector_list.size() / 2 : graph_vertex_vector_list.size();
					for (unsigned vertex_id = 0; vertex_id < size; ++vertex_id) {
						double margin = mSGDSVMManager.Predict(graph_vertex_vector_list[vertex_id]);
						if (mpParameters->mGraphType == "DIRECTED")
							margin += mSGDSVMManager.Predict(graph_vertex_vector_list[vertex_id + size]);
						ofs << instance_counter << " " << vertex_id << " " << margin << endl;
					}
					if (!mpParameters->mMinimalOutput)
						pb.Count();
					instance_counter++;
				} else
					valid_input = false;
			}
				break;
			case SPARSE_VECTOR:
				throw range_error("ERROR TestPartManager::Main: Cannot process directly sparse vector representations for TEST_PART");
				break;
			default:
				throw range_error("ERROR TestPartManager::Main: File type not recognized: " + mpParameters->mFileType);
		}
	}
}
