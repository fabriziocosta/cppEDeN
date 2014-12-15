#include "FeatureManager.h"

FeatureManager::FeatureManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void FeatureManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mpData->mKernel.ParametersSetup();
	mMinHashEncoder.Init(apParameters, apData);
}

void FeatureManager::Exec() {
	if (mpParameters->mSmooth == false) {
		InputOutputManager();
	} else {
		mpData->LoadData(false,false,false);
		FeatureOutputManager();
	}
}

void FeatureManager::InputOutputManager() {
	//output
	OutputManager om("feature", mpParameters->mDirectoryPath);

	//input
	igzstream fin;
	fin.open(mpParameters->mInputDataFileName.c_str());
	if (!fin)
		throw range_error("ERROR2.11: Cannot open file: " + mpParameters->mInputDataFileName);
	//perform online action
	if (!mpParameters->mMinimalOutput)
		cout << "Processing I/O file: " << mpParameters->mInputDataFileName << endl;
	Main(fin, om.mOut);
	if (!mpParameters->mMinimalOutput)
		cout << "Result saved in file " << om.GetFullPathFileName() << endl;
}

void FeatureManager::Main(istream& fin, ostream& ofs) {
	ProgressBar pb;
	unsigned instance_counter = 0;
	bool valid_input = true;
	while (!fin.eof() && valid_input) {
		vector<GraphClass> g_list(BUFFER_SIZE);
		unsigned i = 0;
		while (i < BUFFER_SIZE && !fin.eof() && valid_input) {
			mpData->SetGraphFromFile(fin, g_list[i]);
			if (g_list[i].IsEmpty())
				valid_input = false;
			else
				i++;
		}
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic) ordered
#endif
		for (unsigned j = 0; j < i; j++) {
#ifdef USEMULTITHREAD
#pragma omp ordered
#endif
			{
				SVector x(pow(2, mpParameters->mHashBitSize));
				mpData->mKernel.GenerateFeatureVector(g_list[j], x);
				if (mpParameters->mBinaryFormat) {
					ofs << printSparseBinary(x);
				} else {
					ofs << printSparse(x);
				}
				pb.Count();
				instance_counter++;
			}
		}
	}
}

void FeatureManager::FeatureOutputManager() {
	cout << SEP << endl << "Explicit Feature extraction using K-Nearest-Neighborsmoothing" << endl << SEP << endl;
	ProgressBar pb;
	pb.Count();

	//output
	OutputManager om("feature", mpParameters->mDirectoryPath);

	Main(om.mOut);
	if (!mpParameters->mMinimalOutput)
		cout << "Result saved in file " << om.GetFullPathFileName() << endl;
}

void FeatureManager::Main(ostream& out) {
	if (mpData->mRowIndexList.size() == 0) {
		cout << "No row index list specified. Assuming all " << mpData->Size() << " row indices as valid." << endl;
		for (unsigned i = 0; i < mpData->Size(); ++i)
			mpData->mRowIndexList.push_back(i);
	}
	if (mpData->mColIndexList.size() == 0) {
		cout << "No col index list specified. Assuming all " << mpData->Size() << " col indices as valid." << endl;
		for (unsigned i = 0; i < mpData->Size(); ++i)
			mpData->mColIndexList.push_back(i);
	}

	mMinHashEncoder.ComputeInverseIndex();

	cout << "Computing " << mpParameters->mNumNearestNeighbors << " nearest neighbors smoother for " << mpData->mRowIndexList.size() << " instances from a  database of size: " << mpData->mColIndexList.size() << endl;

	ProgressBar ppb;
	for (unsigned i = 0; i < mpData->mRowIndexList.size(); ++i) {
		unsigned ii = mpData->mRowIndexList[i];
		SVector x = mpData->mVectorList[ii];
		ComputeSmoothedFeatureRepresentation(x);
		if (mpParameters->mBinaryFormat)
			out << printSparseBinary(x);
		else
			out << printSparse(x);
		ppb.Count();
	}
}

void FeatureManager::ComputeSmoothedFeatureRepresentation(SVector& x) {
	vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(x);
	vector<unsigned> approximate_neighborhood = mMinHashEncoder.ComputeApproximateNeighborhood(signature);
	vector<pair<double, unsigned> > rank_list(approximate_neighborhood.size());
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
	for (unsigned j = 0; j < approximate_neighborhood.size(); ++j) {
		unsigned neighbor_id = approximate_neighborhood[j];
		double k = mpData->mKernel.ComputeKernel(mpData->mVectorList[neighbor_id], x);
		rank_list[j] = make_pair(-k, neighbor_id); //use -k to rank from most similar to least similar
	}
	unsigned effective_size = min((unsigned) rank_list.size(), mpParameters->mNumNearestNeighbors);
	partial_sort(rank_list.begin(), rank_list.begin() + effective_size, rank_list.end());
	double ncounter = 0;
	SVector acc(pow(2, mpParameters->mHashBitSize));
	;
	for (unsigned j = 0; j < effective_size; j++) {
		unsigned neighbor_id = rank_list[j].second;
		double k = -rank_list[j].first; //invert value to re-obtain kernel value
		if (k < 1 && k > mpParameters->mSmootherParam) {
			ncounter++;
			SVector& z = mpData->mVectorList[neighbor_id];
			acc += z;
		}
	}
	if (ncounter != 0)
		acc /= ncounter;
	ReHash(acc, 42); //magic number to rebase the neighborhood vector
	x += acc;
	x /= x.norm();
}

//TODO
void FeatureManager::ReHash(SVector& x, unsigned aReHashCode) {
	throw range_error("Error: ReHash with SVector not implemented");
//		vector<pair<int, double> > vec = x.unpack();
//		vector<unsigned> hash_vec(2, 0);
//		SVector z(pow(2,mpParameters->mHashBitSize));;
//		for (unsigned i = 0; i < vec.size(); ++i) {
//			int key = vec[i].first;
//			double val = vec[i].second;
//			hash_vec[0] = (unsigned) (key);
//			hash_vec[1] = aReHashCode;
//			unsigned code = HashFunc(hash_vec, (2 << (mpParameters->mHashBitSize - 1)) - 1);
//			z.set(code, val);
//		}
//		x = z;
}

//------------------------------------------------------------------------------------------------------------------------
FeaturePartManager::FeaturePartManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void FeaturePartManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mpData->mKernel.ParametersSetup();
}

void FeaturePartManager::Exec() {
	InputOutputManager();
}

void FeaturePartManager::InputOutputManager() {
	//output
	OutputManager om("feature_part", mpParameters->mDirectoryPath);

	//input
	igzstream fin;
	fin.open(mpParameters->mInputDataFileName.c_str());
	if (!fin)
		throw range_error("ERROR2.11: Cannot open file: " + mpParameters->mInputDataFileName);
	//perform online action
	if (!mpParameters->mMinimalOutput)
		cout << "Processing I/O file: " << mpParameters->mInputDataFileName << endl;
	Main(fin, om.mOut);
	if (!mpParameters->mMinimalOutput)
		cout << "Result saved in file " << om.GetFullPathFileName() << endl;
}

void FeaturePartManager::Main(istream& fin, ostream& ofs) {
	ProgressBar pb;
	unsigned instance_counter = 0;
	bool valid_input = true;
	while (!fin.eof() && valid_input) {
		vector<GraphClass> g_list(BUFFER_SIZE);
		unsigned i = 0;
		while (i < BUFFER_SIZE && !fin.eof() && valid_input) {
			mpData->SetGraphFromFile(fin, g_list[i]);
			if (g_list[i].IsEmpty())
				valid_input = false;
			else {
				i++;
				pb.Count();
				instance_counter++;
			}
		}
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic,100) ordered
#endif
		for (unsigned j = 0; j < i; j++) {
			vector<SVector> graph_vertex_vector_list;
			mpData->mKernel.GenerateVertexFeatureVector(g_list[j], graph_vertex_vector_list);
			unsigned size = mpParameters->mGraphType == "DIRECTED" ? graph_vertex_vector_list.size() / 2 : graph_vertex_vector_list.size();
#ifdef USEMULTITHREAD
#pragma omp ordered
#endif
			{
				for (unsigned vertex_id = 0; vertex_id < size; ++vertex_id) {
					SVector x = graph_vertex_vector_list[vertex_id];
					if (mpParameters->mGraphType == "DIRECTED")
						x += graph_vertex_vector_list[vertex_id + size];
					ofs << instance_counter -i +j << " " << vertex_id << " ";
					ofs << printSparse(x);
				}
			}
		}
	}
}

//------------------------------------------------------------------------------------------------------------------------
FeatureScaledManager::FeatureScaledManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
	Init(apParameters, apData);
}

void FeatureScaledManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mpData->mKernel.ParametersSetup();
	mSGDSVM.Init(mpParameters, mpData);
}

void FeatureScaledManager::LoadModel() {
	string filename = mpParameters->mModelFileName + mpParameters->mSuffix;
	ifstream ifs;
	ifs.open(filename.c_str());
	if (!ifs)
		throw range_error("ERROR2.23: Cannot open file:" + filename);
	if (!mpParameters->mMinimalOutput)
		cout << endl << "Loading model file: " << filename << endl;
	mSGDSVM.Load(ifs);
	if (!mpParameters->mMinimalOutput) {
		mSGDSVM.OutputModelInfo();
		cout << endl;
	}
}

void FeatureScaledManager::Exec() {
	LoadModel();
	InputOutputManager();
}

void FeatureScaledManager::InputOutputManager() {
	//output
	OutputManager om("feature_scaled", mpParameters->mDirectoryPath);

	//input
	igzstream fin;
	fin.open(mpParameters->mInputDataFileName.c_str());
	if (!fin)
		throw range_error("ERROR2.11: Cannot open file: " + mpParameters->mInputDataFileName);
	//perform online action
	if (!mpParameters->mMinimalOutput)
		cout << "Processing I/O file: " << mpParameters->mInputDataFileName << endl;
	Main(fin, om.mOut);
	if (!mpParameters->mMinimalOutput)
		cout << "Result saved in file " << om.GetFullPathFileName() << endl;
}

void FeatureScaledManager::Main(istream& fin, ostream& ofs) {
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
			vector<GraphClass> g_list(BUFFER_SIZE);
			unsigned i = 0;
			while (i < BUFFER_SIZE && !fin.eof() && valid_input) {
				mpData->SetGraphFromFile(fin, g_list[i]);
				if (g_list[i].IsEmpty())
					valid_input = false;
				else
					i++;
			}
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic) ordered
#endif
			for (unsigned j = 0; j < i; j++) {
				SVector x(pow(2, mpParameters->mHashBitSize));
				mpData->mKernel.GenerateFeatureVector(g_list[j], x);
				mSGDSVM.VectorElementwiseProductWithModel(x);
#ifdef USEMULTITHREAD
#pragma omp ordered
#endif
				{
					if (mpParameters->mBinaryFormat) {
						ofs << printSparseBinary(x);
					} else {
						ofs << printSparse(x);
					}
					pb.Count();
					instance_counter++;
				}
			}
		}
			break;
		case SPARSE_VECTOR: {
			SVector x(pow(2, mpParameters->mHashBitSize));
			;
			if (mpParameters->mBinaryFormat)
				mpData->SetVectorFromSparseVectorBinaryFile(fin, x);
			else
				mpData->SetVectorFromSparseVectorAsciiFile(fin, x);
			mSGDSVM.VectorElementwiseProductWithModel(x);
			if (mpParameters->mBinaryFormat) {
				ofs << printSparseBinary(x);
			} else {
				ofs << printSparse(x);
			}
			pb.Count();
			instance_counter++;
		}
			break;
		default:
			throw range_error("ERROR2.45: file type not recognized: " + mpParameters->mFileType);
		}
	}
}
