#include "PreClusterFilterManager.h"

PreClusterFilterManager::PreClusterFilterManager() :
		BaseManager(0, 0), mThreshold(0) {
}

PreClusterFilterManager::PreClusterFilterManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData), mThreshold(0) {
	Init(apParameters, apData);
}

void PreClusterFilterManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mpData->mKernel.ParametersSetup();
	mMinHashEncoder.Init(apParameters, apData);
}

void PreClusterFilterManager::Exec() {
	OutputManager om("filtered", mpParameters->mDirectoryPath);
	OutputManager omi("filtered_index", mpParameters->mDirectoryPath);

	//Phase 1: build inverse index
	{
		igzstream fin;
		fin.open(mpParameters->mInputDataFileName.c_str());
		if (!fin)
			throw range_error("ERROR2.117: Cannot open file: " + mpParameters->mInputDataFileName);
		cout << "Phase 1: Build inverse index from " << mpParameters->mInputDataFileName << endl;
		OnlineDataProcessing(fin, om.mOut, omi.mOut, "INVERSE_INDEX_UPDATE");
	}

	//Phase 2: find threshold value
	{
		igzstream fin;
		fin.open(mpParameters->mInputDataFileName.c_str());
		if (!mpParameters->mMinimalOutput)
			cout << "Phase 2: Get threshold value to select top " << mpParameters->mNumOutputInstances << " instances from " << mpParameters->mInputDataFileName << endl;
		OnlineDataProcessing(fin, om.mOut, omi.mOut, "GET_FILTER_THRESHOLD");
		cout << "Threshold value:" << mThreshold << endl;
		if (mThreshold==0)
			cerr<<"Warning: Something seems wrong: obtained null threshold"<<endl;
	}

	//Phase 3: use threshold value to filter data
	{
		igzstream fin;
		fin.open(mpParameters->mInputDataFileName.c_str());
		if (!mpParameters->mMinimalOutput)
			cout << "Phase 3: Filter instances from " << mpParameters->mInputDataFileName << endl;
		OnlineDataProcessing(fin, om.mOut, omi.mOut, "FILTER");
		cout << "Filtered instances saved in file " << om.GetFullPathFileName() << endl;
		cout << "Original instances ids saved in file " << omi.GetFullPathFileName() << endl;
	}
}

void PreClusterFilterManager::OnlineDataProcessing(istream& fin, ostream& out, ostream& outi, const string& aActionCode) {
	vector<int> size_list;

	ProgressBar pb;
	bool valid_input = true;
	unsigned instance_counter = 0;
	while (!fin.eof() && valid_input) {
		switch (mpParameters->mFileTypeCode) {
		case GRAPH:
#ifdef USEOBABEL
			case MOLECULAR_GRAPH:
#endif
			case STRINGSEQ:
			case SEQUENCE: {
			vector<GraphClass> g_list(BUFFER_SIZE);
			unsigned i = 0;
			while (i < BUFFER_SIZE && !fin.eof() && valid_input) {
				mpData->SetGraphFromFile(fin, g_list[i]);
				if (g_list[i].IsEmpty()) {
					valid_input = false;
				} else {
					i++;
					instance_counter++;
				}
			}
			// store vectors calculated from buffer elements
			vector<SVector> v_list(i);
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
			for (unsigned j = 0; j < i; j++) {
				SVector x(pow(2, mpParameters->mHashBitSize));
				mpData->mKernel.GenerateFeatureVector(g_list[j], x);
				v_list[j] = x;
				pb.Count();
			}

			if (aActionCode == "INVERSE_INDEX_UPDATE") {
				//for all vector encodings update the inverse index
				for (unsigned j = 0; j < i; j++) {
					unsigned index = instance_counter - i + j;
					vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(v_list[j]);
					mMinHashEncoder.UpdateInverseIndex(signature, index);
				}
			} else if (aActionCode == "GET_FILTER_THRESHOLD") {
				for (unsigned j = 0; j < i; j++) {
					vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(v_list[j]);
					vector<unsigned> signature_size = mMinHashEncoder.ComputeHashSignatureSize(signature);
					//compute statistics
					VectorClass stats_vec;
					for (unsigned t = 0; t < signature_size.size(); t++)
						stats_vec.PushBack(signature_size[t]);
					unsigned median = (unsigned) stats_vec.Median();
					size_list.push_back((int) (-median));
				}
			} else if (aActionCode == "FILTER") {
				//for all vector encodings update the inverse index
				for (unsigned j = 0; j < i; j++) {
					vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(v_list[j]);
					vector<unsigned> signature_size = mMinHashEncoder.ComputeHashSignatureSize(signature);
					//compute statistics
					VectorClass stats_vec;
					for (unsigned t = 0; t < signature_size.size(); t++)
						stats_vec.PushBack(signature_size[t]);
					unsigned median = (unsigned) stats_vec.Median();

					if (median >= mThreshold) {
						unsigned index = instance_counter - i + j;
						outi << index << endl;
						if (mpParameters->mBinaryFormat)
							out << printSparseBinary(v_list[j]);
						else
							out << printSparse(v_list[j]);
					}
				}

			}
		}
			break;
		case SPARSE_VECTOR: {
			bool success_status = false;
			SVector x(pow(2, mpParameters->mHashBitSize));
			if (mpParameters->mBinaryFormat)
				success_status = mpData->SetVectorFromSparseVectorBinaryFile(fin, x);
			else
				success_status = mpData->SetVectorFromSparseVectorAsciiFile(fin, x);
			if (success_status) {
				if (aActionCode == "INVERSE_INDEX_UPDATE") {
					vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(x);
					mMinHashEncoder.UpdateInverseIndex(signature, instance_counter);
				} else if (aActionCode == "GET_FILTER_THRESHOLD") {
					vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(x);
					vector<unsigned> signature_size = mMinHashEncoder.ComputeHashSignatureSize(signature);
					//compute statistics
					VectorClass stats_vec;
					for (unsigned t = 0; t < signature_size.size(); t++)
						stats_vec.PushBack(signature_size[t]);
					unsigned median = (unsigned) stats_vec.Median();
					size_list.push_back((int) (-median));
				} else if (aActionCode == "FILTER") {
					vector<unsigned> signature = mMinHashEncoder.ComputeHashSignature(x);
					vector<unsigned> signature_size = mMinHashEncoder.ComputeHashSignatureSize(signature);
					//compute statistics
					VectorClass stats_vec;
					for (unsigned j = 0; j < signature_size.size(); j++)
						stats_vec.PushBack(signature_size[j]);
					unsigned median = (unsigned) stats_vec.Median();
					if (median >= mThreshold) {
						outi << instance_counter << endl;
						if (mpParameters->mBinaryFormat)
							out << printSparseBinary(x);
						else
							out << printSparse(x);
					}
				}

				instance_counter++;
				pb.Count();
			}
		}
			break;
		default:
			throw range_error("ERROR2.45: file type not recognized: " + mpParameters->mFileType);
		}
	}

	if (aActionCode == "GET_FILTER_THRESHOLD") {
		if (size_list.size() == 0)
			throw range_error("ERROR: Something went wrong: expecting a non-empty list of indices to sort in threshold determination phase!");
		sort(size_list.begin(), size_list.end());
		if (mpParameters->mNumOutputInstances - 1 > (unsigned) size_list.size())
			cerr<<endl<<"Warning: requested more instances than there are. There is no point in filtering."<<endl;
		unsigned effective_index = std::min(mpParameters->mNumOutputInstances - 1, (unsigned) size_list.size());
		unsigned threshold = (unsigned) (-size_list[effective_index]);
		mThreshold = threshold;
		VectorClass stats_vec(size_list);
		cout<<endl<<"Score statistics on entire dataset (with sign reversed):"<<endl;
		stats_vec.OutputStatistics(cout);
	} else 	if (aActionCode == "INVERSE_INDEX_UPDATE") {
		mMinHashEncoder.CleanUpInverseIndex();
	}
}
