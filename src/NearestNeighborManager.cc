#include "NearestNeighborManager.h"

NearestNeighborManager::NearestNeighborManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData),mNearestNeighbor(apParameters, apData) {
}

void NearestNeighborManager::Exec() {
	cout << SEP << endl << "K-Nearest-Neighbor phase" << endl << SEP << endl;
	ProgressBar pb;
	pb.Count();

	OutputManager om_knn("knn", mpParameters->mDirectoryPath);
	OutputManager om_k("knn_kernel_value", mpParameters->mDirectoryPath);
	OutputManager om_t("knn_target_value", mpParameters->mDirectoryPath);
	OutputManager om_f("knn_feature", mpParameters->mDirectoryPath);

	Main(om_knn.mOut, om_k.mOut, om_t.mOut, om_f.mOut);

	cout << "Nearest neighbor list saved in file " << om_knn.GetFullPathFileName() << endl;
	cout << "Nearest neighbor kernel value list saved in file " << om_k.GetFullPathFileName() << endl;
	if (mpParameters->mTargetFileName != "")
		cout << "Nearest neighbor target value list saved in file " << om_t.GetFullPathFileName() << endl;
	cout << "Nearest neighbor feature representation saved in file " << om_f.GetFullPathFileName() << endl;
}

void NearestNeighborManager::Main(ostream& out_knn, ostream& out_k, ostream& out_t, ostream& out_f) {
	cout << "Extracting " << mpParameters->mNumNearestNeighbors << " nearest neighbors in a matrix of size [" << mpData->mRowIndexList.size() << " rows x " << mpData->mColIndexList.size() << " columns]=" << mpData->mRowIndexList.size() * mpData->mColIndexList.size() << " (equivalent) kernel evaluations." << endl;

	ProgressBar ppb;
	for (unsigned i = 0; i < mpData->mRowIndexList.size(); ++i) {
		unsigned ii = mpData->mRowIndexList[i];
		//compute neighbors
		vector<unsigned> neighborhood_list = mNearestNeighbor.ComputeNeighborhood(ii);
		vector<pair<double, unsigned> > extended_neighbor_list(neighborhood_list.size());
		//compute similarity of neighbors wrt root instance
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
		for (unsigned j = 0; j < neighborhood_list.size(); ++j) {
			unsigned jj=neighborhood_list[j];
			double k_ii_jj = mpData->ComputeKernel(ii, jj);
			extended_neighbor_list[j] = make_pair(k_ii_jj, jj);
		}

		//output all info: id of neighbour, similarity of neighbor, target of neighbor
		unsigned effective_size = min((unsigned) extended_neighbor_list.size(), mpParameters->mNumNearestNeighbors);
		out_knn << ii << "    ";
		out_k << ii << "    ";
		if (mpParameters->mTargetFileName != "")
			out_t << ii << "    ";

		for (unsigned t = 0; t < effective_size; ++t) {
			double neighbor_kernel = extended_neighbor_list[t].first;
			unsigned neighbor_id = extended_neighbor_list[t].second;
			out_knn << neighbor_id << " ";
			out_k << neighbor_kernel << " ";
			if (mpParameters->mTargetFileName != "")
				out_t << mpData->mTargetList[neighbor_id] << " ";
		}
		out_knn << endl;
		out_k << endl;
		if (mpParameters->mTargetFileName != "")
			out_t << endl;

		//make a list of <id,similarity value> pairs and sort by id
		//output the neighbor list as a novel feature representation
		vector<pair<unsigned, double> > neighbor_feature_list;
		for (unsigned t = 0; t < effective_size; ++t) {
			double neighbor_kernel = extended_neighbor_list[t].first;
			unsigned neighbor_id = extended_neighbor_list[t].second;
			neighbor_feature_list.push_back(make_pair(neighbor_id, neighbor_kernel));
		}
		sort(neighbor_feature_list.begin(), neighbor_feature_list.end());
		for (unsigned t = 0; t < effective_size; ++t) {
			out_f << neighbor_feature_list[t].first + 1 << ":" << neighbor_feature_list[t].second << " "; //NOTE: add 1 to the instance ID in order to avoid feature with id=0
		}
		out_f << endl;
		ppb.Count();
	}
}
