#include "ClusterManager.h"
#include <ostream>

ClusterManager::ClusterManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData), mNearestNeighbor(apParameters, apData) {
}

double ClusterManager::ComputeDensity(unsigned aID) {
	vector<unsigned> neighborhood = mNearestNeighbor.ComputeNeighborhood(aID);
	double density = ComputeDensity(aID, neighborhood);
	return density;
}

double ClusterManager::ComputeDensity(unsigned aID, vector<unsigned>& aNeighborhood) {
	double density = 0;
	//compute kernel pairs between i and all elements in aApproximateNeighborhood
	for (unsigned j = 0; j < aNeighborhood.size(); j++) {
		unsigned u = aID;
		unsigned v = aNeighborhood[j];
		if (u != v) {
			double k_uv = mpData->ComputeKernel(u, v);
			if (mpParameters->mSharedNeighborhood) //if we use the shared neighborhood weighting than we multiply the similarity by a corrective factor given by the fraction of shared neighbors between the two instances
				k_uv *= mNearestNeighbor.ComputeSharedNeighborhoodSimilarity(u, v);
			density += k_uv;
		}
	}
	density = density / (aNeighborhood.size() - 1); //-1 as the similarity of an instance to itself (which is the closest of the elements in the neighborhood)  is excluded
	return density;
}

double ClusterManager::ComputeDensityStandardDeviation(unsigned aID, double aAverageDensity, vector<unsigned>& aNeighborhood) {
	double variance = 0;
	//compute kernel pairs between i and all elements in aApproximateNeighborhood
	for (unsigned j = 0; j < aNeighborhood.size(); j++) {
		unsigned u = aID;
		unsigned v = aNeighborhood[j];
		if (u != v) {
			double k_uv = mpData->ComputeKernel(u, v);
			if (mpParameters->mSharedNeighborhood) //if we use the shared neighborhood weighting than we multiply the similarity by a corrective factor given by the fraction of shared neighbors between the two instances
				k_uv *= mNearestNeighbor.ComputeSharedNeighborhoodSimilarity(u, v);
			variance += (k_uv - aAverageDensity) * (k_uv - aAverageDensity);
		}
	}
	variance = variance / (aNeighborhood.size() - 1); //-1 as the similarity of an instance to itself (which is the closest of the elements in the neighborhood)  is excluded
	return sqrt(variance);
}

void ClusterManager::ComputeDensity(vector<unsigned>& aSelectedIndexList, vector<pair<double, unsigned> >& oDensityList) {
	cout << "Computing density information for random sample of " << aSelectedIndexList.size() << " instances" << endl; ////

	ProgressBar progress_bar;

	oDensityList.clear();
	oDensityList.resize(aSelectedIndexList.size());

#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
	for (unsigned j = 0; j < aSelectedIndexList.size(); ++j) {
		unsigned i = aSelectedIndexList[j];
		double density = ComputeDensity(i);
		oDensityList[j] = make_pair(-density, i);
		progress_bar.Count();
	}
}

pair<double, double> ClusterManager::ComputeLocalCentrality(unsigned aID, vector<vector<unsigned> >& aApproximateNeighborhoodList) {
	double centrality = ComputeDensity(aID, aApproximateNeighborhoodList[aID]);
	double std = ComputeDensityStandardDeviation(aID, centrality, aApproximateNeighborhoodList[aID]);
	return make_pair(centrality, std);
}

//---------------------------------------------------------------------------------------------------
DensityClusterManager::DensityClusterManager(Parameters* apParameters, Data* apData) :
		ClusterManager(apParameters, apData) {
}

void DensityClusterManager::Exec() {
	ProgressBar pb;

	OutputManager om("fast_cluster", mpParameters->mDirectoryPath);
	OutputManager om_sim("fast_cluster_inner_similarity", mpParameters->mDirectoryPath);
	OutputManager om_ecc("fast_cluster_eccess_neighbors", mpParameters->mDirectoryPath);

	DenseCluster(om.mOut, om_sim.mOut, om_ecc.mOut);

	cout << endl << "Dense center cluster results written in file " << om.GetFullPathFileName() << endl;
	cout << endl << "Similarity matrix results written in file " << om_sim.GetFullPathFileName() << endl;
	cout << endl << "Eccess neighbors for dense centers written in file " << om_ecc.GetFullPathFileName() << endl;
	pb.Count();
	cout << "Total clustering time:" << endl;
}

void DensityClusterManager::DenseCluster(ostream& out, ostream& out_sim, ostream& out_ecc) {
	vector<unsigned> density_center_list;
	density_center_list = ComputeMinimallyOverlappingHighDensityCenterList(mpParameters->mSampleSize, mpParameters->mFractionCenterScan, mpParameters->mNeighborhoodIntersectionSize);
	cout << "Compute neighborhood for selected " << density_center_list.size() << " cluster centers." << endl;
	{
		ProgressBar progress_bar(1);
		for (unsigned idc = 0; idc < density_center_list.size(); ++idc) {
			unsigned id = density_center_list[idc];
			vector<unsigned> neighborhood = mNearestNeighbor.ComputeNeighborhood(id);
			progress_bar.Count();
			OutputDenseClusterNeighbors(out, neighborhood);
			OutputDenseClusterNeighborsSimilarity(out_sim, neighborhood);

			//output excess neighborhood
			if (mpParameters->mNoNeighborhoodCache == false)
				mNearestNeighbor.CacheReset();
			unsigned num_nearest_neighbors = mpParameters->mNumNearestNeighbors;
			mpParameters->mNumNearestNeighbors = mpParameters->mEccessNeighborSizeFactor * mpParameters->mNumNearestNeighbors;
			vector<unsigned> neighborhood_eccess;
			neighborhood_eccess = mNearestNeighbor.ComputeNeighborhood(id);
			OutputDenseClusterNeighbors(out_ecc, neighborhood_eccess);
			mpParameters->mNumNearestNeighbors = num_nearest_neighbors;
		}
	}
}

vector<unsigned> DensityClusterManager::ComputeMinimallyOverlappingHighDensityCenterList(unsigned aSampleSize, double aFractionCenterScan, unsigned aMaxIntersectionSize) {
	//compute density estimate for random fraction of instances in dataset
	vector<unsigned> result;
	vector<unsigned> high_density_center_list = ComputeHighDensityCenterList(aSampleSize, aFractionCenterScan);
	set<unsigned> active_neighborhood;
	ProgressBar progress_bar(1);
	cout << "Computing minimally overlapping high density center list for up to " << aSampleSize << " centers." << endl; ////
	for (unsigned i = 0; i < high_density_center_list.size() && result.size() < aSampleSize; i++) {
		unsigned id = high_density_center_list[i];
		vector<unsigned> neighborhood = mNearestNeighbor.ComputeNeighborhood(id);
		set<unsigned> neighborhood_set;
		neighborhood_set.insert(neighborhood.begin(), neighborhood.end());
		set<unsigned> intersection;
		set_intersection(active_neighborhood.begin(), active_neighborhood.end(), neighborhood_set.begin(), neighborhood_set.end(), inserter(intersection, intersection.begin()));
		if (i == 0 || intersection.size() <= aMaxIntersectionSize) { //if the intersection between the neighborhood of the current center and the union of all active neighborhoods is less than a defined constant (eg. 0) then accept the new center in the active set
			active_neighborhood.insert(neighborhood.begin(), neighborhood.end());
			result.push_back(id);
			progress_bar.Count();
		}
	}
	return result;
}

vector<unsigned> DensityClusterManager::SampleID(double aFractionCenterScan) {
	unsigned data_size = mpData->mRowIndexList.size() > 0 ? mpData->mRowIndexList.size() : mpData->Size();
	unsigned effective_size = floor(data_size * aFractionCenterScan);
	vector<unsigned> selected_index_list;
	//select either all the available centers
	if (aFractionCenterScan == 1) {
		selected_index_list.insert(selected_index_list.begin(), mpData->mRowIndexList.begin(), mpData->mRowIndexList.end());
	} else {
		//or select a random subset of candidate centers of size effective_size
		vector<unsigned> index_list = mpData->mRowIndexList;
		for (unsigned i = 0; i < data_size; ++i) {
			unsigned j = randomUnsigned(data_size);
			swap(index_list[i], index_list[j]);
		}
		for (unsigned i = 0; i < effective_size; ++i)
			selected_index_list.push_back(index_list[i]);
	}
	return selected_index_list;
}

vector<unsigned> DensityClusterManager::ComputeHighDensityCenterList(unsigned aSampleSize, double aFractionCenterScan) {
	//compute density estimate for random fraction of instances in dataset
	if (aFractionCenterScan == 1)
		cout << "Selecting all the " << mpData->mRowIndexList.size() << " instances" << endl;
	else
		cout << "Selecting a random " << aFractionCenterScan * 100 << " % of " << mpData->mRowIndexList.size() << " instances" << endl;
	//random selection of instances
	vector<unsigned> selected_index_list = SampleID(aFractionCenterScan);

	//compute density estimate
	vector<pair<double, unsigned> > density_list;
	ComputeDensity(selected_index_list, density_list);
	//select non overlapping centers in decreasing order of density
	sort(density_list.begin(), density_list.end());

	vector<unsigned> result;
	for (unsigned i = 0; i < density_list.size() && result.size() < aSampleSize; i++) {
		unsigned id = density_list[i].second;
		result.push_back(id);
	}
	return result;
}

void DensityClusterManager::OutputDenseClusterNeighbors(ostream& out, vector<unsigned>& aNeighborhood) {
	for (unsigned i = 0; i < aNeighborhood.size(); i++) {
		unsigned nid = aNeighborhood[i];
		out << nid << " ";
	}
	out << endl;
}

void DensityClusterManager::OutputDenseClusterNeighborsSimilarity(ostream& out, vector<unsigned>& aNeighborhood) {
	for (unsigned i = 0; i < aNeighborhood.size(); i++) {
		unsigned nid = aNeighborhood[i];
		for (unsigned j = i + 1; j < aNeighborhood.size(); j++) {
			unsigned njd = aNeighborhood[j];
			double k = mpData->ComputeKernel(nid, njd);
			out << nid << ":" << njd << ":" << k << " ";
		}
		out << " ";
	}
	out << endl;
}

//---------------------------------------------------------------------------------------------------
ConnectedDensityClusterManager::ConnectedDensityClusterManager(Parameters* apParameters, Data* apData) :
		DensityClusterManager(apParameters, apData) {
}

void ConnectedDensityClusterManager::Exec() {
	ProgressBar pb;

	OutputManager om("cluster", mpParameters->mDirectoryPath);
	OutputManager om_sim("cluster_inner_similarity", mpParameters->mDirectoryPath);
	DenseCluster(om.mOut, om_sim.mOut);

	cout << endl << "Connected dense clustering results written in file " << om.GetFullPathFileName() << endl;
	cout << endl << "Connected dense clustering inner similarity written in file " << om_sim.GetFullPathFileName() << endl;
	pb.Count();
	cout << "Total clustering time:" << endl;
}

void ConnectedDensityClusterManager::DenseCluster(ostream& out, ostream& out_sim) {
	vector<unsigned> density_center_list = ComputeHighDensityCenterList(mpParameters->mSampleSize, mpParameters->mFractionCenterScan);
	cout << "Compute neighborhood overlap for selected " << density_center_list.size() << " top dense cluster centers." << endl;
	vector<vector<unsigned> > adjacency_list(mpData->Size());
	{
		ProgressBar pb(1);
		for (unsigned i = 0; i < density_center_list.size(); ++i) {
			unsigned id_i = density_center_list[i];
			for (unsigned j = 0; j < i; ++j) {
				unsigned id_j = density_center_list[j];
				unsigned intersection_size = mNearestNeighbor.ComputeNeighborhoodIntersection(id_i, id_j);
				if (intersection_size >= mpParameters->mNeighborhoodIntersectionSize) {
					adjacency_list[id_i].push_back(id_j);
					adjacency_list[id_j].push_back(id_i);
				}
			}
			pb.Count();
		}
	}

	//find connected components
	vector<vector<unsigned> > connected_components = ConnectedComponents(adjacency_list, density_center_list);
	//expand each center with its neighborhood and put everything in a set (to remove duplicates)
	//output clusters as the list of the merged neighborhoods
	for (unsigned i = 0; i < connected_components.size(); ++i) {
		vector<unsigned>& component = connected_components[i];
		if (component.size() > 1) { //NOTE: here we can require that only components of large size be output
			set<unsigned> cluster;
			for (unsigned j = 0; j < component.size(); j++) {
				unsigned id = component[j];
				vector<unsigned> neighborhood = mNearestNeighbor.ComputeNeighborhood(id);
				cluster.insert(neighborhood.begin(), neighborhood.end());
			}
			//output cluster id
			for (set<unsigned>::const_iterator it = cluster.begin(); it != cluster.end(); ++it) {
				out << (*it) << " ";
			}
			out << endl;

			//output inner similarity
			for (set<unsigned>::const_iterator it = cluster.begin(); it != cluster.end(); ++it) {
				unsigned id_i = (*it);
				for (set<unsigned>::const_iterator jt = cluster.begin(); jt != cluster.end(); ++jt) {
					unsigned id_j = (*jt);
					if (id_i < id_j) {
						double k = mpData->ComputeKernel(id_i, id_j);
						out_sim << id_i << ":" << id_j << ":" << k << " ";
					}
				}
			}
			out_sim << endl;
		}
	}
}

vector<vector<unsigned> > ConnectedDensityClusterManager::ConnectedComponents(vector<vector<unsigned> >& aAdjacencyList, vector<unsigned>& aElementList) {
	vector<vector<unsigned> > result;
	map<int, bool> already_explored;
	//for all element of the init_list if they have not already been visited perform BF, at the end return set of visited elements
	for (unsigned t = 0; t < aElementList.size(); ++t) {
		unsigned id = aElementList[t];
		if (already_explored[t] == true) { // do nothing
		} else {
			vector<unsigned> component;
			component.push_back(id);
			already_explored[id] = true;
			queue<int> q;
			q.push(id);
			while (q.empty() == false) {
				int u = q.front();
				for (unsigned j = 0; j < aAdjacencyList[u].size(); j++) {
					int v = aAdjacencyList[u][j];
					if (already_explored[v] == true) { //do nothing, ignore the vertex
					} else {
						component.push_back(v);
						already_explored[v] = true;
						q.push(v);
					}
				}
				q.pop();
			}
			result.push_back(component);
		}
	}
	return result;
}

//---------------------------------------------------------------------------------------------------
KQuickShiftClusterManager::KQuickShiftClusterManager(Parameters* apParameters, Data* apData) :
		ClusterManager(apParameters, apData) {
}

void KQuickShiftClusterManager::Exec() {
	ProgressBar pb;

	OutputManager om("kquickshift_cluster", mpParameters->mDirectoryPath);
	OutputManager om_sp("kquickshift_cluster_parent_relation", mpParameters->mDirectoryPath);
	OutputManager om_ext("kquickshift_cluster_neighbor_extension", mpParameters->mDirectoryPath);
	OutputManager om_sim("kquickshift_cluster_inner_similarity", mpParameters->mDirectoryPath);

	KQuickShiftCluster(om.mOut, om_sp.mOut, om_ext.mOut, om_sim.mOut);

	cout << endl << "Clustering results written in file " << om.GetFullPathFileName() << endl;
	cout << endl << "Parent relations written in file " << om_sp.GetFullPathFileName() << endl;
	cout << endl << "Nearest Neighbor Extension Clustering results written in file " << om_ext.GetFullPathFileName() << endl;
	cout << endl << "Similarity results for each cluster written in file " << om_sim.GetFullPathFileName() << endl;

	pb.Count();
	cout << "Total clustering time:" << endl;
}

void KQuickShiftClusterManager::KQuickShiftCluster(ostream& out, ostream& out_parent, ostream& out_ext, ostream& out_sim) {
//compute approximate neighborhood and cache it
	vector<vector<unsigned> > approximate_neighborhood_list(mpData->Size());
	if (mpParameters->mUseApproximate)
		cout << "Compute approximate neighborhood for " << mpData->mRowIndexList.size() << " instances." << endl;
	else
		cout << "Compute neighborhood for " << mpData->mRowIndexList.size() << " instances." << endl;

	{
		ProgressBar progress_bar;
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
		for (unsigned ii = 0; ii < mpData->mRowIndexList.size(); ii++) {
			unsigned i = mpData->mRowIndexList[ii];
			vector<unsigned> approximate_neighborhood = mNearestNeighbor.ComputeNeighborhood(i);
#ifdef USEMULTITHREAD
#pragma omp critical
#endif
			{
				approximate_neighborhood_list[i] = approximate_neighborhood;
				progress_bar.Count();
			}
		}
	}

//compute local centrality and cache it
	vector<double> local_centrality_list(mpData->Size());
	vector<double> local_centrality_std_list(mpData->Size());
	cout << "Compute local centality for " << mpData->mRowIndexList.size() << " instances." << endl;
	{
		ProgressBar progress_bar;
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
		for (unsigned ii = 0; ii < mpData->mRowIndexList.size(); ii++) {
			unsigned i = mpData->mRowIndexList[ii];
			pair<double, double> stats = ComputeLocalCentrality(i, approximate_neighborhood_list);
#ifdef USEMULTITHREAD
#pragma omp critical
#endif
			{
				double local_centralty = stats.first;
				double local_centrality_std = stats.second;
				local_centrality_list[i] = local_centralty;
				local_centrality_std_list[i] = local_centrality_std;
				progress_bar.Count();
			}
		}
	}

//find closest more central neighbor and store parent pointer
	vector<int> parent_pointer_list(mpData->Size(), -1);
	cout << "Compute parent pointer for " << mpData->mRowIndexList.size() << " instances." << endl;
	{
		ProgressBar progress_bar;
		for (unsigned ii = 0; ii < mpData->mRowIndexList.size(); ii++) {
			unsigned i = mpData->mRowIndexList[ii];
			unsigned parent_id = i;
			double local_centrality = local_centrality_list[i];
			for (unsigned j = 0; j < approximate_neighborhood_list[i].size(); j++) {
				unsigned neighbor_id = approximate_neighborhood_list[i][j];
				if (local_centrality_list[neighbor_id] > local_centrality) {
					parent_id = neighbor_id;
					break;
				}
			}
			parent_pointer_list[i] = parent_id;
			out_parent << i << " " << parent_id << endl;

			progress_bar.Count();
		}
	}

//invert parent  relationship and compute child relationship
	vector<vector<unsigned> > child_pointer_list(mpData->Size());
	for (unsigned ii = 0; ii < mpData->mRowIndexList.size(); ii++) {
		unsigned i = mpData->mRowIndexList[ii];
		int parent_id = parent_pointer_list[i];
		if (parent_id != -1) //if it is not a blacklisted parent
			if (parent_id != (int) i) { //if it is not a pointer to self
				child_pointer_list[parent_id].push_back(i);
			}
	}

//for every instance compare average neighbor similarity and remove all parent edges such that parent has similarity below x normalized value
	vector<unsigned> root_pointer_list;
	vector<int> filtered_parent_pointer_list(mpData->Size());

	for (unsigned ii = 0; ii < mpData->mRowIndexList.size(); ii++) {
		unsigned i = mpData->mRowIndexList[ii];
		int parent_id = parent_pointer_list[i];
		if (parent_id == -1) {
		} //skip any blacklisted instance
		else {
			double average = local_centrality_list[i];
			double standard_deviation = local_centrality_std_list[i];
			double k_parent = mpData->ComputeKernel(i, parent_id);

			if (parent_id == (int) i) { //self loops (i.e. roots) are preserved
				filtered_parent_pointer_list[i] = i;
				root_pointer_list.push_back(i);
			} else if (child_pointer_list.size() == 0) { //leaf are preserved if the parent similarity is compatible with the average similarity around the parent
				double parent_average = local_centrality_list[parent_id];
				double parent_standard_deviation = local_centrality_std_list[parent_id];
				if (fabs(k_parent - parent_average) / parent_standard_deviation > 1 / mpParameters->mClusterThreshold) {
					filtered_parent_pointer_list[i] = parent_id;
				} else {
					filtered_parent_pointer_list[i] = i;
					root_pointer_list.push_back(i);
				}
			} else if (fabs(k_parent - average) / standard_deviation > 1 / mpParameters->mClusterThreshold) {
				filtered_parent_pointer_list[i] = parent_id;
			} else { //else make a new root
				filtered_parent_pointer_list[i] = i;
				root_pointer_list.push_back(i);
			}
		}
	}

//invert parent  relationship and compute filtered child relationship
	vector<vector<unsigned> > filtered_child_pointer_list(mpData->Size());
	for (unsigned i = 0; i < mpData->Size(); i++) {
		int parent_id = filtered_parent_pointer_list[i];
		if (parent_id != -1) //if it is not a blacklisted parent
			if (parent_id != (int) i) //if it is not a pointer to self
				filtered_child_pointer_list[parent_id].push_back(i);
	}

//for each root visit the dominated tree and output all instances as members of the cluster
	for (unsigned i = 0; i < root_pointer_list.size(); ++i) {
		unsigned root_id = root_pointer_list[i];
		vector<unsigned> dominated_tree;
		TreeVisit(root_id, filtered_child_pointer_list, dominated_tree);
		//for each instance compute the average similarity wrt the instances in the cluster and sort by that
		vector<pair<double, unsigned> > element_list;
		for (unsigned j = 0; j < dominated_tree.size(); ++j) {
			double k_sum = 0;
			unsigned id_j = dominated_tree[j];
			for (unsigned m = 0; m < dominated_tree.size(); ++m) {
				unsigned id_m = dominated_tree[m];
				if (j != m)
					k_sum += mpData->ComputeKernel(id_j, id_m);
			}
			element_list.push_back(make_pair(-k_sum, id_j));
		}
		sort(element_list.begin(), element_list.end());
		//output instances sorted by average similarity
		for (unsigned j = 0; j < element_list.size(); ++j) {
			out << element_list[j].second << " ";
		}
		out << endl;
	}

//for each elements in the kquickshift clusters, find knn and output them
	for (unsigned i = 0; i < root_pointer_list.size(); ++i) {
		set<unsigned> cluster_id_list;

		unsigned root_id = root_pointer_list[i];
		cluster_id_list.insert(approximate_neighborhood_list[root_id].begin(), approximate_neighborhood_list[root_id].end());

		vector<unsigned> dominated_tree;
		TreeVisit(root_id, filtered_child_pointer_list, dominated_tree);
		//add unique copy of nearest neighbor
		for (unsigned j = 0; j < dominated_tree.size(); ++j) {
			unsigned id = dominated_tree[j];
			cluster_id_list.insert(approximate_neighborhood_list[id].begin(), approximate_neighborhood_list[id].end());
		}
		vector<unsigned> id_list;
		for (set<unsigned>::iterator it = cluster_id_list.begin(); it != cluster_id_list.end(); ++it)
			id_list.push_back(*it);
		//sort by internal similarity
		vector<pair<double, unsigned> > element_list;
		for (unsigned j = 0; j < id_list.size(); ++j) {
			double k_sum = 0;
			unsigned id_j = id_list[j];
			for (unsigned m = 0; m < id_list.size(); ++m) {
				unsigned id_m = id_list[m];
				if (j != m)
					k_sum += mpData->ComputeKernel(id_j, id_m);
			}
			element_list.push_back(make_pair(-k_sum, id_j));
		}
		sort(element_list.begin(), element_list.end());
		//output instances sorted by average similarity
		for (unsigned j = 0; j < element_list.size(); ++j) {
			out_ext << element_list[j].second << " ";
		}
		out_ext << endl;
	}

	//for each elements in the kquickshift clusters, output their pairwise similarity
	for (unsigned i = 0; i < root_pointer_list.size(); ++i) {
		unsigned root_id = root_pointer_list[i];
		vector<unsigned> dominated_tree;
		TreeVisit(root_id, filtered_child_pointer_list, dominated_tree);
		if (dominated_tree.size() == 1) {
			out_sim << root_id << ":" << root_id << ":1" << endl;
		} else {
			for (unsigned j = 0; j < dominated_tree.size(); ++j) {
				unsigned idx = dominated_tree[j];
				for (unsigned z = j + 1; z < dominated_tree.size(); ++z) {
					unsigned idz = dominated_tree[z];
					double k = mpData->ComputeKernel(idx, idz);
					out_sim << idx << ":" << idz << ":" << k << " ";
				}
			}
			out_sim << endl;
		}
	}
}

void KQuickShiftClusterManager::TreeVisit(unsigned aID, vector<vector<unsigned> >& aChildPointerList, vector<unsigned>& oDominatedTree) {
	oDominatedTree.push_back(aID);
	for (unsigned i = 0; i < aChildPointerList[aID].size(); ++i) {
		unsigned child_id = aChildPointerList[aID][i];
		TreeVisit(child_id, aChildPointerList, oDominatedTree);
	}
}

//---------------------------------------------------------------------------------------------------
AccuracyClusterManager::AccuracyClusterManager(Parameters* apParameters, Data* apData) :
		ClusterManager(apParameters, apData) {
}

void AccuracyClusterManager::Exec() {
	ProgressBar pb;

	OutputManager oma("nn_accuracy", mpParameters->mDirectoryPath);
	Accuracy(oma.mOut);
	cout << endl << "Accuracy results written in file " << oma.GetFullPathFileName() << endl;

	OutputManager oms("signature_size", mpParameters->mDirectoryPath);
	SignatureSize(oms.mOut);
	cout << endl << "Signature size results written in file " << oms.GetFullPathFileName() << endl;

	OutputManager omd("density", mpParameters->mDirectoryPath);
	Density(omd.mOut);
	cout << endl << "Density results written in file " << omd.GetFullPathFileName() << endl;

	//NOTE: remove comments to extract the median number of approximate nearest neighbors
	//OutputManager omas("approximate_nn_statistics", mpParameters->mDirectoryPath);
	//ApproximateNNStatistics(omas.mOut);
	//cout << endl << "Approximate NN statistics results written in file " << omas.GetFullPathFileName() << endl;

	pb.Count();
	cout << "Total time:" << endl;
}

void AccuracyClusterManager::ApproximateNNStatistics(ostream& out) {
	cout << "Compute signature size statistics" << endl;
	ProgressBar progress_bar;
	for (unsigned i = 0; i < mpData->Size(); ++i) {
		vector<unsigned> signature = mNearestNeighbor.mMinHashEncoder.ComputeHashSignature(i);
		vector<unsigned> approximate_NN = mNearestNeighbor.mMinHashEncoder.ComputeApproximateNeighborhood(signature);
		out << i << " approx_num_neighbors: " << approximate_NN.size() << endl;
	}
}

void AccuracyClusterManager::SignatureSize(ostream& out) {
	cout << "Compute signature size statistics" << endl;
	ProgressBar progress_bar;

	for (unsigned i = 0; i < mpData->Size(); ++i) {
		//extract signature size
		vector<unsigned> signature = mNearestNeighbor.mMinHashEncoder.ComputeHashSignature(i);
		vector<unsigned> signature_size = mNearestNeighbor.mMinHashEncoder.ComputeHashSignatureSize(signature);
		//compute statistics
		VectorClass stats_vec;
		for (unsigned j = 0; j < signature_size.size(); j++)
			stats_vec.PushBack(signature_size[j]);
		out << i << " ";
		stats_vec.OutputStatistics(out);
		out << endl;
	}
}

void AccuracyClusterManager::Density(ostream& out) {
	cout << "Compute density" << endl;
	ProgressBar progress_bar;

	vector<unsigned> selected_index_list(mpData->Size());
	iota(selected_index_list.begin(), selected_index_list.end(), 0);

	//compute density estimate
	vector<pair<double, unsigned> > density_list;
	ComputeDensity(selected_index_list, density_list);

	for (unsigned i = 0; i < density_list.size(); ++i)
		out << -density_list[i].first << endl;
}

void AccuracyClusterManager::Accuracy(ostream& out) {
	cout << "Compute neighborhood accuracy" << endl;
	ProgressBar progress_bar;
	double cum = 0;
	unsigned effective_neighborhood_size = min(mpParameters->mNumNearestNeighbors, (unsigned) mpData->Size());
	for (unsigned u = 0; u < mpData->Size(); ++u) {
		progress_bar.Count();
		vector<unsigned> approximate_neighborhood = mNearestNeighbor.ComputeApproximateNeighborhood(u);
		set<unsigned> approximate_neighborhood_set;
		approximate_neighborhood_set.insert(approximate_neighborhood.begin(), approximate_neighborhood.end());
		vector<unsigned> true_neighborhood = mNearestNeighbor.ComputeTrueNeighborhood(u);
		set<unsigned> true_neighborhood_set;
		true_neighborhood_set.insert(true_neighborhood.begin(), true_neighborhood.end());
		set<unsigned> intersection;
		set_intersection(approximate_neighborhood_set.begin(), approximate_neighborhood_set.end(), true_neighborhood_set.begin(), true_neighborhood_set.end(), inserter(intersection, intersection.begin()));
		double val = (double) intersection.size() / effective_neighborhood_size;
		out << val << endl;
		cum += val;
	}
	cout << endl << "Accuracy: " << cum / mpData->Size() << endl;
}
