#include "GramMatrixManager.h"

GramMatrixManager::GramMatrixManager(Parameters* apParameters, Data* apData) :
		BaseManager(apParameters, apData) {
}

void GramMatrixManager::Load() {
	mpData->LoadData(true,true,false);
}

void GramMatrixManager::Exec() {
	Load();
	ComputeAndOutput();
}

void GramMatrixManager::ComputeAndOutput() {
	cout << SEP << endl << "Gram matrix phase" << endl << SEP << endl;
	ProgressBar pb;
	pb.Count();
	OutputManager om( "matrix", mpParameters->mDirectoryPath);
	ComputeAndOutput(om.mOut);
	cout << "Gram matrix saved in file " << om.GetFullPathFileName() << endl;
}

void GramMatrixManager::ComputeAndOutput(ostream& out) {
	if (!mpParameters->mMinimalOutput) {
		cout << "Computing Gram matrix for [" << mpData->mRowIndexList.size() << " rows x " << mpData->mColIndexList.size() << " columns]=" << mpData->mRowIndexList.size() * mpData->mColIndexList.size() << " pairs of instances." << endl;
	}

	double k_row_row = 0;
	double k_col_col = 0;
	double k_row_col = 0;
	double avg_k_row_row;
	double avg_k_col_col;
	double avg_k_row_col;
	{
		ProgressBar ppb;
		unsigned counter = 0;
		for (unsigned i = 0; i < mpData->mRowIndexList.size(); ++i) {
			unsigned ii = mpData->mRowIndexList[i];
			vector<double> k_val_list(mpData->mColIndexList.size());
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
			for (unsigned j = 0; j < mpData->mColIndexList.size(); ++j) {
				unsigned jj = mpData->mColIndexList[j];
				k_val_list[j] = mpData->ComputeKernel(ii, jj);
			}
			for (unsigned t = 0; t < k_val_list.size(); ++t) {
				out << k_val_list[t] << " ";
				k_row_col += k_val_list[t];
				counter++;
			}
			out << endl;
			ppb.Count();
		}
		avg_k_row_col = k_row_col / (double) counter;
	}

	if (mpParameters->mExtendedMatrixInformation) {
		{
			ProgressBar ppb;
			unsigned counter = 0;
			cout << "Computing Gram matrix for [" << mpData->mRowIndexList.size() << " rows x " << mpData->mRowIndexList.size() << " rows]=" << mpData->mRowIndexList.size() * mpData->mRowIndexList.size() << " pairs of instances." << endl;
			for (unsigned i = 0; i < mpData->mRowIndexList.size(); ++i) {
				unsigned ii = mpData->mRowIndexList[i];
				vector<double> k_val_list(mpData->mRowIndexList.size());
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
				for (unsigned j = 0; j < mpData->mRowIndexList.size(); ++j) {
					unsigned jj = mpData->mRowIndexList[j];
					k_val_list[j] = mpData->ComputeKernel(ii, jj);
				}
				for (unsigned j = 0; j < mpData->mRowIndexList.size(); ++j) {
					unsigned jj = mpData->mRowIndexList[j];
					if (ii != jj) {
						double k = k_val_list[j];
						k_row_row += k;
						counter++;
					}
				}
				out << endl;
				ppb.Count();
			}
			avg_k_row_row = k_row_row / (double) counter;
		}
		{
			ProgressBar ppb;
			unsigned counter = 0;
			cout << "Computing Gram matrix for [" << mpData->mColIndexList.size() << " cols x " << mpData->mColIndexList.size() << " cols]=" << mpData->mColIndexList.size() * mpData->mColIndexList.size() << " pairs of instances." << endl;
			for (unsigned i = 0; i < mpData->mColIndexList.size(); ++i) {
				unsigned ii = mpData->mColIndexList[i];
				vector<double> k_val_list(mpData->mColIndexList.size());
#ifdef USEMULTITHREAD
#pragma omp parallel for schedule(dynamic)
#endif
				for (unsigned j = 0; j < mpData->mColIndexList.size(); ++j) {
					unsigned jj = mpData->mColIndexList[j];
					k_val_list[j] = mpData->ComputeKernel(ii, jj);
				}
				for (unsigned j = 0; j < mpData->mColIndexList.size(); ++j) {
					unsigned jj = mpData->mColIndexList[j];
					if (ii != jj) {
						double k = k_val_list[j];
						k_col_col += k;
						counter++;
					}
				}
				out << endl;
				ppb.Count();
			}
			avg_k_col_col = k_col_col / (double) counter;
		}
		cout << endl << "Average kernel value: " << avg_k_row_col << endl;
		cout << endl << "Average row kernel value: " << avg_k_row_row << endl;
		cout << endl << "Average col kernel value: " << avg_k_col_col << endl;
		double avg_k_sim = avg_k_row_col / sqrt(avg_k_row_row * avg_k_col_col);
		cout << endl << "Average kernel similarity row vs. col: " << avg_k_sim << endl << endl;
	}
}
