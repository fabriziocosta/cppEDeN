#include "SGDSVM.h"

StochasticGradientDescentSupportVectorMachine::StochasticGradientDescentSupportVectorMachine() :
		mpParameters(0), mpData(0), mWScale(0), mBias(0) {
}

StochasticGradientDescentSupportVectorMachine::StochasticGradientDescentSupportVectorMachine(Parameters* apParameters, Data* apData) {
	Init(apParameters,apData);
}

void StochasticGradientDescentSupportVectorMachine::Init(Parameters* apParameters, Data* apData) {
	mpParameters = apParameters;
	mpData = apData;
	Clear();
}

void StochasticGradientDescentSupportVectorMachine::Clear() {
	mWScale = 1;
	mBias = 0;
	mW = SVector(pow(2, mpParameters->mHashBitSize));
}

void StochasticGradientDescentSupportVectorMachine::VectorElementwiseProductWithModel(SVector& oX) {
	SVector x = elementwise_product_clamped(oX, mW);
	x /= x.norm();
	oX = x;
}

/**
 Prints several informative measures given a list of predictions and a list of true targets
 */
void StochasticGradientDescentSupportVectorMachine::OutputPerformanceMeasures(ostream& out, const vector<double>& aMarginList, const vector<double>& aTargetList) {
	assert(aMarginList.size() == aTargetList.size());
	unsigned size = aMarginList.size();
	unsigned error = 0;
	unsigned correct = 0;
	unsigned tp, tn, fp, fn;
	tp = tn = fp = fn = 0;
	for (unsigned i = 0; i < aMarginList.size(); ++i) {
		double margin = aMarginList[i];
		double prediction = margin > 0 ? 1 : -1;
		double target = aTargetList[i];
		if (prediction != target)
			error++;
		if (prediction == target)
			correct++;
		if (prediction > 0 && target > 0)
			tp++;
		if (prediction > 0 && target < 0)
			fp++;
		if (prediction < 0 && target > 0)
			fn++;
		if (prediction < 0 && target < 0)
			tn++;
	}

	double pprecision = (double) tp / (tp + fp);
	double precall = (double) tp / (tp + fn);
	double pfmeasure = 2 * pprecision * precall / (pprecision + precall);

	double nprecision = (double) tn / (tn + fn);
	double nrecall = (double) tn / (tn + fp);
	double nfmeasure = 2 * nprecision * nrecall / (nprecision + nrecall);

	double bprecision = (pprecision + nprecision) / 2;
	double brecall = (precall + nrecall) / 2;
	double bfmeasure = (pfmeasure + nfmeasure) / 2;

	out << TAB << "Size: " << size << endl;
	out << TAB << "Correct: " << correct << " ( " << correct * 100 / (double) (correct + error) << " %)" << endl;
	out << TAB << "Error: " << error << " ( " << error * 100 / (double) (correct + error) << " %)" << endl;
	out << TAB << "Confusion table:" << endl;
	out << TAB << "TP:" << tp << " FP:" << fp << endl;
	out << TAB << "FN:" << fn << " TN:" << tn << endl;
	out << TAB << "+Precision:" << pprecision << " +Recall:" << precall << " +F-measure:" << pfmeasure << endl;
	out << TAB << "-Precision:" << nprecision << " -Recall:" << nrecall << " -F-measure:" << nfmeasure << endl;
	out << TAB << "bPrecision:" << bprecision << " bRecall:" << brecall << " bF-measure:" << bfmeasure << endl;
}

double StochasticGradientDescentSupportVectorMachine::ComputeBalancedFMeasure(const vector<double>& aMarginList, const vector<double>& aTargetList) {
	assert(aMarginList.size() == aTargetList.size());
	unsigned error = 0;
	unsigned correct = 0;
	unsigned tp, tn, fp, fn;
	tp = tn = fp = fn = 0;
	for (unsigned i = 0; i < aMarginList.size(); ++i) {
		double margin = aMarginList[i];
		double prediction = margin > 0 ? 1 : -1;
		double target = aTargetList[i];
		if (prediction != target)
			error++;
		if (prediction == target)
			correct++;
		if (prediction > 0 && target > 0)
			tp++;
		if (prediction > 0 && target < 0)
			fp++;
		if (prediction < 0 && target > 0)
			fn++;
		if (prediction < 0 && target < 0)
			tn++;
	}

	double pprecision = (double) tp / (tp + fp);
	double precall = (double) tp / (tp + fn);
	double pfmeasure = 2 * pprecision * precall / (pprecision + precall);

	double nprecision = (double) tn / (tn + fn);
	double nrecall = (double) tn / (tn + fp);
	double nfmeasure = 2 * nprecision * nrecall / (nprecision + nrecall);

	double bfmeasure = (pfmeasure + nfmeasure) / 2;

	return bfmeasure;
}

void StochasticGradientDescentSupportVectorMachine::Save(ostream& out) {
	out << "bias " << std::setprecision(9)<< mBias << endl;
	out << "wscale " << std::setprecision(9)<< mWScale << endl;
	out << "w " << printSparse(mW);
}

void StochasticGradientDescentSupportVectorMachine::Save(string aLocalSuffix) {
	string filename = mpParameters->mModelFileName + aLocalSuffix + mpParameters->mSuffix;
	OutputManager om(filename, mpParameters->mDirectoryPath);
	Save(om.mOut);
	cout << "Model file saved in file " << om.GetFullPathFileName() << endl;
}

void StochasticGradientDescentSupportVectorMachine::Load(istream& in) {
	string attribute = "";
	string expected = "";
	in >> attribute >> mBias;
	expected = "bias";
	if (attribute != expected)
		throw range_error("ERROR StochasticGradientDescentSupportVectorMachine::Load: Format error: expecting [" + expected + "] but found [" + attribute + "]");
	in >> attribute >> mWScale;
	expected = "wscale";
	if (attribute != expected)
		throw range_error("ERROR StochasticGradientDescentSupportVectorMachine::Load: Format error: expecting [" + expected + "] but found [" + attribute + "]");
	in >> attribute;
	assert(attribute == "w");
	loadSparse(in, mW);
}

vector<double> StochasticGradientDescentSupportVectorMachine::Train(vector<double>& aTargetList, vector<unsigned>& aTrainsetIDList) {
	if (aTrainsetIDList.size() != aTargetList.size())
		throw range_error("ERROR StochasticGradientDescentSupportVectorMachine::Train: Data list and Target list have not the same size: #data:" + stream_cast<string>(aTrainsetIDList.size()) + " #targets:" + stream_cast<string>(aTargetList.size()));

	vector<SVector*> sv_data_list;
	for (unsigned i = 0; i < aTrainsetIDList.size(); ++i) {
		unsigned id = aTrainsetIDList[i];
		sv_data_list.push_back(&mpData->mVectorList[id]);
	}

	//allocate local target list and compute positive/negative target counts
	vector<double> target_list;
	unsigned p, n;
	p = n = 0;
	for (unsigned i = 0; i < aTargetList.size(); ++i) {
		if (aTargetList[i] > 0)
			p++;
		else
			n++;
		target_list.push_back(aTargetList[i]);
	}

	//if no instance has negative class then generate negative instances with opposite features wrt positive instances
	vector<SVector> synth_neg_sv_data_list(aTrainsetIDList.size());
	if (n == 0) {
		if (!mpParameters->mMinimalOutput)
			cout << "No negative instances: proceeding to generate " << aTrainsetIDList.size() << " negative instances with opposite features wrt positive instances" << endl;
		for (unsigned i = 0; i < aTrainsetIDList.size(); ++i) {
			synth_neg_sv_data_list[i] = -1 * *sv_data_list[i];
			sv_data_list.push_back(&(synth_neg_sv_data_list[i]));
			target_list.push_back(-1);
		}
	}

	//clear margin list and allocate memory
	vector<double> margin_list(target_list.size());
	//...rebalance classes use pointers array to scramble and oversample
	vector<TrainItem> balanced_dataset;
	BalanceDataset(aTrainsetIDList, target_list, margin_list, sv_data_list, balanced_dataset);

	//train on balanced train data
	CoreTrain(balanced_dataset);

	if (!mpParameters->mMinimalOutput) {
		//output statistics on original train data (no class balance)
		OutputModelInfo();
		cout << "Performance on train set:" << endl;
		vector<double> train_margin_list=Test(aTrainsetIDList);
		OutputPerformanceMeasures(cout, train_margin_list, target_list);
	}
	if (n == 0) { //if no instance has negative class then re-test model only on real training data to extract margins and predictions
		margin_list = Test(aTrainsetIDList);
	}
	return margin_list;
}

void StochasticGradientDescentSupportVectorMachine::BalanceDataset(vector<unsigned>& aDatasetIDList, vector<double>& aTargetList, vector<double>& oMarginList, vector<SVector*>& aSVDataList, vector<TrainItem>& oDataset) {
	//compute class distribution
	unsigned p, n;
	p = n = 0;
	for (unsigned i = 0; i < aTargetList.size(); ++i)
		if (aTargetList[i] > 0)
			p++;
		else
			n++;
	if (!mpParameters->mMinimalOutput)
		cout << "Class distribution: " << p + n << " (+:" << p << " -:" << n << ") " << "[+:" << (double) p / (p + n) << " -:" << (double) n / (p + n) << "]" << endl;

	//separate positive from negative instances
	vector<TrainItem> positive_data_list;
	vector<TrainItem> negative_data_list;
	if (aTargetList.size() != aSVDataList.size())
		throw range_error("ERROR: StochasticGradientDescentSupportVectorMachine::BalanceDataset: number of target values: " + stream_cast<string>(aTargetList.size()) + " is different from dataset size:" + stream_cast<string>(aSVDataList.size()));
	for (unsigned i = 0; i < aTargetList.size(); ++i) {
		TrainItem ti;
		ti.mTarget = aTargetList[i];
		ti.mpInstance = aSVDataList[i];
		ti.mpMargin = &oMarginList[i];
		if (i < aDatasetIDList.size()) { //Synthesized instances are appended after the real instances, so the size information of the original id_list marks the start of the syntesized instances
			ti.mInstanceID = aDatasetIDList[i];
		} else
			ti.mInstanceID = -1; //if the instance has been synthesized then it has no correspondent original graph
		if (aTargetList[i] == 1)
			positive_data_list.push_back(ti);
		else if (aTargetList[i] == -1)
			negative_data_list.push_back(ti);
		else
			throw range_error("ERROR: StochasticGradientDescentSupportVectorMachine::BalanceDataset: target has to be 1 or -1; cannot be: " + stream_cast<string>(aTargetList[i]));
	}
	//randomly shuffle data
	std::random_shuffle ( positive_data_list.begin(), positive_data_list.end() );
	std::random_shuffle ( negative_data_list.begin(), negative_data_list.end() );

	//over-sample minority class only if there is an imbalance higher than MIN_KFOLD_IMBALANCE and if there is at least one instance for the minority class
	vector<TrainItem> balanced_positive_data_list;
	vector<TrainItem> balanced_negative_data_list;
	const double MIN_KFOLD_IMBALANCE = 1;
	if (p != 0 && p < n / MIN_KFOLD_IMBALANCE) {
		if (!mpParameters->mMinimalOutput)
			cout << "Oversampling positive factor: " << n / (double) p << endl;
		unsigned ratio = n / p;
		unsigned reminder = n % p;
		//duplicate a number of times equal to ratio the datastaset itself
		for (unsigned i = 0; i < ratio; i++)
			balanced_positive_data_list.insert(balanced_positive_data_list.end(), positive_data_list.begin(), positive_data_list.end());
		//add the remainder instances
		for (unsigned i = 0; i < reminder; i++)
			balanced_positive_data_list.push_back(positive_data_list[i]);
		balanced_negative_data_list = negative_data_list;
	} else if (n != 0 && n < p / MIN_KFOLD_IMBALANCE) {
		if (!mpParameters->mMinimalOutput)
			cout << "Oversampling negative factor: " << p / (double) n << endl;
		unsigned ratio = p / n;
		unsigned reminder = p % n;
		for (unsigned i = 0; i < ratio; i++)
			balanced_negative_data_list.insert(balanced_negative_data_list.end(), negative_data_list.begin(), negative_data_list.end());
		for (unsigned i = 0; i < reminder; i++)
			balanced_negative_data_list.push_back(negative_data_list[i]);
		balanced_positive_data_list = positive_data_list;
	} else {
		balanced_positive_data_list = positive_data_list;
		balanced_negative_data_list = negative_data_list;
	}

	//compose dataset by alternating positive and negative examples
	unsigned i;
	for (i = 0; i < balanced_positive_data_list.size(); i++) {
		oDataset.push_back(balanced_positive_data_list[i]);
		if (i < balanced_negative_data_list.size())
			oDataset.push_back(balanced_negative_data_list[i]);
	}
	for (unsigned j = i; j < balanced_negative_data_list.size(); j++)
		oDataset.push_back(balanced_negative_data_list[i]);

	//compute new class ratio
	unsigned bp = 0, bn = 0;
	for (unsigned i = 0; i < oDataset.size(); i++)
		if (oDataset[i].mTarget > 0)
			bp++;
		else
			bn++;
	if (!mpParameters->mMinimalOutput)
		cout << "Rebalanced dataset: " << bp + bn << " (+:" << bp << " -:" << bn << ")" << endl;
}

void StochasticGradientDescentSupportVectorMachine::CoreTrain(vector<TrainItem>& aDataset) {
#ifdef DEBUGON
	VectorClass stat;
#endif

	SVector w_sparsifier(pow(2, mpParameters->mHashBitSize));
	;
	SVector w_sparsifier_binarized(pow(2,mpParameters->mHashBitSize));
	;
	unsigned num_original_fetures = 0;

	//Iterate epochs times in gradient descent
	ProgressBar pb(1);
	if (!mpParameters->mMinimalOutput) {
		OutputTrainingInfo();
		cout << "Training for " << mpParameters->mEpochs << " epochs." << endl;
	}

	//---------------------------------------------------------------------------------------------------------------
	//iterate for several sparsification iterations
	for (unsigned s = 0; s <= mpParameters->mSparsificationNumIterations; s++) {
		Clear();
		// Shift t in order to have a reasonable initial learning rate. This assumes |x| \approx 1.
		double maxw = 1.0 / sqrt(mpParameters->mLambda);
		double typw = sqrt(maxw);
		double eta0 = typw / max(1.0, dloss(-typw));
		double t = 1 / (eta0 * mpParameters->mLambda);

		if (s != 0 && s == mpParameters->mSparsificationNumIterations) { //for the last iteration (if mSparsificationNumIterations is not just 0 ) use a 0-1 binarized version of the mWSparse
			w_sparsifier_binarized = binarize(w_sparsifier);
			if (!mpParameters->mMinimalOutput)
				cout << endl << "Feature filtering step: num features retained: " << w_sparsifier_binarized.nonZeros() << endl;
		}

		//---------------------------------------------------------------------------------------------------------------
		//iterate for several epochs
		for (unsigned e = 0; e < mpParameters->mEpochs; e++) {

			if (e > 0 && s != mpParameters->mSparsificationNumIterations) //after first epoch and excluding last sparsification iteration
				if (mpParameters->mTopologicalRegularizationNumNeighbors != 0) { //if topological regularization is required
					FeatureTopologicalRegularization(1.0 / (mpParameters->mLambda * t));
				}

			if (!mpParameters->mMinimalOutput)
				pb.Count();

			//---------------------------------------------------------------------------------------------------------------
			//iterate over all train instances
			for (unsigned i = 0; i < aDataset.size(); ++i) {
				double eta = 1.0 / (mpParameters->mLambda * t);
				double scale = 1 - eta * mpParameters->mLambda;
				mWScale *= scale;
				if (mWScale < 1e-9) {
					mW *= mWScale;
					mWScale = 1;
				}

				//const SVector &x = (*aDataset[i].mpInstance);
				SVector x = (*aDataset[i].mpInstance);

				//iterative sparsification for approximate L0 norm regularization
				if (mpParameters->mSparsificationNumIterations > 0) {
					if (s == mpParameters->mSparsificationNumIterations) { //at last iteration use binarized selection
						assert(w_sparsifier.nonZeros() > 0);
						x = elementwise_product_clamped(x, w_sparsifier_binarized);
					} else if (s > 0) {
						assert(w_sparsifier.nonZeros() > 0);
						x = elementwise_product_clamped(x, w_sparsifier);
					} else {
						//do not sparsify the very first iteration
					}
				}

				double y = aDataset[i].mTarget;
				double wx = mW.dot(x) * mWScale;
				double margin = (wx + mBias);
				(*(aDataset[i].mpMargin)) = margin;
				double z = y * margin;
#if LOSS < LOGLOSS
				if (z < 1)
#endif
						{
					double etd = eta * dloss(z);
					mW += x * etd * y / mWScale;
#if BIAS
					// Slower rate on the bias because it learns at each iteration.
					mBias += etd * y * 0.01;
#endif
				}
				t += 1;
			}
			//---------------------------------------------------------------------------------------------------------------
			//end of iteration over all train instances
			if (s == 0)
				num_original_fetures = mW.nonZeros();
#ifdef DEBUGON
			stat.PushBack((double) mW.nonZeros());
#endif
		}
#ifdef DEBUGON
		cout << endl << "W size statistics: ";
		stat.OutputStatistics(cout);
		cout << endl;
#endif

		if (s != mpParameters->mSparsificationNumIterations) { //skip the last iteration
			//iterative sparsification for approximate L0 norm regularization
			if (s == 0) {
				w_sparsifier = mW * mWScale;
				//w_sparsifier.normalize();
			} else {
				SVector w_sparsifier_new = mW * mWScale;
				w_sparsifier = elementwise_product_clamped(w_sparsifier, w_sparsifier_new, 1e3); //NOTE:set the constant as a hard limit on any elelemnt size
				//w_sparsifier.normalize();
			}
			if (!mpParameters->mMinimalOutput)
				cout << endl << "Iteration: " << s << endl;
			cout << " w_sparsifier norm: " << w_sparsifier.norm();
			cout << endl;
			cout << " num features retained: " << w_sparsifier.nonZeros() << "/" << num_original_fetures << " (" << (double) (w_sparsifier.nonZeros()) / double(num_original_fetures) << ")" << endl;
		} //end of iterative sparsification

	} //for s
}

void StochasticGradientDescentSupportVectorMachine::FeatureTopologicalRegularization(double aGamma) {
	SVector w_new(pow(2, mpParameters->mHashBitSize));
	;
	for (map<unsigned, SVector>::iterator it = mpData->mFeatureCorrelationMatrix.begin(); it != mpData->mFeatureCorrelationMatrix.end(); ++it) {
		unsigned key = it->first;
		SVector& f = it->second;
		double value = f.dot(mW);
		if (value > 1e-9)
			w_new.insert(key) = value;
	}
	//cout << endl << "w norm before topological regularization:" << sqrt(dot(mW, mW)) * mWScale  << endl; /////////////////
	//cout<<"L*gamma="<<mpParameters->mTopologicalRegularizationRate*aGamma<<endl;//////////////
	mW = mW * w_new * -mpParameters->mTopologicalRegularizationRate * aGamma;
	//cout << "w norm after topological regularization:" << sqrt(dot(mW, mW)) * mWScale << endl; /////////////////
}

vector<double> StochasticGradientDescentSupportVectorMachine::Test(vector<unsigned>& aTestSetIDList) {
	vector<double> margin_list;
	for (unsigned i = 0; i < aTestSetIDList.size(); ++i) {
		unsigned gid = aTestSetIDList[i];
		SVector& x = mpData->mVectorList[gid];
		double y = Predict(x);
		margin_list.push_back(y);
	}
	return margin_list;
}

double StochasticGradientDescentSupportVectorMachine::Predict(const SVector& x) {
	return mW.dot(x) * mWScale + mBias;
}

void StochasticGradientDescentSupportVectorMachine::OutputTrainingInfo() {
	cout << SEP << endl;
	cout << "Training information" << endl;
	cout << SEP << endl;
	cout << "Lambda: " << mpParameters->mLambda << endl;
	cout << "Epochs: " << mpParameters->mEpochs << endl;
	cout << SEP << endl;
}

void StochasticGradientDescentSupportVectorMachine::OutputModelInfo() {
	cout << SEP << endl;
	cout << "Model information" << endl;
	cout << SEP << endl;
	cout << "W Norm: " << std::setprecision(9) << sqrt(mW.dot(mW)) * mWScale << endl;
	cout << "Bias: "  << std::setprecision(9) << mBias << endl;
	cout << SEP << endl;
}
