#include "ParametersOptimizationManager.h"

ParametersOptimizationManager::ParametersOptimizationManager(Parameters* apParameters, Data* apData): BaseManager(apParameters, apData){
	Init(apParameters,apData);
}

void ParametersOptimizationManager::Init(Parameters* apParameters, Data* apData) {
	BaseManager::Init(apParameters, apData);
	mCrossValidationManager.Init(apParameters, apData);

	mCurrentBFMeasure = 0;
	mBestBFMeasure = 0;
	mLambdaLimit = mpParameters->mLambda;
	mEpochsLimit = mpParameters->mEpochs;
	mRadiusLimit = mpParameters->mRadius;
	mDistanceLimit = mpParameters->mDistance;
	mTopologicalRegularizationNumNeighborsLimit = mpParameters->mTopologicalRegularizationNumNeighbors;
	mTopologicalRegularizationRateLimit = mpParameters->mTopologicalRegularizationRate;
	mSparsificationNumIterationsLimit = mpParameters->mSparsificationNumIterations;
	mTreeLambdaLimit = mpParameters->mTreeLambda;
	mRadiusTwoLimit = mpParameters->mRadiusTwo;

}

void ParametersOptimizationManager::Exec() {
	mpData->LoadTarget();
	OutputManager();
}

void ParametersOptimizationManager::OutputManager() {
	string output_filename = mpParameters->mInputDataFileName + ".opt_param" + mpParameters->mSuffix;
	ofstream ofs;
	ofs.open(output_filename.c_str());
	if (!ofs)
		throw range_error("ERROR2.45: Cannot open file:" + output_filename);

	ParametersOptimization(ofs);

	cout << "Best parameter configuration obtains a balanced F-measure= " << mBestBFMeasure << endl;
	cout << "For best predictive performance use the following parameter setting:" << endl;
	OutputParameters(cout);
	OutputParameters(ofs);
	cout << "Optimal parameters configuration saved in file " << output_filename << endl;
}

void ParametersOptimizationManager::ParametersOptimization(ostream& ofs) {
	ProgressBar pbt;
	pbt.Count();

	SetDefaultParameters();
	cout << "Initial parameters configuration: ";
	OutputParameters(cout);
	mpData->LoadData(true,true,false);

	cout << "Line search parameters optimization: " << mpParameters->mNumLineSearchIterations << " iterations." << endl;
	for (unsigned line_search_iteration = 0; line_search_iteration < mpParameters->mNumLineSearchIterations; line_search_iteration++) {
		cout << "Iteration " << line_search_iteration + 1 << "/" << mpParameters->mNumLineSearchIterations << endl;
		OptimizeLambda();
		OptimizeEpochs();
		OptimizeDistance();
		OptimizeRadius();
		OptimizeTopologicalRegularizationNumNeighbors();
		OptimizeTopologicalRegularizationRate();
		OptimizeSparsificationNumIterations();
		if (mpParameters->mKernelType == "DDK" || mpParameters->mKernelType == "NSDDK" || mpParameters->mKernelType == "ANSDDK") {
			OptimizeTreeLambda();
		}
		if (mpParameters->mKernelType == "NSDDK" || mpParameters->mKernelType == "ANSDDK") {
			OptimizeRadiusTwo();
		}

	}
}

//Required for DDK kernel family
void ParametersOptimizationManager::OptimizeTreeLambda() {
	mBestBFMeasure = 0;
	double lambda_best = mpParameters->mTreeLambda;
	for (double l = 0.2; l <= mTreeLambdaLimit; l = l + 0.2) {
		mpParameters->mTreeLambda = l;
		mpData->LoadData(true,true,false);
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			lambda_best = l;
			cout << "*";
		}
		cout << "tree_lambda:" << l << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mTreeLambda = lambda_best;
	//mpData->LoadData();
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}
void ParametersOptimizationManager::OptimizeRadiusTwo() {
	mBestBFMeasure = 0;
	unsigned r_best = mpParameters->mRadiusTwo;
	for (unsigned r = 0; r <= mRadiusTwoLimit; r++) {
		mpParameters->mRadiusTwo = r;
		mpData->LoadData(true,true,false);
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			r_best = r;
			cout << "*";
		}
		cout << "radius_two:" << r << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mRadiusTwo = r_best;
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}
//-----------
void ParametersOptimizationManager::SetDefaultParameters() {
	//default values
	mpParameters->mLambda = 1e-4;
	mpParameters->mEpochs = 5;
	mpParameters->mRadius = 1;
	mpParameters->mDistance = 2;
	mpParameters->mTopologicalRegularizationNumNeighbors = 0;
	mpParameters->mTopologicalRegularizationRate = 0.001;
	mpParameters->mSparsificationNumIterations = 0;
	mpParameters->mRadiusTwo = 1;
	mpParameters->mTreeLambda = 1.2;

}

void ParametersOptimizationManager::OptimizeLambda() {
	mBestBFMeasure = 0;
	double lambda_best = mpParameters->mLambda;
	const double LAMBDA_UPPER_BOUND = 0.01;
	//double lambda_step = exp((log(LAMBDA_UPPER_BOUND) - log(lambda_limit)) / (2 * mpParameters->mLearningCurveNumPoints));
	double lambda_step = 10;
	for (double lambda = mLambdaLimit; lambda <= LAMBDA_UPPER_BOUND; lambda *= lambda_step) {
		mpParameters->mLambda = lambda;
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			lambda_best = lambda;
			cout << "*";
		}
		cout << "l:" << lambda << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mLambda = lambda_best;
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}

void ParametersOptimizationManager::OptimizeEpochs() {
	mBestBFMeasure = 0;
	unsigned epochs_best = mpParameters->mEpochs;
	unsigned epochs_step = (double) mEpochsLimit / (double) mpParameters->mNumPoints;
	epochs_step = epochs_step == 0 ? 1 : epochs_step;
	for (unsigned epochs = epochs_step; epochs <= mEpochsLimit; epochs += epochs_step) {
		mpParameters->mEpochs = epochs;
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			epochs_best = epochs;
			cout << "*";
		}
		cout << "e:" << epochs << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mEpochs = epochs_best;
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}

void ParametersOptimizationManager::OptimizeDistance() {
	mBestBFMeasure = 0;
	unsigned d_best = mpParameters->mDistance;
	for (unsigned d = 0; d <= mDistanceLimit; d++) {
		mpParameters->mDistance = d;
		mpData->LoadData(true,true,false);
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			d_best = d;
			cout << "*";
		}
		cout << "d:" << d << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mDistance = d_best;
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}

void ParametersOptimizationManager::OptimizeRadius() {
	mBestBFMeasure = 0;
	unsigned r_best = mpParameters->mRadius;
	for (unsigned r = 0; r <= mRadiusLimit; r++) {
		mpParameters->mRadius = r;
		mpData->LoadData(true,true,false);
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			r_best = r;
			cout << "*";
		}
		cout << "r:" << r << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mRadius = r_best;
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}

void ParametersOptimizationManager::OptimizeTopologicalRegularizationNumNeighbors() {
	mBestBFMeasure = 0;
	unsigned topological_regularization_num_neighbors_best = mpParameters->mTopologicalRegularizationNumNeighbors;
	unsigned topological_regularization_num_neighbors_step = (double) mTopologicalRegularizationNumNeighborsLimit / (double) mpParameters->mNumPoints;
	topological_regularization_num_neighbors_step = topological_regularization_num_neighbors_step == 0 ? 1 : topological_regularization_num_neighbors_step;
	for (unsigned topological_regularization_num_neighbors = 0; topological_regularization_num_neighbors <= mTopologicalRegularizationNumNeighborsLimit; topological_regularization_num_neighbors += topological_regularization_num_neighbors_step) {
		mpParameters->mTopologicalRegularizationNumNeighbors = topological_regularization_num_neighbors;
		mpData->LoadData(true,true,false);
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			topological_regularization_num_neighbors_best = topological_regularization_num_neighbors;
			cout << "*";
		}
		cout << "C:" << topological_regularization_num_neighbors << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mTopologicalRegularizationNumNeighbors = topological_regularization_num_neighbors_best;
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}

void ParametersOptimizationManager::OptimizeTopologicalRegularizationRate() {
	mBestBFMeasure = 0;
	double topological_regularization_rate_best = mpParameters->mTopologicalRegularizationRate;
	const double TOPOLOGICAL_REGULARIZATION_RATE_UPPER_BOUND = .001;
	//double topological_regularization_rate_step = (double) topological_regularization_rate_limit / (double) mpParameters->mLearningCurveNumPoints;
	double topological_regularization_rate_step = 10;
	for (double topological_regularization_rate = mTopologicalRegularizationRateLimit; topological_regularization_rate <= TOPOLOGICAL_REGULARIZATION_RATE_UPPER_BOUND; topological_regularization_rate *= topological_regularization_rate_step) {
		mpParameters->mTopologicalRegularizationRate = topological_regularization_rate;
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			topological_regularization_rate_best = topological_regularization_rate;
			cout << "*";
		}
		cout << "L:" << topological_regularization_rate << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mTopologicalRegularizationRate = topological_regularization_rate_best;
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}

void ParametersOptimizationManager::OptimizeSparsificationNumIterations() {
	mBestBFMeasure = 0;
	double sparsification_num_iterations_best = mpParameters->mSparsificationNumIterations;
	unsigned sparsification_num_iterations_step = (double) mSparsificationNumIterationsLimit / (double) mpParameters->mNumPoints;
	sparsification_num_iterations_step = sparsification_num_iterations_step == 0 ? 1 : sparsification_num_iterations_step;
	for (unsigned sparsification_num_iterations = 0; sparsification_num_iterations <= mSparsificationNumIterationsLimit; sparsification_num_iterations += sparsification_num_iterations_step) {
		mpParameters->mSparsificationNumIterations = sparsification_num_iterations;
		mCurrentBFMeasure = mCrossValidationManager.GetBalancedFMeasure();
		if (mCurrentBFMeasure > mBestBFMeasure) {
			mBestBFMeasure = mCurrentBFMeasure;
			sparsification_num_iterations_best = sparsification_num_iterations;
			cout << "*";
		}
		cout << "O:" << sparsification_num_iterations << " bf:" << mCurrentBFMeasure << " " << endl;
	}
	mpParameters->mSparsificationNumIterations = sparsification_num_iterations_best;
	cout << "bFmeasure:" << mBestBFMeasure << " current best parameters configuration:";
	OutputParameters(cout);
}

void ParametersOptimizationManager::OutputParameters(ostream& out) {
	out << " -r " << mpParameters->mRadius;
	out << " -d " << mpParameters->mDistance;
	out << " -e " << mpParameters->mEpochs;
	out << " -l " << mpParameters->mLambda;
	out << " -O " << mpParameters->mSparsificationNumIterations;
	out << " -L " << mpParameters->mTopologicalRegularizationRate;
	out << " -C " << mpParameters->mTopologicalRegularizationNumNeighbors;
	if (mpParameters->mKernelType == "DDK" || mpParameters->mKernelType == "NSDDK" || mpParameters->mKernelType == "ANSDDK")
		out << " --tree_lambda " << mpParameters->mTreeLambda;
	if (mpParameters->mKernelType == "DDK" || mpParameters->mKernelType == "NSDDK" || mpParameters->mKernelType == "ANSDDK")
		out << " --radius_two " << mpParameters->mRadiusTwo;

	out << endl;
}
