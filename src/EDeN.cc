#include "Utility.h"
#include "Parameters.h"
#include "Kernel.h"
#include "Data.h"
#include "SGDSVM.h"
#include "GramMatrixManager.h"
#include "MinHashEncoder.h"
#include "EmbedManager.h"
#include "TargetAlignmentManager.h"
#include "SGDSVMManager.h"
#include "CrossValidationManager.h"
#include "LearningCurveManager.h"
#include "BiasVarianceDecompositionManager.h"
#include "ParametersOptimizationManager.h"
#include "TestManager.h"
#include "FeatureManager.h"
#include "MinHashManager.h"
#include "NearestNeighborManager.h"
#include "ClusterManager.h"
#include "SemiSupervisedManager.h"
#include "PreClusterFilterManager.h"
#include "DensityManager.h"
#include "TrainMultilabelMLManager.h"
#include "TestMultilabelMLManager.h"


using namespace std;

FlagsService& The_FlagsService = FlagsService::get_instance();

//------------------------------------------------------------------------------------------------------------------------
class Dispatcher {
	protected:
		Parameters mParameters;
		Data mData;

	public:
		Dispatcher() {
		}

		void Init(int argc, const char **argv) {
			mParameters.Init(argc, argv);
			srand(mParameters.mRandomSeed);
			mData.Init(&mParameters);
		}

		void Exec() {
			ProgressBar pb;

			if (!mParameters.mMinimalOutput)
				cout << SEP << endl << PROG_NAME << endl << "Version: " << PROG_VERSION << endl << "Last Update: " << PROG_DATE << endl << PROG_CREDIT << endl << SEP << endl;

			switch (mParameters.mActionCode) {
				case TRAIN: {
					StochasticGradientDescentSupportVectorMachineManager sgdsvm_manager(&mParameters, &mData);
					sgdsvm_manager.Exec();
				}
					break;
				case CROSS_VALIDATION: {
					CrossValidationManager cross_validation_manager(&mParameters, &mData);
					cross_validation_manager.Exec();
				}
					break;
				case BIAS_VARIANCE_DECOMPOSITION: {
					BiasVarianceDecompositionManager bias_variance_decomposition_manager(&mParameters, &mData);
					bias_variance_decomposition_manager.Exec();
				}
					break;
				case PARAMETERS_OPTIMIZATION: {
					ParametersOptimizationManager parameters_optimization_manager(&mParameters, &mData);
					parameters_optimization_manager.Exec();
				}
					break;
				case LEARNING_CURVE: {
					LearningCurveManager learning_curve_manager(&mParameters, &mData);
					learning_curve_manager.Exec();
				}
					break;
				case TARGET_ALIGNMENT: {
					TargetAlignmentManager target_alignment_manager(&mParameters, &mData);
					target_alignment_manager.Exec();
				}
					break;
				case MATRIX: {
					GramMatrixManager gram_matrix_manager(&mParameters, &mData);
					gram_matrix_manager.Exec();
				}
					break;
				case NEAREST_NEIGHBOR: {
					NearestNeighborManager nearest_neighbor_manager(&mParameters, &mData);
					nearest_neighbor_manager.Exec();
				}
					break;
				case EMBED: {
					EmbedManager embed_manager(&mParameters, &mData);
					embed_manager.Exec();
				}
					break;
				case TEST: {
					TestManager test_manager(&mParameters, &mData);
					test_manager.Exec();
				}
					break;
				case TEST_PART: {
					TestPartManager test_part_manager(&mParameters, &mData);
					test_part_manager.Exec();
				}
					break;
				case FEATURE: {
					FeatureManager feature_manager(&mParameters, &mData);
					feature_manager.Exec();
				}
					break;
				case FEATURE_PART: {
					FeaturePartManager feature_part_manager(&mParameters, &mData);
					feature_part_manager.Exec();
				}
					break;
				case FEATURE_SCALED: {
					FeatureScaledManager feature_scaled_manager(&mParameters, &mData);
					feature_scaled_manager.Exec();
				}
					break;
				case CLUSTER: {
					if (mParameters.mClusterType == "DENSE_CENTERS") {
						DensityClusterManager cluster_manager(&mParameters, &mData);
						cluster_manager.Exec();
					} else if (mParameters.mClusterType == "DENSE_CONNECTED_CENTERS") {
						ConnectedDensityClusterManager cluster_manager(&mParameters, &mData);
						cluster_manager.Exec();
					} else if (mParameters.mClusterType == "K_QUICK_SHIFT") {
						KQuickShiftClusterManager cluster_manager(&mParameters, &mData);
						cluster_manager.Exec();
					} else if (mParameters.mClusterType == "APPROXIMATION_ACCURACY") {
						mParameters.mUseApproximate = true;
						AccuracyClusterManager cluster_manager(&mParameters, &mData);
						cluster_manager.Exec();
					} else
						throw range_error("ERROR: Unknown cluster type: " + mParameters.mClusterType);
				}
					break;
				case MIN_HASH: {
					MinHashManager min_hash_manager(&mParameters, &mData);
					min_hash_manager.Exec();
				}
					break;
				case SEMI_SUPERVISED: {
					SemiSupervisedManager semi_supervised_manager(&mParameters, &mData);
					semi_supervised_manager.Exec();
				}
					break;
				case PRE_CLUSTER_FILTER: {
					PreClusterFilterManager pre_cluster_filter_manager(&mParameters, &mData);
					pre_cluster_filter_manager.Exec();
				}
					break;
				case DENSITY: {
					DensityManager density_manager(&mParameters, &mData);
					density_manager.Exec();
				}
					break;
				case TRAIN_MULTILABEL: {
					TrainMultilabelMLManager train_multilabel_ml_manager(&mParameters, &mData);
					train_multilabel_ml_manager.Exec();
				}
					break;
				case TEST_MULTILABEL: {
					TestMultilabelMLManager test_multilabel_ml_manager(&mParameters, &mData);
					test_multilabel_ml_manager.Exec();
				}
					break;

				default:
					throw range_error("ERROR2.2: Unknown action parameter: " + mParameters.mAction);
			}
			pb.Count();
			cout << "Total run-time:" << endl;
		}
};

//------------------------------------------------------------------------------------------------------------------------
int main(int argc, const char **argv) {
	try {
		// initialize Eigen to be called from multiple threads
		Eigen::initParallel();
		Dispatcher dispatcher;
		dispatcher.Init(argc, argv);
		dispatcher.Exec();
	} catch (exception& e) {
		cerr << e.what() << endl;
	}
	return 0;
}

//TODO: make a better data manager that can be used also for in-line processing with an external callable function inbetween
//TODO: make parameter optimization
//TODO: make a principled object oriented organization of the code, so that the learning algorithm is a module, the instance representation is abstracted (i.e. accessible only through properties)
