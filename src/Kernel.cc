#include "Kernel.h"

void Kernel::Init(Parameters* apParameters) {
	mpParameters = apParameters;

	// use a single feature generator
#ifndef USEMULTITHREAD
	if (mpParameters->mKernelType == "NSPDK")
	mpFeatureGenerator = new ANSPDK_FeatureGenerator("anspdk");
	else if (mpParameters->mKernelType == "PBK")
	mpFeatureGenerator = new PBK_FeatureGenerator("pbk");
	else if (mpParameters->mKernelType == "WDK")
	mpFeatureGenerator = new WDK_FeatureGenerator("wdk");
	else if (mpParameters->mKernelType == "USPK")
	mpFeatureGenerator = new USPK_FeatureGenerator("uspk");
	else if (mpParameters->mKernelType == "DDK")
	mpFeatureGenerator = new DDkernel_FeatureGeneratorNew("ddk");
	else if (mpParameters->mKernelType == "NSDDK")
	mpFeatureGenerator = new NSDDkernel_FeatureGenerator("nsddk");
	else if (mpParameters->mKernelType == "ANSDDK")
	mpFeatureGenerator = new ANSDDkernel_FeatureGenerator("ansddk");
	else if (mpParameters->mKernelType == "SK")
	mpFeatureGenerator = new SK_FeatureGenerator("sk");
	else if (mpParameters->mKernelType == "STRING")
	mpFeatureGenerator = new String_FeatureGenerator("string");
	else
	throw range_error("ERROR2.1: Unknown kernel type: " + mpParameters->mKernelType);
#endif

	// use a vector of feature generators
#ifdef USEMULTITHREAD
	// prepare as many NSPDK_FeatureGenerator objects as there can be threads
	int n_threads = omp_get_max_threads();
	for (int i = 0; i < n_threads; i++) {
		// TODO: use C++11 to_string() starting from 2018
		stringstream i_ss;
		i_ss << i;
		if (mpParameters->mKernelType == "NSPDK")
			mpFeatureGeneratorList.push_back(new ANSPDK_FeatureGenerator("anspdk_" + i_ss.str()));
		else if (mpParameters->mKernelType == "PBK")
			mpFeatureGeneratorList.push_back(new PBK_FeatureGenerator("pbk_" + i_ss.str()));
		else if (mpParameters->mKernelType == "WDK")
			mpFeatureGeneratorList.push_back(new WDK_FeatureGenerator("wdk_" + i_ss.str()));
		else if (mpParameters->mKernelType == "USPK")
			mpFeatureGeneratorList.push_back(new USPK_FeatureGenerator("uspk_" + i_ss.str()));
		else if (mpParameters->mKernelType == "DDK")
			mpFeatureGeneratorList.push_back(new DDkernel_FeatureGeneratorNew("ddk_" + i_ss.str()));
		else if (mpParameters->mKernelType == "NSDDK")
			mpFeatureGeneratorList.push_back(new NSDDkernel_FeatureGenerator("nsddk_" + i_ss.str()));
		else if (mpParameters->mKernelType == "ANSDDK")
			mpFeatureGeneratorList.push_back(new ANSDDkernel_FeatureGenerator("ansddk_" + i_ss.str()));
		else if (mpParameters->mKernelType == "SK")
			mpFeatureGeneratorList.push_back(new SK_FeatureGenerator("sk_" + i_ss.str()));
		else if (mpParameters->mKernelType == "STRING")
			mpFeatureGeneratorList.push_back(new String_FeatureGenerator("string_" + i_ss.str()));
		else
			throw range_error("ERROR2.1: Unknown kernel type: " + mpParameters->mKernelType);
	}
#endif

	ParametersSetup();
}

void Kernel::OutputParameters(ostream& out) {
#ifndef USEMULTITHREAD
	cout << SEP << endl << "Kernel parameters" << endl << SEP << endl;
	mpFeatureGenerator->OutputParameters(cout);
	cout << SEP << endl;
#endif
#ifdef USEMULTITHREAD
	cout << SEP << endl << "Kernel parameters" << endl << SEP << endl;
	mpFeatureGeneratorList[0]->OutputParameters(cout);
//	unsigned counter=0;
//	for (vector<NSPDK_FeatureGenerator*>::iterator fgen_it = mpFeatureGeneratorList.begin(); fgen_it != mpFeatureGeneratorList.end(); fgen_it++) {
//		cout<<endl<<"Kernel for thread #"<<counter<<endl;
//		(*fgen_it)->OutputParameters(cout);
//		counter++;
//	}
	cout << SEP << endl;
#endif
}

void Kernel::ParametersSetup() {
#ifdef DEBUGON
	mpFeatureGenerator->set_flag("verbosity", stream_cast<string>(1));
#endif

#ifndef USEMULTITHREAD

	if (mpParameters->mVerbose)
	mpFeatureGenerator->set_flag("verbosity", stream_cast<string>(1));
	if (mpParameters->mMinKernel)
	mpFeatureGenerator->set_flag("min_kernel", "true");
	if (mpParameters->mKernelNoNormalization)
	mpFeatureGenerator->set_flag("normalization", "false");
	mpFeatureGenerator->set_flag("radius", stream_cast<string>(mpParameters->mRadius));
	mpFeatureGenerator->set_flag("distance", stream_cast<string>(mpParameters->mDistance));
	mpFeatureGenerator->set_flag("hash_bit_size", stream_cast<string>(mpParameters->mHashBitSize));
	unsigned bitmask = (2 << (mpParameters->mHashBitSize - 1)) - 1;
	mpFeatureGenerator->set_flag("hash_bit_mask", stream_cast<string>(bitmask));
	if (mpParameters->mUseRealVectorInformation) {
		mpFeatureGenerator->set_flag("use_real_vector_information", "true");
		mpFeatureGenerator->set_flag("num_rand_projections",stream_cast<string>(mpParameters->mNumRandomProjections));
	}

	//if type of kernel PBK then also perform the following initializations
	if (mpParameters->mKernelType == "PBK") {
		mpFeatureGenerator->set_flag("lower_vertex_degree_threshold", stream_cast<string>(mpParameters->mVertexDegreeThreshold));
		mpFeatureGenerator->set_flag("vertex_degree_threshold", stream_cast<string>(mpParameters->mVertexDegreeThreshold));

		if (mpParameters->mMinKernel)
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("min_kernel", "true");
		if (mpParameters->mKernelNoNormalization)
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("normalization", "false");
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("radius", stream_cast<string>(mpParameters->mRadius));
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("distance", stream_cast<string>(mpParameters->mDistance));
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("hash_bit_size", stream_cast<string>(mpParameters->mHashBitSize));
		bitmask = (2 << (mpParameters->mHashBitSize - 1)) - 1;
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("hash_bit_mask", stream_cast<string>(bitmask));
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("lower_vertex_degree_threshold", stream_cast<string>(mpParameters->mVertexDegreeThreshold));
		if (mpParameters->mUseRealVectorInformation) {
			dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("use_real_vector_information", "true");
			dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mWDK.set_flag("num_rand_projections",stream_cast<string>(mpParameters->mNumRandomProjections));
		}
		if (mpParameters->mMinKernel)
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("min_kernel", "true");
		if (mpParameters->mKernelNoNormalization)
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("normalization", "false");
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("radius", stream_cast<string>(mpParameters->mRadius));
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("distance", stream_cast<string>(mpParameters->mDistance));
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("hash_bit_size", stream_cast<string>(mpParameters->mHashBitSize));
		bitmask = (2 << (mpParameters->mHashBitSize - 1)) - 1;
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("hash_bit_mask", stream_cast<string>(bitmask));
		dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("vertex_degree_threshold", stream_cast<string>(mpParameters->mVertexDegreeThreshold));
		if (mpParameters->mUseRealVectorInformation) {
			dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("use_real_vector_information", "true");
			dynamic_cast<PBK_FeatureGenerator*>(mpFeatureGenerator)->mANSPDK.set_flag("num_rand_projections",stream_cast<string>(mpParameters->mNumRandomProjections));
		}
	}

	//DD kernel family additional parameters
	if (mpParameters->mKernelType == "DDK") {
		dynamic_cast<DDkernel_FeatureGeneratorNew*>(mpFeatureGenerator)->set_flag("mTreeLambda", stream_cast<string>(mpParameters->mTreeLambda));
	}
	if (mpParameters->mKernelType == "NSDDK" || mpParameters->mKernelType == "ANSDDK") {
		dynamic_cast<NSDDkernel_FeatureGenerator*>(mpFeatureGenerator)->set_flag("mTreeLambda", stream_cast<string>(mpParameters->mTreeLambda));
		dynamic_cast<NSDDkernel_FeatureGenerator*>(mpFeatureGenerator)->set_flag("mRadiusTwo", stream_cast<string>(mpParameters->mRadiusTwo));

	}
#endif

#ifdef USEMULTITHREAD
	for (vector<NSPDK_FeatureGenerator*>::iterator fgen_it = mpFeatureGeneratorList.begin(); fgen_it != mpFeatureGeneratorList.end(); fgen_it++) {
		if (mpParameters->mVerbose)
			(*fgen_it)->set_flag("verbosity", stream_cast<string>(1));
		if (mpParameters->mMinKernel)
			(*fgen_it)->set_flag("min_kernel", "true");
		if (mpParameters->mKernelNoNormalization)
			(*fgen_it)->set_flag("normalization", "false");
		(*fgen_it)->set_flag("radius", stream_cast<string>(mpParameters->mRadius));
		(*fgen_it)->set_flag("distance", stream_cast<string>(mpParameters->mDistance));
		(*fgen_it)->set_flag("hash_bit_size", stream_cast<string>(mpParameters->mHashBitSize));
		unsigned bitmask = (2 << (mpParameters->mHashBitSize - 1)) - 1;
		(*fgen_it)->set_flag("hash_bit_mask", stream_cast<string>(bitmask));
		if (mpParameters->mUseRealVectorInformation) {
			(*fgen_it)->set_flag("use_real_vector_information", "true");
			(*fgen_it)->set_flag("num_rand_projections", stream_cast<string>(mpParameters->mNumRandomProjections));
		}
		//if type of kernel PBK then also perform the following initializations
		if (mpParameters->mKernelType == "PBK") {
			(*fgen_it)->set_flag("lower_vertex_degree_threshold", stream_cast<string>(mpParameters->mVertexDegreeThreshold));
			(*fgen_it)->set_flag("vertex_degree_threshold", stream_cast<string>(mpParameters->mVertexDegreeThreshold));

			if (mpParameters->mMinKernel)
				dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("min_kernel", "true");
			if (mpParameters->mKernelNoNormalization)
				dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("normalization", "false");
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("radius", stream_cast<string>(mpParameters->mRadius));
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("distance", stream_cast<string>(mpParameters->mDistance));
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("hash_bit_size", stream_cast<string>(mpParameters->mHashBitSize));
			bitmask = (2 << (mpParameters->mHashBitSize - 1)) - 1;
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("hash_bit_mask", stream_cast<string>(bitmask));
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("lower_vertex_degree_threshold", stream_cast<string>(mpParameters->mVertexDegreeThreshold));
			if (mpParameters->mUseRealVectorInformation) {
				dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("use_real_vector_information", "true");
				dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mWDK.set_flag("num_rand_projections", stream_cast<string>(mpParameters->mNumRandomProjections));
			}

			if (mpParameters->mMinKernel)
				dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("min_kernel", "true");
			if (mpParameters->mKernelNoNormalization)
				dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("normalization", "false");
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("radius", stream_cast<string>(mpParameters->mRadius));
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("distance", stream_cast<string>(mpParameters->mDistance));
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("hash_bit_size", stream_cast<string>(mpParameters->mHashBitSize));
			bitmask = (2 << (mpParameters->mHashBitSize - 1)) - 1;
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("hash_bit_mask", stream_cast<string>(bitmask));
			dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("vertex_degree_threshold", stream_cast<string>(mpParameters->mVertexDegreeThreshold));
			if (mpParameters->mUseRealVectorInformation) {
				dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("use_real_vector_information", "true");
				dynamic_cast<PBK_FeatureGenerator*>(*fgen_it)->mANSPDK.set_flag("num_rand_projections", stream_cast<string>(mpParameters->mNumRandomProjections));
			}
		}

		//DD kernel family additional parameters
		if (mpParameters->mKernelType == "DDK") {
			dynamic_cast<DDkernel_FeatureGeneratorNew*>(*fgen_it)->set_flag("mTreeLambda", stream_cast<string>(mpParameters->mTreeLambda));
		}
		if (mpParameters->mKernelType == "NSDDK" || mpParameters->mKernelType == "ANSDDK") {
			dynamic_cast<NSDDkernel_FeatureGenerator*>(*fgen_it)->set_flag("mTreeLambda", stream_cast<string>(mpParameters->mTreeLambda));
			dynamic_cast<NSDDkernel_FeatureGenerator*>(*fgen_it)->set_flag("mRadiusTwo", stream_cast<string>(mpParameters->mRadiusTwo));

		}

		/*		if (!mpParameters->mMinimalOutput) {
		 cout << SEP << endl << "Kernel parameters" << endl << SEP << endl;
		 (*fgen_it)->OutputParameters(cout);
		 cout << SEP << endl;
		 }
		 */
	}
#endif
}

void Kernel::GenerateFeatureVector(GraphClass& aG, SVector& oX) {
#ifndef USEMULTITHREAD
	mpFeatureGenerator->generate_feature_vector(aG, oX);
#endif
#ifdef USEMULTITHREAD
	mpFeatureGeneratorList[omp_get_thread_num()]->generate_feature_vector(aG, oX);
#endif
}

void Kernel::GenerateVertexFeatureVector(GraphClass& aG, vector<SVector>& oXList) {
#ifndef USEMULTITHREAD
	mpFeatureGenerator->generate_vertex_feature_vector(aG, oXList);
#endif
#ifdef USEMULTITHREAD
	mpFeatureGeneratorList[omp_get_thread_num()]->generate_vertex_feature_vector(aG, oXList);
#endif
}

double Kernel::ComputeKernel(GraphClass& aG, GraphClass& aM) {
	SVector xg(pow(2, mpParameters->mHashBitSize));
	SVector xm(pow(2, mpParameters->mHashBitSize));
	GenerateFeatureVector(aG, xg);
	GenerateFeatureVector(aM, xm);
	return ComputeKernel(xg, xm);
}

double Kernel::ComputeKernel(const SVector& aX, const SVector& aZ) {
	return aX.dot(aZ);
}
