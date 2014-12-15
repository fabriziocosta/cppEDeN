# SVM with stochastic gradient

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11

#CHECKLIMITS activates the limit check for graph operations

CXX=g++
OPTS=-g -O3 -Wno-deprecated -DLOSS=1 -static -fopenmp -DUSEMULTITHREAD -DEIGEN_DONT_PARALLELIZE # -DDEBUGON # -DCHECKLIMITS
#OPTS=-g -Wno-deprecated -DLOSS=1 -static -DEIGEN_DONT_PARALLELIZE # -DDEBUGON # -DCHECKLIMITS
#OPTS=-g -Wno-deprecated -DLOSS=1 #-DDEBUGON
#OPTS=-g -Wno-deprecated -static -pg
CXXFLAGS=${OPTS} -Wall -I eigen-eigen-3.20  #-mtune=native -march=native 
#OBABEL=-I /usr/local/openbabel/2.3.1/include/openbabel-2.0  -L /usr/local/openbabel/2.3.1/lib -DUSEOBABEL
#OBABEL=-I /usr/include/openbabel-2.0 -DUSEOBABEL
#CXXFLAGS= ${OPTS} 
LIBS=-lm -lz #-lopenbabel 

PROGRAMS=EDeN 

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS}
	-rm *.o
	
objects := $(patsubst %.cc,%.o,$(wildcard *.cc))

EDeN : $(objects)
	${CXX} ${CXXFLAGS} ${OBABEL} $(objects) ${LIBS} -o $@

EDeN.o: EDeN.cc vectors.h gzstream.h OpenBabelConverter.cc
	 ${CXX} ${CXXFLAGS} -c EDeN.cc ${OBABEL} -o EDeN.o
	 
TestMultilabelMLManager.o:TestMultilabelMLManager.h SGDSVMManager.cc

TrainMultilabelMLManager.o:TrainMultilabelMLManager.h SGDSVMManager.cc

DensityManager.o:DensityManager.h BaseManager.cc NearestNeighbor.cc

SemiSupervisedManager.o:SemiSupervisedManager.h MinHashEncoder.h Data.h Kernel.h BaseManager.cc

ClusterManager.o:ClusterManager.h NearestNeighbor.cc

NearestNeighborManager.o:NearestNeighborManager.h NearestNeighbor.cc

NearestNeighbor.o:NearestNeighbor.h MinHashEncoder.h BaseManager.cc

PreClusterFilterManager.o:PreClusterFilterManager.h MinHashEncoder.h BaseManager.cc

MinHashManager.o:MinHashManager.h MinHashEncoder.h BaseManager.cc

FeatureManager.o:FeatureManager.h SGDSVMManager.o MinHashEncoder.h Data.h Kernel.h BaseManager.cc

TestManager.o:TestManager.h SGDSVMManager.cc BaseManager.cc

ParametersOptimizationManager.o:ParametersOptimizationManager.h CrossValidationManager.h BaseManager.cc

BiasVarianceDecompositionManager.o:BiasVarianceDecompositionManager.h CrossValidationManager.h BaseManager.cc

LearningCurveManager.o:LearningCurveManager.h SGDSVMManager.o BaseManager.cc

CrossValidationManager.o:CrossValidationManager.h SGDSVMManager.o BaseManager.cc

SGDSVMManager.o:SGDSVMManager.h SGDSVM.o BaseManager.cc

TargetAlignmentManager.o:TargetAlignmentManager.h BaseManager.cc

EmbedManager.o:EmbedManager.h NearestNeighbor.o BaseManager.cc 

MinHashEncoder.o:MinHashEncoder.h Data.h Kernel.h

GramMatrixManager.o:GramMatrixManager.h BaseManager.cc

BaseManager.o:BaseManager.h Data.h Kernel.h

SGDSVM.o:SGDSVM.h Data.h Kernel.h vectors.cc

Data.o:Data.h Kernel.h vectors.cc

Kernel.o:Kernel.h Utility.h GraphClass.h BaseGraphClass.cc FlagsService.cc DDK.o

DDK.o: DDK_FeatureGenerator.h DDK_FeatureGenerator.cc NSPDK_FeatureGenerator.cc
	${CXX} ${CXXFLAGS} -c DDK_FeatureGenerator.cc $(LIBS) $(OBABEL) -o DDK.o
	
NSPDK_FeatureGenerator.o:NSPDK_FeatureGenerator.h GraphClass.cc 

FlagsService.o:FlagsService.h

GraphClass.o:GraphClass.h BaseGraphClass.cc 

BaseGraphClass.o:BaseGraphClass.h 

OpenBabelConverter.o:OpenBabelConverter.h GraphClass.h BaseGraphClass.cc
	${CXX} ${CXXFLAGS} -c OpenBabelConverter.cc ${OBABEL} -o OpenBabelConverter.o

Parameters.o:Parameters.h Utility.h

Utility.o:Utility.h vectors.h

vectors.o: vectors.cc vectors.h  wrapper.h

gzstream.o: gzstream.cc gzstream.h 
