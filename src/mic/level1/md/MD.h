// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Jeffrey Vetter <vetter@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011-2013, UT-Battelle, LLC
// Copyright (c) 2013, Intel Corporation
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//   
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Oak Ridge National Laboratory, nor UT-Battelle, LLC, 
//    nor the names of its contributors may be used to endorse or promote 
//    products derived from this software without specific prior written 
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
// THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __MD_H
#define __MD_H

#include "ResultDatabase.h"
#include "OptionParser.h"
#include <string>
#include <list>

static const float lj1 = 1.5;  // LJ constants
static const float lj2 = 2.0;

typedef struct float3 {
   float x;
   float y;
   float z;
   float w;
} float3;

typedef struct double3 {
   double x;
   double y;
   double z;
   double w;
} double3;

template <class T, class posVecType>
T distance(const posVecType* position, const int i, const int j);

template <class T>
void insertInOrder(std::list<T>& currDist, std::list<int>& currList,
        const int j, const T distIJ, const int maxNeighbors);

template <class T, class posVecType>
int buildNeighborList(const int nAtom, const posVecType* position,
        int* neighborList, double cutsq, int maxNeighbors);

template <class T>
int populateNeighborList(std::list<T>& currDist,
        std::list<int>& currList, const int j, const int nAtom,
        int* neighborList, double cutsq, int maxNeighbors);

template <class T, class forceVecType, class posVecType, bool useMIC>
void runTest(const string& testName, ResultDatabase& resultDB, 
        OptionParser& op);

#endif // __MD_H
