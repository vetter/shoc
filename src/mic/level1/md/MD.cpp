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

// ****************************************************************************
// File: MD.cpp
//
// Purpose:
//   Contains performance test for a (somewhat) simplified molecular dynamics
//   kernel. This kernel is based on the Lennard-Jones potential in LAMMPS.
//
// Programmer:  Kyle Spafford
// Creation:    November 19, 2010
//
// ****************************************************************************
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <list>

#include "offload.h"
#include "omp.h"

#include "MD.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

#ifdef __MIC2__
#include <pthread.h>
#endif

using namespace std;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line options
//
// Arguments:
//      op: reference to the OptionParser object
//
// Returns: nothing
//
// Programmer: Kyle Spafford
// Creation: November 24, 2010
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser& op) {
   // Problem Constants
   op.addOption("nAtom", OPT_INT, "0", "number of atoms");
   op.addOption("cutsq", OPT_FLOAT, "16.0", "cutoff distance squared");
   op.addOption("maxNeighbors", OPT_INT, "128", "max length of neighbor list");
   op.addOption("domain", OPT_FLOAT, "20.0", "edge length of the cubic domain");
   op.addOption("eps", OPT_FLOAT, "0.1", "relative error tolerance");
   op.addOption("iterations", OPT_INT, "16", "number of kernel calls per pass");
}

// ****************************************************************************
// Function: compute_lj_force
//
// Purpose:
//   Kernel to calculate Lennard Jones force
//
// Arguments:
//      force3:     array to store the calculated forces
//      position:   positions of atoms
//      neighCount: number of neighbors for each atom to consider
//      neighList:  atom neighbor list
//      cutsq:      cutoff distance squared
//      lj1, lj2:   LJ force constants
//      inum:       total number of atoms
//
// Returns: nothing
//
// Programmer: Kyle Spafford
// Creation: November 24, 2010
//
// Modifications:
//
// ****************************************************************************
template <class T, class forceVecType, class posVecType>
__declspec(target(mic)) void compute_lj_force(forceVecType* force3,
                      const posVecType* position,
                      const int neighCount,
                      const int* neighList,
                      const T cutsq,
                      const T lj1,
                      const T lj2,
                      const int inum,
                      const int maxNeighbors)
{
    #pragma omp parallel for shared(position, neighList, force3) \
    schedule(dynamic, 10)
    for (int i = 0; i < inum; i++)
    {
        T iposx = position[i].x;
        T iposy = position[i].y;
        T iposz = position[i].z;

        T fx = 0.0f;
        T fy = 0.0f;
        T fz = 0.0f;

        int jidx;
        T jposx;
        T jposy;
        T jposz;

        int dis = 16;
        int nei = maxNeighbors / dis;
        for(int j = 0;j < nei; j++)
        {
            _mm_prefetch ((void *)&position[neighList[dis*j +
                    maxNeighbors * i]],3);
            _mm_prefetch ((void *)&position[neighList[dis*j + 1 +
                    maxNeighbors * i]],3);
            _mm_prefetch ((void *)&position[neighList[dis*j + 2 + 
                    maxNeighbors * i]],3);
            _mm_prefetch ((void *)&position[neighList[dis*j + 3 +
                    maxNeighbors * i]],3);
            _mm_prefetch ((void *)&position[neighList[dis*j + 4 +
                    maxNeighbors * i]],3);
            _mm_prefetch ((void *)&position[neighList[dis*j + 5 +
                    maxNeighbors * i]],3);
            _mm_prefetch ((void *)&position[neighList[dis*j + 6 +
                    maxNeighbors * i]],3);
            _mm_prefetch ((void *)&position[neighList[dis*j + 7 + 
                    maxNeighbors * i]],3);
            _mm_prefetch ((void *)&position[neighList[dis*j + 8 + 
                    maxNeighbors * i]],3);  
            _mm_prefetch ((void *)&position[neighList[dis*j + 9 +
                    maxNeighbors * i]],3);  
            _mm_prefetch ((void *)&position[neighList[dis*j + 10 +
                    maxNeighbors * i]],3);  
            _mm_prefetch ((void *)&position[neighList[dis*j + 11 +
                    maxNeighbors * i]],3);  
            _mm_prefetch ((void *)&position[neighList[dis*j + 12 +
                    maxNeighbors * i]],3);  
            _mm_prefetch ((void *)&position[neighList[dis*j + 13 +
                    maxNeighbors * i]],3);  
            _mm_prefetch ((void *)&position[neighList[dis*j + 14 +
                    maxNeighbors * i]],3);  
            _mm_prefetch ((void *)&position[neighList[dis*j + 15 +
                    maxNeighbors * i]],3);  

            #pragma simd
            #pragma unroll
            #pragma ivdep
            for(int k = 0; k < dis; k++)
            {
                jposx = position[neighList[j*dis + k + maxNeighbors * i]].x;
                jposy = position[neighList[j*dis + k + maxNeighbors * i]].y;
                jposz = position[neighList[j*dis + k + maxNeighbors * i]].z;

                T delx = iposx - jposx;
                T dely = iposy - jposy;
                T delz = iposz - jposz;
                T r2inv = delx*delx + dely*dely + delz*delz;
                if (r2inv < cutsq) 
                {
                    r2inv = 1.0f/r2inv;
                    T r6inv = r2inv * r2inv * r2inv;
                    T force = r2inv*r6inv*(lj1*r6inv - lj2);
                    fx += delx * force;
                    fy += dely * force;
                    fz += delz * force;
                }
            }    
        }
        force3[i].x = fx;
        force3[i].y = fy;
        force3[i].z = fz;
    }
}

template <class T, class forceVecType, class posVecType>
bool checkResults(forceVecType* d_force, posVecType *position,
                  int *neighList, int nAtom, double eps, int maxNeighbors,
                  double cutsq)
{
    for (int i = 0; i < nAtom; i++)
    {
        posVecType ipos = position[i];
        forceVecType f = {0.0f, 0.0f, 0.0f};
        int j = 0;
        while (j < maxNeighbors)
        {
            int jidx = neighList[j + maxNeighbors * i];
            posVecType jpos = position[jidx];
            // Calculate distance
            T delx = ipos.x - jpos.x;
            T dely = ipos.y - jpos.y;
            T delz = ipos.z - jpos.z;
            T r2inv = delx*delx + dely*dely + delz*delz;

            // If distance is less than cutoff, calculate force
            if (r2inv < cutsq) {

                r2inv = 1.0f/r2inv;
                T r6inv = r2inv * r2inv * r2inv;
                T force = r2inv*r6inv*(lj1*r6inv - lj2);

                f.x += delx * force;
                f.y += dely * force;
                f.z += delz * force;
            }
            j++;
        }
        // Check the results
        T diffx = (d_force[i].x - f.x) / d_force[i].x;
        T diffy = (d_force[i].y - f.y) / d_force[i].y;
        T diffz = (d_force[i].z - f.z) / d_force[i].z;
        T err = sqrt(diffx*diffx) + sqrt(diffy*diffy) + sqrt(diffz*diffz);
        if (err > (3.0 * eps))
        {
            cout << "Test Failed\n";
            return false;
        }
    }
    cout << "Test Passed\n";
    return true;
}

void
RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
   runTest<float,   float3,  float3, true>("MD-LJ",    resultDB, op);
   runTest<double, double3, double3, true>("MD-LJ-DP", resultDB, op);
}

template <class T, class forceVecType, class posVecType, bool useMIC>
void runTest(const string& testName, ResultDatabase& resultDB, OptionParser& op)
{
    __declspec(target(mic))    posVecType* position;
    __declspec(target(mic)) forceVecType* force ;
    __declspec(target(mic)) int* neighborList;


    // Problem Parameters
    const int probSizes[4] = { 12288, 24576, 36864, 73728 };
    int sizeClass = op.getOptionInt("size");
    assert(sizeClass >= 0 && sizeClass < 5);
    int nAtom = probSizes[sizeClass - 1];

    // If a custom number of atoms is specified on command line...
    if (op.getOptionInt("nAtom") != 0)
    {
       nAtom = op.getOptionInt("nAtom");
    }
    
    // Problem Constants
    const float  cutsq        = op.getOptionFloat("cutsq");
    const int    maxNeighbors = op.getOptionInt("maxNeighbors");
    const double domainEdge   = op.getOptionFloat("domain");
    const double eps          = op.getOptionFloat("eps");

    // Allocate problem data on host
    position = (posVecType *) _mm_malloc(nAtom*sizeof(posVecType),
            64);
    force = (forceVecType*) 
        _mm_malloc(nAtom*sizeof(forceVecType), 64);
    neighborList = (int*) _mm_malloc(nAtom*maxNeighbors*sizeof(int), 64);

    cout << "Initializing test problem (this can take several "
            "minutes for large problems)\n";

    // Seed random number generator
    srand48(8650341L);

    // Initialize positions -- random distribution in cubic domain
    // domainEdge constant specifies edge length    
    #pragma omp parallel for
    for (int i = 0; i < nAtom; i++)
    {
        position[i].x = (T)(drand48() * domainEdge);
        position[i].y = (T)(drand48() * domainEdge);
        position[i].z = (T)(drand48() * domainEdge);
    }

    // Keep track of how many atoms are within the cutoff distance to
    // accurately calculate FLOPS later
    int totalPairs = buildNeighborList<T, posVecType>(nAtom, position,
            neighborList, cutsq, maxNeighbors);

    cout << "Finished.\n";
    cout << totalPairs << " of " << nAtom*maxNeighbors <<
            " pairs within cutoff distance = " <<
            100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %" << endl;

    //Pick one device when a system has multiple cards
    const int micdev = op.getOptionInt("device");


    // Do device memory allocation without transfer
    size_t nl_length = nAtom * maxNeighbors;
    #pragma offload target(mic:micdev) if(useMIC) \
        nocopy(position:length(nAtom) alloc_if(1) free_if(0))         \
        nocopy(neighborList:length(nl_length) alloc_if(1)  free_if(0)) \
        nocopy(force:length(nAtom) alloc_if(1) free_if(0))
    {
    }

    // Time the transfer of input data to the GPU
    int txToDevTimerHandle = Timer::Start();
    #pragma offload target(mic:micdev) if(useMIC) \
        in(position:length(nAtom) alloc_if(0) free_if(0))         \
        in(neighborList:length(nl_length) alloc_if(0) free_if(0)) \
        nocopy(force:length(nAtom) alloc_if(0) free_if(0))
    {
    }
    double transferTime = Timer::Stop(txToDevTimerHandle, "tx to dev");
    
    // Warm up the kernel and check correctness
    #pragma offload target(mic:micdev) if(useMIC) \
        nocopy(position:length(nAtom) alloc_if(0) free_if(0))         \
        nocopy(neighborList:length(nl_length) alloc_if(0) free_if(0)) \
        nocopy(force:length(nAtom) alloc_if(0) free_if(0))
    compute_lj_force<T, forceVecType, posVecType>
                    (force, position, maxNeighbors, neighborList,cutsq, lj1, lj2, nAtom, maxNeighbors);
    
    // Copy back forces
    int txFromDevTimerHandle = Timer::Start();
    #pragma offload target(mic:micdev) if(useMIC) \
        nocopy(position:length(nAtom) alloc_if(0) free_if(0))         \
        nocopy(neighborList:length(nl_length) alloc_if(0) free_if(0)) \
        out(force:length(nAtom) alloc_if(0) free_if(0))
    {
    }
    transferTime += Timer::Stop(txFromDevTimerHandle, "tx from device" );
    
    // If results are incorrect, skip the performance tests
    cout << "Performing Correctness Check (can take several minutes)\n";
    if (!checkResults<T, forceVecType, posVecType>
            (force, position, neighborList, nAtom, eps, maxNeighbors, cutsq))
    {
        cerr << "Correctness check failed, skipping perf tests." << endl;
        return;
    }



    // Begin performance tests
    cout << "Starting Performance Tests" << endl;
    
    int passes = op.getOptionInt("passes");
    int iter  = op.getOptionInt("iterations");


    for (int i = 0; i < passes; i++)
    {
        int kernelTimerHandle = Timer::Start();

        #pragma offload target(mic:micdev) if(useMIC) \
            nocopy(position:length(nAtom) alloc_if(0) free_if(0))         \
            nocopy(neighborList:length(nl_length) alloc_if(0) free_if(0)) \
            nocopy(force:length(nAtom) alloc_if(0) free_if(0))
        {
            for (int j = 0; j < iter; j++)
            {
                compute_lj_force<T, forceVecType, posVecType>
                    (force, position, maxNeighbors, neighborList, cutsq,lj1, lj2, nAtom, maxNeighbors);
            }
        }
        double kernelTime = Timer::Stop(kernelTimerHandle, "md") / (double)iter;
        double totalTime = kernelTime + transferTime;

        // Total number of flops
        // Every pair of atoms compute distance - 8 flops
        // totalPairs with distance < cutsq perform an additional 13
        // for force calculation
        double gflops = ((8 * nAtom * maxNeighbors) + (totalPairs * 13)) * 1e-9;

        char atts[64];
        sprintf(atts, "%d_atoms", nAtom);
        resultDB.AddResult(testName, atts, "GFLOPS", gflops / kernelTime);
        resultDB.AddResult(testName + "-PCIe", atts, "GFLOPS", 
            gflops / totalTime);

        int numPairs = nAtom * maxNeighbors;
        long int nbytes = (3 * sizeof(T) * (1+numPairs)) + // position data
                          (3 * sizeof(T) * nAtom) + // force for each atom
                          (sizeof(int) * numPairs); // neighbor list 
        double gbytes = (double)nbytes / (1000. * 1000. * 1000.);
        resultDB.AddResult(testName + "-Bandwidth", atts, "GB/s", gbytes /
                kernelTime);

        resultDB.AddResult(testName + "-Bandwidth_PCIe", atts, "GB/s",
                gbytes / totalTime);

        resultDB.AddResult(testName+"_Parity", atts, "N",
                (transferTime) / kernelTime);
    }
    
    // Clean up MIC device memory
    #pragma offload target(mic:micdev) if(useMIC) \
        in(position:length(nAtom) alloc_if(0) free_if(1))       \
        in(neighborList:length(nl_length) alloc_if(0) free_if(1)) \
        out(force:length(nAtom) alloc_if(0) free_if(1) )
    {
    }

    // Clean up host
    _mm_free(position);
    _mm_free(force);
    _mm_free(neighborList);
}

// ********************************************************
// Function: distance
//
// Purpose:
//   Calculates distance squared between two atoms
//
// Arguments:
//   position: atom position information
//   i, j: indexes of the two atoms
//
// Returns:  the computed distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T, class posVecType>
inline T distance(const posVecType* position, const int i, const int j)
{
    posVecType ipos = position[i];
    posVecType jpos = position[j];
    T delx = ipos.x - jpos.x;
    T dely = ipos.y - jpos.y;
    T delz = ipos.z - jpos.z;
    T r2inv = delx * delx + dely * dely + delz * delz;
    return r2inv;
}

// ********************************************************
// Function: insertInOrder
//
// Purpose:
//   Adds atom j to current neighbor list and distance list
//   if it's distance is low enough.
//
// Arguments:
//   currDist: distance between current atom and each of its neighbors in the
//             current list, sorted in ascending order
//   currList: neighbor list for current atom, sorted by distance in asc. order
//   j:        atom to insert into neighbor list
//   distIJ:   distance between current atom and atom J
//   maxNeighbors: max length of neighbor list
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T>
inline void insertInOrder(list<T>& currDist, list<int>& currList,
        const int j, const T distIJ, const int maxNeighbors)
{

    typename list<T>::iterator   it;
    typename list<int>::iterator it2;

    it2 = currList.begin();

    T currMax = currDist.back();

    if (distIJ > currMax) return;

    for (it=currDist.begin(); it!=currDist.end(); it++)
    {
        if (distIJ < (*it))
        {
            // Insert into appropriate place in list
            currDist.insert(it,distIJ);
            currList.insert(it2, j);

            // Trim end of list
            currList.resize(maxNeighbors);
            currDist.resize(maxNeighbors);
            return;
        }
        it2++;
    }
}
// ********************************************************
// Function: buildNeighborList
//
// Purpose:
//   Builds the neighbor list structure for all atoms
//   and counts the number of pairs within the cutoff distance, so
//   the benchmark gets an accurate FLOPS count
//
// Arguments:
//   nAtom:    total number of atoms
//   position: pointer to the atom's position information
//   neighborList: pointer to neighbor list data structure
//
// Returns:  number of pairs of atoms within cutoff distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//   Jeremy Meredith, Tue Oct  9 17:35:16 EDT 2012
//   On some slow systems and without optimization, this
//   could take a while.  Give users a rough completion
//   percentage so they don't give up.
//
// ********************************************************
template <class T, class posVecType>
inline int buildNeighborList(const int nAtom, const posVecType* position,
        int* neighborList, double cutsq, int maxNeighbors)
{
    int totalPairs = 0;
    // Build Neighbor List
    // Find the nearest N atoms to each other atom, where N = maxNeighbors
    for (int i = 0; i < nAtom; i++)
    {
        // Print progress every 10% completion.
        if (int((i+1)/(nAtom/10)) > int(i/(nAtom/10)))
            cout << "  " << 10*int((i+1)/(nAtom/10)) << "% done\n";

        // Current neighbor list for atom i, initialized to -1
        list<int>   currList(maxNeighbors, -1);
        // Distance to those neighbors.  We're populating this with the
        // closest neighbors, so initialize to FLT_MAX
        list<T> currDist(maxNeighbors, FLT_MAX);

        for (int j = 0; j < nAtom; j++)
        {
            if (i == j) continue; // An atom cannot be its own neighbor

            // Calculate distance and insert in order into the current lists
            T distIJ = distance<T, posVecType>(position, i, j);
            insertInOrder<T>(currDist, currList, j, distIJ, maxNeighbors);
        }
        // We should now have the closest maxNeighbors neighbors and their
        // distances to atom i. Populate the neighbor list data structure
        // for GPU coalesced reads.
        // The populate method returns how many of the maxNeighbors closest
        // neighbors are within the cutoff distance.  This will be used to
        // calculate GFLOPS later.
        totalPairs += populateNeighborList<T>(currDist, currList, i, nAtom,
                neighborList, cutsq, maxNeighbors);
    }
    return totalPairs;
}


// ********************************************************
// Function: populateNeighborList
//
// Purpose:
//   Populates the neighbor list structure for a *single* atom for
//   GPU coalesced reads and counts the number of pairs within the cutoff
//   distance, (for current atom) so the benchmark gets an accurate FLOPS count
//
// Arguments:
//   currDist: distance between current atom and each of its maxNeighbors
//             neighbors
//   currList: current list of neighbors
//   i:        current atom
//   nAtom:    total number of atoms
//   neighborList: pointer to neighbor list data structure
//
// Returns:  number of pairs of atoms within cutoff distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T>
inline int populateNeighborList(list<T>& currDist,
        list<int>& currList, const int i, const int nAtom,
        int* neighborList, double cutsq,int maxNeighbors)
{
    int idx = 0;
    int validPairs = 0; // Pairs of atoms closer together than the cutoff

    // Iterate across distance and neighbor list
    typename list<T>::iterator distanceIter = currDist.begin();
    for (list<int>::iterator neighborIter = currList.begin();
            neighborIter != currList.end(); neighborIter++)
    {
        // Populate packed neighbor list
        neighborList[idx + maxNeighbors * i] = *neighborIter;

        // If the distance is less than cutoff, increment valid counter
        if (*distanceIter < cutsq)
            validPairs++;

        // Increment idx and distance iterator
        idx++;
        distanceIter++;
    }
    return validPairs;
}
