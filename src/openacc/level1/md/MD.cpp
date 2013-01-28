#include <cassert>
#include <cfloat>
#include <iostream>
#include <list>
#include <math.h>
#include <stdlib.h>

#include "constants.h"
#include "MD.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Utility.h"

using namespace std;




// ****************************************************************************
// Function: checkResults
//
// Purpose:
//   Check device results against cpu results
//
// Arguments:
//      d_force:   forces calculated on the device
//      position:  positions of atoms
//      neighList: atom neighbor list
//      nAtom:     number of atoms
// Returns:  true if results match, false otherwise
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
template <class T, class forceVecType, class posVecType>
bool checkResults(forceVecType* d_force, posVecType *position,
                  int *neighList, int nAtom)
{
    for (int i = 0; i < nAtom; i++)
    {
        posVecType ipos = position[i];
        forceVecType f = {0.0f, 0.0f, 0.0f};
        int j = 0;
        while (j < maxNeighbors)
        {
            int jidx = neighList[j*nAtom + i];
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
        if (err > (3.0 * EPSILON))
        {
            cout << "Test Failed, idx: " << i << " diff: " << err << "\n";
            cout << "f.x: " << f.x << " df.x: " << d_force[i].x << "\n";
            cout << "f.y: " << f.y << " df.y: " << d_force[i].y << "\n";
            cout << "f.z: " << f.z << " df.z: " << d_force[i].z << "\n";
            cout << "Test FAILED\n";
            return false;
        }
    }
    cout << "Test Passed\n";
    return true;
}


// ********************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ********************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "1",
                     "specify MD kernel iterations", 'r');
}

// ********************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the md benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
// 1/25/13 - KS: Ported driver from CUDA to OpenACC
// ********************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    // TODO Test to see if this device supports double precision
    cout << "Running single precision test" << endl;
    runTest<float, float3, float4>("MD-LJ", resultDB, op);
        
    cout << "Running double precision test" << endl;
    runTest<double, double3, double4>("MD-LJ-DP", resultDB, op);
}

template <class T, class forceVecType, class posVecType>
void runTest(const string& testName, ResultDatabase& resultDB, OptionParser& op)
{
    // Problem Parameters
    const int probSizes[4] = { 12288, 24576, 36864, 73728 };
    int sizeClass = op.getOptionInt("size");
    assert(sizeClass >= 0 && sizeClass < 5);
    int nAtom = probSizes[sizeClass - 1];

    // Allocate problem data on host
    posVecType* position = new posVecType[nAtom];
    forceVecType* force  = new forceVecType[nAtom];
    int* neighborList    = new int[nAtom * maxNeighbors];

    cout << "Initializing test problem (this can take several "
            "minutes for large problems)\n";

    // Seed random number generator
    srand48(8650341L);

    // Initialize positions -- random distribution in cubic domain
    // domainEdge constant specifies edge length
    for (int i = 0; i < nAtom; i++)
    {
        position[i].x = (T)(drand48() * domainEdge);
        position[i].y = (T)(drand48() * domainEdge);
        position[i].z = (T)(drand48() * domainEdge);
    }

    // Keep track of how many atoms are within the cutoff distance to
    // accurately calculate FLOPS later
    int totalPairs = buildNeighborList<T, posVecType>(nAtom, position,
            neighborList);

    cout << "Finished.\n";
    cout << totalPairs << " of " << nAtom*maxNeighbors <<
            " pairs within cutoff distance = " <<
            100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %" << endl;

    double totalTime, kernelTime;
    
    // Warmup Kernel call...
    lj(1, nAtom, maxNeighbors, force, position,
       neighborList, &kernelTime, &totalTime);
                         
    // If results are incorrect, skip the performance tests
    cout << "Performing Correctness Check (can take several minutes)\n";
    if (!checkResults<T, forceVecType, posVecType>
            (force, position, neighborList, nAtom))
    {
        return;
    }
    
    int passes = op.getOptionInt("passes");
    int iter   = op.getOptionInt("iterations");
    for (int i = 0; i < passes; i++)
    {
        
        lj(iter, nAtom, maxNeighbors, force, position,
             neighborList, &kernelTime, &totalTime);
        
        double transferTime = totalTime - kernelTime;
        kernelTime /= (double)iter;

        // Total number of flops
        // Every pair of atoms compute distance - 8 flops
        // totalPairs with distance < cutsq perform an additional 13
        // for force calculation
        double gflops = ((8 * nAtom * maxNeighbors) + (totalPairs * 13)) * 1e-9;

        char atts[64];
        sprintf(atts, "%d_atoms", nAtom);;
        resultDB.AddResult(testName, atts, "GFLOPS", gflops / kernelTime);
        resultDB.AddResult(testName+"_PCIe", atts, "GFLOPS",
                gflops / (kernelTime + transferTime));

        int numPairs = nAtom * maxNeighbors;
        long int nbytes = (3 * sizeof(T) * (1+numPairs)) + // position data
                          (3 * sizeof(T) * nAtom) + // force for each atom
                          (sizeof(int) * numPairs); // neighbor list 
        double gbytes = (double)nbytes / (1000. * 1000. * 1000.);
        resultDB.AddResult(testName + "-Bandwidth", atts, "GB/s", gbytes /
                kernelTime);
        resultDB.AddResult(testName + "-Bandwidth_PCIe", atts, "GB/s",
                gbytes / (kernelTime + transferTime));

        resultDB.AddResult(testName+"_Parity", atts, "N",
                transferTime / kernelTime);
    }
    // Clean up
    delete[] position;
    delete[] force;
    delete[] neighborList;
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
//   Builds the neighbor list structure for all atoms for GPU coalesced reads
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
        int* neighborList)
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
                neighborList);
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
        int* neighborList)
{
    int idx = 0;
    int validPairs = 0; // Pairs of atoms closer together than the cutoff

    // Iterate across distance and neighbor list
    typename list<T>::iterator distanceIter = currDist.begin();
    for (list<int>::iterator neighborIter = currList.begin();
            neighborIter != currList.end(); neighborIter++)
    {
        // Populate packed neighbor list
        neighborList[(idx * nAtom) + i] = *neighborIter;

        // If the distance is less than cutoff, increment valid counter
        if (*distanceIter < cutsq)
            validPairs++;

        // Increment idx and distance iterator
        idx++;
        distanceIter++;
    }
    return validPairs;
}
