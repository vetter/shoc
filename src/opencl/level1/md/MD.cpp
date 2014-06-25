#include <cassert>
#include <cfloat>
#include <list>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "Event.h"
#include "MD.h"
#include "OpenCLDeviceInfo.h"
#include "OptionParser.h"
#include "support.h"
#include "ResultDatabase.h"

using namespace std;

// Forward Declarations
template <class T, class forceVecType, class posVecType>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        string compileFlags);

template <class T, class posVecType>
inline T distance(const posVecType* position, const int i, const int j);

template <class T>
inline void insertInOrder(std::list<T>& currDist, std::list<int>& currList,
        const int j, const T distIJ, const int maxNeighbors);

template <class T, class posVecType>
inline int buildNeighborList(const int nAtom, const posVecType* position,
        int* neighborList);

template <class T>
inline int populateNeighborList(std::list<T>& currDist,
        std::list<int>& currList, const int j, const int nAtom,
        int* neighborList);


// ****************************************************************************
// Function: checkResults
//
// Purpose:
//   Check device results against cpu results -- this is the CPU equivalent of
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


// ****************************************************************************
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
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "8",
            "specify MD kernel iterations", 'r');
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the md benchmark
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
extern const char *cl_source_md;

void
RunBenchmark(cl_device_id dev,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    string spMacros = "-DSINGLE_PRECISION";
    runTest<float, float4, float4>
        ("MD-LJ", dev, ctx, queue, resultDB, op, spMacros);

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DK_DOUBLE_PRECISION ";
        runTest<double, double4, double4>
                ("MD-LJ-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DAMD_DOUBLE_PRECISION ";
        runTest<double, double4, double4>
        ("MD-LJ-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else
    {
        cout << "DP Not Supported\n";
        char atts[32] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (int i = 0; i < passes; i++) {
            resultDB.AddResult("MD-LJ-DP" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("MD-LJ-DP_PCIe" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("MD-LJ-DP-Bandwidth", atts, "GB/s", FLT_MAX);
            resultDB.AddResult("MD-LJ-DP-Bandwidth_PCIe", atts, "GB/s", FLT_MAX);
            resultDB.AddResult("MD-LJ-DP_Parity" , atts, "GB/s", FLT_MAX);
        }
    }
}

template <class T, class forceVecType, class posVecType>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        string compileFlags)
{
    // Problem Parameters
    const int probSizes[4] = { 12288, 24576, 36864, 73728 };
    int sizeClass = op.getOptionInt("size");
    assert(sizeClass >= 0 && sizeClass < 5);
    int nAtom = probSizes[sizeClass - 1];

    // Allocate problem data on host
    cl_mem h_pos, h_force, h_neigh;
    posVecType*   position;
    forceVecType* force;
    int* neighborList;
    int passes = op.getOptionInt("passes");
    int iter   = op.getOptionInt("iterations");

    // Allocate and map pinned host memory
    int err = 0;
    // Position
    h_pos = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(posVecType)*nAtom, NULL, &err);
    CL_CHECK_ERROR(err);
    position = (posVecType*)clEnqueueMapBuffer(queue, h_pos, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, sizeof(posVecType)*nAtom , 0,
            NULL, NULL, &err);
    CL_CHECK_ERROR(err);
    // Force
    h_force = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(forceVecType)*nAtom, NULL, &err);
    CL_CHECK_ERROR(err);
    force = (forceVecType*)clEnqueueMapBuffer(queue, h_force, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, sizeof(forceVecType)*nAtom , 0,
            NULL, NULL, &err);
    CL_CHECK_ERROR(err);
    // Neighbor List
    h_neigh = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(int) * nAtom * maxNeighbors, NULL, &err);
    CL_CHECK_ERROR(err);
    neighborList = (int*)clEnqueueMapBuffer(queue, h_neigh, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, sizeof(int) * nAtom * maxNeighbors, 0,
            NULL, NULL, &err);
    CL_CHECK_ERROR(err);

    // Allocate device memory
    cl_mem d_force = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            nAtom * sizeof(forceVecType), NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem d_position = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            nAtom * sizeof(posVecType), NULL, &err);
    CL_CHECK_ERROR(err);
    // Allocate device memory neighbor list
    cl_mem d_neighborList = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            maxNeighbors * nAtom * sizeof(int), NULL, &err);
    CL_CHECK_ERROR(err);

    size_t maxGroupSize = getMaxWorkGroupSize(dev);
    if (maxGroupSize < 128)
    {
        cout << "MD requires a work group size of at least 128" << endl;
        // Add special values to the results database
        char atts[1024];
        sprintf(atts, "GSize_Not_Supported");
        for (int i=0 ; i<passes ; ++i) {
            resultDB.AddResult(testName, atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult(testName + "_PCIe", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult(testName+"-Bandwidth", atts, "GB/s", FLT_MAX);
            resultDB.AddResult(testName+"-Bandwidth_PCIe", atts, "GB/s", FLT_MAX);
            resultDB.AddResult(testName+"_Parity", atts, "N", FLT_MAX);
        }
        return;
    }
    size_t localSize  = 128;
    size_t globalSize = nAtom;

    cout << "Initializing test problem (this can take several "
            "minutes for large problems).\n                   ";

    // Seed random number generator
    srand48(8650341L);

    // Initialize positions -- random distribution in cubic domain
    for (int i = 0; i < nAtom; i++)
    {
        position[i].x = (drand48() * domainEdge);
        position[i].y = (drand48() * domainEdge);
        position[i].z = (drand48() * domainEdge);
    }

    // Copy position to GPU
    Event evTransfer("h->d transfer");
    err = clEnqueueWriteBuffer(queue, d_position, true, 0,
            nAtom * sizeof(posVecType), position, 0, NULL,
            &evTransfer.CLEvent());
    CL_CHECK_ERROR(err);
    err = clFinish(queue);
    CL_CHECK_ERROR(err);

    evTransfer.FillTimingInfo();
    long transferTime = evTransfer.StartEndRuntime();

    // Keep track of how many atoms are within the cutoff distance to
    // accurately calculate FLOPS later
    int totalPairs = buildNeighborList<T, posVecType>(nAtom, position,
            neighborList);

    cout << "Finished.\n";
    cout << totalPairs << " of " << nAtom*maxNeighbors <<
            " pairs within cutoff distance = " <<
            100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %" << endl;

    // Copy data to GPU
    err = clEnqueueWriteBuffer(queue, d_neighborList, true, 0,
            maxNeighbors * nAtom * sizeof(int), neighborList, 0, NULL,
            &evTransfer.CLEvent());
    CL_CHECK_ERROR(err);
    clFinish(queue);
    evTransfer.FillTimingInfo();
    transferTime += evTransfer.StartEndRuntime();

    // Build the openCL kernel
    cl_program prog = clCreateProgramWithSource(ctx, 1, &cl_source_md, NULL,
            &err);
    CL_CHECK_ERROR(err);

    err = clBuildProgram(prog, 1, &dev, compileFlags.c_str(), NULL, NULL);
    CL_CHECK_ERROR(err);

    // If there is a build error, print the output and return
    if (err != CL_SUCCESS)
    {
        char log[5000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 50000
                * sizeof(char), log, &retsize);
        CL_CHECK_ERROR(err);
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return;
    }

    // Extract out the kernels
    cl_kernel lj_kernel = clCreateKernel(prog, "compute_lj_force",
            &err);
    CL_CHECK_ERROR(err);

    T lj1_t = (T) lj1;
    T lj2_t = (T) lj2;
    T cutsq_t = (T) cutsq;

    // Set kernel arguments
    err = clSetKernelArg(lj_kernel, 0, sizeof(cl_mem),
            (void*) &d_force);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(lj_kernel, 1, sizeof(cl_mem),
            (void*) &d_position);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(lj_kernel, 2, sizeof(cl_int),
            (void*) &maxNeighbors);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(lj_kernel, 3, sizeof(cl_mem),
            (void*) &d_neighborList);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(lj_kernel, 4, sizeof(T),
            (void*) &cutsq_t);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(lj_kernel, 5, sizeof(T),
            (void*) &lj1_t);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(lj_kernel, 6, sizeof(T),
            (void*) &lj2_t);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(lj_kernel, 7, sizeof(cl_int),
            (void*) &nAtom);
    CL_CHECK_ERROR(err);

    Event evLJ("computeLJ");

    // Warm up the kernel and check correctness
    err = clEnqueueNDRangeKernel(queue, lj_kernel, 1, NULL, &globalSize,
            &localSize, 0, NULL, &evLJ.CLEvent());
    CL_CHECK_ERROR(err);
    err = clFinish(queue);
    CL_CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, d_force, true, 0,
            nAtom * sizeof(forceVecType), force, 0, NULL,
            &evTransfer.CLEvent());
    CL_CHECK_ERROR(err);
    err = clFinish(queue);
    CL_CHECK_ERROR(err);
    evTransfer.FillTimingInfo();
    transferTime += evTransfer.StartEndRuntime();


    cout << "Performing Correctness Check (can take several minutes)\n";

    // If results are correct, skip the performance tests
    if (!checkResults<T, forceVecType, posVecType>(force, position,
            neighborList, nAtom))
    {
        return;
    }


    for (int i = 0; i < passes; i++)
    {
        double total_time = 0.0;
        for (int j = 0; j < iter; j++)
        {
            //Launch Kernels
            err = clEnqueueNDRangeKernel(queue, lj_kernel, 1, NULL,
                    &globalSize, &localSize, 0, NULL,
                    &evLJ.CLEvent());
            CL_CHECK_ERROR(err);
            err = clFinish(queue);
            CL_CHECK_ERROR(err);

            // Collect timing info from events
            evLJ.FillTimingInfo();
            total_time += evLJ.SubmitEndRuntime();
        }

        char atts[1024];
        long int nflops = (8 * nAtom * maxNeighbors) + (totalPairs * 13);
        sprintf(atts, "%d_atoms", nAtom);
        total_time /= (double) iter;
        resultDB.AddResult(testName, atts, "GFLOPS",
                ((double) nflops) / total_time);
        resultDB.AddResult(testName + "_PCIe", atts, "GFLOPS",
                        ((double) nflops) / (total_time + transferTime));

        long int numPairs = nAtom * maxNeighbors;
        long int nbytes = (3 * sizeof(T) * (1+numPairs)) + // position data
                          (3 * sizeof(T) * nAtom) + // force for each atom
                          (sizeof(int) * numPairs); // neighbor list
        double gbytes = (double)nbytes / (1000. * 1000. * 1000.);
        double seconds = total_time / 1.e9;
        resultDB.AddResult(testName+"-Bandwidth", atts, "GB/s", gbytes /
                        seconds);
        resultDB.AddResult(testName+"-Bandwidth_PCIe", atts, "GB/s", gbytes /
                           (seconds + (transferTime / 1.e9)));
        resultDB.AddResult(testName+"_Parity", atts, "N",
                                    (transferTime / 1.e9) / seconds);
    }

    // Clean up
    // Host memory
    err = clEnqueueUnmapMemObject(queue, h_pos,   position, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueUnmapMemObject(queue, h_force, force, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueUnmapMemObject(queue, h_neigh, neighborList, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_pos);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_force);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_neigh);
    CL_CHECK_ERROR(err);

    // Program Objects
    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(lj_kernel);
    CL_CHECK_ERROR(err);

    // Device Memory
    err = clReleaseMemObject(d_force);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_position);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_neighborList);
    CL_CHECK_ERROR(err);

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
