#ifdef PARALLEL
// When using MPICH and MPICH-derived MPI implementations, there is a
// naming conflict between stdio.h and MPI's C++ binding.
// Since we do not use the C++ MPI binding, we can avoid the ordering
// issue by ignoring the C++ MPI binding headers.
// This #define should be quietly ignored when using other MPI implementations.
#define MPICH_SKIP_MPICXX
#include <mpi.h>
#endif
#include <iostream>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ResultDatabase.h"
#include "OptionParser.h"
#include "Utility.h"

#ifdef PARALLEL
#include <ParallelResultDatabase.h>
#include <ParallelHelpers.h>
#include <ParallelMerge.h>
#include <NodeInfo.h>
#endif

using namespace std;

// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op);

// ****************************************************************************
// Function: EnumerateDevicesAndChoose
//
// Purpose:
//   This function queries cuda about the available gpus in the system, prints
//   those results to standard out, and selects a device for use in the
//   benchmark.
//
// Arguments:
//   chooseDevice: logical number for the desired device
//   verbose: whether or not to print verbose output
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation:
//
// Modifications:
//   Jeremy Meredith, Tue Oct  9 17:27:04 EDT 2012
//   Added a windows-specific --noprompt, which unless the user passes it,
//   prompts the user to press enter before the program exits on Windows.
//   This is because on Windows, the console disappears when the program
//   exits, but our results go to the console.
//
// ****************************************************************************
void EnumerateDevicesAndChoose(int chooseDevice, bool verbose)
{
    cudaSetDevice(chooseDevice);
    int actualdevice;
    cudaGetDevice(&actualdevice);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (verbose)
    {
        cout << "Number of devices = " << deviceCount << "\n";
    }
    string deviceName = "";
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        if (device == actualdevice)
            deviceName = deviceProp.name;
        if (verbose)
        {
            cout << "Device " << device << ":\n";
            cout << "  name               = '" << deviceProp.name << "'"
                    << endl;
            cout << "  totalGlobalMem     = " << HumanReadable(
                    deviceProp.totalGlobalMem) << endl;
            cout << "  sharedMemPerBlock  = " << HumanReadable(
                    deviceProp.sharedMemPerBlock) << endl;
            cout << "  regsPerBlock       = " << deviceProp.regsPerBlock
                    << endl;
            cout << "  warpSize           = " << deviceProp.warpSize << endl;
            cout << "  memPitch           = " << HumanReadable(
                    deviceProp.memPitch) << endl;
            cout << "  maxThreadsPerBlock = " << deviceProp.maxThreadsPerBlock
                    << endl;
            cout << "  maxThreadsDim[3]   = " << deviceProp.maxThreadsDim[0]
                    << "," << deviceProp.maxThreadsDim[1] << ","
                    << deviceProp.maxThreadsDim[2] << endl;
            cout << "  maxGridSize[3]     = " << deviceProp.maxGridSize[0]
                    << "," << deviceProp.maxGridSize[1] << ","
                    << deviceProp.maxGridSize[2] << endl;
            cout << "  totalConstMem      = " << HumanReadable(
                    deviceProp.totalConstMem) << endl;
            cout << "  major (hw version) = " << deviceProp.major << endl;
            cout << "  minor (hw version) = " << deviceProp.minor << endl;
            cout << "  clockRate          = " << deviceProp.clockRate << endl;
            cout << "  textureAlignment   = " << deviceProp.textureAlignment
                    << endl;
        }
    }
    cout << "Chose device:"
         << " name='"<<deviceName<<"'"
         << " index="<<actualdevice
         << endl;
}

// ****************************************************************************
// Function: main
//
// Purpose:
//   The main function takes care of initialization (device and MPI),  then
//   performs the benchmark and prints results.
//
// Arguments:
//
//
// Programmer: Jeremy Meredith
// Creation:
//
// Modifications:
//   Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//   Split timing reports into detailed and summary.  For serial code, we
//   report all trial values, and for parallel, skip the per-process vals.
//   Also detect and print outliers from parallel runs.
//
// ****************************************************************************
int main(int argc, char *argv[])
{
    int ret = 0;
    bool noprompt = false;

    try
    {
#ifdef PARALLEL
        int rank, size;
        MPI_Init(&argc,&argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        cerr << "MPI Task " << rank << "/" << size - 1 << " starting....\n";
#endif

        // Get args
        OptionParser op;

        //Add shared options to the parser
        op.addOption("device", OPT_VECINT, "0",
                "specify device(s) to run on", 'd');
        op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
        op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
        op.addOption("size", OPT_INT, "1", "specify problem size", 's');
        op.addOption("infoDevices", OPT_BOOL, "",
                "show info for available platforms and devices", 'i');
        op.addOption("quiet", OPT_BOOL, "", "write minimum necessary to standard output", 'q');
#ifdef _WIN32
        op.addOption("noprompt", OPT_BOOL, "", "don't wait for prompt at program exit");
#endif

        addBenchmarkSpecOptions(op);

        if (!op.parse(argc, argv))
        {
#ifdef PARALLEL
            if (rank == 0)
                op.usage();
            MPI_Finalize();
#else
            op.usage();
#endif
            return (op.HelpRequested() ? 0 : 1);
        }

        bool verbose = op.getOptionBool("verbose");
        bool infoDev = op.getOptionBool("infoDevices");
#ifdef _WIN32
        noprompt = op.getOptionBool("noprompt");
#endif

        int device;
#ifdef PARALLEL
        NodeInfo ni;
        int myNodeRank = ni.nodeRank();
        vector<long long> deviceVec = op.getOptionVecInt("device");
        if (myNodeRank >= deviceVec.size()) {
            // Default is for task i to test device i
            device = myNodeRank;
        } else {
            device = deviceVec[myNodeRank];
        }
#else
        device = op.getOptionVecInt("device")[0];
#endif
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (device >= deviceCount) {
            cerr << "Warning: device index: " << device <<
            " out of range, defaulting to device 0.\n";
            device = 0;
        }

        // Initialization
        EnumerateDevicesAndChoose(device, infoDev);
        if( infoDev )
        {
            return 0;
        }
        ResultDatabase resultDB;

        // Run the benchmark
        RunBenchmark(resultDB, op);

#ifndef PARALLEL
        resultDB.DumpDetailed(cout);
#else
        ParallelResultDatabase pardb;
        pardb.MergeSerialDatabases(resultDB,MPI_COMM_WORLD);
        if (rank==0)
        {
            pardb.DumpSummary(cout);
            pardb.DumpOutliers(cout);
        }
#endif

    }
    catch( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
        ret = 1;
    }
    catch( ... )
    {
        ret = 1;
    }


#ifdef PARALLEL
    MPI_Finalize();
#endif

#ifdef _WIN32
    if (!noprompt)
    {
        cout << "Press return to exit\n";
        cin.get();
    }
#endif

    return ret;
}
