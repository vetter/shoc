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
#include <stdlib.h>

#include "OpenCLDeviceInfo.h"
#include "OpenCLNodePlatformContainer.h"
#include "MultiNodeContainer.h"
#include "support.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

#ifdef PARALLEL
#include <ParallelResultDatabase.h>
#include <ParallelHelpers.h>
#include <ParallelMerge.h>
#include <NodeInfo.h>
#endif


using namespace std;
using namespace SHOC;

typedef MultiNodeContainer<OpenCLNodePlatformContainer> OpenCLMultiNodeContainer;

void addBenchmarkSpecOptions(OptionParser &op);

void RunBenchmark(cl_device_id devID,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op);

// ****************************************************************************
// Method:  main()
//
// Purpose:
//   serial and parallel main for OpenCL level0 benchmarks
//
// Arguments:
//   argc, argv
//
// Programmer:  SHOC Team
// Creation:    The Epoch
//
// Modifications:
//   Jeremy Meredith, Tue Jan 12 15:09:33 EST 2010
//   Changed the way device selection works.  It now defaults to the device
//   index corresponding to the process's rank within a node if no devices
//   are specified on the command command line, and otherwise, round-robins
//   the list of devices among the tasks.
//
//   Gabriel Marin, Tue Jun 01 15:38 EST 2010
//   Check that we have valid (not NULL) context and queue objects before
//   running the benchmarks. Errors inside CreateContextFromSingleDevice or
//   CreateCommandQueueForContextAndDevice were not propagated out to the main
//   program.
//
//   Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//   Split timing reports into detailed and summary.  For serial code, we
//   report all trial values, and for parallel, skip the per-process vals.
//   Also detect and print outliers from parallel runs.
//
// ****************************************************************************
int main(int argc, char *argv[])
{
    int ret = 0;

    try
    {
#ifdef PARALLEL
        int rank, size;
        MPI_Init(&argc,&argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        cout << "MPI Task "<< rank << "/" << size - 1 << " starting....\n";
#endif

        OptionParser op;

        //Add shared options to the parser
        op.addOption("platform", OPT_INT, "0", "specify OpenCL platform to use",
                'p');
        op.addOption("device", OPT_VECINT, "", "specify device(s) to run on", 'd');
        op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
        op.addOption("size", OPT_VECINT, "1", "specify problem size", 's');
        op.addOption("infoDevices", OPT_BOOL, "",
                "show info for available platforms and devices", 'i');
        op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
        op.addOption("quiet", OPT_BOOL, "", "write minimum necessary to standard output", 'q');

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
            return (op.HelpRequested() ? 0 : 1 );
        }

        if (op.getOptionBool("infoDevices"))
        {
#define DEBUG_DEVICE_CONTAINER 0
#ifdef PARALLEL
            // execute following code only if I am the process of lowest
            // rank on this node
            NodeInfo NI;
            int mynoderank = NI.nodeRank();
            if (mynoderank==0)
            {
                int nlrrank, nlrsize;
                MPI_Comm nlrcomm = NI.getNLRComm();
                MPI_Comm_size(nlrcomm, &nlrsize);
                MPI_Comm_rank(nlrcomm, &nlrrank);

                OpenCLNodePlatformContainer ndc1;
                OpenCLMultiNodeContainer localMnc(ndc1);
                localMnc.doMerge (nlrrank, nlrsize, nlrcomm);
                if (rank==0)  // I am the global rank 0, print all configurations
                    localMnc.Print (cout);
            }
#else
            OpenCLNodePlatformContainer ndc1;
            ndc1.Print (cout);
#if DEBUG_DEVICE_CONTAINER
            OpenCLMultiNodeContainer mnc1(ndc1), mnc2;
            mnc1.Print (cout);
            ostringstream oss;
            mnc1.writeObject (oss);
            std::string temp(oss.str());
            cout << "Serialized MultiNodeContainer:\n" << temp;
            istringstream iss(temp);
            mnc2.readObject (iss);
            cout << "Unserialized object2:\n";
            mnc2.Print (cout);
            mnc1.merge (mnc2);
            cout << "==============\nObject1 after merging 1:\n";
            mnc1.Print (cout);
            mnc1.merge (mnc2);
            cout << "==============\nObject1 after merging 2:\n";
            mnc1.Print (cout);
#endif  // DEBUG
#endif  // PARALLEL
            return (0);
        }

        bool verbose = op.getOptionBool("verbose");

        // The device option supports specifying more than one device
        // for now, just choose the first one.
        int platform = op.getOptionInt("platform");

#ifdef PARALLEL
        NodeInfo ni;
        int myNodeRank = ni.nodeRank();
        if (verbose)
        cout << "Global rank "<<rank<<" is local rank "<<myNodeRank << endl;
#else
        int myNodeRank = 0;
#endif

        // If they haven't specified any devices, assume they
        // want the process with in-node rank N to use device N
        int deviceIdx = myNodeRank;

        // If they have, then round-robin the list of devices
        // among the processes on a node.
        vector<long long> deviceVec = op.getOptionVecInt("device");
        if (deviceVec.size() > 0)
        {
        int len = deviceVec.size();
            deviceIdx = deviceVec[myNodeRank % len];
        }

        // Check for an erroneous device
        if (deviceIdx >= GetNumOclDevices(platform)) {
            cerr << "Warning: device index: " << deviceIdx
                 << " out of range, defaulting to device 0.\n";
            deviceIdx = 0;
        }

        // Initialization
        if (verbose) cout << ">> initializing\n";
        cl_device_id devID = ListDevicesAndGetDevice(platform, deviceIdx);
        cl_int clErr;
        cl_context ctx = clCreateContext( NULL,     // properties
                                            1,      // number of devices
                                            &devID, // device
                                            NULL,   // notification function
                                            NULL,
                                            &clErr );
        CL_CHECK_ERROR(clErr);
        cl_command_queue queue = clCreateCommandQueue( ctx,
                                                        devID,
                                                        CL_QUEUE_PROFILING_ENABLE,
                                                        &clErr );
        CL_CHECK_ERROR(clErr);
        ResultDatabase resultDB;

        // Run the benchmark
        RunBenchmark(devID, ctx, queue, resultDB, op);

        clReleaseCommandQueue( queue );
        clReleaseContext( ctx );


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
        std::cerr << "unrecognized exception caught" << std::endl;
        ret = 1;
    }

#ifdef PARALLEL
    MPI_Finalize();
#endif

    return ret;
}
