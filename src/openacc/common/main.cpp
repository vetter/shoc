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

#include "ResultDatabase.h"
#include "OptionParser.h"
#include "InvalidArgValue.h"

#ifdef PARALLEL
#include <ParallelResultDatabase.h>
#include <ParallelHelpers.h>
#include <ParallelMerge.h>
#include <NodeInfo.h>
#endif

// The only compiler we support for OpenACC is PGI, and it currently
// (December 2012) does not support OpenACC directives in C++ programs,
// so the openacc.h header is not protected with extern "C" internally.
// We do it here.
extern "C"
{
#include "openacc.h"
}


using namespace std;


void addBenchmarkSpecOptions(OptionParser &op);

void RunBenchmark(ResultDatabase &resultDB,
                  OptionParser &op);

// ****************************************************************************
// Method:  main()
//
// Purpose:
//   Common serial and parallel main for OpenACC benchmarks
//
// Arguments:
//   argc, argv
//
// Programmer:  SHOC Team
// Creation:    The Epoch
//
// Modifications:
//
// ****************************************************************************
int
main(int argc, char *argv[])
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
#ifdef PARALLEL
            // execute following code only if I am the process of lowest 
            // rank on this node
            NodeInfo NI;
            int mynoderank = NI.nodeRank();
            if( mynoderank == 0 )
            {
#endif

                std::cout << "num \'default\' devices" 
                        << acc_get_num_devices( acc_device_default ) << '\n'
                    << "num \'host\' devices"
                        << acc_get_num_devices( acc_device_host ) << '\n'
                    << "num \'non-host\' devices"
                        << acc_get_num_devices( acc_device_not_host ) << '\n'
                    << std::endl;

#ifdef PARALLEL
            }
#endif // ifdef PARALLEL

            return (0);
        }

        bool verbose = op.getOptionBool("verbose");
        
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
        int device = myNodeRank;

        // If they have, then round-robin the list of devices
        // among the processes on a node.
        vector<long long> deviceVec = op.getOptionVecInt("device");
        if (deviceVec.size() > 0)
        {
        int len = deviceVec.size();
            device = deviceVec[myNodeRank % len];
        }

        // Check for an erroneous device
        if (device >= acc_get_num_devices( acc_device_not_host ) ) {
            cerr << "Warning: device index: " << device
                 << " out of range, defaulting to device 0.\n";
            device = 0;
        }

        // Initialization
        if (verbose) cout << ">> initializing\n";
        acc_set_device_num( device, acc_device_not_host );
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
    catch( InvalidArgValue& e )
    {
        std::cerr << e.what() << ": " << e.GetMessage() << std::endl;
        ret = 1;
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

