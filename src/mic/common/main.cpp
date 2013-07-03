// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Kyle Spafford <kys@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011, UT-Battelle, LLC
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
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <string>
#include "omp.h"

#include "offload.h"
#include "Timer.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "InvalidArgValue.h"

#ifdef  __MIC__ || __MIC2__
//#include <lmmintrin.h>
#include <pthread.h>
//#include <pthread_affinity_np.h>
#endif


// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(OptionParser& op, ResultDatabase& resultDB);


// ****************************************************************************
// Function: EnumerateDevicesAndChoose
//
// Purpose:
//   This function queries cuda about the available gpus in the system, prints
//   those results to standard out, and selects a device for use in the
//   benchmark.
//
// Arguments:
//   chosenDevice: logical number for the desired device
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
//   Philip C. Roth, Wed Jul  3 14:03:42 EDT 2013
//   Adapted for Intel Xeon Phi (MIC) from CUDA version.
//
// ****************************************************************************
void
EnumerateDevicesAndChoose(int chosenDevice, bool verbose)
{
    int deviceCount = _Offload_number_of_devices();
    if (verbose)
    {
        cout << "Number of devices = " << deviceCount << "\n";
    }

    // Unlike CUDA and OpenCL, the Intel Xeon Phi infrastructure 
    // doesn't seem to provide much of an API to query the characteristics
    // of each device.
    for( int device = 0; device < deviceCount; device++ )
    {
        // TODO might be able to do a little better on Linux systems if
        // we parse the output of lspci for 'Co-processor' and extract
        // model number.
        std::ostringstream devNameStr;
        devNameStr << "MIC " << device;

        if( verbose )
        {
            std::cout << "Device " << device << ":\n";
            std::cout << "  name         = " << devNameStr.str() << '\n';
            std::cout << "  max threads  = " << omp_get_max_threads_target( TARGET_MIC, device ) << '\n';
            std::cout << "  num procs    = " << omp_get_num_procs_target( TARGET_MIC, device ) << '\n';
            std::cout << "  dynamic      = " << omp_get_dynamic_target( TARGET_MIC, device ) << '\n';
            std::cout << "  nested       = " << omp_get_nested_target( TARGET_MIC, device ) << '\n';
            std::cout << std::endl;
        }
    }
    std::cout << "Chosen device:"
                << "  index=" << chosenDevice
                << std::endl;
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
//   Philip Roth, Wed Jul  3 14:00:12 EDT 2013
//   Adapted for Intel Xeon Phi offload programming model.
//
// ****************************************************************************
int
main(int argc, char *argv[])
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
        int deviceCount = _Offload_number_of_devices();
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
        RunBenchmark(op, resultDB);

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

