#include <iostream>
#include <stdlib.h>


#include "ResultDatabase.h"
#include "OptionParser.h"
#include "Utility.h"

using namespace std;
//using namespace SHOC;

#include <cuda.h>
#include <cuda_runtime.h>

void addBenchmarkSpecOptions(OptionParser &op);

void RunBenchmark(ResultDatabase &resultDB,
                  OptionParser &op);

void EnumerateDevicesAndChoose(int chooseDevice, bool verbose, const char* prefix = NULL)
{
    cudaSetDevice(chooseDevice);
    int actualdevice;
    cudaGetDevice(&actualdevice);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    string deviceName = "";
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        if (device == actualdevice)
            deviceName = deviceProp.name;

	if (verbose)
	{
            cout << "Device "<<device<<":\n";
            cout << "  name               = '"<<deviceProp.name<<"'"<<endl;
            cout << "  totalGlobalMem     = "<<HumanReadable(deviceProp.totalGlobalMem)<<endl;
            cout << "  sharedMemPerBlock  = "<<HumanReadable(deviceProp.sharedMemPerBlock)<<endl;
            cout << "  regsPerBlock       = "<<deviceProp.regsPerBlock<<endl;
            cout << "  warpSize           = "<<deviceProp.warpSize<<endl;
            cout << "  memPitch           = "<<HumanReadable(deviceProp.memPitch)<<endl;
            cout << "  maxThreadsPerBlock = "<<deviceProp.maxThreadsPerBlock<<endl;
            cout << "  maxThreadsDim[3]   = "<<deviceProp.maxThreadsDim[0]<<","<<deviceProp.maxThreadsDim[1]<<","<<deviceProp.maxThreadsDim[2]<<endl;
            cout << "  maxGridSize[3]     = "<<deviceProp.maxGridSize[0]<<","<<deviceProp.maxGridSize[1]<<","<<deviceProp.maxGridSize[2]<<endl;
            cout << "  totalConstMem      = "<<HumanReadable(deviceProp.totalConstMem)<<endl;
            cout << "  major (hw version) = "<<deviceProp.major<<endl;
            cout << "  minor (hw version) = "<<deviceProp.minor<<endl;
            cout << "  clockRate          = "<<deviceProp.clockRate<<endl;
            cout << "  textureAlignment   = "<<deviceProp.textureAlignment<<endl;
	}
    }

    std::ostringstream chosenDevStr;
    if( prefix != NULL )
    {
        chosenDevStr << prefix;
    }
    chosenDevStr << "Chose device:"
         << " name='"<<deviceName<<"'"
         << " index="<<actualdevice;
    std::cout << chosenDevStr.str() << std::endl;
}


// ****************************************************************************
// Function: GPUSetup
//
// Purpose:
//  do the necessary OpenCL setup for GPU part of the test
//
// Arguments:
//   op: the options parser / parameter database
//   mympirank: for printing errors in case of failure
//   mynoderank: this is typically the device ID (the mapping done in main)
//
// Returns:  success/failure
//
// Creation: 2009
//
// Modifications:
//
// ****************************************************************************
//
int GPUSetup(OptionParser &op, int mympirank, int mynoderank)
{
    //op.addOption("device", OPT_VECINT, "0", "specify device(s) to run on", 'd');
    //op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
    addBenchmarkSpecOptions(op);

    // The device option supports specifying more than one device
    int deviceIdx = mynoderank;
    if( deviceIdx >= op.getOptionVecInt( "device" ).size() )
    {
        std::ostringstream estr;
        estr << "Warning: not enough devices specified with --device flag for task "
            << mympirank
            << " ( node rank " << mynoderank
            << ") to claim its own device; forcing to use first device ";
        std::cerr << estr.str() << std::endl;
        deviceIdx = 0;
    }
    int device = op.getOptionVecInt("device")[deviceIdx];
    bool verbose = op.getOptionBool("verbose");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (device >= deviceCount) {
        cerr << "Warning: device index: " << device <<
        "out of range, defaulting to device 0.\n";
        device = 0;
    }

    std::ostringstream pstr;
    pstr << mympirank << ": ";

    // Initialization
    EnumerateDevicesAndChoose(device,verbose, pstr.str().c_str());
    return 0;
}


// ****************************************************************************
// Function: GPUCleanup
//
// Purpose:
//  do the necessary OpenCL cleanup for GPU part of the test
//
// Arguments:
//  op: option parser (to be removed)
//
// Returns: success/failure
//
// Creation: 2009
//
// Modifications:
//
// ****************************************************************************
//
int GPUCleanup(OptionParser &op)
{
    return 0;
}

// ****************************************************************************
// Function: GPUDriver
//
// Purpose:
//  drive the GPU test in both sequential and simultaneous run (with MPI)
//
// Arguments:
//  op: benchmark options to be passed down to the gpu benchmark
//
// Returns:  success/failure
//
// Creation: 2009
//
// Modifications:
//
// ****************************************************************************
//
int GPUDriver(OptionParser &op, ResultDatabase &resultDB)
{
    // Run the benchmark
    RunBenchmark(resultDB, op);
    return 0;
}
