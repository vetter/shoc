#include <iostream>
#include <stdlib.h>

#include "OpenCLDeviceInfo.h"
#include "OpenCLNodePlatformContainer.h"
#include "Event.h"
#include "support.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

using namespace std;
using namespace SHOC;

void addBenchmarkSpecOptions(OptionParser &op);

void RunBenchmark(cl_device_id id,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op);


cl_device_id* _mpicontention_ocldev = NULL;
cl_context* _mpicontention_ocldriver_ctx = NULL;
cl_command_queue* _mpicontention_ocldriver_queue = NULL;
OptionParser _mpicontention_gpuop;
ResultDatabase _mpicontention_gpuseqrdb, _mpicontention_gpuwuprdb, _mpicontention_gpusimrdb;

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
// Returns: success/failure
//
// Creation: 2009
//
// Modifications:
//
// ****************************************************************************
//
int GPUSetup(OptionParser &op, int mympirank, int mynoderank)
{
    addBenchmarkSpecOptions(op);

    if (op.getOptionBool("infoDevices"))
    {
        OpenCLNodePlatformContainer ndc1;
        ndc1.Print (cout);
        return (0);
    }

    // The device option supports specifying more than one device
    int platform = op.getOptionInt("platform");
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

    // Initialization
    cl_int clErr;
    _mpicontention_ocldev = new cl_device_id( ListDevicesAndGetDevice(platform, device) );
    cl_context ctx = clCreateContext( NULL,     // properties
                                        1,      // number of devices
                                        _mpicontention_ocldev,  // device
                                        NULL,   // notification function
                                        NULL,   // notification function data
                                        &clErr );
    CL_CHECK_ERROR(clErr);
    _mpicontention_ocldriver_ctx = new cl_context(ctx);

    cl_command_queue queue = clCreateCommandQueue(ctx,
                                                *_mpicontention_ocldev,
                                                CL_QUEUE_PROFILING_ENABLE,
                                                &clErr);
    CL_CHECK_ERROR(clErr);
    _mpicontention_ocldriver_queue = new cl_command_queue(queue);
    _mpicontention_gpuop = op;
    return 0;
}

// ****************************************************************************
// Function: GPUCleanup
//
// Purpose:
//  do the necessary OpenCL cleanup for GPU part of the test
//
// Arguments:
//
// Returns:  nothing
//
// Creation: 2009
//
// Modifications:
//
// ****************************************************************************
//
int GPUCleanup()
{
    if( _mpicontention_ocldriver_queue != NULL )
    {
        clReleaseCommandQueue( *_mpicontention_ocldriver_queue );
        delete _mpicontention_ocldriver_queue;
        _mpicontention_ocldriver_queue = NULL;
    }
    if( _mpicontention_ocldriver_ctx != NULL )
    {
        clReleaseContext( *_mpicontention_ocldriver_ctx );
        delete _mpicontention_ocldriver_ctx;
        _mpicontention_ocldriver_ctx = NULL;
    }
    delete _mpicontention_ocldev;
    _mpicontention_ocldev = NULL;

    return 0;
}

// ****************************************************************************
// Function: GPUDriverwrmup
//
// Purpose:
//  drive the GPU test for the warmup run (no simultaneous MPI)
//
// Arguments:
//
// Returns:  nothing
//
// Creation: 2010
//
// Modifications:
//
// ****************************************************************************
//
void GPUDriverwrmup()
{
    // Run the benchmark
    RunBenchmark(*_mpicontention_ocldev, *_mpicontention_ocldriver_ctx,
                    *_mpicontention_ocldriver_queue, _mpicontention_gpuwuprdb,
                    _mpicontention_gpuop);
}


// ****************************************************************************
// Function: GPUDriverseq
//
// Purpose:
//  drive the GPU test in the standalone case (no simultaneous MPI)
//
// Arguments:
//
// Returns:  nothing
//
// Creation: 2009
//
// Modifications:
//
// ****************************************************************************
//
void GPUDriverseq()
{
    // Run the benchmark
    RunBenchmark( *_mpicontention_ocldev, *_mpicontention_ocldriver_ctx,
                    *_mpicontention_ocldriver_queue, _mpicontention_gpuseqrdb,
                    _mpicontention_gpuop);
}

// ****************************************************************************
// Function: GPUDriversim
//
// Purpose:
//  drive the GPU test in the simultaneous run (with MPI)
//
// Arguments:
//
// Returns:  nothing
//
// Creation: 2009
//
// Modifications:
//
// ****************************************************************************
//
void GPUDriversim()
{
    // Run the benchmark
    RunBenchmark( *_mpicontention_ocldev, *_mpicontention_ocldriver_ctx,
                    *_mpicontention_ocldriver_queue, _mpicontention_gpusimrdb,
                    _mpicontention_gpuop);

}

ResultDatabase &GPUGetsimrdb()
{
    return _mpicontention_gpusimrdb;
}

ResultDatabase &GPUGetseqrdb()
{
    return _mpicontention_gpuseqrdb;
}
