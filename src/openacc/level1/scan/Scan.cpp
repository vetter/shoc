#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"


template <class T>
void RunTest(const std::string& testName, 
                ResultDatabase& resultDB,
                OptionParser& op);

// ****************************************************************************
// Function: VerifyResult
//
// Purpose:
//   Verify that result computed on device matches a "gold standard" result.
//   Uses relative error.
//
// Arguments:
//   devResult: result computed on device
//   data : the input data
//   nItems : number of Ts (items) in the input
//
// Returns:  true if device-computed result passes, false otherwise
//
// Programmer: Philip C. Roth
// Creation: 2013-01-17, based on OpenCL version of Scan
//
// Modifications:
//
// ****************************************************************************
template <class T>
bool
VerifyResult(T* devResult, T* idata, const unsigned int nItems, bool beVerbose)
{
    bool ok = true;
    T* refResult = new T[nItems];

    // compute the reference result - the "gold standard" against
    // which we will compare the device's result
    T runningVal = 0.0f;
    for( unsigned int i = 0; i < nItems; i++ )
    {
        refResult[i] = idata[i] + runningVal;
        runningVal = refResult[i];
    }

    // now check the CPU result against the device result
    // we compute the relative error in the device's result and
    // if that is greater than our threshold, we consider the test
    // failed (regardless of the number of bad results in the array)
    double err;
    for( unsigned int i = 0; i < nItems; i++ )
    {
        if( refResult[i] != 0 )
        {
            err = fabs( (refResult[i] - devResult[i]) / refResult[i] );
        }
        else
        {
            // we cannot compute a relative error - 
            // use absolute error
            if( beVerbose )
            {
                std::cerr << "Warning: reference result item is 0.0; using absolute error" << std::endl;
            }
            err = fabs(refResult[i] - devResult[i]);
        }

        double threshold = 1.0e-8;
        if( err > threshold )
        {
            std::cerr << "Err (refResult[" << i << "]=" << refResult[i] << ", devResult[" << i << "]=" << devResult[i] << std::endl;
            ok = false;
            break;
        }
    }

    if( ok )
    {
        std::cout << "PASSED" << std::endl;
    }
    else
    {
        std::cout << "FAILED" << std::endl;
    }

    return ok;
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
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "256",
                 "specify scan iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the scan benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Philip Roth
// Creation: 2013-01-17
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &opts)
{
    // Always run single precision test
    RunTest<float>("Scan", resultDB, opts);

    // TODO is there a way to check if double precision is supported by device?
    // Or does implementation always fall back to executing on
    // CPU if available accelerators don't support double precision?
    // If double precision is supported, run the DP test
    RunTest<double>("Scan-DP", resultDB, opts);
}



// ****************************************************************************
// Function: runtest<T>
//
// Purpose:
//   Executes the scan benchmark
//
// Arguments:
//   testName: name of the test as reported via the results database
//   resultDB: results from the benchmark are stored in this db
//   opts: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Philip Roth
// Creation: 2013-01-17 (based on existing SHOC OpenCL and CUDA Scan implementations)
//
// Modifications:
//
// ****************************************************************************
extern "C" void DoScanDoublesIters( unsigned int nIters,
                                        void* restrict idata, 
                                        unsigned int nItems, 
                                        void* restrict odata,
                                        double* itersScanTime,
                                        double* totalScanTime );
extern "C" void DoScanFloatsIters( unsigned int nIters,
                                        void* restrict idata, 
                                        unsigned int nItems, 
                                        void* restrict odata,
                                        double* itersScanTime,
                                        double* totalScanTime );


template <class T>
void
RunTest(const std::string& testName, 
                ResultDatabase& resultDB,
                OptionParser& opts)
{
    // As of Dec 2012, the available compilers with OpenACC support
    // do not support OpenACC from C++ programs.  We have to call out to
    // C routines with the OpenACC directives, but we leave the benchmark
    // skeleton in C++ so we can reuse classes like the ResultsDatabase
    // and OptionParser.
    // 
    // Once compilers start supporting C++, the separate C function with
    // OpenACC directives can be inlined into this templatized function.
    //
    // Determine which function we will use, based on type of T.
    // We assume that we will only be called with doubles and floats.
    // Note that our test for type of T is nowhere close to bullet proof -
    // for example, it would recognize T=uint64_t as a 64-bit double.
    // Also note that the signature of our C function has to take the
    // data as a void* since it must handle both types.
    // Likewise, our scan functions return via an argument rather than
    // a return value, so that they can have the correct type for the 
    // output variable.
    //
    void (*scanfunc)( unsigned int, 
                        void* restrict, 
                        unsigned int, 
                        unsigned int, 
                        void* restrict, 
                        double*, 
                        double* );
    if( sizeof(T) == sizeof(double) )
    {
        scanfunc = DoScanDoublesIters;
    }
    else if( sizeof(T) == sizeof(float) )
    {
        scanfunc = DoScanFloatsIters;
    }
    else
    {
        // Our assumption was wrong - T is not a double or a float.
        std::cerr << "unsupported type in runTest; ignoring" << std::endl;
        return;
    }

    // Determine the problem sizes
    int probSizes[4] = { 1, 8, 32, 64 };    // in megabytes

    int size = probSizes[opts.getOptionInt("size")-1];
    unsigned int nItems = (size * 1024 * 1024) / sizeof(T);

    // Initialize input
    std::cout << "Initializing input." << std::endl;
    T* idata = new T[nItems];
    for( unsigned int i = 0; i < nItems; i++ )
    {
        idata[i] = i % 3; //Fill with some pattern
    }

    // run the benchmark
    std::cout << "Running benchmark" << std::endl;
    int nPasses = opts.getOptionInt("passes");
    int nIters  = opts.getOptionInt("iterations");
    T* devResult = new T[nItems];

    for( int pass = 0; pass < nPasses; pass++ )
    {

        double itersScanTime = 0.0;
        double totalScanTime = 0.0;
        (*scanfunc)( nIters, 
                        idata, 
                        nItems, 
                        devResult, 
                        &itersScanTime, 
                        &totalScanTime );

        // verify result
        bool beVerbose = opts.getOptionBool("verbose");
        bool verified = VerifyResult( devResult, idata, nItems, beVerbose );
        if( !verified )
        {
            // result computed on device does not match
            // result computed on CPU; do not report results.
            std::cerr << "scan failed" << std::endl;
            return;
        }

        // record results
        // avgTime is in seconds, since that is the units returned
        // by the Timer class.
        double itersAvgTime = itersScanTime / nIters;
        double totalAvgTime = totalScanTime / nIters;
        double gbytes = (double)(nItems*sizeof(T)) / (1000. * 1000. * 1000.);

        std::ostringstream attrstr;
        attrstr << nItems << "_items";

        std::string txTestName = testName + "_PCIe";

        resultDB.AddResult(testName, attrstr.str(), "GB/s", gbytes / itersAvgTime);
        resultDB.AddResult(txTestName, attrstr.str(), "GB/s", gbytes / totalAvgTime);
    }
}

