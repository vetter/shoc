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
// Returns:  sum of the data
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//    2012-12-06: Philip C. Roth: some renaming/reformatting, removal of 
//    code from OpenCL version not needed for OpenACC version.
//
// ****************************************************************************
template <class T>
bool
VerifyResult(T devResult, T* idata, const unsigned int nItems)
{
    // compute the reference result - the "gold standard" against
    // which we will compare the device's result
    T refResult = 0.0f;
    for( unsigned int i = 0; i < nItems; i++ )
    {
        refResult += idata[i];
    }

    // compute the relative error in the device's result
    double err;
    if( refResult != 0.0 )
    {
        err = fabs( (refResult - devResult) / refResult );
    }
    else
    {
        // we cannot compute a relative error
        // use absolute error
        std::cerr << "Warning: reference result is 0.0; using absolute error" << std::endl;
        err = fabs(refResult - devResult);
    }
    
    double threshold = 1.0e-8;

    bool ret = false;
    std::cout << "TEST ";
    if( err < threshold )
    {
        std::cout << "PASSED";
        ret = true;
    }
    else
    {
        std::cout << "FAILED\n"
            << "RelErr: " << err;
    }
    std::cout << std::endl;
    return ret;
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
                 "specify reduction iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the reduction (sum) benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Philip Roth
// Creation: 2012-12-06
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &opts)
{
    // Always run single precision test
    RunTest<float>("Reduction", resultDB, opts);

    // TODO is there a way to check if double precision is supported by device?
    // Or does implementation always fall back to executing on
    // CPU if available accelerators don't support double precision?
    // If double precision is supported, run the DP test
    RunTest<double>("Reduction-DP", resultDB, opts);
}



// ****************************************************************************
// Function: runtest<T>
//
// Purpose:
//   Executes the reduction (sum) benchmark
//
// Arguments:
//   testName: name of the test as reported via the results database
//   resultDB: results from the benchmark are stored in this db
//   opts: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Philip Roth
// Creation: 2012-12-06 (based on existing SHOC OpenCL and CUDA Reduction implementations)
//
// Modifications:
//
// ****************************************************************************
extern "C" void DoReduceDoublesIters( unsigned int nIters,
                                        void* idata, 
                                        unsigned int nItems, 
                                        void* ores,
                                        double* itersReduceTime,
                                        double* totalReduceTime,
                                        void (*gredfunc)(void*,void*) );
extern "C" void DoReduceFloatsIters( unsigned int nIters,
                                        void* idata, 
                                        unsigned int nItems, 
                                        void* ores,
                                        double* itersReduceTime,
                                        double* totalReduceTime,
                                        void (*gredfunc)(void*,void*) );


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
    // Likewise, our reduce functions return via an argument rather than
    // a return value, so that they can have the correct type for the 
    // output variable.
    //
    void (*reducefunc)( unsigned int, void*, unsigned int, void*, double*, double*, void (*func)(void*, void*) );
    void (*greducefunc)( void*, void* );
    if( sizeof(T) == sizeof(double) )
    {
        reducefunc = DoReduceDoublesIters;
        greducefunc = NULL;
    }
    else if( sizeof(T) == sizeof(float) )
    {
        reducefunc = DoReduceFloatsIters;
        greducefunc = NULL;
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

    for( int pass = 0; pass < nPasses; pass++ )
    {
        T devResult;

        double itersReduceTime = 0.0;
        double totalReduceTime = 0.0;
        (*reducefunc)( nIters, 
                        idata, 
                        nItems, 
                        &devResult, 
                        &itersReduceTime, 
                        &totalReduceTime, 
                        greducefunc );

        // verify result
        bool verified = VerifyResult( devResult, idata, nItems );
        if( !verified )
        {
            // result computed on device does not match
            // result computed on CPU; do not report results.
            std::cerr << "reduction failed" << std::endl;
            return;
        }

        // record results
        // avgTime is in seconds, since that is the units returned
        // by the Timer class.
        double itersAvgTime = itersReduceTime / nIters;
        double totalAvgTime = totalReduceTime / nIters;
        double gbytes = (double)(nItems*sizeof(T)) / (1000. * 1000. * 1000.);

        std::ostringstream attrstr;
        attrstr << nItems << "_items";

        std::string txTestName = testName + "_PCIe";

        resultDB.AddResult(testName, attrstr.str(), "GB/s", gbytes / itersAvgTime);
        resultDB.AddResult(txTestName, attrstr.str(), "GB/s", gbytes / totalAvgTime);
    }
}

