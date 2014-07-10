#include <iostream>
#include <sstream>
#include <string>
    
#include <math.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

using namespace std;

template <class T>
void RunTest(const std::string& testName, ResultDatabase& resultDB, OptionParser& op);

template <class T>
inline void devTriad(unsigned int nItems, unsigned int blockSize,
        const T *Adata, const T *Bdata, const T sdata, T *Cres,
        double* TriadTime);

// ****************************************************************************
// // Function: VerifyResult
// //
// // Purpose:
// //   Verify that result computed on device matches a "gold standard" result.
// //   Uses relative error.
// //
// // Arguments:
// //   devResult: result computed on device
// //   i_A: input A vector
// //   i_B: input B vector
// //   i_s: input s scalar
// //   i_n: size of inputs A and B
// //
// // Returns:  pass or fail
// //
// // Programmer: Graham Lopez (modeled after KS/PR's OpenACC reduction)
// // Creation: March, 2013
// //
// // Modifications:
// //
// // ****************************************************************************
template <class T>
bool
VerifyResult(const T *devResult, const T* i_A, const T* i_B, const T i_s, const unsigned int i_n)
{
    // compute the reference result - the "gold standard" against
    // which we will compare the device's result
    T refResult[i_n]; 
    for( unsigned int i = 0; i < i_n; i++ )
    {
        refResult[i] = i_A[i] + i_s*i_B[i];
    }

    // compute the relative error in the device's result for each element
    double err = 0.0f;
    double threshold = 1.0e-8;
    int n_failed = 0;
    for( unsigned int i = 0; i < i_n; i++ )
    {
        err = fabs( (refResult[i] - devResult[i]) / refResult[i] );
        if( err > threshold )
        {
            n_failed = n_failed + 1;
        }
    }

    bool ret = false;
    if( !n_failed )
    {
        ret = true;
    }
    else
    {
        std::cout << "TEST FAILED\n"
            << "RelErr: " << err << std::endl;
    }
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
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
    ;
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Run triad operation asynchronously mainly to test bidirectional bandwidth
//   performance.
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Graham Lopez
// Creation: March, 2013
//
// Modifications:
//
// ****************************************************************************

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &opts)
{
    // Always run single precision test
    RunTest<float>("Triad", resultDB, opts);

    // TODO is there a way to check if double precision is supported by device?
    // Or does implementation always fall back to executing on
    // CPU if available accelerators don't support double precision?
    // If double precision is supported, run the DP test
    RunTest<double>("Triad-DP", resultDB, opts);
}

// ****************************************************************************
// Function: runtest<T>
//
// Purpose:
//   Executes the triad benchmark
//
// Arguments:
//   testName: name of the test as reported via the results database
//   resultDB: results from the benchmark are stored in this db
//   opts: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Graham Lopez
// Creation: 2013-03-29 (based on existing SHOC benchmark codes)
//
// Modifications:
//
// ****************************************************************************
extern "C" void DoTriadDoubles( unsigned int nItems, 
                                        unsigned int blockSize,
                                        const double* Adata, 
                                        const double* Bdata, 
                                        const double sdata,
                                        double* Cres,
                                        double* TriadTime);
extern "C" void DoTriadFloats( unsigned int nItems, 
                                        unsigned int blockSize,
                                        const float* Adata, 
                                        const float* Bdata, 
                                        const float sdata,
                                        float * Cres,
                                        double* TriadTime);


template <class T>
void RunTest(const std::string& testName, 
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

    const bool verbose = opts.getOptionBool("verbose");
    int nPasses = 1;

    // 256k through 8M bytes
    const int nSizes = 9;
    const size_t blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 }; //in kB
    unsigned int nItems = (blockSizes[nSizes-1] * 1024 ) / sizeof(T);
    unsigned int nItemsInBlock;

    // Initialize input
    std::cout << "Initializing input." << std::endl;
    T* Adata = new T[nItems];
    T* Bdata = new T[nItems];
    T* Cres = new T[nItems];
    T sdata = 2.0;
    for( unsigned int i = 0; i < nItems; i++ )
    {
        Adata[i] = i % 3; //Fill with some pattern
        Bdata[i] = i % 2; //Fill with some pattern
    }

    // run the benchmark
    std::cout << "Running benchmark" << std::endl;

    char sizeStr[256];
    for( int pass = 0; pass < nPasses; pass++ )
    {
        double TriadTime = 0.0;

        for( int sz = 0; sz < nSizes; sz++ )
        {
            sprintf(sizeStr, "Block:%05ldKB", blockSizes[sz]);
            nItemsInBlock = blockSizes[sz] * 1024 / sizeof(T);
            if(verbose) printf("nItemsInBlock = %d | nItems = %d\n", nItemsInBlock, nItems);

            if (verbose)
                cout << ">> Executing Triad with vectors of length "
                << nItems << " and block size of "
                << nItemsInBlock << " elements." << "\n";

            devTriad<T>( nItems, nItemsInBlock, Adata, Bdata, sdata, 
                        Cres, &TriadTime);

            // verify result
            bool verified = VerifyResult( Cres, Adata, Bdata, sdata, nItems );
            if( !verified )
            {
                // result computed on device does not match
                // result computed on CPU; do not report results.
                std::cerr << "triad failed" << std::endl;
                return;
            }

            // record results
            std::string TestName = testName + "_FLOPs";

            double triad = ((double)nItems * 2.0) / (TriadTime*1e9);
            resultDB.AddResult(TestName, sizeStr, "GFLOP/s", triad);

            std::string txTestName = testName + "_BW";

            double bdwth = ((double)nItems*sizeof(T)*4.0)
                / (TriadTime*1000.*1000.*1000.);
            resultDB.AddResult(txTestName, sizeStr, "GB/s", bdwth);
        }

    }
}

template<>
inline void devTriad<double>(unsigned int nItems, 
                             unsigned int blockSize,
                             const double* Adata, 
                             const double* Bdata, 
                             const double sdata,
                             double* Cres,
                             double* TriadTime)
{
    DoTriadDoubles( nItems, blockSize, Adata, Bdata, sdata,
                         Cres, TriadTime);
}

template<>
inline void devTriad<float>(unsigned int nItems, 
                            unsigned int blockSize,
                             const float* Adata, 
                             const float* Bdata, 
                             const float sdata,
                             float* Cres,
                             double* TriadTime)
{
    DoTriadFloats( nItems, blockSize, Adata, Bdata, sdata,
                         Cres, TriadTime);
}
