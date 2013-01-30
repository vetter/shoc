#include <stdio.h>

#include "OptionParser.h"
#include "ResultDatabase.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line argument parsing.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
// 1/30/13 - KS: Modified for OpenACC 
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    ; // No specific options.
}

// Forward declaration for OpenACC function.
extern "C" void readbackFunc(const long long numBytes, float* data,
        double* totalTime);

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   OpenACC device.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
// 1/30/13 - KS: Modified for OpenACC 
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    const bool verbose = op.getOptionBool("verbose");

    // Sizes are in kb
    int nSizes  = 20;
    int sizes[20] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
		     32768,65536,131072,262144,524288};
    
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create some host memory pattern
    float *hostMem = new float[numMaxFloats];
    for (int i = 0; i < numMaxFloats; i++)
    {
        hostMem[i] = i % 77;
    }

    const unsigned int passes = op.getOptionInt("passes");

    // Three passes, forward and backward both
    for (int pass = 0; pass < passes; pass++)
    {
        // Step through sizes forward on even passes and backward on odd
        for (int i = 0; i < nSizes; i++)
        {
            int sizeIndex;
            if ((pass % 2) == 0)
                sizeIndex = i;
            else
                sizeIndex = (nSizes - 1) - i;

            int nbytes = sizes[sizeIndex] * 1024;

            double transferTime = 0;
            readbackFunc(nbytes, hostMem, &transferTime);
            transferTime *= 1000.0; // seconds -> ms

            // Convert to GB/sec
            if (verbose)
            {
                cerr << "size " << sizes[sizeIndex] << "k took " << 
                    transferTime << " ms\n";
            }

            double speed = (double(sizes[sizeIndex]) * 1024. / (1000*1000)) / 
                transferTime;
            char sizeStr[256];
            sprintf(sizeStr, "% 7dkB", sizes[sizeIndex]);
            resultDB.AddResult("ReadbackSpeed", sizeStr, "GB/sec", speed);
            resultDB.AddResult("ReadbackTime", sizeStr, "ms", transferTime);
        }
    }

    delete[] hostMem;
}
