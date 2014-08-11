#include <iostream>
    
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

extern "C" void radixSort(uint *, int, double *, double *);

// ****************************************************************************
// Function: verifySort
//
// Purpose:
//   Simple cpu routine to verify device results
//
// Arguments:
//
//
// Returns:  nothing, prints relevant info to stdout
//
// Programmer: Graham Lopez 
// Creation: August 21, 2014
//
// Modifications:
//
// ****************************************************************************
bool verifySort(uint *keys, const size_t size)
{
    bool passed = true;

    for (unsigned int i = 0; i < size - 1; i++)
    {
        if (keys[i] > keys[i + 1])
        {
            passed = false;
#ifdef VERBOSE_OUTPUT
            cout << "Failure: at idx: " << i << endl;
            cout << "Key: " << keys[i] << endl;
            cout << "Idx: " << i + 1 << " Key: " << keys[i + 1] << endl;
#endif
        }
    }
    cout << "Test ";
    if (passed)
        cout << "Passed" << endl;
    else
        cout << "Failed" << endl;
    return passed;
}

void
addBenchmarkSpecOptions(OptionParser &op)
{
   ;
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the radix sort (LSD) benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Graham Lopez 
// Creation: August 21, 2014
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &opts)
{

    //Number of key-value pairs to sort, must be a multiple of 1024
    int probSizes[4] = { 1, 8, 48, 96 };

    int size = probSizes[opts.getOptionInt("size")-1];
    // Convert to MB
    size = (size * 1024 * 1024) / sizeof(uint);

    int iterations = opts.getOptionInt("passes");

    // Size of the keys & vals buffers in bytes
    uint bytes = size * sizeof(uint);

    // allocate input data memory on CPU
    uint *hKeys;
    hKeys = (uint *)malloc(sizeof(*hKeys)*bytes);

    double SortTime = 0.0;
    double TransferTime = 0.0;

    for (int it = 0; it < iterations; it++)
    {
        // Initialize host memory to some pattern
        for (uint i = 0; i < size; i++)
            hKeys[i] = i % 1024;

        // Radix sort on the device
        radixSort(hKeys, size, &SortTime, &TransferTime);

        //verify results
        if(! verifySort(hKeys, size) )
        {
            return;
        }

        //record results
        char atts[1024];
        sprintf(atts, "%ditems", size);
        double gb = (bytes * 2.) / (1000. * 1000. * 1000.);
        resultDB.AddResult("Sort-Rate", atts, "GB/s", gb / SortTime);
        // resultDB.AddResult("Sort-Rate_PCIe", atts, "GB/s", gb / (SortTime + TransferTime));
        // resultDB.AddResult("Sort-Rate_Parity", atts, "N", TransferTime / SortTime);
        
        resultDB.AddResult("Sort-Rate_PCIe", atts, "GB/s", -1.0); //need OpenACC profiling 
                                                                  //support to get accurate
        resultDB.AddResult("Sort-Rate_Parity", atts, "N", -1.0);  //transfer times for these
    }



    // Clean up
    free(hKeys);

}
