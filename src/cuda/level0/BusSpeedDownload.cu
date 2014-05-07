#include <stdio.h>
#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line argument parsing.
//
//   -nopinned
//   This option controls whether page-locked or "pinned" memory is used.
//   The use of pinned memory typically results in higher bandwidth for data
//   transfer between host and device.
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
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("nopinned", OPT_BOOL, "",
                 "disable usage of pinned (pagelocked) memory", 'p');
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   OpenCL device.  This benchmark repeatedly transfers data chunks of various
//   sizes across the bus to the OpenCL device, and calculates the bandwidth.
//
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
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB,
                  OptionParser &op)
{
    const bool verbose = op.getOptionBool("verbose");
    const bool pinned = !op.getOptionBool("nopinned");

    // Sizes are in kb
    int nSizes  = 20;
    int sizes[20] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
		     32768,65536,131072,262144,524288};
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create some host memory pattern
    float *hostMem = NULL;
    if (pinned)
    {
        cudaMallocHost((void**)&hostMem, sizeof(float) * numMaxFloats);
        while (cudaGetLastError() != cudaSuccess)
        {
 	    // drop the size and try again
	    if (verbose) cout << " - dropping size allocating pinned mem\n";
	    --nSizes;
	    if (nSizes < 1)
	    {
		cerr << "Error: Couldn't allocated any pinned buffer\n";
		return;
	    }
	    numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
            cudaMallocHost((void**)&hostMem, sizeof(float) * numMaxFloats);
        }
    }
    else
    {
        hostMem = new float[numMaxFloats];
    }

    for (int i = 0; i < numMaxFloats; i++)
    {
        hostMem[i] = i % 77;
    }

    float *device;
    cudaMalloc((void**)&device, sizeof(float) * numMaxFloats);
    while (cudaGetLastError() != cudaSuccess)
    {
	// drop the size and try again
	if (verbose) cout << " - dropping size allocating device mem\n";
	--nSizes;
	if (nSizes < 1)
	{
	    cerr << "Error: Couldn't allocated any device buffer\n";
	    return;
	}
	numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
        cudaMalloc((void**)&device, sizeof(float) * numMaxFloats);
    }

    const unsigned int passes = op.getOptionInt("passes");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CHECK_CUDA_ERROR();

    // Three passes, forward and backward both
    for (int pass = 0; pass < passes; pass++)
    {
        // store the times temporarily to estimate latency
        //float times[nSizes];
        // Step through sizes forward on even passes and backward on odd
        for (int i = 0; i < nSizes; i++)
        {
            int sizeIndex;
            if ((pass % 2) == 0)
                sizeIndex = i;
            else
                sizeIndex = (nSizes - 1) - i;

            int nbytes = sizes[sizeIndex] * 1024;

            cudaEventRecord(start, 0);
            cudaMemcpy(device, hostMem, nbytes, cudaMemcpyHostToDevice);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = 0;
            cudaEventElapsedTime(&t, start, stop);
            //times[sizeIndex] = t;

            // Convert to GB/sec
            if (verbose)
            {
                cerr << "size " << sizes[sizeIndex] << "k took " << t <<
                        " ms\n";
            }

            double speed = (double(sizes[sizeIndex]) * 1024. / (1000*1000)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "% 7dkB", sizes[sizeIndex]);
            resultDB.AddResult("DownloadSpeed", sizeStr, "GB/sec", speed);
            resultDB.AddResult("DownloadTime", sizeStr, "ms", t);
        }
	//resultDB.AddResult("DownloadLatencyEstimate", "1-2kb", "ms", times[0]-(times[1]-times[0])/1.);
	//resultDB.AddResult("DownloadLatencyEstimate", "1-4kb", "ms", times[0]-(times[2]-times[0])/3.);
	//resultDB.AddResult("DownloadLatencyEstimate", "2-4kb", "ms", times[1]-(times[2]-times[1])/1.);
    }

    // Cleanup
    cudaFree((void*)device);
    CHECK_CUDA_ERROR();
    if (pinned)
    {
        cudaFreeHost((void*)hostMem);
        CHECK_CUDA_ERROR();
    }
    else
    {
        delete[] hostMem;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
