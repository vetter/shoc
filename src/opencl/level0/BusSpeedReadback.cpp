#include <iostream>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

using namespace std;

void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("nopinned", OPT_BOOL, "",
                 "disable usage of pinned (pagelocked) memory");
}

// Modifications:
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
void RunBenchmark(cl_device_id id,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    bool verbose = op.getOptionBool("verbose");
    bool pinned = !op.getOptionBool("nopinned");
    int  npasses = op.getOptionInt("passes");
    const bool waitForEvents = true;

    // Sizes are in kb
    int nSizes  = 20;
    int sizes[20] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
		     32768,65536,131072,262144,524288};

    // Max sure we don't surpass the OpenCL limit.
    cl_long maxAllocSizeBytes = 0;
    clGetDeviceInfo(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(cl_long), &maxAllocSizeBytes, NULL);
    while (sizes[nSizes-1]*1024 > 0.90 * maxAllocSizeBytes)
    {
        --nSizes;
        if (verbose) cout << " - dropping allocation size to keep under reported limit.\n";
        if (nSizes < 1)
        {
            cerr << "Error: OpenCL reported a max allocation size less than 1kB.\b";
            return;
        }
    }

    // Create some host memory pattern
    if (verbose) cout << ">> creating host mem pattern\n";
    int err;
    float *hostMem1;
    float *hostMem2;
    cl_mem hostMemObj1;
    cl_mem hostMemObj2;
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
    if (pinned)
    {
	int err1, err2;
        hostMemObj1 = clCreateBuffer(ctx,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(float)*numMaxFloats, NULL, &err1);
        if (err1 == CL_SUCCESS)
        {
            hostMem1 = (float*)clEnqueueMapBuffer(queue, hostMemObj1, true,
                                                  CL_MAP_READ|CL_MAP_WRITE,
                                                  0,sizeof(float)*numMaxFloats,0,
                                                  NULL,NULL,&err1);
        }
        hostMemObj2 = clCreateBuffer(ctx,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(float)*numMaxFloats, NULL, &err2);
        if (err2 == CL_SUCCESS)
        {
            hostMem2 = (float*)clEnqueueMapBuffer(queue, hostMemObj2, true,
                                                  CL_MAP_READ|CL_MAP_WRITE,
                                                  0,sizeof(float)*numMaxFloats,0,
                                                  NULL,NULL,&err2);
        }
	while (err1 != CL_SUCCESS || err2 != CL_SUCCESS)
	{
	    // free the first buffer if only the second failed
	    if (err1 == CL_SUCCESS)
		clReleaseMemObject(hostMemObj1);

	    // drop the size and try again
	    if (verbose) cout << " - dropping size allocating pinned mem\n";
	    --nSizes;
	    if (nSizes < 1)
	    {
		cerr << "Error: Couldn't allocated any pinned buffer\n";
		return;
	    }
	    numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
	    hostMemObj1 = clCreateBuffer(ctx,
					 CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
					 sizeof(float)*numMaxFloats, NULL, &err1);
            if (err1 == CL_SUCCESS)
            {
                hostMem1 = (float*)clEnqueueMapBuffer(queue, hostMemObj1, true,
                                                      CL_MAP_READ|CL_MAP_WRITE,
                                                      0,sizeof(float)*numMaxFloats,0,
                                                      NULL,NULL,&err1);
            }
	    hostMemObj2 = clCreateBuffer(ctx,
					 CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
					 sizeof(float)*numMaxFloats, NULL, &err2);
            if (err2 == CL_SUCCESS)
            {
                hostMem2 = (float*)clEnqueueMapBuffer(queue, hostMemObj2, true,
                                                      CL_MAP_READ|CL_MAP_WRITE,
                                                      0,sizeof(float)*numMaxFloats,0,
                                                      NULL,NULL,&err2);
            }
	}
    }
    else
    {
        hostMem1 = new float[numMaxFloats];
        hostMem2 = new float[numMaxFloats];
    }

    for (int i=0; i<numMaxFloats; i++) {
        hostMem1[i] = i % 77;
        hostMem2[i] = -1;
    }

    // Allocate some device memory
    if (verbose) cout << ">> allocating device mem\n";
    cl_mem mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 sizeof(float)*numMaxFloats, NULL, &err);
    while (err != CL_SUCCESS)
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
	mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			      sizeof(float)*numMaxFloats, NULL, &err);
    }
    if (verbose) cout << ">> filling device mem to force allocation\n";
    Event evDownloadPrime("DownloadPrime");
    err = clEnqueueWriteBuffer(queue, mem1, false, 0,
                               numMaxFloats*sizeof(float), hostMem1,
                               0, NULL, &evDownloadPrime.CLEvent());
    CL_CHECK_ERROR(err);
    if (verbose) cout << ">> waiting for download to finish\n";
    err = clWaitForEvents(1, &evDownloadPrime.CLEvent());
    CL_CHECK_ERROR(err);

    // Three passes, forward and backward both
    for (int pass = 0; pass < npasses; pass++)
    {
        // store the times temporarily to estimate latency
        //float times[nSizes];
        // Step through sizes forward on even passes and backward on odd
        for (int i = 0; i < nSizes; i++)
        {
            int sizeIndex;
            if ((pass%2) == 0)
                sizeIndex = i;
            else
                sizeIndex = (nSizes-1) - i;

            // Read memory back from the device
            if (verbose) cout << ">> reading from device "<<sizes[sizeIndex]<<"kB\n";
            Event evReadback("Readback");
            err = clEnqueueReadBuffer(queue, mem1, false, 0,
                                       sizes[sizeIndex]*1024, hostMem2,
                                       0, NULL, &evReadback.CLEvent());
            CL_CHECK_ERROR(err);

            // Wait for event to finish
            if (verbose) cout << ">> waiting for readback to finish\n";
            err = clWaitForEvents(1, &evReadback.CLEvent());
            CL_CHECK_ERROR(err);

            if (verbose) cout << ">> finish!";
            if (verbose) cout << endl;

            // Get timings
            err = clFlush(queue);
            CL_CHECK_ERROR(err);
            evReadback.FillTimingInfo();
            if (verbose) evReadback.Print(cerr);

            double t = evReadback.SubmitEndRuntime() / 1.e6; // in ms
            //times[sizeIndex] = t;

            // Add timings to database
            double speed = (double(sizes[sizeIndex] * 1024.) /  (1000.*1000.)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "% 7dkB", sizes[sizeIndex]);
            resultDB.AddResult("ReadbackSpeed", sizeStr, "GB/sec", speed);

            // Add timings to database
            double delay = evReadback.SubmitStartDelay() / 1.e6;
            resultDB.AddResult("ReadbackDelay", sizeStr, "ms", delay);
            resultDB.AddResult("ReadbackTime", sizeStr, "ms", t);
        }
	//resultDB.AddResult("ReadbackLatencyEstimate", "1-2kb", "ms", times[0]-(times[1]-times[0])/1.);
	//resultDB.AddResult("ReadbackLatencyEstimate", "1-4kb", "ms", times[0]-(times[2]-times[0])/3.);
	//resultDB.AddResult("ReadbackLatencyEstimate", "2-4kb", "ms", times[1]-(times[2]-times[1])/1.);
    }

    // Cleanup
    err = clReleaseMemObject(mem1);
    CL_CHECK_ERROR(err);
    if (pinned)
    {
        err = clReleaseMemObject(hostMemObj1);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(hostMemObj2);
        CL_CHECK_ERROR(err);
    }
    else
    {
        delete[] hostMem1;
        delete[] hostMem2;
    }
}
