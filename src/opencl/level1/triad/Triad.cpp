#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

using namespace std;

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
// Programmer: Gabriel Marin
// Creation: September 25, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    // do not use pinned memory option
    op.addOption("nopinned", OPT_BOOL, "",
                 "disable usage of pinned (pagelocked) memory", 'P');
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Implements the Streams Triad benchmark in OpenCL.  This benchmark
//   is designed to test the OpenCL's data transfer speed. It executes
//   a vector add operation with no temporal reuse. Thus data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector add computation for one tile with
//   the data download for next tile and results upload for previous
//   tile. However, since data transfer from host to device and vice-versa
//   is much more expensive than the simple vector add computation, data
//   transfer operations completely dominate the execution time.
//   Using large tiles gives better performance because transfering data
//   in big chunks is faster than piecemeal transfers.
//
// Arguments:
//   devid: the opencl device id
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Gabriel Marin
// Creation: July 13, 2009
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(cl_device_id devid,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    bool verbose = op.getOptionBool("verbose");
    int n_passes = op.getOptionInt("passes");
    bool pinned = true; // !op.getOptionBool("nopinned");

    const bool waitForEvents = true;
    int err;

    // 256k through 8M bytes
    const int nSizes  = 9;
    int blockSizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int memSize = 16384;
    int numMaxFloats = 1024 * memSize / 4,
        halfNumFloats = numMaxFloats / 2;

    // Create some host memory pattern
    srand48(8650341L);
    float *hostMem = NULL;
    cl_mem hostMemObj = NULL;

    if (pinned)
    {
        hostMemObj = clCreateBuffer(ctx,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(float)*numMaxFloats, NULL, &err);
        CL_CHECK_ERROR(err);
        hostMem = (float*)clEnqueueMapBuffer(queue, hostMemObj, true,
                                             CL_MAP_READ|CL_MAP_WRITE,
                                             0,sizeof(float)*numMaxFloats,0,
                                             NULL,NULL,&err);
        CL_CHECK_ERROR(err);
    }
    else
    {
        hostMem = new float[numMaxFloats];
    }

    // Allocate some device memory
    cl_mem memA0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 blockSizes[nSizes-1]*1024,
                                 NULL, &err);
    CL_CHECK_ERROR(err);
    Event evDownloadPrimeA0("DownloadPrimeA0");
    err = clEnqueueWriteBuffer(queue, memA0, false, 0,
                               blockSizes[nSizes-1]*1024,
                               hostMem, 0, NULL,
                               &evDownloadPrimeA0.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents(1, &evDownloadPrimeA0.CLEvent());
    CL_CHECK_ERROR(err);

    cl_mem memB0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 blockSizes[nSizes-1]*1024,
                                 NULL, &err);
    CL_CHECK_ERROR(err);
    Event evDownloadPrimeB0("DownloadPrimeB0");
    err = clEnqueueWriteBuffer(queue, memB0, false, 0,
                               blockSizes[nSizes-1]*1024,
                               hostMem, 0, NULL,
                               &evDownloadPrimeB0.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents(1, &evDownloadPrimeB0.CLEvent());
    CL_CHECK_ERROR(err);

    cl_mem memC0 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                 blockSizes[nSizes-1]*1024,
                                 NULL, &err);
    CL_CHECK_ERROR(err);
    Event evDownloadPrimeC0("DownloadPrimeC0");
    err = clEnqueueWriteBuffer(queue, memC0, false, 0,
                               blockSizes[nSizes-1]*1024,
                               hostMem, 0, NULL,
                               &evDownloadPrimeC0.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents(1, &evDownloadPrimeC0.CLEvent());
    CL_CHECK_ERROR(err);

    cl_mem memA1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 blockSizes[nSizes-1]*1024,
                                 NULL, &err);
    CL_CHECK_ERROR(err);
    Event evDownloadPrimeA1("DownloadPrimeA1");
    err = clEnqueueWriteBuffer(queue, memA1, false, 0,
                               blockSizes[nSizes-1]*1024,
                               hostMem, 0, NULL,
                               &evDownloadPrimeA1.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents(1, &evDownloadPrimeA1.CLEvent());
    CL_CHECK_ERROR(err);

    cl_mem memB1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 blockSizes[nSizes-1]*1024,
                                 NULL, &err);
    CL_CHECK_ERROR(err);
    Event evDownloadPrimeB1("DownloadPrimeB1");
    err = clEnqueueWriteBuffer(queue, memB1, false, 0,
                               blockSizes[nSizes-1]*1024,
                               hostMem, 0, NULL,
                               &evDownloadPrimeB1.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents(1, &evDownloadPrimeB1.CLEvent());
    CL_CHECK_ERROR(err);

    cl_mem memC1 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                 blockSizes[nSizes-1]*1024,
                                 NULL, &err);
    CL_CHECK_ERROR(err);
    Event evDownloadPrimeC1("DownloadPrimeC1");
    err = clEnqueueWriteBuffer(queue, memC1, false, 0,
                               blockSizes[nSizes-1]*1024,
                               hostMem, 0, NULL,
                               &evDownloadPrimeC1.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents(1, &evDownloadPrimeC1.CLEvent());
    CL_CHECK_ERROR(err);

    // Create a Triad OpenCL program
    if (verbose) cout << ">> building the kernel\n";
    const char *mTriadCLSource[] = {
         "__kernel void Triad(__global const float *memA, ",
         "__global const float *memB, __global float *memC, ",
         "const float s)",
         "{",
         "    int gid = get_global_id(0);",
         "    memC[gid] = memA[gid] + s*memB[gid];",
         "}"
    };

    cl_program prog = clCreateProgramWithSource (ctx, 7, mTriadCLSource, NULL, &err);
    CL_CHECK_ERROR (err);

    // Compile the program
    err = clBuildProgram (prog, 0, NULL, NULL, NULL, NULL);
    CL_CHECK_ERROR (err);

    // check if we have to dump the PTX
    char* dumpPtx = getenv("DUMP_PTX");
    if (dumpPtx && !strcmp(dumpPtx, "1")) {  // must dump the PTX
       dumpPTXCode(ctx, prog, "TriadOCL");
    }

    // Extract out "Triad" kernel
    cl_kernel kernel_triad_0 = clCreateKernel(prog, "Triad", &err);
    CL_CHECK_ERROR(err);
    cl_kernel kernel_triad_1 = clCreateKernel(prog, "Triad", &err);
    CL_CHECK_ERROR(err);

    size_t maxGroupSize = getMaxWorkGroupSize(ctx, kernel_triad_0);
    size_t localWorkSize = (maxGroupSize<128?maxGroupSize:128);

    // Number of passes. Use a large number for stress testing.
    // A small value is sufficient for computing sustained performance.
    char sizeStr[256];
    for (int pass=0 ; pass<n_passes ; ++pass)
    {
        // Step through sizes forward
        for (int i = 0; i < nSizes; ++i)
        {
            int elemsInBlock = blockSizes[i]*1024 / sizeof(float);
            for (int j=0; j<halfNumFloats; ++j)
              hostMem[j] = hostMem[halfNumFloats+j] = (float)(drand48()*10.0);

            // Copy input memory to the device
            if (verbose)
                cout << ">> Executing Triad with vectors of length "
                     << numMaxFloats << " and block size of " << elemsInBlock << " elements." << "\n";
            sprintf (sizeStr, "Block:%05dKB", blockSizes[i]);

            // start submitting blocks of data of size elemsInBlock
            // overlap the computation of one block with the data
            // download for the next block and the results upload for
            // the previous block
            int crtIdx = 0;
            size_t globalWorkSize = elemsInBlock;

            Event evDownload_0(sizeStr, 3);
            Event evDownload_1(sizeStr, 3);
            err = clEnqueueWriteBuffer(queue, memA0, false, 0,
                                       blockSizes[i]*1024, hostMem,
                                       0, NULL, &evDownload_0.CLEvent(0));
            CL_CHECK_ERROR(err);
            err = clEnqueueWriteBuffer(queue, memB0, false, 0,
                                       blockSizes[i]*1024, hostMem,
                                       0, NULL, &evDownload_0.CLEvent(1));
            CL_CHECK_ERROR(err);

            Event evKernel_0("TriadExec_0");
            Event evKernel_1("TriadExec_1");

            // Set the arguments
            float scalar = 1.75f;
            err = clSetKernelArg(kernel_triad_0, 0, sizeof(cl_mem), (void*)&memA0);
            CL_CHECK_ERROR(err);
            err = clSetKernelArg(kernel_triad_0, 1, sizeof(cl_mem), (void*)&memB0);
            CL_CHECK_ERROR(err);
            err = clSetKernelArg(kernel_triad_0, 2, sizeof(cl_mem), (void*)&memC0);
            CL_CHECK_ERROR(err);
            err = clSetKernelArg(kernel_triad_0, 3, sizeof(cl_float), (void*)&scalar);
            CL_CHECK_ERROR(err);

            err = clEnqueueNDRangeKernel(queue, kernel_triad_0, 1, NULL,
                                 &globalWorkSize, &localWorkSize,
                                 2, evDownload_0.CLEvents(),
                                 &evKernel_0.CLEvent());
            CL_CHECK_ERROR(err);

            if (elemsInBlock < numMaxFloats)
            {
                // start downloading data for next block
                err = clEnqueueWriteBuffer(queue, memA1, false, 0,
                                       blockSizes[i]*1024, hostMem+elemsInBlock,
                                       0, NULL, &evDownload_1.CLEvent(0));
                CL_CHECK_ERROR(err);
                err = clEnqueueWriteBuffer(queue, memB1, false, 0,
                                       blockSizes[i]*1024, hostMem+elemsInBlock,
                                       0, NULL, &evDownload_1.CLEvent(1));
                CL_CHECK_ERROR(err);
            }

            cl_ulong minQueueTime;

            // Read the result device memory back to the host
            int blockIdx = 1;
            while (crtIdx < numMaxFloats)
            {
                // this is the steady state
                Event *downEv, *kernelEv, *pKernelEv, *npDownEv;
                cl_mem *prevC, *memA, *memB, *memC, *nextA, *nextB;
                cl_kernel *p_kernel;
                if (blockIdx&1)
                {
                    pKernelEv = &evKernel_0;  // kernel event for previous block
                    downEv = &evDownload_1;  // download for current block
                    kernelEv = &evKernel_1;  // kernel event for this block
                    npDownEv = &evDownload_0;  // download for next block and upload for previous block
                    prevC = &memC0;
                    memA = &memA1;
                    memB = &memB1;
                    memC = &memC1;
                    nextA = &memA0;
                    nextB = &memB0;
                    p_kernel = &kernel_triad_1;
                } else
                {
                    pKernelEv = &evKernel_1;  // kernel event for previous block
                    downEv = &evDownload_0;  // download for current block
                    kernelEv = &evKernel_0;  // kernel event for this block
                    npDownEv = &evDownload_1;  // download for next block and upload for previous block
                    prevC = &memC1;
                    memA = &memA0;
                    memB = &memB0;
                    memC = &memC0;
                    nextA = &memA1;
                    nextB = &memB1;
                    p_kernel = &kernel_triad_0;
                }

                // download results for previous block
                if (blockIdx>2)
                {
                    clReleaseEvent(npDownEv->CLEvent(2));
                }
                err = clEnqueueReadBuffer(queue, *prevC, false, 0,
                              elemsInBlock*sizeof(float), hostMem+crtIdx,
                              1, pKernelEv->CLEvents(),
                              &(npDownEv->CLEvent(2)));
                CL_CHECK_ERROR(err);

                // if this is the first block, I want to grab the start time
                // before the events get overwritten
                if (crtIdx==0)
                {
                    err = clWaitForEvents (2, evDownload_0.CLEvents());
                    CL_CHECK_ERROR (err);
                    evDownload_0.FillTimingInfo(0);
                    evDownload_0.FillTimingInfo(1);

                    minQueueTime = evDownload_0.QueuedTime(0);
                    if (minQueueTime > evDownload_0.QueuedTime(1))
                        minQueueTime = evDownload_0.QueuedTime(1);
                }
                crtIdx += elemsInBlock;

                if (crtIdx < numMaxFloats)
                {
                    // Set the arguments
                    err = clSetKernelArg(*p_kernel, 0,
                                  sizeof(cl_mem), (void*)memA);
                    CL_CHECK_ERROR(err);
                    err = clSetKernelArg(*p_kernel, 1,
                                  sizeof(cl_mem), (void*)memB);
                    CL_CHECK_ERROR(err);
                    err = clSetKernelArg(*p_kernel, 2,
                                  sizeof(cl_mem), (void*)memC);
                    CL_CHECK_ERROR(err);
                    err = clSetKernelArg(*p_kernel, 3,
                                  sizeof(cl_float), (void*)&scalar);
                    CL_CHECK_ERROR(err);

                    int num_depends = 2;
                    if (blockIdx>1)
                    {
                        num_depends = 3;
                        clReleaseEvent(kernelEv->CLEvent());

//                        err = clWaitForEvents (1, &(downEv->CLEvent(2)));
//                        CL_CHECK_ERROR (err);
                    }
                    err = clEnqueueNDRangeKernel(queue, *p_kernel,
                                 1, NULL,
                                 &globalWorkSize, &localWorkSize,
                                 num_depends, downEv->CLEvents(),
                                 &(kernelEv->CLEvent()));
                    CL_CHECK_ERROR(err);
                }

                if (crtIdx+elemsInBlock < numMaxFloats)
                {
                    clReleaseEvent(npDownEv->CLEvent(0));
                    clReleaseEvent(npDownEv->CLEvent(1));
                    // download data for next block
                    err = clEnqueueWriteBuffer(queue, *nextA, false, 0,
                                       blockSizes[i]*1024,
                                       hostMem+crtIdx+elemsInBlock,
                                       1, pKernelEv->CLEvents(),
                                       &(npDownEv->CLEvent(0)));
                    CL_CHECK_ERROR(err);
                    err = clEnqueueWriteBuffer(queue, *nextB, false, 0,
                                       blockSizes[i]*1024,
                                       hostMem+crtIdx+elemsInBlock,
                                       1, pKernelEv->CLEvents(),
                                       &(npDownEv->CLEvent(1)));
                    CL_CHECK_ERROR(err);
                }

                blockIdx += 1;
            }

            Event *lastEv = (blockIdx&1?&evDownload_1:&evDownload_0);
            // Wait for event to finish
            if (waitForEvents)
            {
               if (verbose) cout << ">> waiting for Triad to finish\n";
               err = clWaitForEvents(1, &(lastEv->CLEvent(2)));
               CL_CHECK_ERROR(err);
            }

            // Get timings
            err = clFlush(queue);
            CL_CHECK_ERROR(err);
            lastEv->FillTimingInfo(2);

            double triad = (numMaxFloats*2) /
                     double(lastEv->EndTime(2) - minQueueTime);
            resultDB.AddResult("TriadFlops", sizeStr, "GFLOP/s", triad);

            double bdwth = (numMaxFloats*sizeof(float)*3) /
                     double(lastEv->EndTime(2) - minQueueTime);
            resultDB.AddResult("TriadBdwth", sizeStr, "GB/s", bdwth);

            // Checking memory for correctness. The two halves of the array
            // should have the same results.
            if (verbose) cout << ">> checking memory\n";
            for (int j=0 ; j<halfNumFloats ; ++j)
            {
               if (hostMem[j] != hostMem[j+halfNumFloats])
               {
                  cout << "Error; hostMem[" << j << "]=" << hostMem[j]
                       << " is different from its twin element hostMem["
                       << (j+halfNumFloats) << "]; stopping check\n";
                  break;
               }
            }
            if (verbose) cout << ">> finish!" << endl;

            // Zero out the test host memory
            for (int j=0 ; j<numMaxFloats ; ++j)
               hostMem[j] = 0;
        }
    }

    // Cleanup
    err = clReleaseKernel(kernel_triad_0);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(kernel_triad_1);
    CL_CHECK_ERROR(err);

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memA0);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memB0);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memC0);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memA1);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memB1);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memC1);
    CL_CHECK_ERROR(err);

    if (pinned)
    {
        err = clReleaseMemObject(hostMemObj);
        CL_CHECK_ERROR(err);
    }
    else
    {
        delete[] hostMem;
    }
}
