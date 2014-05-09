#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "support.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "Timer.h"

using namespace std;

const char *short_source =
"__kernel void Triad(__global const float *memA, __global const float *memB, __global float *memC)""\n"
"{""\n"
"    int gid = get_global_id(0);""\n"
"    memC[gid] = memA[gid] + memB[gid];""\n"
"}""\n";


const char *long_source =
"__kernel void""\n"
"uniformAdd(__global float *g_data,""\n"
"           __global float *uniforms,""\n"
"           int n,""\n"
"           int blockOffset,""\n"
"           int baseIndex)""\n"
"{""\n"
"    float uni = 0.0f;""\n"
"\n"
"    uni = uniforms[get_group_id(0) + blockOffset];""\n"
"    unsigned int address = (get_group_id(0) * (get_local_size(0) << 1)) +""\n"
"                           baseIndex + get_local_id(0);""\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);""\n"
"\n"
"    g_data[address] += uni;""\n"
"    if (get_local_id(0) + get_local_size(0) < n)""\n"
"    {""\n"
"        g_data[address + get_local_size(0)] +=  uni;""\n"
"    }""\n"
"}""\n"
"\n"
"__kernel void""\n"
"scan(__global float *g_odata,""\n"
"     __global float *g_idata,""\n"
"     __global float *g_blockSums,""\n"
"     int n,""\n"
"     int blockIndex,""\n"
"     int baseIndex,""\n"
"     int storeSum,""\n"
"     __local float *s_data)""\n"
"{""\n"
"    int ai, bi;""\n"
"    int mem_ai, mem_bi;""\n"
"    int bIndex;""\n"
"\n"
"    // load data into shared memory""\n"
"    if (baseIndex == 0)""\n"
"    {""\n"
"        bIndex = get_group_id(0) * (get_local_size(0) << 1);""\n"
"    }""\n"
"    else""\n"
"    {""\n"
"        bIndex = baseIndex;""\n"
"    }""\n"
"\n"
"    int thid = get_local_id(0);""\n"
"    mem_ai = bIndex + thid;""\n"
"    mem_bi = mem_ai + get_local_size(0);""\n"
"\n"
"    ai = thid;""\n"
"    bi = thid + get_local_size(0);""\n"
"\n"
"    // Cache the computational window in shared memory""\n"
"    // pad values beyond n with zeros""\n"
"    s_data[ai] = g_idata[mem_ai];""\n"
"    if (bi < n)""\n"
"    {""\n"
"        s_data[bi] = g_idata[mem_bi];""\n"
"    }""\n"
"    else""\n"
"    {""\n"
"        s_data[bi] = 0.0f;""\n"
"    }""\n"
"\n"
"    unsigned int stride = 1;""\n"
"\n"
"    // build the sum in place up the tree""\n"
"    for (int d = get_local_size(0); d > 0; d >>= 1)""\n"
"    {""\n"
"        barrier(CLK_LOCAL_MEM_FENCE);""\n"
"        if (thid < d)""\n"
"        {""\n"
"            int i  = 2 * stride * thid;""\n"
"            int aii = i + stride - 1;""\n"
"            int bii = aii + stride;""\n"
"\n"
"            s_data[bii] += s_data[aii];""\n"
"        }""\n"
"        stride *= 2;""\n"
"    }""\n"
"\n"
"    bIndex = (blockIndex == 0) ? get_group_id(0) : blockIndex;""\n"
"\n"
"    if (get_local_id(0) == 0)""\n"
"    {""\n"
"        int index = (get_local_size(0) << 1) - 1;""\n"
"\n"
"        if (storeSum == 1)""\n"
"        {""\n"
"            // write this block's total sum to the corresponding""\n"
"            // index in the blockSums array""\n"
"            g_blockSums[bIndex] = s_data[index];""\n"
"        }""\n"
"\n"
"        // zero the last element in the scan so it will propagate""\n"
"        // back to the front""\n"
"        s_data[index] = 0;""\n"
"    }""\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);""\n"
"\n"
"    // traverse down the tree building the scan in place""\n"
"    for (int d = 1; d <= get_local_size(0); d *= 2)""\n"
"    {""\n"
"        stride >>= 1;""\n"
"        barrier(CLK_LOCAL_MEM_FENCE);""\n"
"\n"
"        if (thid < d)""\n"
"        {""\n"
"            int i  = 2 * stride * thid;""\n"
"            int aii = i + stride - 1;""\n"
"            int bii = aii + stride;""\n"
"\n"
"            float t  = s_data[aii];""\n"
"            s_data[aii] = s_data[bii];""\n"
"            s_data[bii] += t;""\n"
"        }""\n"
"    }""\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);""\n"
"\n"
"    // write results to global memory""\n"
"    g_odata[mem_ai] = s_data[ai];""\n"
"    if (bi < n)""\n"
"    {""\n"
"        g_odata[mem_bi] = s_data[bi];""\n"
"    }""\n"
"}""\n";


void addBenchmarkSpecOptions(OptionParser &op) {
   ;
}

void RunBenchmark(cl_device_id id,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    int n_passes = op.getOptionInt("passes");
    int err;

    for (int pass=0 ; pass<n_passes ; ++pass)
    {
        //
        // Short kernel
        //

        // Create the program
        int TH_short_create = Timer::Start();
        cl_program short_prog = clCreateProgramWithSource(ctx, 1,
                                                          &short_source, NULL,
                                                          &err);
        double len_short_create = Timer::Stop(TH_short_create, "TH_short_create");
        CL_CHECK_ERROR(err);

        int TH_short_build = Timer::Start();
        // Compile the program
        err = clBuildProgram (short_prog, 0, NULL, NULL, NULL, NULL);
        double len_short_build = Timer::Stop(TH_short_build, "TH_short_build");
        CL_CHECK_ERROR(err);

        // Extract out kernel
        int TH_short_extract = Timer::Start();
        cl_kernel short_kernel = clCreateKernel(short_prog, "Triad", &err);
        double len_short_extract = Timer::Stop(TH_short_extract, "TH_short_extract");
        CL_CHECK_ERROR(err);

        // Cleanup
        int TH_short_cleanup = Timer::Start();
        err = clReleaseKernel(short_kernel);
        CL_CHECK_ERROR(err);
        err = clReleaseProgram(short_prog);
        CL_CHECK_ERROR(err);
        double len_short_cleanup = Timer::Stop(TH_short_cleanup, "TH_short_cleanup");

        //
        // Long kernel
        //

        // Create the program
        int TH_long_create = Timer::Start();
        cl_program long_prog = clCreateProgramWithSource(ctx, 1,
                                                         &long_source, NULL,
                                                         &err);
        double len_long_create = Timer::Stop(TH_long_create, "TH_long_create");
        CL_CHECK_ERROR(err);

        // Compile the program
        int TH_long_build = Timer::Start();
        err = clBuildProgram (long_prog, 0, NULL, NULL, NULL, NULL);
        double len_long_build = Timer::Stop(TH_long_build, "TH_long_build");
        CL_CHECK_ERROR(err);

        // Extract out kernel
        int TH_long_extract = Timer::Start();
        cl_kernel long_kernel = clCreateKernel(long_prog, "scan", &err);
        double len_long_extract = Timer::Stop(TH_long_extract, "TH_long_extract");
        CL_CHECK_ERROR(err);

        // Cleanup
        int TH_long_cleanup = Timer::Start();
        err = clReleaseKernel(long_kernel);
        CL_CHECK_ERROR(err);
        err = clReleaseProgram(long_prog);
        CL_CHECK_ERROR(err);
        double len_long_cleanup = Timer::Stop(TH_long_cleanup, "TH_long_cleanup");

        resultDB.AddResult("CreateFromSource","short_kernel", "sec", len_short_create);
        resultDB.AddResult("BuildProgram",    "short_kernel", "sec", len_short_build);
        resultDB.AddResult("ExtractKernel",   "short_kernel", "sec", len_short_extract);
        resultDB.AddResult("Cleanup",         "short_kernel", "sec", len_short_cleanup);

        resultDB.AddResult("CreateFromSource","long_kernel", "sec", len_long_create);
        resultDB.AddResult("BuildProgram",    "long_kernel", "sec", len_long_build);
        resultDB.AddResult("ExtractKernel",   "long_kernel", "sec", len_long_extract);
        resultDB.AddResult("Cleanup",         "long_kernel", "sec", len_long_cleanup);
    }
}
