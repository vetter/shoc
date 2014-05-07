#ifdef SINGLE_PRECISION
#define FPTYPE float
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#endif

__kernel void
reduce(__global const FPTYPE *g_idata, __global FPTYPE *g_odata,
       __local FPTYPE* sdata, const unsigned int n)
{
    const unsigned int tid = get_local_id(0);
    unsigned int i = (get_group_id(0)*(get_local_size(0)*2)) + tid;
    const unsigned int gridSize = get_local_size(0)*2*get_num_groups(0);
    const unsigned int blockSize = get_local_size(0);

    sdata[tid] = 0;

    // Reduce multiple elements per thread, strided by grid size
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result back to global memory
    if (tid == 0)
    {
        g_odata[get_group_id(0)] = sdata[0];
    }
}


// Currently, CPUs on Snow Leopard only support a work group size of 1
// So, we have a separate version of the kernel which doesn't use
// local memory. This version is only used when the maximum
// supported local group size is 1.
__kernel void
reduceNoLocal(__global FPTYPE *g_idata, __global FPTYPE *g_odata,
       unsigned int n)
{
    FPTYPE sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += g_idata[i];
    }
    g_odata[0] = sum;
}
