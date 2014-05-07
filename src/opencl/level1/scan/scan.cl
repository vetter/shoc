#ifdef SINGLE_PRECISION
#define FPTYPE float
#define FPVECTYPE float4
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#define FPVECTYPE double4
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#define FPVECTYPE double4
#endif

__kernel void
reduce(__global const FPTYPE * in,
       __global FPTYPE * isums,
       const int n,
       __local FPTYPE * lmem)
{
    // First, calculate the bounds of the region of the array
    // that this block will sum.  We need these regions to match
    // perfectly with those in the bottom-level scan, so we index
    // as if vector types of length 4 were in use.  This prevents
    // errors due to slightly misaligned regions.
    int region_size = ((n / 4) / get_num_groups(0)) * 4;
    int block_start = get_group_id(0) * region_size;

    // Give the last block any extra elements
    int block_stop  = (get_group_id(0) == get_num_groups(0) - 1) ?
        n : block_start + region_size;

    // Calculate starting index for this thread/work item
    int tid = get_local_id(0);
    int i = block_start + tid;

    FPTYPE sum = 0.0f;

    // Reduce multiple elements per thread
    while (i < block_stop)
    {
        sum += in[i];
        i += get_local_size(0);
    }
    // Load this thread's sum into local/shared memory
    lmem[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce the contents of shared/local memory
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            lmem[tid] += lmem[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // Write result for this block to global memory
    if (tid == 0)
    {
        isums[get_group_id(0)] = lmem[0];
    }
}

// This kernel scans the contents of local memory using a work
// inefficient, but highly parallel Kogge-Stone style scan.
// Set exclusive to 1 for an exclusive scan or 0 for an inclusive scan
inline FPTYPE scanLocalMem(FPTYPE val, __local FPTYPE* lmem, int exclusive)
{
    // Set first half of local memory to zero to make room for scanning
    int idx = get_local_id(0);
    lmem[idx] = 0.0f;

    // Set second half to block sums from global memory, but don't go out
    // of bounds
    idx += get_local_size(0);
    lmem[idx] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Now, perform Kogge-Stone scan
    FPTYPE t;
    for (int i = 1; i < get_local_size(0); i *= 2)
    {
        t = lmem[idx -  i]; barrier(CLK_LOCAL_MEM_FENCE);
        lmem[idx] += t;     barrier(CLK_LOCAL_MEM_FENCE);
    }
    return lmem[idx-exclusive];
}

__kernel void
top_scan(__global FPTYPE * isums, const int n, __local FPTYPE * lmem)
{
    FPTYPE val = get_local_id(0) < n ? isums[get_local_id(0)] : 0.0f;
    val = scanLocalMem(val, lmem, 1);

    if (get_local_id(0) < n)
    {
        isums[get_local_id(0)] = val;
    }
}

__kernel void
bottom_scan(__global const FPTYPE * in,
            __global const FPTYPE * isums,
            __global FPTYPE * out,
            const int n,
            __local FPTYPE * lmem)
{
    __local FPTYPE s_seed;
    s_seed = 0;

    // Prepare for reading 4-element vectors
    // Assume n is divisible by 4
    __global FPVECTYPE *in4  = (__global FPVECTYPE*) in;
    __global FPVECTYPE *out4 = (__global FPVECTYPE*) out;
    int n4 = n / 4; //vector type is 4 wide

    int region_size = n4 / get_num_groups(0);
    int block_start = get_group_id(0) * region_size;
    // Give the last block any extra elements
    int block_stop  = (get_group_id(0) == get_num_groups(0) - 1) ?
        n4 : block_start + region_size;

    // Calculate starting index for this thread/work item
    int i = block_start + get_local_id(0);
    unsigned int window = block_start;

    // Seed the bottom scan with the results from the top scan (i.e. load the per
    // block sums from the previous kernel)
    FPTYPE seed = isums[get_group_id(0)];

    // Scan multiple elements per thread
    while (window < block_stop) {
        FPVECTYPE val_4;
        if (i < block_stop) {
            val_4 = in4[i];
        } else {
            val_4.x = 0.0f;
            val_4.y = 0.0f;
            val_4.z = 0.0f;
            val_4.w = 0.0f;
        }

        // Serial scan in registers
        val_4.y += val_4.x;
        val_4.z += val_4.y;
        val_4.w += val_4.z;

        // ExScan sums in local memory
        FPTYPE res = scanLocalMem(val_4.w, lmem, 1);

        // Update and write out to global memory
        val_4.x += res + seed;
        val_4.y += res + seed;
        val_4.z += res + seed;
        val_4.w += res + seed;

        if (i < block_stop)
        {
            out4[i] = val_4;
        }

        // Next seed will be the last value
        // Last thread puts seed into smem.
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) == get_local_size(0)-1) {
              s_seed = val_4.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Broadcast seed to other threads
        seed = s_seed;

        // Advance window
        window += get_local_size(0);
        i += get_local_size(0);
    }
}

