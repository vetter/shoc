#define FPTYPE uint
#define FPVECTYPE uint4

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable


// Compute a per block histogram of the occurrences of each
// digit, using a 4-bit radix (i.e. 16 possible digits).
__kernel void
reduce(__global const FPTYPE * in,
       __global FPTYPE * isums,
       const int n,
       __local FPTYPE * lmem,
       const int shift)
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

    // The per thread histogram, initially 0's.
    int digit_counts[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0 };

    // Reduce multiple elements per thread
    while (i < block_stop)
    {
        // This statement
        // 1) Loads the value in from global memory
        // 2) Shifts to the right to have the 4 bits of interest
        //    in the least significant places
        // 3) Masks any more significant bits away. This leaves us
        // with the relevant digit (which is also the index into the
        // histogram). Next increment the histogram to count this occurrence.
        digit_counts[(in[i] >> shift) & 0xFU]++;
        i += get_local_size(0);
    }

    for (int d = 0; d < 16; d++)
    {
        // Load this thread's sum into local/shared memory
        lmem[tid] = digit_counts[d];
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

        // Write result for this block to global memory
        if (tid == 0)
        {
            isums[(d * get_num_groups(0)) + get_group_id(0)] = lmem[0];
        }
    }
}

// This kernel scans the contents of local memory using a work
// inefficient, but highly parallel Kogge-Stone style scan.
// Set exclusive to 1 for an exclusive scan or 0 for an inclusive scan
inline FPTYPE scanLocalMem(FPTYPE val, __local FPTYPE* lmem, int exclusive)
{
    // Set first half of local memory to zero to make room for scanning
    int idx = get_local_id(0);
    lmem[idx] = 0;

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

// This single group kernel takes the per block histograms
// from the reduction and performs an exclusive scan on them.
__kernel void
top_scan(__global FPTYPE * isums,
         const int n,
         __local FPTYPE * lmem)
{
    __local int s_seed;
    s_seed = 0; barrier(CLK_LOCAL_MEM_FENCE);

    // Decide if this is the last thread that needs to
    // propagate the seed value
    int last_thread = (get_local_id(0) < n &&
                      (get_local_id(0)+1) == n) ? 1 : 0;

    for (int d = 0; d < 16; d++)
    {
        FPTYPE val = 0;
        // Load each block's count for digit d
        if (get_local_id(0) < n)
        {
            val = isums[(n * d) + get_local_id(0)];
        }
        // Exclusive scan the counts in local memory
        FPTYPE res = scanLocalMem(val, lmem, 1);
        // Write scanned value out to global
        if (get_local_id(0) < n)
        {
            isums[(n * d) + get_local_id(0)] = res + s_seed;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (last_thread)
        {
            s_seed += res + val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void
bottom_scan(__global const FPTYPE * in,
            __global const FPTYPE * isums,
            __global FPTYPE * out,
            const int n,
            __local FPTYPE * lmem,
            const int shift)
{
    // Use local memory to cache the scanned seeds
    __local FPTYPE l_scanned_seeds[16];

    // Keep a shared histogram of all instances seen by the current
    // block
    __local FPTYPE l_block_counts[16];

    // Keep a private histogram as well
    __private int histogram[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0  };

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
    int window = block_start;

    // Set the histogram in local memory to zero
    // and read in the scanned seeds from gmem
    if (get_local_id(0) < 16)
    {
        l_block_counts[get_local_id(0)] = 0;
        l_scanned_seeds[get_local_id(0)] =
            isums[(get_local_id(0)*get_num_groups(0))+get_group_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Scan multiple elements per thread
    while (window < block_stop)
    {
        // Reset histogram
        for (int q = 0; q < 16; q++) histogram[q] = 0;
        FPVECTYPE val_4;
        FPVECTYPE key_4;

        if (i < block_stop) // Make sure we don't read out of bounds
        {
            val_4 = in4[i];

            // Mask the keys to get the appropriate digit
            key_4.x = (val_4.x >> shift) & 0xFU;
            key_4.y = (val_4.y >> shift) & 0xFU;
            key_4.z = (val_4.z >> shift) & 0xFU;
            key_4.w = (val_4.w >> shift) & 0xFU;

            // Update the histogram
            histogram[key_4.x]++;
            histogram[key_4.y]++;
            histogram[key_4.z]++;
            histogram[key_4.w]++;
        }

        // Scan the digit counts in local memory
        for (int digit = 0; digit < 16; digit++)
        {
            histogram[digit] = scanLocalMem(histogram[digit], lmem, 1);
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i < block_stop) // Make sure we don't write out of bounds
        {
            int address;
            address = histogram[key_4.x] + l_scanned_seeds[key_4.x] + l_block_counts[key_4.x];
            out[address] = val_4.x;
            histogram[key_4.x]++;

            address = histogram[key_4.y] + l_scanned_seeds[key_4.y] + l_block_counts[key_4.y];
            out[address] = val_4.y;
            histogram[key_4.y]++;

            address = histogram[key_4.z] + l_scanned_seeds[key_4.z] + l_block_counts[key_4.z];
            out[address] = val_4.z;
            histogram[key_4.z]++;

            address = histogram[key_4.w] + l_scanned_seeds[key_4.w] + l_block_counts[key_4.w];
            out[address] = val_4.w;
            histogram[key_4.w]++;
        }

        // Before proceeding, make sure everyone has finished their current
        // indexing computations.
        barrier(CLK_LOCAL_MEM_FENCE);
        // Now update the seed array.
        if (get_local_id(0) == get_local_size(0)-1)
        {
            for (int q = 0; q < 16; q++)
            {
                 l_block_counts[q] += histogram[q];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Advance window
        window += get_local_size(0);
        i += get_local_size(0);
    }
}

