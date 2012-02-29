#ifndef SCAN_KERNEL_H_
#define SCAN_KERNEL_H_

#include <cuda.h>

// NB: The following class is a workaround for using dynamically sized
// shared memory in templated code. Without this workaround, the
// compiler would generate two shared memory arrays (one for SP
// and one for DP) of the same name and would generate an error.
template <class T>
class SharedMem
{
    public:
      __device__ inline T* getPointer() 
      { 
          extern __shared__ T s[]; 
          return s; 
      };
};

// Specialization for double
template <>
class SharedMem <double>
{
    public:
      __device__ inline double* getPointer() 
      { 
          extern __shared__ double s_double[]; 
          return s_double; 
      }
};

// Specialization for float
template <>
class SharedMem <float>
{
    public:
      __device__ inline float* getPointer() 
      { 
          extern __shared__ float s_float[]; 
          return s_float; 
      }
};

// Reduction Kernel
// Note: This kernel is slightly different than the kernel in the
// reduction benchmark, in which blocks read in a coalesced, but
// non-contiguous pattern (i.e. the accesses are coalesced, but
// strided by the grid size). This kernel essentially assigns
// each block a contiguous region of the input array.
template <class T, int blockSize>
__global__ void
reduce(const T* __restrict__ g_idata, T* __restrict__ g_odata,
       int n)
{

    // First, calculate the bounds of the region of the array 
    // that this block will sum.  We need these regions to match
    // perfectly with those in the bottom-level scan, so we index
    // as if vector types of length 4 were in use.  This prevents
    // errors due to slightly misaligned regions.
    int region_size = ((n / 4) / gridDim.x) * 4;
    int block_start = blockIdx.x * region_size;
    
    // Give the last block any extra elements
    int block_stop  = (blockIdx.x == gridDim.x - 1) ? 
        n : block_start + region_size;
    
    // Calculate starting index for this thread
    int tid = threadIdx.x;
    int i = block_start + tid;

    // Shared memory will be used for intrablock summation
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.
#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[]; 
#else
    SharedMem<T> shared;
    volatile T* sdata = shared.getPointer();
#endif
    // This thread's sum
    T sum = 0.0f;

    // Reduce multiple elements per thread
    while (i < block_stop)
    {
        sum += g_idata[i];
        i += blockSize;
    }
    // Load this thread's sum into shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduce the contents of shared memory
    // NB: This is an unrolled loop, and assumes warp-syncrhonous
    // execution.
    if (blockSize >= 512) 
    { 
        if (tid < 256) 
        { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads();
    }
    if (blockSize >= 256) 
    { 
        if (tid < 128) 
        { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) 
    { 
        if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads(); 
    }
    if (tid < warpSize) 
    {
        // NB2: This section would also need __sync calls if warp
        // synchronous execution were not assumed
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
    }
    
    // Write result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <class T>
__device__ T scanLocalMem(const T val, volatile T* s_data)
{
    // Set second half of shared memory to zero,
    // to make room for scanning
    s_data[threadIdx.x + blockDim.x] = 0.0f;
    __syncthreads();

    // Compute the number of elems each raking thread will handle.
    // Assumes blockDim.x is divisible by warpSize (32)
    int nsteps = blockDim.x / warpSize;
    // Compute pad for shared memory to reduce bank conflicts
    int pad = threadIdx.x / nsteps;

    // Set first half to the initial values
    s_data[threadIdx.x + pad] = val;
    __syncthreads();

    // "Rake" warp 0 across the shared memory
    if (threadIdx.x < warpSize)
    {
        int start = threadIdx.x * (nsteps+1); //+1 is due to pads
        int stop  = start + nsteps;
        T sum = 0.0f;
        for (int i = start+1; i < stop; i++) {
            sum = s_data[i] += s_data[i-1];
        }
        // Sums are finished, copy into smem's highest 32 words
        int idx = ((blockDim.x*2) - warpSize) + threadIdx.x;
        s_data[idx] = sum;
        
        // Now, perform Kogge-Stone warpscan on those upper 32 words
        T t;
        t = s_data[idx -  1]; 
        s_data[idx] += t;     
        t = s_data[idx -  2]; 
        s_data[idx] += t;      
        t = s_data[idx -  4];  
        s_data[idx] += t;      
        t = s_data[idx -  8];  
        s_data[idx] += t;      
        t = s_data[idx - 16];  
        s_data[idx] += t;      
    }
    __syncthreads();
    // Update smem with the result of the warpscan
    s_data[threadIdx.x+pad] += s_data[((blockDim.x*2) - warpSize) + (threadIdx.x / nsteps)-1];
    __syncthreads();

    // Make the scan exclusive and factor the padding into the
    // address calculation.
    int pad_factor = (threadIdx.x % nsteps == 0) ? 2 : 1;
    return (threadIdx.x == 0) ? 0.0f : s_data[threadIdx.x+pad-pad_factor];
}

template <class T, int blockSize>
__global__ void scan_single_block(T* __restrict__ g_block_sums, const int n)
{
    // Shared memory will be used for intrablock scanning
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.

#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float s_data[]; 
#else
    SharedMem<T> shared;
    volatile T* s_data = shared.getPointer();
#endif
    
    T val = (threadIdx.x < n) ? g_block_sums[threadIdx.x] : 0.0f;
    
    val = scanLocalMem(val, (volatile T*) s_data);

    // Write out to global memory
    if (threadIdx.x < n)
    {
        g_block_sums[threadIdx.x] = val;
    }
}

template <class T, class vecT>
__global__ void
bottom_scan(const T* __restrict__ g_idata, 
                  T* __restrict__ g_odata,
            const T* __restrict__ g_block_sums,
            int n)
{
    // Shared memory will be used for intrablock summation
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.

#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[]; 
#else
    SharedMem<T> shared;
    volatile T* sdata = shared.getPointer();
#endif

    __shared__ T s_seed;

    // Prepare for reading 4-element vectors
    // Assume n is divisible by 4
    vecT *g_idata4 = (vecT*)g_idata;
    vecT *g_odata4 = (vecT*)g_odata;
    n /= 4; //vecT is four wide
    
    // Calculate the bounds of the region of the array 
    // that this block will scan
    int region_size = n / gridDim.x;
    int block_start = blockIdx.x * region_size;
    int block_stop  = (blockIdx.x == gridDim.x - 1) ? n : block_start + region_size;
    
    // Calculate initial starting point for this thread
    int i = block_start + threadIdx.x;
    int window = block_start;

    // Seed the bottom scan with the results from the top scan (i.e. load the per
    // block sums from the previous kernel)
    T seed = g_block_sums[blockIdx.x];

    // Scan multiple elements per thread
    while (window < block_stop)
    {
        vecT val_4;
        if (i < block_stop) // Make sure we don't read out of bounds
        {
            val_4 = g_idata4[i];
        } 
        else
        {
            val_4.x = 0.0f;
            val_4.y = 0.0f;
            val_4.z = 0.0f;
            val_4.w = 0.0f;
        }
        
        // Serial scan in registers
        //val_4.x += seed;
        val_4.y += val_4.x;
        val_4.z += val_4.y;
        val_4.w += val_4.z;
        
        // ExScan sums in shared memory
        T res = scanLocalMem(val_4.w, (volatile T*) sdata);
        __syncthreads();

        // Update and write out to global memory
        val_4.x += res + seed;
        val_4.y += res + seed;
        val_4.z += res + seed;
        val_4.w += res + seed;

        if (i < block_stop) // Make sure we don't write out of bounds
        {
            g_odata4[i] = val_4;
        }
                
        // Next seed will be the last value
        // Last thread puts seed into smem.
        if (threadIdx.x == blockDim.x-1) s_seed = val_4.w;
        __syncthreads();
        
        // Broadcast seed to all threads
        seed = s_seed;

        // Advance window
        window += blockDim.x;
        i += blockDim.x;
    }
}

#endif // SCAN_KERNEL_H_
