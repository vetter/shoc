#include "bfs_kernel.h"

// BFS depends on atomic instructions.  NVCC will generate errors if
// the code is compiled for CC < 1.2.  So, we use this macro and stubs
// so the code will compile cleanly.  If run on CC < 1.2, it will
// return a "NoResult" flag.
#if __CUDA_ARCH__ >= 120

//Sungpack Hong, Sang Kyun Kim, Tayo Oguntebi, and Kunle Olukotun. 2011.
//Accelerating CUDA graph algorithms at maximum warp.
//In Proceedings of the 16th ACM symposium on Principles and practice of
//parallel programming (PPoPP '11). ACM, New York, NY, USA, 267-276.
// ****************************************************************************
// Function: BFS_kernel_warp
//
// Purpose:
//   Perform BFS on the given graph
//
// Arguments:
//   levels: array that stores the level of vertices
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   W_SZ: the warp size to use to process vertices
//   CHUNK_SZ: the number of vertices each warp processes
//   numVertices: number of vertices in the given graph
//   curr: the current BFS level
//   flag: set when more vertices remain to be traversed
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__global__ void BFS_kernel_warp(
        unsigned int *levels,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        int W_SZ,
        int CHUNK_SZ,
        unsigned int numVertices,
        int curr,
        int *flag)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int W_OFF = tid % W_SZ;
    int W_ID = tid / W_SZ;
    int NUM_WARPS = blockDim.x * gridDim.x/W_SZ;
    int v1= W_ID * CHUNK_SZ;
    int chk_sz=CHUNK_SZ+1;

    if((v1+CHUNK_SZ)>=numVertices)
    {
        chk_sz =  numVertices-v1+1;
        if(chk_sz<0)
            chk_sz=0;
    }

    for(int v=v1; v< chk_sz-1+v1; v++)
    {
        if(levels[v] == curr)
        {
            unsigned int num_nbr = edgeArray[v+1]-edgeArray[v];
            unsigned int nbr_off = edgeArray[v];
            for(int i=W_OFF; i<num_nbr; i+=W_SZ)
            {
               int v = edgeArrayAux[i + nbr_off];
               if(levels[v]==UINT_MAX)
               {
                    levels[v] = curr + 1;
                    *flag = 1;
               }
            }
        }
    }
}


//the global mutex for global barrier function
volatile __device__ int g_mutex=0;
//Store the last used value of g_mutex2
volatile __device__ int g_mutex2=0;

//S. Xiao and W. Feng, .Inter-block GPU communication
//via fast barrier synchronization, Technical Report TR-09-19,
//Dept. of Computer Science, Virginia Tech
// ****************************************************************************
// Function: __gpu_sync
//
// Purpose:
//   Implements global barrier synchronization across thread blocks. Thread
//   blocks must be limited to number of multiprocessors available
//
// Arguments:
//   blocks_to_synch: the number of blocks across which to implement the barrier
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__device__ void __gpu_sync(int blocks_to_synch)
{
    __syncthreads();
    //thread ID in a block
    int tid_in_block= threadIdx.x;


    // only thread 0 is used for synchronization
    if (tid_in_block == 0)
    {
        atomicAdd((int *)&g_mutex, 1);
        //only when all blocks add 1 to g_mutex will
        //g_mutex equal to blocks_to_synch
        while(g_mutex < blocks_to_synch);
    }
    __syncthreads();
}

//store the frontier length
volatile __device__ int g_q_offsets[1]={0};
//store the frontier length for the next iteration
volatile __device__ int g_q_size[1]={0};


// ****************************************************************************
// Function: Reset_kernel_parameters
//
// Purpose:
//   Reset global variables
//
// Arguments:
//   frontier_length: length of the frontier array
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__global__ void Reset_kernel_parameters(unsigned int *frontier_length)
{
    g_mutex=0;
    g_mutex2=0;
    *frontier_length=0;
    *g_q_offsets=0;
    g_q_size[0]=0;
}

// ****************************************************************************
// Function: Frontier_copy
//
// Purpose:
//   Copy frontier2 data to frontier & reset global variables
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level
//   frontier2: used with frontier in even odd loops
//   frontier_length: length of the new frontier array
//
// Returns:  nothing
//
// Programmer:
// Creation:
//
// Modifications:
//
// ****************************************************************************
__global__ void Frontier_copy(
        unsigned int *frontier,
        unsigned int *frontier2,
        unsigned int *frontier_length)
{
    unsigned int tid=threadIdx.x + blockDim.x * blockIdx.x;

    if(tid<*frontier_length)
    {
        frontier[tid]=frontier2[tid];
    }
    if(tid==0)
    {
        g_mutex=0;
        g_mutex2=0;
        *g_q_offsets=0;
        *g_q_size=0;
    }
}


//An Effective GPU Implementation of Breadth-First Search, Lijuan Luo,
//Martin Wong,Wen-mei Hwu ,
//Department of Electrical and Computer Engineering,
//University of Illinois at Urbana-Champaign

// ****************************************************************************
// Function: BFS_kernel_one_block
//
// Purpose:
//   Perform BFS on the given graph when the frontier length is within one
//   thread block (i.e max number of threads per block)
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level
//   frontier_len: length of the given frontier array
//   cost: array that stores the cost to visit each vertex
//   visited: mask that tells if a vertex is currently in frontier
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   numVertices: number of vertices in the given graph
//   numEdges: number of edges in the given graph
//   frontier_length: length of the new frontier array
//   max_local_mem: max size of the shared memory queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__global__ void BFS_kernel_one_block_spill(
        volatile unsigned int *frontier,
        unsigned int frontier_len,
        volatile unsigned int *cost,
        volatile int *visited,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        unsigned int numVertices,
        unsigned int numEdges,
        volatile unsigned int *frontier_length,
        const unsigned int max_local_mem)
{

    extern volatile __shared__ unsigned int s_mem[];

    //block queues
    unsigned int *b_q=(unsigned int *)&s_mem[0];
    unsigned int *b_q2=(unsigned int *)&s_mem[max_local_mem];

    volatile __shared__ unsigned int b_offset[1];
    volatile __shared__ unsigned int b_q_length[1];
    //get the threadId
    unsigned int tid=threadIdx.x;
    //copy frontier queue from global queue to local block queue
    if(tid<frontier_len)
    {
        b_q[tid]=frontier[tid];
    }

    unsigned int f_len=frontier_len;
    while(1)
    {
        //Initialize the block queue size to 0
        if(tid==0)
        {
            b_q_length[0]=0;
            b_offset[0]=0;
        }
        __syncthreads();
        if(tid<f_len)
        {
            //get the nodes to traverse from block queue
            unsigned int node_to_process=*(volatile unsigned int *)&b_q[tid];
            //remove from frontier
            visited[node_to_process]=0;
            //get the offsets of the vertex in the edge list
            unsigned int offset = edgeArray[node_to_process];
            unsigned int next   = edgeArray[node_to_process+1];

            //Iterate through the neighbors of the vertex
            while(offset<next)
            {
                //get neighbor
                unsigned int nid=edgeArrayAux[offset];
                //get its cost
                unsigned int v=atomicMin((unsigned int *)&cost[nid],
                        cost[node_to_process]+1);
                //if cost is less than previously set add to frontier
                if(v>cost[node_to_process]+1)
                {
                    int is_in_frontier=atomicExch((int *)&visited[nid],1);
                    //if node already in frontier do nothing
                    if(is_in_frontier==0)
                    {
                        //increment the warp queue size
                        unsigned int t=
                            atomicAdd((unsigned int *)&b_q_length[0],1);
                        if(t< max_local_mem)
                        {
                            b_q2[t]=nid;
                        }
                        //write to global memory if shared memory full
                        else
                        {
                            int off=atomicAdd((unsigned int *)&b_offset[0],
                                    1);
                            frontier[off]=nid;
                        }
                    }
                }
                offset++;
            }
        }
        __syncthreads();

        if(tid<max_local_mem)
            b_q[tid]=*(volatile unsigned int *)&b_q2[tid];

        __syncthreads();
        //Traversal complete exit
        if(b_q_length[0]==0)
        {
            if(tid==0)
                frontier_length[0]=0;
            return;
        }
        // If frontier exceeds one block in size copy warp queues to
        //global frontier queue and exit
        else if( b_q_length[0] > blockDim.x || b_q_length[0] > max_local_mem)
        {
            if(tid<(b_q_length[0]-b_offset[0]))
                frontier[b_offset[0]+tid]= *(volatile unsigned int *)&b_q[tid];
            if(tid==0)
            {
                frontier_length[0] = b_q_length[0];
            }
            return;
        }
        f_len=b_q_length[0];
        __syncthreads();
    }
}

// ****************************************************************************
// Function: BFS_kernel_SM_block
//
// Purpose:
//   Perform BFS on the given graph when the frontier length is greater than
//   one thread block but less than number of Streaming Multiprocessor(SM)
//   thread blocks (i.e max threads per block * SM blocks)
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level
//   frontier2: array that stores the vertices to visit in the next level
//   frontier_len: length of the given frontier array
//   cost: array that stores the cost to visit each vertex
//   visited: mask that tells if a vertex is currently in frontier
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   numVertices: number of vertices in the given graph
//   numEdges: number of edges in the given graph
//   frontier_length: length of the new frontier array
//   max_local_mem: max size of the shared memory queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__global__ void BFS_kernel_SM_block_spill(
        volatile unsigned int *frontier,
        volatile unsigned int *frontier2,
        unsigned int frontier_len,
        volatile unsigned int *cost,
        volatile int *visited,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        unsigned int numVertices,
        unsigned int numEdges,
        volatile unsigned int *frontier_length,
        const unsigned int max_local_mem)
{
    extern volatile __shared__ unsigned int b_q[];

    volatile __shared__ unsigned int b_q_length[1];
    volatile __shared__ unsigned int b_offset[1];

    //get the threadId
    unsigned int tid=threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int lid=threadIdx.x;

    int loop_index=0;
    unsigned int l_mutex=g_mutex2;
    unsigned int f_len=frontier_len;
    while(1)
    {
        //initialize the block queue length and warp queue offset
        if (lid==0)
        {
            b_q_length[0]=0;
            b_offset[0]=0;
        }
        __syncthreads();
        //Initialize the warp queue sizes to 0
        if(tid<f_len)
        {
            //get the nodes to traverse from block queue
            unsigned int node_to_process;

            if(loop_index==0)
               node_to_process=frontier[tid];
            else
               node_to_process=frontier2[tid];

            //remove from frontier
            visited[node_to_process]=0;
            //get the offsets of the vertex in the edge list
            unsigned int offset=edgeArray[node_to_process];
            unsigned int next=edgeArray[node_to_process+1];

            //Iterate through the neighbors of the vertex
            while(offset<next)
            {
                //get neighbor
                unsigned int nid=edgeArrayAux[offset];
                //get its cost
                unsigned int v=atomicMin((unsigned int *)&cost[nid],
                        cost[node_to_process]+1);
                //if cost is less than previously set add to frontier
                if(v>cost[node_to_process]+1)
                {
                    int is_in_frontier=atomicExch((int *)&visited[nid],1);
                    //if node already in frontier do nothing
                    if(is_in_frontier==0)
                    {
                        //increment the warp queue size
                        unsigned int t=atomicAdd((unsigned int *)&b_q_length[0],
                                1);
                        if(t<max_local_mem)
                        {
                            b_q[t]=nid;
                        }
                        //write to global memory if shared memory full
                        else
                        {
                            int off=atomicAdd((unsigned int *)g_q_offsets,1);
                            if(loop_index==0)
                                frontier2[off]=nid;
                            else
                                frontier[off]=nid;
                        }
                    }
                }
                offset++;
            }
        }
        //get offset of block queue in global queue
        __syncthreads();
        if(lid==0)
        {
            if(b_q_length[0] > max_local_mem)
            {
                b_q_length[0] = max_local_mem;
            }
            b_offset[0]=atomicAdd((unsigned int *)g_q_offsets,b_q_length[0]);
        }
        __syncthreads();

        l_mutex+=gridDim.x;
        __gpu_sync(l_mutex);

        //store frontier size
        if(tid==0)
        {
            g_q_size[0]=g_q_offsets[0];
            g_q_offsets[0]=0;
        }

        //copy block queue to global queue
        if(lid < b_q_length[0])
        {
            if(loop_index==0)
                frontier2[lid+b_offset[0]]=b_q[lid];
            else
                frontier[lid+b_offset[0]]=b_q[lid];
        }

        l_mutex+=gridDim.x;
        __gpu_sync(l_mutex);

        //if frontier exceeds SM blocks or less than 1 block exit
        if(g_q_size[0] < blockDim.x ||
                g_q_size[0] > blockDim.x * gridDim.x)
        {

            //TODO:Call the 1-block bfs right here
            break;
        }
        loop_index=(loop_index+1)%2;
        //store the current frontier size
        f_len=g_q_size[0];
    }

    if(loop_index==0)
    {
        for(int i=tid;i<g_q_size[0];i += blockDim.x*gridDim.x)
               frontier[i]=frontier2[i];
    }

    if(tid==0)
    {
        frontier_length[0]=g_q_size[0];
    }
}


// ****************************************************************************
// Function: BFS_kernel_multi_block
//
// Purpose:
//   Perform BFS on the given graph when the frontier length is greater than
//   than number of Streaming Multiprocessor(SM) thread blocks
//   (i.e max threads per block * SM blocks)
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level
//   frontier2: array that stores the vertices to visit in the next level
//   frontier_len: length of the given frontier array
//   cost: array that stores the cost to visit each vertex
//   visited: mask that tells if a vertex has been visited
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   numVertices: number of vertices in the given graph
//   numEdges: number of edges in the given graph
//   frontier_length: length of the new frontier array
//   max_local_mem: max size of the shared memory queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__global__ void BFS_kernel_multi_block_spill(
        volatile unsigned int *frontier,
        volatile unsigned int *frontier2,
        unsigned int frontier_len,
        volatile unsigned int *cost,
        volatile int *visited,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        unsigned int numVertices,
        unsigned int numEdges,
        volatile unsigned int *frontier_length,
        const unsigned int max_local_mem)
{

    extern volatile __shared__ unsigned int b_q[];

    volatile __shared__ unsigned int b_q_length[1];
    volatile __shared__ unsigned int b_offset[1];
    //get the threadId
    unsigned int tid=threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int lid=threadIdx.x;

    //initialize the block queue length and warp queue offset
    if (lid == 0 )
    {
        b_q_length[0]=0;
        b_offset[0]=0;
    }

    __syncthreads();
    //Initialize the warp queue sizes to 0
    if(tid<frontier_len)
    {
        //get the nodes to traverse from block queue
        unsigned int node_to_process=frontier[tid];
        visited[node_to_process]=0;
        //get the offsets of the vertex in the edge list
        unsigned int offset=edgeArray[node_to_process];
        unsigned int next=edgeArray[node_to_process+1];

        //Iterate through the neighbors of the vertex
        while(offset<next)
        {
            //get neighbor
            unsigned int nid=edgeArrayAux[offset];
            //get its cost
            unsigned int v=atomicMin((unsigned int *)&cost[nid],
                    cost[node_to_process]+1);
            //if cost is less than previously set add to frontier
            if(v>cost[node_to_process]+1)
            {
                int is_in_frontier=atomicExch((int *)&visited[nid],1);
                //if node already in frontier do nothing
                if(is_in_frontier==0)
                {
                    //increment the warp queue size
                    unsigned int t=atomicAdd((unsigned int *)&b_q_length[0],
                            1);
                    if(t<max_local_mem)
                    {
                        b_q[t]=nid;
                    }
                    //write to global memory if shared memory full
                    else
                    {
                        int off=atomicAdd((unsigned int *)frontier_length,
                                1);
                        frontier2[off]=nid;
                    }
                }
            }
            offset++;
        }
    }

    __syncthreads();

    //get block queue offset in global queue
    if(lid==0)
    {
        if(b_q_length[0] > max_local_mem)
        {
            b_q_length[0]=max_local_mem;
        }
        b_offset[0]=atomicAdd((unsigned int *)frontier_length,b_q_length[0]);
    }
    __syncthreads();

    //copy block queue to frontier
    if(lid < b_q_length[0])
        frontier2[lid+b_offset[0]]=b_q[lid];
}
#else
// No atomics are available, compile with stubs.
__global__ void BFS_kernel_warp(
        unsigned int *levels,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        int W_SZ,
        int CHUNK_SZ,
        unsigned int numVertices,
        int curr,
        int *flag) { ; }

volatile __device__ int g_mutex=0;
volatile __device__ int g_mutex2=0;
//store the frontier length
volatile __device__ int g_q_offsets[1]={0};
//store the frontier length for the next iteration
volatile __device__ int g_q_size[1]={0};

__global__ void BFS_kernel_one_block(
	volatile unsigned int *frontier,
	unsigned int frontier_len,
	volatile unsigned int *cost,
	volatile int *visited,
	unsigned int *edgeArray,
	unsigned int *edgeArrayAux,
	unsigned int numVertices,
	unsigned int numEdges,
	volatile unsigned int *frontier_length,
    unsigned int num_p_per_mp,
    unsigned int w_q_size) { ; }

__global__ void BFS_kernel_SM_block(
	volatile unsigned int *frontier,
	volatile unsigned int *frontier2,
	unsigned int frontier_len,
	volatile unsigned int *cost,
	volatile int *visited,
	unsigned int *edgeArray,
	unsigned int *edgeArrayAux,
	unsigned int numVertices,
	unsigned int numEdges,
	volatile unsigned int *frontier_length,
    unsigned int num_p_per_mp,
    unsigned int w_q_size) { ; }


__global__ void BFS_kernel_multi_block(
        volatile unsigned int *frontier,
        volatile unsigned int *frontier2,
        unsigned int frontier_len,
        volatile unsigned int *cost,
        volatile int *visited,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        unsigned int numVertices,
        unsigned int numEdges,
        volatile unsigned int *frontier_length,
        unsigned int NUM_P_PER_MP,
        unsigned int W_Q_SIZE) { ; }

__global__ void Reset_kernel_parameters(unsigned int *frontier_length) { ; }

__global__ void Frontier_copy(
	unsigned int *frontier,
	unsigned int *frontier2,
	unsigned int *frontier_length) { ; }

__global__ void BFS_kernel_one_block_spill(
    volatile unsigned int *frontier,
    unsigned int frontier_len,
    volatile unsigned int *cost,
    volatile int *visited,
    unsigned int *edgeArray,
    unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile unsigned int *frontier_length,
    const unsigned int max_mem) { ; }

__global__ void BFS_kernel_SM_block_spill(
    volatile unsigned int *frontier,
    volatile unsigned int *frontier2,
    unsigned int frontier_len,
    volatile unsigned int *cost,
    volatile int *visited,
    unsigned int *edgeArray,
    unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile unsigned int *frontier_length,
    const unsigned int max_mem) { ; }

__global__ void BFS_kernel_multi_block_spill(
    volatile unsigned int *frontier,
    volatile unsigned int *frontier2,
    unsigned int frontier_len,
    volatile unsigned int *cost,
    volatile int *visited,
    unsigned int *edgeArray,
    unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile unsigned int *frontier_length,
    const unsigned int max_mem) { ; }
#endif
