#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

#define get_queue_index(tid) ((tid%NUM_P_PER_MP))
#define get_queue_offset(tid) ((tid%NUM_P_PER_MP)*W_Q_SIZE)

//S. Xiao and W. Feng, .Inter-block GPU communication via fast barrier
//synchronization,.Technical Report TR-09-19,
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
//   g_mutex: keeps track of number of blocks that are at barrier
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void __gpu_sync(int blocks_to_synch , volatile __global unsigned int *g_mutex)
{
    //thread ID in a block
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    int tid_in_block= get_local_id(0);


    // only thread 0 is used for synchronization
    if (tid_in_block == 0)
    {
        atomic_add(g_mutex, 1);
        //only when all blocks add 1 to g_mutex will
        //g_mutex equal to blocks_to_synch
        while(g_mutex[0] < blocks_to_synch)
        {
        }

    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
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
//   visited: mask that tells if a vertex is currently in frontier
//   cost: array that stores the cost to visit each vertex
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   numVertices: number of vertices in the given graph
//   numEdges: number of edges in the given graph
//   frontier_length: length of the new frontier array
//   max_local_mem: max size of the shared memory queue
//   b_q: block level queue
//   b_q2: alterante block level queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void BFS_kernel_one_block(

    volatile __global unsigned int *frontier,
    unsigned int frontier_len,
    volatile __global int *visited,
    volatile __global unsigned int *cost,
    __global unsigned int *edgeArray,
    __global unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile __global unsigned int *frontier_length,
    const unsigned int max_local_mem,

    //the block queues of size MAX_THREADS_PER_BLOCK
    volatile __local unsigned int *b_q,
    volatile __local unsigned int *b_q2)
{
    volatile __local unsigned int b_offset[1];
    volatile __local unsigned int b_q_length[1];

    //get the threadId
    unsigned int tid = get_local_id(0);
    //copy frontier queue from global queue to local block queue
    if(tid<frontier_len)
    {
        b_q[tid]=frontier[tid];
    }

    unsigned int f_len=frontier_len;
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    while(1)
    {
        //Initialize the block queue size to 0
        if(tid==0)
        {
            b_q_length[0]=0;
            b_offset[0]=0;
        }
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        if(tid<f_len)
        {
            //get the nodes to traverse from block queue
            unsigned int node_to_process=b_q[tid];

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
                unsigned int v=atomic_min(&cost[nid],cost[node_to_process]+1);
                //if cost is less than previously set add to frontier
                if(v>cost[node_to_process]+1)
                {
                    int is_in_frontier=atomic_xchg(&visited[nid],1);
                    //if node already in frontier do nothing
                    if(is_in_frontier==0)
                    {
                            //increment the local queue size
                            unsigned int t=atomic_add(&b_q_length[0],1);
                            if(t< max_local_mem)
                            {
                                b_q2[t]=nid;
                            }
                            //write to global memory if shared memory full
                            else
                            {
                                int off=atomic_add(&b_offset[0],1);
                                frontier[off]=nid;
                            }
                        }
                }
                offset++;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        //copy block queue from b_q2 to b_q
        if(tid<max_local_mem)
            b_q[tid]=b_q2[tid];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        //if traversal complete exit
        if(b_q_length[0]==0)
        {
            if(tid==0)
                frontier_length[0]=0;

            return;
        }
        // if frontier exceeds one block in size copy block queue to
        //global queue and exit
        else if( b_q_length[0] > get_local_size(0) ||
                 b_q_length[0] > max_local_mem)
        {
            if(tid<(b_q_length[0]-b_offset[0]))
                frontier[b_offset[0]+tid]=b_q[tid];
            if(tid==0)
            {
                frontier_length[0] = b_q_length[0];
            }
            return;
        }
        f_len=b_q_length[0];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
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
//   frontier_len: length of the given frontier array
//   frontier2: alternate frontier array
//   visited: mask that tells if a vertex is currently in frontier
//   cost: array that stores the cost to visit each vertex
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   numVertices: number of vertices in the given graph
//   numEdges: number of edges in the given graph
//   frontier_length: length of the new frontier array
//   g_mutex: mutex for implementing global barrier
//   g_mutex2: gives the starting value of the g_mutex used in global barrier
//   g_q_offsets: gives the offset of a block in the global queue
//   g_q_size: keeps track of the size of frontier in intermediate iterations
//   max_local_mem: max size of the shared memory queue
//   b_q: block level queue
//
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void BFS_kernel_SM_block(

    volatile __global unsigned int *frontier,
    unsigned int frontier_len,
    volatile __global unsigned int *frontier2,
    volatile __global int *visited,
    volatile __global unsigned int *cost,
    __global unsigned int *edgeArray,
    __global unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile __global unsigned int *frontier_length,
    volatile __global unsigned int *g_mutex,
    volatile __global unsigned int *g_mutex2,
    volatile __global unsigned int *g_q_offsets,
    volatile __global unsigned int *g_q_size,
    const unsigned int max_local_mem,

    //block queue
    volatile __local unsigned int *b_q)
{

    volatile __local unsigned int b_q_length[1];
    volatile __local unsigned int b_offset[1];
    //get the threadId
    unsigned int tid=get_global_id(0);
    unsigned int lid=get_local_id(0);

    int loop_index=0;
    unsigned int l_mutex=g_mutex2[0];
    unsigned int f_len=frontier_len;
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    while(1)
    {
        //Initialize the block queue size to 0
        if (lid==0)
        {
            b_q_length[0]=0;
            b_offset[0]=0;
        }
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        if(tid<f_len)
        {
            unsigned int node_to_process;

            //get the node to traverse from block queue
            if(loop_index==0)
               node_to_process=frontier[tid];
            else
               node_to_process=frontier2[tid];

            //node removed from frontier
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
                unsigned int v=atomic_min(&cost[nid],cost[node_to_process]+1);
                //if cost is less than previously set add to frontier
                if(v>cost[node_to_process]+1)
                {
                    int is_in_frontier=atomic_xchg(&visited[nid],1);
                    //if node already in frontier do nothing
                    if(is_in_frontier==0)
                    {
                        //increment the warp queue size
                        unsigned int t=atomic_add(&b_q_length[0],1);
                        if(t<max_local_mem)
                        {
                            b_q[t]=nid;
                        }
                        //write to global memory if shared memory full
                        else
                        {
                            int off=atomic_add(g_q_offsets,1);
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

        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        //get block queue offset in global queue
        if(lid==0)
        {
            if(b_q_length[0] > max_local_mem)
            {
                b_q_length[0] = max_local_mem;
            }
            b_offset[0]=atomic_add(g_q_offsets,b_q_length[0]);
        }

        //global barrier
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
		l_mutex+=get_num_groups(0);
		__gpu_sync(l_mutex,g_mutex);

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

        //global barrier
		l_mutex+=get_num_groups(0);
		__gpu_sync(l_mutex,g_mutex);

        //exit if frontier size exceeds SM blocks or is less than 1 block
        if(g_q_size[0] < get_local_size(0) ||
            g_q_size[0] > get_local_size(0) * get_num_groups(0))
                break;

        loop_index=(loop_index+1)%2;
        //store the current frontier size
        f_len=g_q_size[0];
    }

    if(loop_index==0)
    {
        for(int i=tid;i<g_q_size[0];i += get_global_size(0))
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
//   frontier: array that stores the vertices to visit in the next level
//   frontier_len: length of the given frontier array
//   frontier2: used with frontier in even odd loops
//   visited: mask that tells if a vertex is currently in frontier
//   cost: array that stores the cost to visit each vertex
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex
//   numVertices: number of vertices in the given graph
//   numEdges: number of edges in the given graph
//   frontier_length: length of the new frontier array
//   max_local_mem: max size of the shared memory queue
//   b_q: block level queue
//
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void BFS_kernel_multi_block(

    volatile __global unsigned int *frontier,
    unsigned int frontier_len,
    volatile __global unsigned int *frontier2,
    volatile __global int *visited,
    volatile __global unsigned int *cost,
    __global unsigned int *edgeArray,
    __global unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile __global unsigned int *frontier_length,
    const unsigned int max_local_mem,

    volatile __local unsigned int *b_q)
{
    volatile __local unsigned int b_q_length[1];
    volatile __local unsigned int b_offset[1];

    //get the threadId
    unsigned int tid=get_global_id(0);
    unsigned int lid=get_local_id(0);

    //initialize the block queue length
    if (lid == 0)
    {
        b_q_length[0]=0;
        b_offset[0]=0;
    }

    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
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
            unsigned int v=atomic_min(&cost[nid],cost[node_to_process]+1);
            //if cost is less than previously set add to frontier
            if(v>cost[node_to_process]+1)
            {
                int is_in_frontier=atomic_xchg(&visited[nid],1);
                //if node already in frontier do nothing
                if(is_in_frontier==0)
                {
                        //increment the warp queue size
                        unsigned int t=atomic_add(&b_q_length[0],1);
                        if(t<max_local_mem)
                        {
                            b_q[t]=nid;
                        }
                        //write to global memory if shared memory full
                        else
                        {
                            int off=atomic_add(frontier_length,1);
                            frontier2[off]=nid;
                        }
                }
            }
            offset++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    //get block queue offset in global queue
    if(lid==0)
    {
        if(b_q_length[0] > max_local_mem)
        {
                b_q_length[0]=max_local_mem;
        }
        b_offset[0]=atomic_add(frontier_length,b_q_length[0]);
    }

    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    //copy block queue to global queue
    if(lid < b_q_length[0])
        frontier2[lid+b_offset[0]]=b_q[lid];

}

// ****************************************************************************
// Function: Reset_kernel_parameters
//
// Purpose:
//   Reset the global variables
//
// Arguments:
//   frontier_length: length of the new frontier array
//   g_mutex: mutex for implementing global barrier
//   g_mutex2: gives the starting value of the g_mutex used in global barrier
//   g_q_offsets: gives the offset of a block in the global queue
//   g_q_size: size of the global queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void Reset_kernel_parameters(

    __global unsigned int *frontier_length,
    __global volatile int *g_mutex,
    __global volatile int *g_mutex2,
    __global volatile int *g_q_offsets,
    __global volatile int *g_q_size)
{
    g_mutex[0]=0;
    g_mutex2[0]=0;
    *frontier_length=0;
    *g_q_offsets=0;
    g_q_size[0]=0;
}

// ****************************************************************************
// Function: Frontier_copy
//
// Purpose:
//   Copy frontier2 data to frontier
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level
//   frontier2: alternate frontier array
//   frontier_length: length of the frontier array
//   g_mutex: mutex for implementing global barrier
//   g_mutex2: gives the starting value of the g_mutex used in global barrier
//   g_q_offsets: gives the offset of a block in the global queue
//   g_q_size: size of the global queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void Frontier_copy(
    __global unsigned int *frontier,
    __global unsigned int *frontier2,
    __global unsigned int *frontier_length,
    __global volatile int *g_mutex,
    __global volatile int *g_mutex2,
    __global volatile int *g_q_offsets,
    __global volatile int *g_q_size)
{
    unsigned int tid=get_global_id(0);

    if(tid<*frontier_length)
    {
        frontier[tid]=frontier2[tid];
    }
}
