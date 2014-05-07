#include <cuda.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string.h>

#include "bfs_kernel.h"
#include "cudacommon.h"
#include "Graph.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"


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
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
//TODO: Check if hostfile option in driver file adds automatically
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("graph_file", OPT_STRING, "random", "name of graph file");
    op.addOption("degree", OPT_INT, "2", "average degree of nodes");
    op.addOption("algo", OPT_INT, "1", "1-IIIT BFS 2-UIUC BFS ");
    op.addOption("dump-pl", OPT_BOOL, "false",
            "enable dump of path lengths to file");
    op.addOption("source_vertex", OPT_INT, "0",
            "vertex to start the traversal from");
    op.addOption("global-barrier", OPT_BOOL, "false",
            "enable the use of global barrier in UIUC BFS");
}


// ****************************************************************************
// Function: verify_results
//
// Purpose:
//  Verify BFS results by comparing the output path lengths from cpu and gpu
//  traversals
//
// Arguments:
//   cpu_cost: path lengths calculated on cpu
//   gpu_cost: path lengths calculated on gpu
//   numVerts: number of vertices in the given graph
//   out_path_lengths: specify if path lengths should be dumped to files
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
unsigned int verify_results(unsigned int *cpu_cost, unsigned int *gpu_cost,
                            unsigned int numVerts,  bool out_path_lengths)
{
    unsigned int unmatched_nodes=0;
    for(int i=0;i<numVerts;i++)
    {
        if(gpu_cost[i]!=cpu_cost[i])
        {
            unmatched_nodes++;
        }
    }

    // If user wants to write path lengths to file
    if(out_path_lengths)
    {
        std::ofstream bfs_out_cpu("bfs_out_cpu.txt");
        std::ofstream bfs_out_gpu("bfs_out_cuda.txt");
        for(int i=0;i<numVerts;i++)
        {
            if(cpu_cost[i]!=UINT_MAX)
                bfs_out_cpu<<cpu_cost[i]<<"\n";
            else
                bfs_out_cpu<<"0\n";

            if(gpu_cost[i]!=UINT_MAX)
                bfs_out_gpu<<gpu_cost[i]<<"\n";
            else
                bfs_out_gpu<<"0\n";
        }
        bfs_out_cpu.close();
        bfs_out_gpu.close();
    }
    return unmatched_nodes;
}

// ****************************************************************************
// Function: RunTest1
//
// Purpose:
//   Runs the BFS benchmark using method 1 (IIIT-BFS method)
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//   G: input graph
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunTest1(ResultDatabase &resultDB, OptionParser &op, Graph *G)
{
    typedef char frontier_type;
    typedef unsigned int cost_type;

    // Get graph info
    unsigned int *edgeArray=G->GetEdgeOffsets();
    unsigned int *edgeArrayAux=G->GetEdgeList();
    unsigned int adj_list_length=G->GetAdjacencyListLength();
    unsigned int numVerts = G->GetNumVertices();
    unsigned int numEdges = G->GetNumEdges();

    int *flag;

    // Allocate pinned memory for frontier and cost arrays on CPU
    cost_type  *costArray;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&costArray,
                                  sizeof(cost_type)*(numVerts)));
    CUDA_SAFE_CALL(cudaMallocHost((void **)&flag,
                                  sizeof(int)));

    // Variables for GPU memory
    // Adjacency lists
    unsigned int *d_edgeArray=NULL,*d_edgeArrayAux=NULL;
    // Cost array
    cost_type  *d_costArray;
    // Flag to check when traversal is complete
    int *d_flag;

    // Allocate memory on GPU
    CUDA_SAFE_CALL(cudaMalloc(&d_costArray,sizeof(cost_type)*numVerts));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeArray,sizeof(unsigned int)*(numVerts+1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeArrayAux,
                                        adj_list_length*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_flag,sizeof(int)));

    // Initialize frontier and cost arrays
    for (int index=0;index<numVerts;index++)
    {
        costArray[index]=UINT_MAX;
    }

    // Set vertex to start traversal from
    const unsigned int source_vertex=op.getOptionInt("source_vertex");
    costArray[source_vertex]=0;

    // Initialize timers
    cudaEvent_t start_cuda_event, stop_cuda_event;
    CUDA_SAFE_CALL(cudaEventCreate(&start_cuda_event));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_cuda_event));

    // Transfer frontier, cost array and adjacency lists on GPU
    cudaEventRecord(start_cuda_event, 0);
    CUDA_SAFE_CALL(cudaMemcpy(d_costArray, costArray,
                   sizeof(cost_type)*numVerts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeArray, edgeArray,
                   sizeof(unsigned int)*(numVerts+1),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeArrayAux,edgeArrayAux,
                 sizeof(unsigned int)*adj_list_length,cudaMemcpyHostToDevice));
    cudaEventRecord(stop_cuda_event,0);
    cudaEventSynchronize(stop_cuda_event);
    float inputTransferTime=0;
    cudaEventElapsedTime(&inputTransferTime,start_cuda_event,stop_cuda_event);

    // Get the device properties for kernel configuration
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,device);

    // Get the kernel configuration
    int numBlocks=0;
    numBlocks=(int)ceil((double)numVerts/(double)devProp.maxThreadsPerBlock);
    if (numBlocks> devProp.maxGridSize[0])
    {
        std::cout<<"Max number of blocks exceeded";
        return;
    }

    unsigned int *cpu_cost = new unsigned int[numVerts];
    // Perform cpu bfs traversal for verifying results
    G->GetVertexLengths(cpu_cost,source_vertex);

    // Start the benchmark
    int passes = op.getOptionInt("passes");
    std::cout<<"Running Benchmark" << endl;
    for (int j=0;j<passes;j++)
    {
        // Initialize kernel timer
        double totalKernelTime=0;
        float outputTransferTime=0;
        float k_time=0;

        // Flag set when there are nodes to be traversed in frontier
        *flag=1;

        // Start CPU Timer to measure total time taken to complete benchmark
        int iters=0;
        int W_SZ=32;
        int CHUNK_SZ=32;
        int cpu_bfs_timer = Timer::Start();
        // While there are nodes to traverse
        while (*flag)
        {
            *flag=0;
            // Set flag to 0
            CUDA_SAFE_CALL(cudaMemcpy(d_flag,flag,
                        sizeof(int),cudaMemcpyHostToDevice));

            cudaEventRecord( start_cuda_event,0);
            BFS_kernel_warp<<<numBlocks,devProp.maxThreadsPerBlock>>>
            (d_costArray,d_edgeArray,d_edgeArrayAux, W_SZ, CHUNK_SZ, numVerts,
                iters,d_flag);
            CHECK_CUDA_ERROR();
            cudaEventRecord(stop_cuda_event,0);
            cudaEventSynchronize(stop_cuda_event);
            k_time=0;
            cudaEventElapsedTime( &k_time, start_cuda_event, stop_cuda_event );
            totalKernelTime += k_time;

            // Read flag
            CUDA_SAFE_CALL(cudaMemcpy(flag,d_flag,
                        sizeof(int),cudaMemcpyDeviceToHost));
            iters++;
        }
        // Stop the CPU Timer
        double result_time = Timer::Stop(cpu_bfs_timer, "cpu_bfs_timer");

        // Copy the cost array from GPU to CPU
        cudaEventRecord(start_cuda_event,0);
        CUDA_SAFE_CALL(cudaMemcpy(costArray,d_costArray,
                       sizeof(cost_type)*numVerts,cudaMemcpyDeviceToHost));
        cudaEventRecord(stop_cuda_event,0);
        cudaEventSynchronize(stop_cuda_event);
        cudaEventElapsedTime(&k_time,start_cuda_event,
                            stop_cuda_event);
        outputTransferTime += k_time;

        // Get the total transfer time
        double totalTransferTime = inputTransferTime + outputTransferTime;

        // Count number of vertices visited
        unsigned int numVisited=0;
        for (int i=0;i<numVerts;i++)
        {
            if (costArray[i]!=UINT_MAX)
                numVisited++;
        }

        bool dump_paths=op.getOptionBool("dump-pl");
        // Verify Results against serial BFS
        unsigned int unmatched_verts=verify_results(cpu_cost,costArray,numVerts,
                                               dump_paths);

        // Total size transferred
        float gbytes = sizeof(cost_type)*numVerts+             //cost array
                       sizeof(unsigned int)*(numVerts+1)+      //edgeArray
                       sizeof(unsigned int)*adj_list_length;   //edgeArrayAux
        gbytes /= (1000. * 1000. * 1000.);

        // Populate the result database
        char atts[1024];
        sprintf(atts,"v:%d_e:%d ", numVerts, adj_list_length);
        if (unmatched_verts==0)
        {
            totalKernelTime *= 1.e-3;
            totalTransferTime *= 1.e-3;
            resultDB.AddResult("BFS_total",atts,"s",result_time);
            resultDB.AddResult("BFS_kernel_time",atts,"s",totalKernelTime);
            resultDB.AddResult("BFS",atts,"GB/s",gbytes/totalKernelTime);
            resultDB.AddResult("BFS_PCIe",atts,"GB/s",
                                gbytes/(totalKernelTime+totalTransferTime));
            resultDB.AddResult("BFS_Parity", atts, "N",
                                totalTransferTime/totalKernelTime);
            resultDB.AddResult("BFS_teps",atts,"Edges/s",
                               numEdges/result_time);
            resultDB.AddResult("BFS_visited_vertices", atts, "N",numVisited);
        }
        else
        {
            resultDB.AddResult("BFS_total",atts,"s",FLT_MAX);
            resultDB.AddResult("BFS_kernel_time",atts,"s",FLT_MAX);
            resultDB.AddResult("BFS",atts,"GB/s",FLT_MAX);
            resultDB.AddResult("BFS_PCIe",atts,"GB/s",FLT_MAX);
            resultDB.AddResult("BFS_Parity",atts,"N",FLT_MAX);
            resultDB.AddResult("BFS_teps",atts,"Edges/s",FLT_MAX);
            resultDB.AddResult("BFS_visited_vertices",atts,"N",FLT_MAX);
        }

        std::cout << "Verification of GPU results: ";
        if (unmatched_verts==0)
        {
            std::cout << "Passed";
        }
        else
        {
            std::cout << "Failed\n";
            return;
        }
        std::cout << endl;

        if (j==passes-1) //if passes completed break;
            break;

        // Reset the arrays to perform BFS again
        for (int index=0;index<numVerts;index++)
        {
            costArray[index]=UINT_MAX;
        }
        costArray[source_vertex]=0;

        CUDA_SAFE_CALL(cudaMemcpy(d_costArray, costArray,
                       sizeof(cost_type)*numVerts, cudaMemcpyHostToDevice));

    }

    // Clean up
    delete[] cpu_cost;
    CUDA_SAFE_CALL(cudaEventDestroy(start_cuda_event));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_cuda_event));

    CUDA_SAFE_CALL(cudaFreeHost(costArray));

    CUDA_SAFE_CALL(cudaFree(d_costArray));
    CUDA_SAFE_CALL(cudaFree(d_edgeArray));
    CUDA_SAFE_CALL(cudaFree(d_edgeArrayAux));
}

// ****************************************************************************
// Function: RunTest2
//
// Purpose:
//   Runs the BFS benchmark using method 2 (UIUC-BFS method)
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//   G: input graph
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunTest2(ResultDatabase &resultDB, OptionParser &op, Graph *G)
{

    typedef unsigned int frontier_type;
    typedef unsigned int cost_type;
    typedef int visited_type;

    // Get graph info
    unsigned int *edgeArray=G->GetEdgeOffsets();
    unsigned int *edgeArrayAux=G->GetEdgeList();
    unsigned int adj_list_length=G->GetAdjacencyListLength();
    unsigned int numVerts = G->GetNumVertices();
    unsigned int numEdges = G->GetNumEdges();

    // Allocate memory for frontier & visited arrays on CPU
    frontier_type *frontier;
    cost_type  *costArray;
    visited_type *visited;

    CUDA_SAFE_CALL(cudaMallocHost((void **)&frontier,
                    sizeof(frontier_type)*(numVerts)));
    CUDA_SAFE_CALL(cudaMallocHost((void **)&costArray ,
                    sizeof(cost_type)*(numVerts)));
    CUDA_SAFE_CALL(cudaMallocHost((void **)&visited,
                    sizeof(visited_type)*(numVerts)));

    // Variables for GPU memory
    // Frontier & visited array, and frontier_length on GPU
    frontier_type *d_frontier, *d_frontier2;
    visited_type *d_visited;
    // Adjacency lists
    unsigned int *d_edgeArray=NULL,*d_edgeArrayAux=NULL;
    // Cost array
    cost_type  *d_costArray;
    // Frontier length
    unsigned int *d_frontier_length;

    // Allocate memory on GPU
    CUDA_SAFE_CALL(cudaMalloc(&d_frontier,sizeof(frontier_type)*numVerts));
    CUDA_SAFE_CALL(cudaMalloc(&d_frontier2,sizeof(frontier_type)*numVerts));
    CUDA_SAFE_CALL(cudaMalloc(&d_costArray,sizeof(cost_type)*numVerts));
    CUDA_SAFE_CALL(cudaMalloc(&d_visited,sizeof(visited_type)*numVerts));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeArray,sizeof(unsigned int)*(numVerts+1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeArrayAux,
                                        adj_list_length*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_frontier_length,sizeof(unsigned int)));

    // Initialize frontier and visited arrays
    for (int index=0;index<numVerts;index++)
    {
        frontier[index]=0;
        costArray[index]=UINT_MAX;
        visited[index]=0;
    }

    // Get vertex to start traversal from
    const unsigned int source_vertex=op.getOptionInt("source_vertex");
    unsigned int frontier_length;

    // Set initial condition for traversal
    frontier_length=1;
    frontier[0]=source_vertex;
    visited[source_vertex]=1;
    costArray[source_vertex]=0;

    // Initialize timers
    cudaEvent_t start_cuda_event, stop_cuda_event;
    CUDA_SAFE_CALL(cudaEventCreate(&start_cuda_event));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_cuda_event));

    // Transfer frontier, visited, cost array and adjacency lists on GPU
    cudaEventRecord(start_cuda_event, 0);
    CUDA_SAFE_CALL(cudaMemcpy(d_frontier, frontier,
                   sizeof(frontier_type)*numVerts , cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_costArray, costArray,
                   sizeof(cost_type)*numVerts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_visited, visited,
                   sizeof(visited_type)*numVerts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeArray, edgeArray,
                   sizeof(unsigned int)*(numVerts+1),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeArrayAux,edgeArrayAux,
                  sizeof(unsigned int)*adj_list_length,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_frontier_length, &costArray[source_vertex],
                   sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaEventRecord(stop_cuda_event,0);
    cudaEventSynchronize(stop_cuda_event);
    float inputTransferTime=0;
    cudaEventElapsedTime(&inputTransferTime,start_cuda_event,stop_cuda_event);

    // Get the device properties for kernel configuration
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,device);

    // Get the kernel configuration
    int numBlocks=0;
    numBlocks=(int)ceil((double)numVerts/(double)devProp.maxThreadsPerBlock);
    if(numBlocks> devProp.maxGridSize[0])
    {
        std::cout<<"Max number of blocks exceeded";
        return;
    }

    // Get the usable shared memory
    unsigned int max_q_size=((devProp.sharedMemPerBlock-
                            (sizeof(unsigned int)*3))/sizeof(unsigned int));

    unsigned int q_size2 = max_q_size;

    if (q_size2 > devProp.maxThreadsPerBlock)
    {
        q_size2 = devProp.maxThreadsPerBlock;
    }

    unsigned int shared_mem2 = q_size2 * sizeof(unsigned int);
    unsigned int q_size1=max_q_size / 2;

    if(q_size1 > devProp.maxThreadsPerBlock)
    {
        q_size1 = devProp.maxThreadsPerBlock;
    }

    unsigned int shared_mem1 = q_size1 * sizeof(unsigned int) * 2;

    // Perform cpu bfs traversal to compare
    unsigned int *cpu_cost = new unsigned int[numVerts];
    G->GetVertexLengths(cpu_cost,source_vertex);

    bool g_barrier=op.getOptionBool("global-barrier");

    // Start the benchmark
    int passes = op.getOptionInt("passes");
    std::cout<<"Running Benchmark" << endl;
    for (int j=0;j<passes;j++)
    {
        //Initialize kernel timer
        double totalKernelTime=0;

        //Start CPU Timer to measure total time taken to complete benchmark
        int cpu_bfs_timer = Timer::Start();

        cudaEventRecord( start_cuda_event,0);
        //While there are nodes to traverse
        while(frontier_length>0)
        {
            //Call Reset_kernel function
            Reset_kernel_parameters<<<1,1>>>(d_frontier_length);
            CHECK_CUDA_ERROR();

            //kernel for frontier length within one block
            if(frontier_length<devProp.maxThreadsPerBlock)
            {
                BFS_kernel_one_block_spill<<<1, devProp.maxThreadsPerBlock,
                    shared_mem1>>>
                (d_frontier,frontier_length,d_costArray,d_visited,
                 d_edgeArray,d_edgeArrayAux,numVerts,numEdges,
                 d_frontier_length,q_size1);
                CHECK_CUDA_ERROR();
            }
            //kernel for frontier length within SM blocks
            else if(g_barrier && frontier_length <
                    devProp.maxThreadsPerBlock*devProp.multiProcessorCount)
            {
                BFS_kernel_SM_block_spill
                <<<devProp.multiProcessorCount, devProp.maxThreadsPerBlock,
                    shared_mem2>>>
                (d_frontier,d_frontier2,frontier_length,d_costArray,
                 d_visited,d_edgeArray,d_edgeArrayAux,numVerts,
                 numEdges,d_frontier_length,q_size2);
                CHECK_CUDA_ERROR();
            }
            //kernel for frontier length greater than SM blocks
            else
            {
                BFS_kernel_multi_block_spill
                <<<numBlocks, devProp.maxThreadsPerBlock,shared_mem2>>>
                (d_frontier,d_frontier2,frontier_length,d_costArray,
                 d_visited,d_edgeArray,d_edgeArrayAux,numVerts,
                 numEdges,d_frontier_length,q_size2);
                CHECK_CUDA_ERROR();

                Frontier_copy<<<numBlocks, devProp.maxThreadsPerBlock>>>(
                    d_frontier,d_frontier2,d_frontier_length);
                CHECK_CUDA_ERROR();
            }
            //Get the current frontier length
            CUDA_SAFE_CALL(cudaMemcpy(&frontier_length,d_frontier_length,
                                sizeof(unsigned int),cudaMemcpyDeviceToHost));
        }
        cudaEventRecord(stop_cuda_event,0);
        cudaEventSynchronize(stop_cuda_event);
        float k_time=0;
        cudaEventElapsedTime( &k_time, start_cuda_event, stop_cuda_event );
        totalKernelTime += k_time;

        //Stop the CPU Timer
        double result_time = Timer::Stop(cpu_bfs_timer, "cpu_bfs_timer");

        //Copy the cost array from GPU to CPU
        cudaEventRecord(start_cuda_event,0);
        CUDA_SAFE_CALL(cudaMemcpy(costArray,d_costArray,
                       sizeof(cost_type)*numVerts,cudaMemcpyDeviceToHost));
        cudaEventRecord(stop_cuda_event,0);
        cudaEventSynchronize(stop_cuda_event);
        float outputTransferTime=0;
        cudaEventElapsedTime(&outputTransferTime,start_cuda_event,
                            stop_cuda_event);

        //get the total transfer time
        double totalTransferTime = inputTransferTime + outputTransferTime;

        //count number of vertices visited
        unsigned int numVisited=0;
        for(int i=0;i<numVerts;i++)
        {
            if(costArray[i]!=UINT_MAX)
                numVisited++;
        }

        bool dump_paths=op.getOptionBool("dump-pl");
        //Verify Results against serial BFS
        unsigned int unmatched_verts=verify_results(cpu_cost,costArray,numVerts,
                                               dump_paths);

        float gbytes=   sizeof(frontier_type)*numVerts*2+       //2 frontiers
                        sizeof(cost_type)*numVerts+             //cost array
                        sizeof(visited_type)*numVerts+          //visited array
                        sizeof(unsigned int)*(numVerts+1)+      //edgeArray
                        sizeof(unsigned int)*adj_list_length;   //edgeArrayAux

        gbytes/=(1000. * 1000. * 1000.);

        //populate the result database
        char atts[1024];
        sprintf(atts,"v:%d_e:%d ",numVerts,adj_list_length);
        if(unmatched_verts==0)
        {
            totalKernelTime *= 1.e-3;
            totalTransferTime *= 1.e-3;
            resultDB.AddResult("BFS_total",atts,"s",result_time);
            resultDB.AddResult("BFS_kernel_time",atts,"s",totalKernelTime);
            resultDB.AddResult("BFS",atts,"GB/s",gbytes/totalKernelTime);
            resultDB.AddResult("BFS_PCIe",atts,"GB/s",
                                gbytes/(totalKernelTime+totalTransferTime));
            resultDB.AddResult("BFS_Parity", atts, "N",
                                totalTransferTime/totalKernelTime);
            resultDB.AddResult("BFS_teps",atts,"Edges/s",
                               numEdges/result_time);
            resultDB.AddResult("BFS_visited_vertices", atts, "N",numVisited);
        }
        else
        {
            resultDB.AddResult("BFS_total",atts,"s",FLT_MAX);
            resultDB.AddResult("BFS_kernel_time",atts,"s",FLT_MAX);
            resultDB.AddResult("BFS",atts,"GB/s",FLT_MAX);
            resultDB.AddResult("BFS_PCIe",atts,"GB/s",FLT_MAX);
            resultDB.AddResult("BFS_Parity",atts,"N",FLT_MAX);
            resultDB.AddResult("BFS_teps",atts,"Edges/s",FLT_MAX);
            resultDB.AddResult("BFS_visited_vertices",atts,"N",FLT_MAX);
        }


        std::cout << endl << "Verification of GPU results: ";
        if(unmatched_verts==0)
        {
            std::cout<<"Passed";
        }
        else
        {
            std::cout<<"Failed\n";
            return;
        }
        cout << endl;
        //if passes completed break;
        if(j==passes-1)
            break;

        //reset the arrays to perform BFS again
        for(int index=0;index<numVerts;index++)
        {
            frontier[index]=0;
            costArray[index]=UINT_MAX;
            visited[index]=0;
        }
        //reset the initial condition
        frontier_length=1;
        frontier[0]=source_vertex;
        visited[source_vertex]=1;
        costArray[source_vertex]=0;

        //transfer the arrays to gpu
        CUDA_SAFE_CALL(cudaMemcpy(d_frontier, frontier,
                       sizeof(frontier_type)*numVerts, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_costArray, costArray,
                       sizeof(cost_type)*numVerts, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_visited, visited,
                       sizeof(visited_type)*numVerts, cudaMemcpyHostToDevice));
    }

    //Clean up
    delete[] cpu_cost;

    CUDA_SAFE_CALL(cudaEventDestroy(start_cuda_event));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_cuda_event));
    CUDA_SAFE_CALL(cudaFreeHost(frontier));
    CUDA_SAFE_CALL(cudaFreeHost(costArray));
    CUDA_SAFE_CALL(cudaFreeHost(visited));

    CUDA_SAFE_CALL(cudaFree(d_frontier));
    CUDA_SAFE_CALL(cudaFree(d_frontier2));
    CUDA_SAFE_CALL(cudaFree(d_visited));
    CUDA_SAFE_CALL(cudaFree(d_costArray));
    CUDA_SAFE_CALL(cudaFree(d_frontier_length));
    CUDA_SAFE_CALL(cudaFree(d_edgeArray));
    CUDA_SAFE_CALL(cudaFree(d_edgeArrayAux));
}



// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the BFS benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{

    // First, check if the device supports atomics, which are required
    // for this benchmark.  If not, return the "NoResult" sentinel.int device;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    if ((deviceProp.major == 1 && deviceProp.minor < 2))
    {
        cerr << "Warning: BFS uses atomics and requires GPUs with CC > 1.2.\n";
        char atts[32] = "Atomics_Unavailable";
        resultDB.AddResult("BFS_total",atts,"s",FLT_MAX);
        resultDB.AddResult("BFS_kernel_time",atts,"s",FLT_MAX);
        resultDB.AddResult("BFS",atts,"GB/s",FLT_MAX);
        resultDB.AddResult("BFS_PCIe",atts,"GB/s",FLT_MAX);
        resultDB.AddResult("BFS_Parity", atts, "N",FLT_MAX);
        resultDB.AddResult("BFS_teps",atts,"Edges/s",FLT_MAX);
        resultDB.AddResult("BFS_visited_vertices",atts,"N",FLT_MAX);
    }

    //adjacency list variables
    //number of vertices and edges in graph
    unsigned int numVerts,numEdges;
    //Get the graph filename
    string inFileName = op.getOptionString("graph_file");
    //max degree in graph
    Graph *G=new Graph();

    unsigned int **edge_ptr1 = G->GetEdgeOffsetsPtr();
    unsigned int **edge_ptr2 = G->GetEdgeListPtr();
    //Load simple k-way tree or from a file
    if (inFileName == "random")
    {
        //Load simple k-way tree
        unsigned int prob_sizes[4] = {1000,10000,100000,1000000};
        numVerts = prob_sizes[op.getOptionInt("size")-1];
        int avg_degree = op.getOptionInt("degree");
        if(avg_degree<1)
            avg_degree=1;

        //allocate memory for adjacency lists
        //edgeArray =new unsigned int[numVerts+1];
        //edgeArrayAux=new unsigned int[numVerts*(avg_degree+1)];

        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr1,
                        sizeof(unsigned int)*(numVerts+1)));
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr2,
                        sizeof(unsigned int)*(numVerts*(avg_degree+1))));

        //Generate simple tree
        G->GenerateSimpleKWayGraph(numVerts,avg_degree);
    }
    else
    {
        //open the graph file
        FILE *fp=fopen(inFileName.c_str(),"r");
        if(fp==NULL)
        {
            std::cerr <<"Error: Graph Input File Not Found." << endl;
            return;
        }

        //get the number of vertices and edges from the first line
        const char delimiters[]=" \n";
        char charBuf[MAX_LINE_LENGTH];
        fgets(charBuf,MAX_LINE_LENGTH,fp);
        char *temp_token = strtok (charBuf, delimiters);
        while(temp_token[0]=='%')
        {
            fgets(charBuf,MAX_LINE_LENGTH,fp);
            temp_token = strtok (charBuf, delimiters);
        }
        numVerts=atoi(temp_token);
        temp_token = strtok (NULL, delimiters);
        numEdges=atoi(temp_token);

        //allocate pinned memory
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr1,
                        sizeof(unsigned int)*(numVerts+1)));
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr2,
                        sizeof(unsigned int)*(numEdges*2)));

        fclose(fp);
        //Load the specified graph
        G->LoadMetisGraph(inFileName.c_str());
    }
    std::cout<<"Vertices: "<<G->GetNumVertices() << endl;
    std::cout<<"Edges: "<<G->GetNumEdges() << endl;


    int algo = op.getOptionInt("algo");
    switch(algo)
    {
        case 1:
                RunTest1(resultDB,op,G);
                break;
        case 2:
                RunTest2(resultDB,op,G);
                break;
    }

    //Clean up
    delete G;
    CUDA_SAFE_CALL(cudaFreeHost(*edge_ptr1));
    CUDA_SAFE_CALL(cudaFreeHost(*edge_ptr2));
}
