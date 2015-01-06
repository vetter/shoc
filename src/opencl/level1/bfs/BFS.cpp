#include<fstream>
#include<iostream>
#include<limits.h>
#include<math.h>
#include<set>
#include<string.h>
#include<sys/time.h>
#include<time.h>

#include "Event.h"
#include "Graph.h"
#include "OpenCLDeviceInfo.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "support.h"
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
// Function: GetWorkSize
//
// Purpose:
//   Get the kernel configuration so that we have one thread mapped to one
//   vertex in the frontier.
//
// Arguments:
//   gws: global work size
//   lws: local work size
//   maxThreadsPerCore: the max work group size for specified device
//   numVerts: number of vertices in the graph
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void GetWorkSize(size_t *gws, size_t *lws, size_t maxThreadsPerCore,
                 cl_uint numVerts)
{
    int temp;
    gws[0]=(size_t)numVerts;
    temp=(int)ceil((float)gws[0]/(float)maxThreadsPerCore);
    gws[0]=temp*maxThreadsPerCore;
    lws[0]=maxThreadsPerCore;
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
cl_uint verify_results(cl_uint *cpu_cost, cl_uint *gpu_cost, cl_uint numVerts,
                       bool out_path_lengths)
{
    cl_uint unmatched_nodes=0;

    //check cpu against gpu results
    for (int i=0;i<numVerts;i++)
    {
        if (gpu_cost[i]!=cpu_cost[i])
        {
            unmatched_nodes++;
        }
    }

    //if user wants to write path lengths to file
    if (out_path_lengths)
    {
        std::ofstream bfs_out_cpu("bfs_out_cpu.txt");
        std::ofstream bfs_out_gpu("bfs_out_ocl.txt");
        for (int i=0;i<numVerts;i++)
        {
            if (cpu_cost[i]!=UINT_MAX)
                bfs_out_cpu<<cpu_cost[i]<<"\n";
            else
                bfs_out_cpu<<"0\n";

            if (gpu_cost[i]!=UINT_MAX)
                bfs_out_gpu<<gpu_cost[i]<<"\n";
            else
                bfs_out_gpu<<"0\n";
        }
        bfs_out_cpu.close();
        bfs_out_gpu.close();
    }
    return unmatched_nodes;
}

extern const char *cl_source_bfs_iiit;

// ****************************************************************************
// Function: RunTest1
//
// Purpose:
//   Runs the BFS benchmark using method 1 (IIIT-BFS method)
//
// Arguments:
//   device: opencl device to run test on
//   context: the opencl context
//   queue: the opencl command queue
//   resultDB: the benchmark stores its results in this ResultDatabase
//   op: the options parser / parameter database
//   G: input graph
//
// Returns:  nothing
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunTest1(cl_device_id device, cl_context context, cl_command_queue queue,
        ResultDatabase& resultDB, OptionParser& op, Graph *G)
{
    typedef cl_uint frontier_type;
    typedef cl_uint cost_type;

    //get graph info
    cl_uint *edgeArray=G->GetEdgeOffsets();
    cl_uint *edgeArrayAux=G->GetEdgeList();
    cl_uint adj_list_length=G->GetAdjacencyListLength();
    cl_uint numVerts = G->GetNumVertices();
    cl_uint numEdges = G->GetNumEdges();

    //variable to get error code
    cl_int err_code;
    //load the kernel code
    cl_program program=clCreateProgramWithSource(context,1,&cl_source_bfs_iiit,
            NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //compile the kernel code
    const char *c_flags="";
    err_code=clBuildProgram(program,1,&device,c_flags,NULL,NULL);

    //if error dump the compile error
    if (err_code!=CL_SUCCESS)
    {
        size_t buildLogSize=0;
        std::cout<<"Kernel compilation error:\n";
        err_code = clGetProgramBuildInfo(program, device,
                CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
        char* buildLog=new char[buildLogSize+1];
        err_code = clGetProgramBuildInfo(program, device,
                CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
        buildLog[buildLogSize] = '\0';
        std::cout<<" : "<<buildLog<<"\n";
        delete[] buildLog;
        return;
    }
    CL_CHECK_ERROR(err_code);

    //allocate pinned memory
    //pinned memory for frontier array
    cl_mem h_f=clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(frontier_type)*numVerts,NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    frontier_type *frontier = (frontier_type *)clEnqueueMapBuffer(queue, h_f,
            true, CL_MAP_READ|CL_MAP_WRITE, 0,
            sizeof(frontier_type)*numVerts,
            0, NULL, NULL, &err_code);
    CL_CHECK_ERROR(err_code);


    //pinned memory for cost array
    cl_mem h_cost=clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(cost_type)*numVerts,
            NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    cost_type *costArray = (cost_type *)clEnqueueMapBuffer(queue,h_cost,true,
            CL_MAP_READ|CL_MAP_WRITE, 0,
            sizeof(cost_type)*numVerts,
            0, NULL, NULL, &err_code);
    CL_CHECK_ERROR(err_code);

    //Initialize frontier and visited arrays
    for (int index=0;index<numVerts;index++)
    {
        frontier[index]=UINT_MAX;
        costArray[index]=UINT_MAX;
    }

    // vertex to start traversal from
    const cl_uint source_vertex=op.getOptionInt("source_vertex");
    //set initial condition to traverse
    frontier[source_vertex]=0;
    costArray[source_vertex]=0;


    //variables for GPU memory
    //frontier arrays
    cl_mem d_frontier;
    //adjacency lists
    cl_mem d_edgeArray,d_edgeArrayAux;
    //flag to check when to stop iteration
    cl_mem d_flag;

    //allocate GPU memory for edge_offsets and edge_list
    d_edgeArray=clCreateBuffer(context,CL_MEM_READ_ONLY,
            (numVerts+1)*sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    d_edgeArrayAux=clCreateBuffer(context,CL_MEM_READ_ONLY,
            adj_list_length*sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //allocate GPU memory for frontier
    d_frontier=clCreateBuffer(context,CL_MEM_READ_WRITE,
            numVerts*sizeof(frontier_type),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //allocate GPU memory for flag variable
    d_flag=clCreateBuffer(context,CL_MEM_READ_WRITE,
            sizeof(cl_int),NULL,&err_code);
    CL_CHECK_ERROR(err_code);


    //initialize timer event
    Event evTransfer("PCIe Transfer");

    //Transfer edge_offsets and edge_list to GPU
    err_code = clEnqueueWriteBuffer(queue, d_edgeArray, CL_TRUE, 0,
            (numVerts+1)*sizeof(cl_uint), (void *)edgeArray,0, NULL,
            &evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    double inputTransferTime = evTransfer.StartEndRuntime();

    err_code = clEnqueueWriteBuffer(queue, d_edgeArrayAux, CL_TRUE, 0,
            adj_list_length*sizeof(cl_uint), (void *)edgeArrayAux,0,
            NULL, &evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    //Transfer frontier to GPU
    err_code = clEnqueueWriteBuffer(queue, d_frontier, CL_TRUE, 0,
            numVerts*sizeof(frontier_type), (void *)frontier,0,
            NULL, &evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    //Create kernel functions
    cl_kernel kernel1=clCreateKernel(program,"BFS_kernel_warp",&err_code);
    CL_CHECK_ERROR(err_code)

    //Get the kernel configuration
    size_t global_work_size,local_work_size;
    size_t maxWorkItemsPerGroup=getMaxWorkGroupSize(context, kernel1);

    GetWorkSize(&global_work_size,&local_work_size,maxWorkItemsPerGroup,
            numVerts);

    //configurable parameters for processing nodes
    cl_int W_SZ=32;
    cl_int CHUNK_SZ=32;

    //index for kernel parameters
    int p=-1;
    //specify  kernel1 parameters
    err_code=clSetKernelArg(kernel1,++p,sizeof(cl_mem),(void *)&d_frontier);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel1,++p,sizeof(cl_mem),(void *)&d_edgeArray);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel1,++p,sizeof(cl_mem),(void *)&d_edgeArrayAux);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel1,++p,sizeof(cl_int),(void *)&W_SZ);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel1,++p,sizeof(cl_int),(void *)&CHUNK_SZ);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel1,++p,sizeof(cl_uint),(void *)&numVerts);
    CL_CHECK_ERROR(err_code);
    //skip for levels
    ++p;
    err_code=clSetKernelArg(kernel1,++p,sizeof(cl_mem),(void *)&d_flag);
    CL_CHECK_ERROR(err_code);

    //Perform cpu bfs traversal for verify results
    cl_uint *cpu_cost = new cl_uint[numVerts];
    G->GetVertexLengths(cpu_cost,source_vertex);

    //timer for kernel execution
    Event evKernel("bfs kernel");
    //number of passes
    int passes = op.getOptionInt("passes");
    //Start the benchmark
    std::cout<<"Running Benchmark\n";
    for (int j=0;j<passes;j++)
    {
        if (j>0)
        {
            //Reset the arrays to perform BFS again
            for(int index=0;index<numVerts;index++)
            {
                frontier[index]=UINT_MAX;
            }
            //reset the initial condition
            frontier [source_vertex]=0;

            //write the arrays to gpu
            err_code = clEnqueueWriteBuffer(queue, d_frontier, CL_TRUE, 0,
                    numVerts*sizeof(frontier_type),(void *)frontier,
                    0,NULL,NULL);
            CL_CHECK_ERROR(err_code);
        }

        //flag set when there are nodes to be traversed.
        int flag=1;
        //Initialize timers
        double totalKernelTime=0;

        //specify the kernel configuration parameters
        size_t gws=0,lws=0;

        //iteration count
        cl_uint iters=0;
        //start CPU Timer to measure total time taken to complete benchmark
        int cpu_bfs_timer = Timer::Start();
        //while there are nodes to traverse
        //flag is set if nodes exist in frontier
        while (flag)
        {
            flag=0;
            err_code = clEnqueueWriteBuffer(queue, d_flag, CL_TRUE, 0,
                    sizeof(cl_int),(void *)&flag,0,NULL,NULL);
            CL_CHECK_ERROR(err_code);

            err_code=clSetKernelArg(kernel1,6,sizeof(cl_uint),(void *)&iters);
            CL_CHECK_ERROR(err_code);

            //Call kernel1
            err_code=clEnqueueNDRangeKernel(queue,kernel1,1,NULL,
                    &global_work_size,&local_work_size,0,NULL,
                    &evKernel.CLEvent());
            CL_CHECK_ERROR(err_code);
            clFinish(queue);
            CL_CHECK_ERROR(err_code);
            evKernel.FillTimingInfo();
            totalKernelTime += evKernel.StartEndRuntime();

            //Read the frontier
            err_code=clEnqueueReadBuffer(queue,d_flag,CL_TRUE,0,
                    sizeof(cl_int),&flag,0,NULL,NULL);
            clFinish(queue);
            CL_CHECK_ERROR(err_code);

            iters++;
        }
        //stop the CPU timer
        double result_time = Timer::Stop(cpu_bfs_timer, "cpu_bfs_timer");

        //copy the cost array from GPU to CPU
        err_code=clEnqueueReadBuffer(queue,d_frontier, CL_TRUE,0,
                sizeof(frontier_type)*numVerts,costArray,0,NULL,
                &evTransfer.CLEvent());
        CL_CHECK_ERROR(err_code);
        err_code = clFinish(queue);
        CL_CHECK_ERROR(err_code);
        evTransfer.FillTimingInfo();
        double outputTransferTime = evTransfer.StartEndRuntime();

        //get the total transfer time
        double totalTransferTime=inputTransferTime + outputTransferTime;

        //count number of visited vertices
        cl_uint numVisited=0;
        for(int i=0;i<numVerts;i++)
        {
            if(costArray[i]!=UINT_MAX)
                numVisited++;
        }

        bool dump_paths=op.getOptionBool("dump-pl");
        //Verify Results against serial BFS
        cl_uint unmatched_verts=verify_results(cpu_cost,costArray,numVerts,
                dump_paths);

        //total size transferred
        float gbytes=
            sizeof(frontier_type)*numVerts*2+   //2 frontiers
            sizeof(cost_type)*numVerts+         //cost array
            sizeof(cl_uint)*(numVerts+1)+       //edgeArray
            sizeof(cl_uint)*adj_list_length;    //edgeArrayAux
        gbytes/=(1000. * 1000. * 1000.);

        //populate the result database
        char atts[1024];
        sprintf(atts,"v:%d_e:%d ",numVerts,adj_list_length);
        if(unmatched_verts==0)
        {
            totalKernelTime *= 1.e-9;
            totalTransferTime *= 1.e-9;
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
            resultDB.AddResult("BFS_Parity", atts, "N",FLT_MAX);
            resultDB.AddResult("BFS_teps",atts,"Edges/s",FLT_MAX);
            resultDB.AddResult("BFS_visited_vertices", atts, "N",FLT_MAX);
            return;
        }

        std::cout<<"Test ";
        if(unmatched_verts==0)
        {
            std::cout<<"Passed\n";
        }
        else
        {
            std::cout<<"Failed\n";
            return;
        }
    }

    //Clean up
    delete[] cpu_cost;

    err_code=clReleaseKernel(kernel1);
    CL_CHECK_ERROR(err_code);

    err_code=clEnqueueUnmapMemObject(queue, h_f, frontier, 0, NULL, NULL);
    CL_CHECK_ERROR(err_code);

    err_code=clEnqueueUnmapMemObject(queue, h_cost, costArray, 0, NULL, NULL);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(h_f);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(h_cost);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_frontier);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_edgeArray);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_edgeArrayAux);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_flag);
    CL_CHECK_ERROR(err_code);

    err_code = clReleaseProgram(program);
    CL_CHECK_ERROR(err_code);

}

extern const char *cl_source_bfs_uiuc_spill;
// ****************************************************************************
// Function: RunTest2
//
// Purpose:
//   Runs the BFS benchmark using method 2 (UIUC-BFS method)
//
// Arguments:
//   device: opencl device to run test on
//   context: the opencl context
//   queue: the opencl command queue
//   resultDB: the benchmark stores its results in this ResultDatabase
//   op: the options parser / parameter database
//   G: input graph
//
// Returns:  nothing
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunTest2(
        cl_device_id device,
        cl_context context,
        cl_command_queue queue,
        ResultDatabase& resultDB,
        OptionParser& op,
        Graph *G)

{
    typedef cl_uint frontier_type;
    typedef int visited_type;
    typedef cl_uint cost_type;

    //get graph info
    cl_uint *edgeArray=G->GetEdgeOffsets();
    cl_uint *edgeArrayAux=G->GetEdgeList();
    cl_uint adj_list_length=G->GetAdjacencyListLength();
    cl_uint numVerts = G->GetNumVertices();
    cl_uint numEdges = G->GetNumEdges();


    //variable to get error code
    cl_int err_code;

    //load the kernel code
    cl_program program=clCreateProgramWithSource(context,1,
            &cl_source_bfs_uiuc_spill,NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //compile the kernel code
    const char *compiler_flags="-g -O0";
    err_code=clBuildProgram(program,1,&device,NULL,NULL,NULL);

    //if error dump the compile error
    if (err_code!=CL_SUCCESS)
    {
        size_t buildLogSize=0;
        std::cout<<"Kernel compilation error.\n";
        err_code = clGetProgramBuildInfo(program, device,
                CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
        char *buildLog =new char[buildLogSize+1];
        err_code = clGetProgramBuildInfo(program, device,
                CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
        buildLog[buildLogSize] = '\0';
        std::cout<<" : "<<buildLog<<"\n";
        delete[] buildLog;
        return;
    }


    //allocate pinned memory on CPU
    //pinned memory for frontier array
    cl_mem h_f=clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(frontier_type)*(numVerts),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    frontier_type *frontier = (frontier_type *)clEnqueueMapBuffer(queue, h_f,
            true, CL_MAP_READ|CL_MAP_WRITE, 0,
            sizeof(frontier_type)*(numVerts),
            0, NULL, NULL, &err_code);
    CL_CHECK_ERROR(err_code);

    //pinned memory for visited array
    cl_mem h_v=clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(visited_type)*(numVerts),
            NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    visited_type *visited = (visited_type *)clEnqueueMapBuffer(queue, h_v,true,
            CL_MAP_READ|CL_MAP_WRITE, 0,
            sizeof(visited_type)*(numVerts),
            0, NULL, NULL, &err_code);
    CL_CHECK_ERROR(err_code);

    //pinned memory for cost array
    cl_mem h_cost=clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(cost_type)*(numVerts),
            NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    cost_type *costArray = (cost_type *)clEnqueueMapBuffer(queue,h_cost,true,
            CL_MAP_READ|CL_MAP_WRITE, 0,
            sizeof(cost_type)*(numVerts),
            0, NULL, NULL, &err_code);
    CL_CHECK_ERROR(err_code);


    //Initialize frontier, visited and costarrays
    for (int index=0;index<numVerts;index++)
    {
        frontier[index]=0;
        costArray[index]=UINT_MAX;
        visited[index]=0;
    }


    //Set vertex to start traversal from
    const cl_uint source_vertex=op.getOptionInt("source_vertex");

    //Set the initial condition to traverse
    cl_uint frontier_length=1;
    frontier[0]=source_vertex;
    visited[source_vertex]=1;
    costArray[source_vertex]=0;

    //variables for GPU memory

    //frontier and visited arrays
    //(d_t_frontier to store frontier for next iteration)
    cl_mem d_frontier,d_t_frontier,d_visited;
    //adjacency lists
    cl_mem d_edgeArray,d_edgeArrayAux;
    //frontier length
    cl_mem d_frontier_length;
    //mutex variables for global synchronization
    cl_mem d_g_mutex,d_g_mutex2;
    //variables to store intermediate frontier lengths while iterating
    cl_mem d_g_q_offsets,d_g_q_size;
    //cost array
    cl_mem d_costArray;

    //allocate GPU memory for adjacency list
    d_edgeArray=clCreateBuffer(context,CL_MEM_READ_ONLY,
            (numVerts+1)*sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    d_edgeArrayAux=clCreateBuffer(context,CL_MEM_READ_ONLY,
            adj_list_length*sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //allocate GPU memory for frontier and d_t_frontier
    d_frontier=clCreateBuffer(context,CL_MEM_READ_WRITE,
            numVerts*sizeof(frontier_type),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    d_t_frontier=clCreateBuffer(context,CL_MEM_READ_WRITE,
            numVerts*sizeof(frontier_type),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //allocate GPU memory for visited
    d_visited=clCreateBuffer(context,CL_MEM_READ_WRITE,
            numVerts* sizeof(visited_type),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //allocate GPU memory for frontier length
    d_frontier_length=clCreateBuffer(context,CL_MEM_READ_WRITE,
            sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //allocate GPU memory for mutex variables and
    //intermediate frontier length variables
    d_g_mutex=clCreateBuffer(context,CL_MEM_READ_WRITE,
            sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    d_g_mutex2=clCreateBuffer(context,CL_MEM_READ_WRITE,
            sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    d_g_q_offsets=clCreateBuffer(context,CL_MEM_READ_WRITE,
            sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    d_g_q_size=clCreateBuffer(context,CL_MEM_READ_WRITE,
            sizeof(cl_uint),NULL,&err_code);
    CL_CHECK_ERROR(err_code);

    //allocate GPU memory for cost array
    d_costArray=clCreateBuffer(context,CL_MEM_READ_WRITE,
            numVerts* sizeof(cost_type),NULL,&err_code);
    CL_CHECK_ERROR(err_code);


    //initialize timer event
    Event evTransfer("PCIe Transfer");

    //Transfer adjacency list to GPU
    err_code = clEnqueueWriteBuffer(queue, d_edgeArray, CL_TRUE, 0,
            (numVerts+1)*sizeof(cl_uint), (void *)edgeArray,0, NULL,
            &evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    double inputTransferTime = evTransfer.StartEndRuntime();

    err_code = clEnqueueWriteBuffer(queue, d_edgeArrayAux, CL_TRUE, 0,
            adj_list_length*sizeof(cl_uint), (void *)edgeArrayAux,0,
            NULL, &evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    //Transfer frontiers to GPU
    err_code = clEnqueueWriteBuffer(queue, d_frontier, CL_TRUE, 0,
            numVerts*sizeof(frontier_type), (void *)frontier,0,
            NULL, &evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    err_code = clEnqueueWriteBuffer(queue, d_t_frontier, CL_TRUE, 0,
            sizeof(frontier_type)*numVerts, (void *)frontier,0,
            NULL, &evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    //Transfer visited array to GPU
    err_code = clEnqueueWriteBuffer(queue, d_visited, CL_TRUE, 0,
            numVerts*sizeof(visited_type), (void*)visited, 0,
            NULL,&evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    //Transfer cost array to GPU
    err_code = clEnqueueWriteBuffer(queue, d_costArray, CL_TRUE, 0,
            numVerts*sizeof(cost_type), (void*)costArray, 0,
            NULL,&evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    //Transfer frontier length to GPU
    err_code = clEnqueueWriteBuffer(queue, d_frontier_length, CL_TRUE, 0,
            sizeof(cl_uint), (void*)&frontier_length, 0,NULL,
            &evTransfer.CLEvent());
    CL_CHECK_ERROR(err_code);
    err_code = clFinish(queue);
    CL_CHECK_ERROR(err_code);
    evTransfer.FillTimingInfo();
    inputTransferTime += evTransfer.StartEndRuntime();

    //Create kernel functions
    cl_kernel kernel_op_1=clCreateKernel(program,"BFS_kernel_one_block",
        &err_code);
    CL_CHECK_ERROR(err_code);
    cl_kernel kernel_op_2=clCreateKernel(program,"BFS_kernel_SM_block",
         &err_code);
    CL_CHECK_ERROR(err_code);
    cl_kernel kernel_op_3=clCreateKernel(program,"BFS_kernel_multi_block",
        &err_code);
    CL_CHECK_ERROR(err_code);
    cl_kernel kernel_op_fcopy=clCreateKernel(program,"Frontier_copy",&err_code);
    CL_CHECK_ERROR(err_code);
    cl_kernel kernel_op_reset=clCreateKernel(program,"Reset_kernel_parameters",
        &err_code);
    CL_CHECK_ERROR(err_code);

    //Get the kernel configuration
    size_t global_work_size;
    size_t local_work_size;
    size_t size_temp;
    int max_compute_units;
    err_code=clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(int),
            &max_compute_units,NULL);
    CL_CHECK_ERROR(err_code);

    size_t maxWorkItemsPerGroup=
        getMaxWorkGroupSize(context, kernel_op_1);
    size_temp=getMaxWorkGroupSize(context, kernel_op_2);
    if(maxWorkItemsPerGroup>size_temp)
        maxWorkItemsPerGroup=size_temp;

    size_temp=getMaxWorkGroupSize(context, kernel_op_3);
    if(maxWorkItemsPerGroup>size_temp)
        maxWorkItemsPerGroup=size_temp;

    size_temp=getMaxWorkGroupSize(context, kernel_op_fcopy);
    if(maxWorkItemsPerGroup>size_temp)
        maxWorkItemsPerGroup=size_temp;

    GetWorkSize(&global_work_size,&local_work_size,maxWorkItemsPerGroup,
            numVerts);

    //calculate the usable shared memory
    size_t max_local_mem =getLocalMemSize(device);
    max_local_mem = max_local_mem - sizeof(cl_uint)*3;

    cl_uint max_q_size=(max_local_mem/sizeof(cl_uint));
    //index for kernel parameters
    int p=-1;


    //Set kernel parameters for BFS_kernel_one_block
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_mem),(void *)&d_frontier);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_uint),
            (void *)&frontier_length);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_mem),(void *)&d_visited);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_mem),
            (void *)&d_costArray);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_mem),
            (void *)&d_edgeArray);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_mem),
            (void *)&d_edgeArrayAux);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_uint),(void *)&numVerts);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_uint),(void *)&numEdges);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_mem),
            (void *)&d_frontier_length);
    CL_CHECK_ERROR(err_code);

    cl_uint q_size=max_q_size/2;
    if(q_size>local_work_size)
        q_size=local_work_size;

    err_code=clSetKernelArg(kernel_op_1,++p,sizeof(cl_uint),(void *)&q_size);
    CL_CHECK_ERROR(err_code);

    //shared memory for BFS_kernel_one_block
    err_code=clSetKernelArg(kernel_op_1,++p,q_size*sizeof(cl_uint), NULL);
    CL_CHECK_ERROR(err_code);

    err_code=clSetKernelArg(kernel_op_1,++p,q_size*sizeof(cl_uint), NULL);
    CL_CHECK_ERROR(err_code);


    //Set kernel parameters for BFS_kernel_SM_block
    p=-1;
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),(void *)&d_frontier);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_uint),
            (void *)&frontier_length);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),
            (void *)&d_t_frontier);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),(void *)&d_visited);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),
            (void *)&d_costArray);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),
            (void *)&d_edgeArray);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),
            (void *)&d_edgeArrayAux);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_uint),(void *)&numVerts);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_uint),(void *)&numEdges);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),
            (void *)&d_frontier_length);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),(void *)&d_g_mutex);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),(void *)&d_g_mutex2);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),
            (void *)&d_g_q_offsets);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_mem),(void *)&d_g_q_size);
    CL_CHECK_ERROR(err_code);

    q_size=max_q_size;
    if(q_size>local_work_size)
        q_size=local_work_size;

    err_code=clSetKernelArg(kernel_op_2,++p,sizeof(cl_uint),(void *)&q_size);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_2,++p, sizeof(cl_uint)*q_size,NULL);
    CL_CHECK_ERROR(err_code);

    //Set kernel parameters for BFS_kernel_multi_block
    p=-1;
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_mem),(void *)&d_frontier);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_uint),
            (void *)&frontier_length);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_mem),
            (void *)&d_t_frontier);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_mem),(void *)&d_visited);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_mem),
            (void *)&d_costArray);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_mem),
            (void *)&d_edgeArray);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_mem),
            (void *)&d_edgeArrayAux);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_uint),(void *)&numVerts);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_uint),(void *)&numEdges);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_mem),
            (void *)&d_frontier_length);
    CL_CHECK_ERROR(err_code);

    q_size=max_q_size;
    if(q_size>local_work_size)
        q_size=local_work_size;

    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_uint),(void *)&q_size);
    CL_CHECK_ERROR(err_code);

    err_code=clSetKernelArg(kernel_op_3,++p,sizeof(cl_uint)*q_size,NULL);
    CL_CHECK_ERROR(err_code);

    //Set kernel parameters for Reset_kernel_parameters
    p=-1;
    err_code=clSetKernelArg(kernel_op_reset,++p,sizeof(cl_mem),
            (void *)&d_frontier_length);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_reset,++p,sizeof(cl_mem),
            (void *)&d_g_mutex);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_reset,++p,sizeof(cl_mem),
            (void *)&d_g_mutex2);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_reset,++p,sizeof(cl_mem),
            (void *)&d_g_q_offsets);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_reset,++p,sizeof(cl_mem),
            (void *)&d_g_q_size);
    CL_CHECK_ERROR(err_code);

    //Set kernel parameters for Frontier_copy
    p=-1;
    err_code=clSetKernelArg(kernel_op_fcopy,++p,sizeof(cl_mem),
            (void *)&d_frontier);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_fcopy,++p,sizeof(cl_mem),
            (void *)&d_t_frontier);
    CL_CHECK_ERROR(err_code);

    err_code=clSetKernelArg(kernel_op_fcopy,++p,sizeof(cl_mem),
            (void *)&d_frontier_length);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_fcopy,++p,sizeof(cl_mem),
            (void *)&d_g_mutex);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_fcopy,++p,sizeof(cl_mem),
            (void *)&d_g_mutex2);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_fcopy,++p,sizeof(cl_mem),
            (void *)&d_g_q_offsets);
    CL_CHECK_ERROR(err_code);
    err_code=clSetKernelArg(kernel_op_fcopy,++p,sizeof(cl_mem),
            (void *)&d_g_q_size);
    CL_CHECK_ERROR(err_code);

    //Perform cpu bfs traversal for verifying results
    cl_uint *cpu_cost = new cl_uint[numVerts];
    G->GetVertexLengths(cpu_cost,source_vertex);

    bool g_barrier=op.getOptionBool("global-barrier");
    //timer for kernel execution
    Event evKernel("bfs kernel");
    //number of passes
    int passes = op.getOptionInt("passes");
    //Start the benchmark
    std::cout<<"Running Benchmark\n";
    for (int j=0;j<passes;j++)
    {

        if (j>0)
        {
            //Reset the arrays to perform BFS again
            for (int index=0;index<numVerts;index++)
            {
                frontier[index]=0;
                costArray[index]=UINT_MAX;
                visited[index]=0;
            }
            //Set the initial condition to traverse
            frontier_length=1;
            frontier[0]=source_vertex;
            visited[source_vertex]=1;
            costArray[source_vertex]=0;
            //write buffers to gpu
            err_code = clEnqueueWriteBuffer(queue, d_frontier, CL_TRUE, 0,
                    numVerts*sizeof(frontier_type),(void *)frontier,
                    0,NULL,NULL);
            CL_CHECK_ERROR(err_code);

            err_code = clEnqueueWriteBuffer(queue, d_visited, CL_TRUE, 0,
                    numVerts*sizeof(visited_type), (void *)visited,
                    0,NULL,NULL);
            CL_CHECK_ERROR(err_code);

            err_code = clEnqueueWriteBuffer(queue, d_costArray, CL_TRUE, 0,
                    numVerts*sizeof(cost_type), (void *)costArray, 0,NULL,NULL);
            CL_CHECK_ERROR(err_code);

            err_code = clEnqueueWriteBuffer(queue, d_t_frontier, CL_TRUE, 0,
                    sizeof(frontier_type),(void *)frontier,0,NULL,NULL);
            CL_CHECK_ERROR(err_code);

            err_code = clEnqueueWriteBuffer(queue, d_frontier_length, CL_TRUE,0,
                    sizeof(cl_uint), (void*)&frontier_length, 0,NULL,NULL);
            CL_CHECK_ERROR(err_code);
        }

        //Initialize timers
        double totalKernelTime=0;
        //specify the kernel configuration parameters
        size_t gws=0,lws=0;
        //start CPU Timer to measure total time taken to complete benchmark
        int cpu_bfs_timer = Timer::Start();
        //while there are nodes to traverse
        while (frontier_length>0)
        {
            //set kernel configuration
            //call Reset_kernel_parameters
            gws=1;
            lws=1;
            err_code=clEnqueueNDRangeKernel(queue,kernel_op_reset,1,NULL,
                    &gws,&lws,0,NULL,&evKernel.CLEvent());
            CL_CHECK_ERROR(err_code);
            clFinish(queue);
            CL_CHECK_ERROR(err_code);
            evKernel.FillTimingInfo();
            totalKernelTime += evKernel.StartEndRuntime();

            //kernel for frontier length within one block
            if (frontier_length<maxWorkItemsPerGroup)
            {
                err_code=clSetKernelArg(kernel_op_1,1,sizeof(cl_uint),
                        (void *)&frontier_length);
                gws=maxWorkItemsPerGroup;
                lws=maxWorkItemsPerGroup;

                err_code=clEnqueueNDRangeKernel(queue,kernel_op_1,1,NULL,
                        &gws,&lws,0,NULL,&evKernel.CLEvent());
                CL_CHECK_ERROR(err_code);
                clFinish(queue);
                CL_CHECK_ERROR(err_code);
                evKernel.FillTimingInfo();
                totalKernelTime += evKernel.StartEndRuntime();
            }
            //kernel for frontier length within SM blocks
            else if (g_barrier &&
                    frontier_length< maxWorkItemsPerGroup * max_compute_units)
            {
                err_code=clSetKernelArg(kernel_op_2,1,sizeof(cl_uint),
                        (void *)&frontier_length);

                gws=maxWorkItemsPerGroup * max_compute_units;
                lws=maxWorkItemsPerGroup;

                err_code=clEnqueueNDRangeKernel(queue,kernel_op_2,1,NULL,
                        &gws,&lws,0,NULL,&evKernel.CLEvent());
                CL_CHECK_ERROR(err_code);
                clFinish(queue);
                CL_CHECK_ERROR(err_code);
                evKernel.FillTimingInfo();
                totalKernelTime += evKernel.StartEndRuntime();
            }
            //kernel for frontier length greater than SM blocks
            else
            {
                err_code=clSetKernelArg(kernel_op_3,1,sizeof(cl_uint),
                        (void *)&frontier_length);

                err_code=clEnqueueNDRangeKernel(queue,kernel_op_3,1,NULL,
                        &global_work_size,&local_work_size,0,NULL,
                        &evKernel.CLEvent());
                CL_CHECK_ERROR(err_code);
                clFinish(queue);
                CL_CHECK_ERROR(err_code);
                evKernel.FillTimingInfo();
                totalKernelTime += evKernel.StartEndRuntime();

                err_code=clEnqueueNDRangeKernel(queue,kernel_op_fcopy,1,NULL,
                        &global_work_size,&local_work_size,0,NULL,
                        &evKernel.CLEvent());
                CL_CHECK_ERROR(err_code);
                clFinish(queue);
                CL_CHECK_ERROR(err_code);
                evKernel.FillTimingInfo();
                totalKernelTime += evKernel.StartEndRuntime();
            }
            //Get the current frontier length
            err_code=clEnqueueReadBuffer(queue,d_frontier_length,CL_TRUE,0,
                    sizeof(cl_uint),&frontier_length,0,NULL,NULL);
            CL_CHECK_ERROR(err_code);
        }
        //stop the CPU timer
        double result_time = Timer::Stop(cpu_bfs_timer, "cpu_bfs_timer");
        //copy the cost array back to CPU
        err_code=clEnqueueReadBuffer(queue,d_costArray, CL_TRUE,0,
                sizeof(cost_type)*numVerts,costArray,0,NULL,
                &evTransfer.CLEvent());
        CL_CHECK_ERROR(err_code);
        err_code = clFinish(queue);
        CL_CHECK_ERROR(err_code);
        evTransfer.FillTimingInfo();
        double outputTransferTime = evTransfer.StartEndRuntime();

        //get the total transfer time
        double totalTransferTime=inputTransferTime + outputTransferTime;

        //count number of visited vertices
        cl_uint numVisited=0;
        for (int i=0;i<numVerts;i++)
        {
            if(costArray[i]!=UINT_MAX)
                numVisited++;
        }

        bool dump_paths=op.getOptionBool("dump-pl");
        //Verify Results against serial BFS
        cl_uint unmatched_verts=verify_results(cpu_cost,costArray,numVerts,
                dump_paths);

        float gbytes=
            sizeof(frontier_type)*numVerts*2+  //2 frontiers
            sizeof(cost_type)*numVerts+        //cost array
            sizeof(visited_type)*numVerts+     //visited mask array
            sizeof(cl_uint)*(numVerts+1)+      //edgeArray
            sizeof(cl_uint)*adj_list_length;   //edgeArrayAux

        gbytes/=(1000. * 1000. * 1000.);

        //populate the result database
        char atts[1024];
        sprintf(atts,"v:%d_e:%d",numVerts,adj_list_length);
        if (unmatched_verts==0)
        {
            totalKernelTime *= 1.e-9;
            totalTransferTime *= 1.e-9;
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
            return;
        }

        std::cout<<"Test ";
        if(unmatched_verts==0)
        {
            std::cout<<"Passed\n";
        }
        else
        {
            std::cout<<"Failed\n";
            return;
        }
    }

    //Clean up

    delete[] cpu_cost;
    err_code = clReleaseKernel(kernel_op_1);
    CL_CHECK_ERROR(err_code);

    err_code = clReleaseKernel(kernel_op_2);
    CL_CHECK_ERROR(err_code);

    err_code = clReleaseKernel(kernel_op_3);
    CL_CHECK_ERROR(err_code);

    err_code = clReleaseKernel(kernel_op_fcopy);
    CL_CHECK_ERROR(err_code);

    err_code = clReleaseKernel(kernel_op_reset);
    CL_CHECK_ERROR(err_code);

    err_code= clEnqueueUnmapMemObject(queue, h_f, frontier, 0, NULL, NULL);
    CL_CHECK_ERROR(err_code);

    err_code= clEnqueueUnmapMemObject(queue, h_v, visited, 0, NULL, NULL);
    CL_CHECK_ERROR(err_code);

    err_code= clEnqueueUnmapMemObject(queue, h_cost, costArray, 0, NULL, NULL);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(h_cost);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(h_f);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(h_v);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_frontier);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_visited);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_t_frontier);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_frontier_length);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_g_mutex);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_g_mutex2);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_g_q_offsets);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_g_q_size);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_costArray);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_edgeArray);
    CL_CHECK_ERROR(err_code);

    err_code=clReleaseMemObject(d_edgeArrayAux);
    CL_CHECK_ERROR(err_code);

    err_code = clReleaseProgram(program);
    CL_CHECK_ERROR(err_code);

}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the BFS benchmark
//
// Arguments:
//   devcpp: opencl device
//   ctxcpp: the opencl context
//   queuecpp: the opencl command queue
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
void RunBenchmark(cl_device_id device,
        cl_context context,
        cl_command_queue queue,
        ResultDatabase &resultDB,
        OptionParser &op)
{
    //adjacency list variables
    cl_mem h_edge,h_edgeAux;
    //number of vertices and edges in graph
    cl_uint numVerts=0,numEdges=0;
    //variable to get error code
    cl_int err_code;
    //Get the graph filename
    string inFileName = op.getOptionString("graph_file");
    //max degree in graph
    unsigned int max_deg=0;

    //Create graph
    Graph *G=new Graph();

    //Get pointers to edge offsets and edge list
    cl_uint **edge_ptr1 = G->GetEdgeOffsetsPtr();
    cl_uint **edge_ptr2 = G->GetEdgeListPtr();

    //Load simple k-way tree or from a file
    if (inFileName == "random")
    {
        //Load simple k-way tree
        //prob size specifies number of vertices
        cl_uint prob_sizes[4] = { 1000,10000,100000,1000000 };
       
        //Check for a valid size option and exit if not found
        if((op.getOptionInt("size") > 4) || (op.getOptionInt("size") <= 0))
        {
          cout<<"Please use a size between 1 and 4"<<endl;
          cout<<"Exiting..."<<endl;
          return;
        }

        numVerts = prob_sizes[op.getOptionInt("size")-1];
        int avg_degree = op.getOptionInt("degree");
        if(avg_degree<1)
            avg_degree=1;

        //allocate pinned memory for adjacency lists
        h_edge = clCreateBuffer(context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                sizeof(cl_uint)*(numVerts+1),NULL,&err_code);
        CL_CHECK_ERROR(err_code);

        *edge_ptr1 = (cl_uint *)clEnqueueMapBuffer(queue, h_edge, true,
                CL_MAP_READ|CL_MAP_WRITE, 0,
                sizeof(cl_uint)*(numVerts+1),
                0, NULL, NULL, &err_code);
        CL_CHECK_ERROR(err_code);

        h_edgeAux = clCreateBuffer(context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                sizeof(cl_uint)*(numVerts*(avg_degree+1)),
                NULL,&err_code);
        CL_CHECK_ERROR(err_code);

        *edge_ptr2 = (cl_uint *)clEnqueueMapBuffer(queue, h_edgeAux,true,
                CL_MAP_READ|CL_MAP_WRITE, 0,
                sizeof(cl_uint)*(numVerts*(avg_degree+1)),
                0, NULL, NULL, &err_code);
        CL_CHECK_ERROR(err_code);

        //Generate simple tree
        G->GenerateSimpleKWayGraph(numVerts,avg_degree);
    }
    else
    {
        //Read number of vertices and edges from first line of graph
        FILE *fp=fopen(inFileName.c_str(),"r");
        if(fp==NULL)
        {
            std::cout<<"\nFile not found!!!";
            return;
        }
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
        fclose(fp);

        //allocate memory for adjacency lists
        h_edge=clCreateBuffer(context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                sizeof(cl_uint)*(numVerts+1),NULL,&err_code);
        CL_CHECK_ERROR(err_code);

        *edge_ptr1 = (cl_uint *)clEnqueueMapBuffer(queue, h_edge, true,
                CL_MAP_READ|CL_MAP_WRITE, 0,
                sizeof(cl_uint)*(numVerts+1),
                0, NULL, NULL, &err_code);
        CL_CHECK_ERROR(err_code);

        h_edgeAux=clCreateBuffer(context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                sizeof(cl_uint)*(numEdges*2),
                NULL,&err_code);
        CL_CHECK_ERROR(err_code);

        *edge_ptr2 = (cl_uint *)clEnqueueMapBuffer(queue, h_edgeAux,true,
                CL_MAP_READ|CL_MAP_WRITE, 0,
                sizeof(cl_uint)*(numEdges*2),
                0, NULL, NULL, &err_code);
        CL_CHECK_ERROR(err_code);

        //Load metis graph
        G->LoadMetisGraph(inFileName.c_str());

    }

    std::cout<<"Vertices: "<<G->GetNumVertices() << endl;
    std::cout<<"Edges: "<<G->GetNumEdges() << endl;
    int algo = op.getOptionInt("algo");
    //Run the test according to specified method
    switch(algo)
    {
        case 1:
            RunTest1(device, context, queue, resultDB, op,G);
            break;
        case 2:
            RunTest2(device, context, queue, resultDB, op,G);
            break;
    }

    //Clean up
    err_code=clEnqueueUnmapMemObject(queue, h_edge, *edge_ptr1, 0, NULL, NULL);
    CL_CHECK_ERROR(err_code);

    err_code=clEnqueueUnmapMemObject(queue, h_edgeAux, *edge_ptr2,
            0, NULL, NULL);
    CL_CHECK_ERROR(err_code);


    clReleaseMemObject(h_edge);
    CL_CHECK_ERROR(err_code);

    clReleaseMemObject(h_edgeAux);
    CL_CHECK_ERROR(err_code);

    delete G;
}
