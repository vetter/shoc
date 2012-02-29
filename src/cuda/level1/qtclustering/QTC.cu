#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "tuningParameters.h"
#include "qtclib.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "support.h"
#include "libdata.h"

#include "cudacommon.h"
#define _USE_MATH_DEFINES
#include <float.h>
#include <cuda_runtime.h>

#include "comm.h"

#if defined(N_SQUARE)
texture<float, 2, cudaReadModeElementType> texWork;
#else
# if defined(USE_TEXTURES)
texture<float, 2, cudaReadModeElementType> texDenseDist;
# endif
#endif

#include "codelets.h"

using namespace std;

#ifdef MIN
# undef MIN
#endif
#define MIN(_X, _Y) ( ((_X) < (_Y)) ? (_X) : (_Y) )

#ifdef MAX
# undef MAX
#endif
#define MAX(_X, _Y) ( ((_X) > (_Y)) ? (_X) : (_Y) )

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in megabytes if they are not using a
//   predefined size (i.e. the -s option).
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op){
    op.addOption("POINTS", OPT_INT, "0", "point count");
    op.addOption("Threshold", OPT_FLOAT, "0.01", "cluster diameter threshold");
    op.addOption("File", OPT_STRING, "data.ssv", "BLAST data input file name");
    op.addOption("SaveOutput", OPT_BOOL, "", "BLAST data input file name");
    op.addOption("Verbose", OPT_BOOL, "", "Print Cluster Cardinalities");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Calls single precision and, if viable, double precision QT-Clustering
//   benchmark.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
//
// ****************************************************************************
void runTest(const string& name, ResultDatabase &resultDB, OptionParser& op);

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op){
    // Test to see if this device supports double precision
    cudaGetDevice(&qtcDevice);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, qtcDevice);

    runTest("SP-QTC", resultDB, op);
}



// ****************************************************************************
// Function: calculate_participants
//
// Purpose:
//   This function decides how many GPUs (up to the maximum requested by the user)
//   and threadblocks per GPU will be used. It also returns the total number of
//   thread-blocks across all GPUs and the number of thread-blocks that are in nodes
//   before the current one.
//   In the future, the behavior of this function should be decided based on
//   auto-tuning instead of arbitrary decisions.
//
// Arguments:
//   The number of nodes requested by the user and the four
//   variables that the function computes (passed by reference)
// 
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: May 25, 2011
//
// ****************************************************************************
void calculate_participants(int point_count, int node_count, int cwrank, int *thread_block_count, int *total_thread_block_count, int *active_node_count, int *node_offset){

    int ac_nd_cnt, offset, thr_blc_cnt, total_thr_blc_cnt;

    ac_nd_cnt = node_count;
    if( point_count <= (node_count-1) * SM_COUNT * GPU_MIN_SATURATION_FACTOR ){
        int K = SM_COUNT * GPU_MIN_SATURATION_FACTOR;
        ac_nd_cnt = (point_count+K-1) / K;
    }

    if( point_count >= ac_nd_cnt * SM_COUNT * OVR_SBSCR_FACTOR ){
        thr_blc_cnt = SM_COUNT * OVR_SBSCR_FACTOR;
        total_thr_blc_cnt = thr_blc_cnt * ac_nd_cnt;
        offset = cwrank*thr_blc_cnt;
    }else{
        thr_blc_cnt = point_count/ac_nd_cnt;
        if( cwrank < point_count%ac_nd_cnt ){
            thr_blc_cnt++;
            offset = cwrank*thr_blc_cnt;
        }else{
            offset = cwrank*thr_blc_cnt + point_count%ac_nd_cnt;
        }
        total_thr_blc_cnt = point_count;
    }


    *active_node_count  = ac_nd_cnt;
    *node_offset = offset;
    *thread_block_count = thr_blc_cnt;
    *total_thread_block_count = total_thr_blc_cnt;

    return;
}

// ****************************************************************************
// Function: runTest
//
// Purpose:
//   This benchmark measures the performance of applying QT-clustering on
//   single precision data.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
//
// ****************************************************************************

void runTest(const string& name, ResultDatabase &resultDB, OptionParser& op)
{    
    int *output;
    ofstream debug_out, debug2_out, seeds_out;
    void *Ai_mask, *cardnl, *ungrpd_pnts_indr, *clustered_pnts_mask, *result, *dist_to_clust;
    bool save_clusters, be_verbose;
    void *degrees; 
#if defined(N_SQUARE)
    cudaArray *work;
    float *source;
#else
# if defined(USE_TEXTURES)
    cudaArray *dense_dist_matrix;
# else
    void *dense_dist_matrix;
# endif
    float *dist_source;
    int *indr_mtrx_host;
#endif
    void *indr_mtrx;
    float threshold;
    int *ungrpd_pnts_indr_host, *cardinalities;
    int i, max_degree, thread_block_count, total_thread_block_count, active_node_count;
    int node_offset, cwrank=0, node_count=1, tpb;
    int max_card, iter=0;
    double t_krn, t_comm, t_trim, t_updt, t_redc;
    unsigned long point_count = 0, max_point_count;
    unsigned long used_bytes;
    float *pnts;

    cwrank = comm_get_rank();
    node_count = comm_get_size();

    point_count = op.getOptionInt("POINTS");
    threshold = op.getOptionFloat("Threshold");
    save_clusters = op.getOptionBool("SaveOutput");
    be_verbose = op.getOptionBool("Verbose");
    if( cwrank == 0 ){
#if defined(N_SQUARE)
        pnts = fake_BLAST_data(&source, &max_degree, threshold, point_count);
#else
        pnts = fake_BLAST_data(&dist_source, &indr_mtrx_host, &max_degree, threshold, point_count);
#endif
    }

    comm_broadcast ( &point_count, 1, COMM_TYPE_INT, 0);
    comm_broadcast ( &max_degree, 1, COMM_TYPE_INT, 0);
    if( cwrank != 0 ){ // For all nodes except zero, in a distributed run.
#if defined(N_SQUARE)
        allocHostBuffer((void **)&source, point_count*point_count*sizeof(float));
#else
        allocHostBuffer((void **)&dist_source, point_count*max_degree*sizeof(float));
        allocHostBuffer((void **)&indr_mtrx_host, point_count*max_degree*sizeof(int));
#endif
    }

#if defined(N_SQUARE)
    comm_broadcast ( source, point_count*point_count, COMM_TYPE_FLOAT, 0);
#else
    comm_broadcast ( dist_source, point_count*max_degree, COMM_TYPE_FLOAT, 0);
    comm_broadcast ( indr_mtrx_host, point_count*max_degree, COMM_TYPE_INT, 0);
#endif
    
    assert( max_degree > 0 );

    init(op);

    calculate_participants(point_count, node_count, cwrank, &thread_block_count, &total_thread_block_count, &active_node_count, &node_offset);

    // allocate host and device memory
    // If we don't have coordinates but we're working with BLAST data, we don't need
    // to allocate "source", because "read_BLAST_data()" will do that for us.
#if defined(N_SQUARE)
    // Allocate an N^2 array for the distance data.
    used_bytes = sizeof(float) * point_count * point_count;
#else
    // Allocate an N*Delta array for the distance data.
    used_bytes = sizeof(float) * point_count * max_degree;
#endif

    allocHostBuffer((void**)&ungrpd_pnts_indr_host, point_count*sizeof(int));
    for(int i=0; i<point_count; i++){
	ungrpd_pnts_indr_host[i] = i;
    }

    allocHostBuffer((void**)&cardinalities, thread_block_count*2*sizeof(int));
    allocHostBuffer((void**)&output, point_count*sizeof(int));

    allocDeviceBuffer(&degrees,             point_count*sizeof(int));
    allocDeviceBuffer(&ungrpd_pnts_indr,    point_count*sizeof(int));
    allocDeviceBuffer(&Ai_mask,             thread_block_count*point_count*sizeof(char));
    allocDeviceBuffer(&dist_to_clust,       thread_block_count*max_degree*sizeof(float));
    allocDeviceBuffer(&clustered_pnts_mask, point_count*sizeof(char));
    allocDeviceBuffer(&cardnl,              thread_block_count*2*sizeof(int));
    allocDeviceBuffer(&result,              point_count*sizeof(int));

#if defined(N_SQUARE)
    // Set texture parameters (default)
    texWork.addressMode[0] = cudaAddressModeClamp;
    texWork.addressMode[1] = cudaAddressModeClamp;
    texWork.filterMode = cudaFilterModePoint;
    texWork.normalized = false; // do not normalize coordinates
    // This is the N*N distance matrix
    cudaMallocArray(&work, &texWork.channelDesc, point_count, point_count);
#else
# if defined(USE_TEXTURES)
    texDenseDist.addressMode[0] = cudaAddressModeClamp;
    texDenseDist.addressMode[1] = cudaAddressModeClamp;
    texDenseDist.filterMode = cudaFilterModePoint;
    texDenseDist.normalized = false; // do not normalize coordinates
    // This is the N*D distance matrix
    cudaMallocArray(&dense_dist_matrix, &texDenseDist.channelDesc, max_degree, point_count);
# else
    allocDeviceBuffer(&dense_dist_matrix, used_bytes);
# endif
#endif
    CHECK_CUDA_ERROR();

    // This is the N*Delta indirection matrix
    allocDeviceBuffer(&indr_mtrx, point_count*max_degree*sizeof(int));

    // Copy to device, and record transfer time
    int pcie_TH = Timer::Start();
#if defined(N_SQUARE)
    cudaMemcpyToArray(work, 0, 0, source, used_bytes, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
    cudaBindTextureToArray(texWork, work);
#else
# if defined(USE_TEXTURES)
    cudaMemcpyToArray(dense_dist_matrix, 0, 0, dist_source, used_bytes, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
    cudaBindTextureToArray(texDenseDist, dense_dist_matrix);
# else
    copyToDevice(dense_dist_matrix, dist_source, used_bytes);
# endif
    copyToDevice(indr_mtrx, indr_mtrx_host, point_count*max_degree*sizeof(int));
#endif

    copyToDevice(ungrpd_pnts_indr, ungrpd_pnts_indr_host, point_count*sizeof(int));
    cudaMemset(clustered_pnts_mask, 0, point_count*sizeof(char));
    cudaMemset(dist_to_clust, 0, max_degree*thread_block_count*sizeof(float));
    double transfer_time = Timer::Stop(pcie_TH, "PCIe Transfer Time");

    tpb = ( point_count > THREADSPERBLOCK )? THREADSPERBLOCK : point_count;
#if defined(N_SQUARE)
    populate_dense_indirection_matrix<<<grid2D(thread_block_count), tpb>>>((int *)indr_mtrx, (float *)work, (int *)degrees, threshold, point_count, max_degree);
#else
    // In this case "populate" is a misnomer, the kernel merely counts the degree of each seed.
    populate_dense_indirection_matrix<<<grid2D(thread_block_count), tpb>>>((int *)indr_mtrx, (int *)degrees, point_count, max_degree);
#endif
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR();

    const char *sizeStr;
    stringstream ss;
    ss << "PointCount=" << (long)point_count;
    sizeStr = strdup(ss.str().c_str());

    if( save_clusters ){
        debug_out.open("p");
        for(i=0; i<point_count; i++){
            debug_out << pnts[2*i] << " " << pnts[2*i+1] << endl;
        }
        debug_out.close();
        seeds_out.open("p_seeds");
    }

    if( 0 == cwrank ){
        cout << "\nInitial ThreadBlockCount: " << thread_block_count;
        cout << " PointCount: " << point_count;
        cout << " Max degree: " << max_degree << "\n" << endl;
        cout.flush();
    }

    max_point_count = point_count;

    tpb = THREADSPERBLOCK;

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Kernel execution

    int TH = Timer::Start();
    do{
        stringstream ss;
        int winner_node=0;
        int winner_index=-1;
        bool this_node_participates = true;
        ++iter;

        calculate_participants(point_count, node_count, cwrank, &thread_block_count, &total_thread_block_count, &active_node_count, &node_offset);

        if( cwrank >= active_node_count ){
            this_node_participates = false;
        }

        comm_update_communicator(cwrank, active_node_count);

        if( !this_node_participates )
            break;

        max_card = 0;

        int Tkernel = Timer::Start();
        dim3 grid = grid2D(thread_block_count);

        //////////////////////////////////////////////////////////////////////////////////////
        ////////// ---------        Entry point to the main kernel        --------- //////////
#if defined(N_SQUARE)
        QTC_device<<<grid, tpb>>>( (float*)work, (char *)Ai_mask, (char *)clustered_pnts_mask,
                                   (int *)indr_mtrx, (int *)cardnl, (int *)ungrpd_pnts_indr, (float *)dist_to_clust,
                                   (int *)degrees, point_count, max_point_count, max_degree, threshold, cwrank, node_offset,
                                   total_thread_block_count);
#else
        QTC_device<<<grid, tpb>>>( (float*)dense_dist_matrix, (char *)Ai_mask, (char *)clustered_pnts_mask,
                                   (int *)indr_mtrx, (int *)cardnl, (int *)ungrpd_pnts_indr, (float *)dist_to_clust,
                                   (int *)degrees, point_count, max_point_count, max_degree, threshold, cwrank, node_offset,
                                   total_thread_block_count);
#endif

        cudaThreadSynchronize();
        CHECK_CUDA_ERROR();
        t_krn += Timer::Stop(Tkernel, "Kernel Only");

        int Tredc = Timer::Start();
        if( thread_block_count > 1 ){
            // We are reducing 128 numbers or less, so one thread should be sufficient.
            reduce_card_device<<<grid2D(1), 1>>>((int *)cardnl, thread_block_count);
            cudaThreadSynchronize();
            CHECK_CUDA_ERROR();
        }

        copyFromDevice( cardinalities, cardnl, 2*sizeof(int) );
        max_card     = cardinalities[0];
        winner_index = cardinalities[1];
        t_redc += Timer::Stop(Tredc, "Reduce Only");

        int Tcomm = Timer::Start();
        comm_find_winner(&max_card, &winner_node, &winner_index, cwrank, node_count);
        t_comm += Timer::Stop(Tcomm, "Comm Only");

        if( be_verbose && cwrank == winner_node){ // for non-parallel cases, both "cwrank" and "winner_node" should be zero.
            cout << "[" << cwrank << "] Cluster Cardinality: " << max_card << " (Node: " << cwrank << ", index: " << winner_index << ")" << endl;
        }

        int Ttrim = Timer::Start();
#if defined(N_SQUARE)
        trim_ungrouped_pnts_indr_array<<<grid2D(1), tpb>>>(winner_index, (int*)ungrpd_pnts_indr, (float*)work,
                                          (int *)result, (char *)Ai_mask, (char *)clustered_pnts_mask,
                                          (int *)indr_mtrx, (int *)cardnl, (float *)dist_to_clust, (int *)degrees,
                                          point_count, max_point_count, max_degree, threshold );
#else
        trim_ungrouped_pnts_indr_array<<<grid2D(1), tpb>>>(winner_index, (int*)ungrpd_pnts_indr, (float*)dense_dist_matrix,
                                          (int *)result, (char *)Ai_mask, (char *)clustered_pnts_mask,
                                          (int *)indr_mtrx, (int *)cardnl, (float *)dist_to_clust, (int *)degrees,
                                          point_count, max_point_count, max_degree, threshold );
#endif
        cudaThreadSynchronize();
        CHECK_CUDA_ERROR();
        t_trim += Timer::Stop(Ttrim, "Trim Only");

        if( cwrank == winner_node){ // for non-parallel cases, these should both be zero.
            if( save_clusters ){
                ss << "p." << iter;
                debug_out.open(ss.str().c_str());
            }

            copyFromDevice(output, (void *)result, max_card*sizeof(int) );

            if( save_clusters ){
                for(int i=0; i<max_card; i++){
                    debug_out << pnts[2*output[i]] << " " << pnts[2*output[i]+1] << endl;
                }
                seeds_out << pnts[2*winner_index] << " " << pnts[2*winner_index+1] << endl;
                debug_out.close();
            }
        }
 
        int Tupdt = Timer::Start();
        update_clustered_pnts_mask<<<grid2D(1), tpb>>>((char *)clustered_pnts_mask, (char *)Ai_mask, max_point_count);
        cudaThreadSynchronize();
        CHECK_CUDA_ERROR();
        t_updt += Timer::Stop(Tupdt, "Update Only");

        point_count -= max_card;

    }while( max_card > 1 && point_count );

    double t = Timer::Stop(TH, "QT_Clustering");

    if( save_clusters ){
        seeds_out.close();
    }
    //
    ////////////////////////////////////////////////////////////////////////////////
    
    if( cwrank == 0){
        cout << "Cluster count: " << iter << endl;
        cout.flush();
    }

    resultDB.AddResult(name+"_comm", sizeStr, "Time", t_comm);
    resultDB.AddResult(name+"_krnl", sizeStr, "Time", t_krn);
    resultDB.AddResult(name+"_trim", sizeStr, "Time", t_trim);
    resultDB.AddResult(name+"_updt", sizeStr, "Time", t_updt);
    resultDB.AddResult(name+"_redc", sizeStr, "Time", t_redc);
    resultDB.AddResult(name, sizeStr, "Time", t);
    resultDB.AddResult(name+"_PCIe", sizeStr, "Time", t+transfer_time);

#if defined(N_SQUARE)
    cudaFreeArray(work);
    freeHostBuffer(source);
    cudaUnbindTexture(texWork);
#else
    freeHostBuffer(dist_source);
    freeHostBuffer(indr_mtrx_host);
# if defined(USE_TEXTURES)
    cudaFreeArray(dense_dist_matrix);
    cudaUnbindTexture(texDenseDist);
# else
    freeDeviceBuffer(dense_dist_matrix);
# endif
#endif
    CHECK_CUDA_ERROR();

    freeDeviceBuffer(indr_mtrx);
    freeDeviceBuffer(Ai_mask);
    freeDeviceBuffer(cardnl);
    freeDeviceBuffer(result);
    freeHostBuffer(output);

    return;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int qtcDevice = -1;

void
init(OptionParser& op)
{
    if (qtcDevice == -1) {
        if (op.getOptionVecInt("device").size() > 0) {
            qtcDevice = op.getOptionVecInt("device")[0];
        }
        else {
            qtcDevice = 0;
        }
        cudaSetDevice(qtcDevice);
        cudaGetDevice(&qtcDevice);
    }
}


void
allocHostBuffer(void** bufferp, unsigned long bytes)
{
    cudaMallocHost(bufferp, bytes);
    CHECK_CUDA_ERROR();
}

void
allocDeviceBuffer(void** bufferp, unsigned long bytes)
{
    cudaMalloc(bufferp, bytes);
    CHECK_CUDA_ERROR();
}

void
freeHostBuffer(void* buffer)
{
    cudaFreeHost(buffer);
    CHECK_CUDA_ERROR();
}


void
freeDeviceBuffer(void* buffer)
{
    cudaFree(buffer);
}

void
copyToDevice(void* to_device, void* from_host, unsigned long bytes)
{
    cudaMemcpy(to_device, from_host, bytes, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
}


void
copyFromDevice(void* to_host, void* from_device, unsigned long bytes)
{
    cudaMemcpy(to_host, from_device, bytes, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR();
}

