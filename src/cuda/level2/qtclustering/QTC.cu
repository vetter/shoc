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

#include "PMSMemMgmt.h"

#include "comm.h"

texture<float, 2, cudaReadModeElementType> texDistance;

using namespace std;

#include "kernels_common.h"
#include "kernels_full_storage.h"
#include "kernels_compact_storage.h"

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
    op.addOption("PointCount", OPT_INT, "4096", "point count");
    op.addOption("DataFile", OPT_STRING, "///", "BLAST data input file name");
    op.addOption("Threshold", OPT_FLOAT, "1", "cluster diameter threshold");
    op.addOption("SaveOutput", OPT_BOOL, "", "BLAST data input file name");
    op.addOption("Verbose", OPT_BOOL, "", "Print cluster cardinalities");
    op.addOption("TextureMem", OPT_BOOL, "0", "Use Texture memory for distance matrix");
    op.addOption("CompactStorage", OPT_BOOL, "0", "Use compact storage distance matrix regardless of problem size");

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

    runTest("QTC", resultDB, op);
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
void calculate_participants(int point_count, int node_count, int cwrank, int *thread_block_count, int *total_thread_block_count, int *active_node_count){

    int ac_nd_cnt, thr_blc_cnt, total_thr_blc_cnt;

    ac_nd_cnt = node_count;
    if( point_count <= (node_count-1) * SM_COUNT * GPU_MIN_SATURATION_FACTOR ){
        int K = SM_COUNT * GPU_MIN_SATURATION_FACTOR;
        ac_nd_cnt = (point_count+K-1) / K;
    }

    if( point_count >= ac_nd_cnt * SM_COUNT * OVR_SBSCR_FACTOR ){
        thr_blc_cnt = SM_COUNT * OVR_SBSCR_FACTOR;
        total_thr_blc_cnt = thr_blc_cnt * ac_nd_cnt;
    }else{
        thr_blc_cnt = point_count/ac_nd_cnt;
        if( cwrank < point_count%ac_nd_cnt ){
            thr_blc_cnt++;
        }
        total_thr_blc_cnt = point_count;
    }

    *active_node_count  = ac_nd_cnt;
    *thread_block_count = thr_blc_cnt;
    *total_thread_block_count = total_thr_blc_cnt;

    return;
}

unsigned long int estimate_memory_for_full_storage(unsigned long int pnt_cnt, float d){
    unsigned long total, thread_block_count, max_degree;
    float density;

    thread_block_count = (unsigned long int)SM_COUNT * OVR_SBSCR_FACTOR;

    // The density calculations assume that we are dealing with generated Euclidean points
    // (as opposed to externally provided scientific data) that are constraint in a 20x20 2D square.
    density = 3.14159*(d*d)/(20.0*20.0);
    if(density > 1.0 ) density = 1.0;
    max_degree = (unsigned long int)((float)pnt_cnt*density); // average number of points in a cirlce with radius d.
    max_degree *= 10; // The distribution of points is not uniform, so throw in a factor of 10 for max/average.
    if( max_degree > pnt_cnt )
        max_degree = pnt_cnt;
    // Due to the point generation algorithm, a cluster can have up to N/30 elements in an arbitratiry small radius.
    if( max_degree < pnt_cnt/30 )
        max_degree = pnt_cnt/30;

    total = 0;
    total += pnt_cnt*pnt_cnt*sizeof(float); // Sparse distance matrix
    total += pnt_cnt*max_degree*sizeof(int); // Indirection matrix
    total += pnt_cnt*thread_block_count*sizeof(char); // Current candidate cluster mask
    total += pnt_cnt*sizeof(int); // Ungrouped elements indirection vector
    total += pnt_cnt*sizeof(int); // Degrees vector
    total += pnt_cnt*sizeof(int); // Result

    return total;
}

void findMemCharacteristics(unsigned long int *gmem, unsigned long int *text){
    int device;
    cudaDeviceProp deviceProp;

    cudaGetDevice(&device);
    CHECK_CUDA_ERROR();

    cudaGetDeviceProperties(&deviceProp, device);
    CHECK_CUDA_ERROR();

    *gmem = (unsigned long int)(0.75*(float)deviceProp.totalGlobalMem);
    *text = (unsigned long int)deviceProp.maxTexture2D[1];

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
    unsigned long int point_count, max_avail_memory, max_texture_dimension, needed_mem;
    int def_size = -1, matrix_type = 0x0;
    float threshold;
    bool use_texture = true, use_compact_storage = false;

    def_size    = op.getOptionInt("size");
    point_count = op.getOptionInt("PointCount");
    threshold   = op.getOptionFloat("Threshold");
    use_texture = op.getOptionBool("TextureMem");
    use_compact_storage = op.getOptionBool("CompactStorage");
    if( use_compact_storage ){
        use_texture = false;
    }

    switch( def_size ){
        case 1:
            // size == 1 should match default values of PointCount,
            // Threshold, TextureMem, and CompactStorage parameters.
            // (i.e., -s 1 is the default)
            point_count = 4*1024;
            threshold   = 1;
            use_texture = false;
            use_compact_storage = false;
            break;
        case 2:
            point_count = 8*1024;
            threshold   = 1;
            use_texture = true;
            use_compact_storage = false;
            break;
        case 3:
            point_count = 16*1024;
            threshold   = 1;
            use_texture = true;
            use_compact_storage = false;
            break;
        case 4:
            point_count = 16*1024;
            threshold   = 4;
            use_texture = true;
            use_compact_storage = false;
            break;
        case 5:
            point_count = 26*1024;
            threshold   = 1;
            use_texture = false;
            use_compact_storage = true;
            break;
        default:
            fprintf( stderr, "unsupported size %d given; terminating\n", def_size );
            return;
    }

    if( 0 == comm_get_rank() ){
        // Make a reasonable estimate of the actual memory I can allocate
        // as well as the max texture size.
        findMemCharacteristics(&max_avail_memory, &max_texture_dimension);

        needed_mem = estimate_memory_for_full_storage(point_count, threshold);

        // see if we can fit the distance matrix in texture memory
        if( (point_count >= max_texture_dimension) || !use_texture ){
            printf("Using global memory for distance matrix\n");
            matrix_type |= GLOBAL_MEMORY;
        }else{
            printf("Using texture memory for distance matrix\n");
            matrix_type |= TEXTUR_MEMORY;
        }

        // find out what type of distance matrix we will be using.
        if( (max_avail_memory > needed_mem) && !use_compact_storage ){
            printf("Using full storage distance matrix algorithm\n");
            matrix_type |= FULL_STORAGE_MATRIX;
        }else{
            printf("Using compact storage distance matrix algorithm\n");
            matrix_type |= COMPACT_STORAGE_MATRIX;
        }
    }
    comm_broadcast ( &matrix_type, 1, COMM_TYPE_INT, 0);

    QTC(name, resultDB, op, matrix_type);

}

////////////////////////////////////////////////////////////////////////////////
//
void QTC(const string& name, ResultDatabase &resultDB, OptionParser& op, int matrix_type){
    ofstream debug_out, seeds_out;
    void *Ai_mask, *cardnl, *ungrpd_pnts_indr, *clustered_pnts_mask, *result, *dist_to_clust;
    void *indr_mtrx, *degrees;
    int *indr_mtrx_host, *ungrpd_pnts_indr_host, *cardinalities, *output;
    bool save_clusters = false;
    bool be_verbose = false;
    bool synthetic_data = true;
    cudaArray *distance_matrix_txt;
    void *distance_matrix_gmem, *distance_matrix;
    float *dist_source, *pnts;
    float threshold;
    int i, max_degree, thread_block_count, total_thread_block_count, active_node_count;
    int cwrank=0, node_count=1, tpb, max_card, iter=0;
    double t_krn=0, t_comm=0, t_trim=0, t_updt=0, t_redc=0, t_sync=0;
    unsigned long int dst_matrix_elems, point_count, max_point_count;
    string fname;

    point_count = op.getOptionInt("PointCount");
    threshold = op.getOptionFloat("Threshold");
    save_clusters = op.getOptionBool("SaveOutput");
    be_verbose = op.getOptionBool("Verbose");
    fname = op.getOptionString("DataFile");
    if( fname.compare("///") == 0 ){
        synthetic_data = true;
    }else{
        synthetic_data = false;
        save_clusters = false;
    }

    bool can_use_texture = !!(matrix_type & TEXTUR_MEMORY);

    // TODO - only deal with this size-switch once
    int def_size = op.getOptionInt("size");
    switch( def_size ) {
        case 1:
            // size == 1 should match default values of PointCount,
            // Threshold, TextureMem, and CompactStorage parameters.
            // (i.e., -s 1 is the default)
            point_count     = 4*1024;
            threshold       = 1;
            break;
        case 2:
            point_count    = 8*1024;
            threshold      = 1;
            break;
        case 3:
            point_count    = 16*1024;
            threshold      = 1;
            break;
        case 4:
            point_count    = 16*1024;
            threshold      = 4;
            break;
        case 5:
            point_count    = 26*1024;
            threshold      = 1;
            break;
        default:
            fprintf( stderr, "unsupported size %d given; terminating\n", def_size );
            return;
    }

    cwrank = comm_get_rank();
    node_count = comm_get_size();

    if( cwrank == 0 ){
        if( synthetic_data )
            pnts = generate_synthetic_data(&dist_source, &indr_mtrx_host, &max_degree, threshold, point_count, matrix_type);
        else
            (void)read_BLAST_data(&dist_source, &indr_mtrx_host, &max_degree, threshold, fname.c_str(), point_count, matrix_type);
    }

    comm_broadcast ( &point_count, 1, COMM_TYPE_INT, 0);
    comm_broadcast ( &max_degree, 1, COMM_TYPE_INT, 0);

    if( matrix_type & FULL_STORAGE_MATRIX ){
        dst_matrix_elems = point_count*point_count;
    }else{
        dst_matrix_elems = point_count*max_degree;
    }

    if( cwrank != 0 ){ // For all nodes except zero, in a distributed run.
        dist_source = pmsAllocHostBuffer<float>( dst_matrix_elems );
        indr_mtrx_host = pmsAllocHostBuffer<int>( point_count*max_degree );
    }
    // If we need to print the actual clusters later on, we'll need to have all points in all nodes.
    if( save_clusters ){
        if( cwrank != 0 ){
            pnts = (float *)malloc( 2*point_count*sizeof(float) );
        }
        comm_broadcast ( pnts, 2*point_count, COMM_TYPE_FLOAT, 0);
    }

    comm_broadcast ( dist_source, dst_matrix_elems, COMM_TYPE_FLOAT, 0);
    comm_broadcast ( indr_mtrx_host, point_count*max_degree, COMM_TYPE_INT, 0);

    assert( max_degree > 0 );

    init(op);

    calculate_participants(point_count, node_count, cwrank, &thread_block_count, &total_thread_block_count, &active_node_count);

    ungrpd_pnts_indr_host = pmsAllocHostBuffer<int>( point_count );
    for(int i=0; i<point_count; i++){
	ungrpd_pnts_indr_host[i] = i;
    }

    cardinalities = pmsAllocHostBuffer<int>(2);
    output = pmsAllocHostBuffer<int>(max_degree);

    if( can_use_texture ){
        texDistance.addressMode[0] = cudaAddressModeClamp;
        texDistance.addressMode[1] = cudaAddressModeClamp;
        texDistance.filterMode = cudaFilterModePoint;
        texDistance.normalized = false; // do not normalize coordinates
        // This is the actual distance matrix (dst_matrix_elems should be "point_count^2, or point_count*max_degree)
        printf("Allocating: %luMB (%lux%lux%lu) bytes in texture memory\n", dst_matrix_elems*sizeof(float)/(1024*1024),
                                                                        dst_matrix_elems/point_count, point_count, (long unsigned int)sizeof(float));
        cudaMallocArray(&distance_matrix_txt, &texDistance.channelDesc, dst_matrix_elems/point_count, point_count);
    }else{
        allocDeviceBuffer(&distance_matrix_gmem, dst_matrix_elems*sizeof(float));
    }
    CHECK_CUDA_ERROR();

    // This is the N*Delta indirection matrix
    allocDeviceBuffer(&indr_mtrx, point_count*max_degree*sizeof(int));

    allocDeviceBuffer(&degrees,             point_count*sizeof(int));
    allocDeviceBuffer(&ungrpd_pnts_indr,    point_count*sizeof(int));
    allocDeviceBuffer(&Ai_mask,             thread_block_count*point_count*sizeof(char));
    allocDeviceBuffer(&dist_to_clust,       thread_block_count*max_degree*sizeof(float));
    allocDeviceBuffer(&clustered_pnts_mask, point_count*sizeof(char));
    allocDeviceBuffer(&cardnl,              thread_block_count*2*sizeof(int));
    allocDeviceBuffer(&result,              point_count*sizeof(int));

    // Copy to device, and record transfer time
    int pcie_TH = Timer::Start();

    if( can_use_texture ){
        cudaMemcpyToArray(distance_matrix_txt, 0, 0, dist_source, dst_matrix_elems*sizeof(float), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR();
        cudaBindTextureToArray(texDistance, distance_matrix_txt);
    }else{
        copyToDevice(distance_matrix_gmem, dist_source, dst_matrix_elems*sizeof(float));
    }

    copyToDevice(indr_mtrx, indr_mtrx_host, point_count*max_degree*sizeof(int));

    copyToDevice(ungrpd_pnts_indr, ungrpd_pnts_indr_host, point_count*sizeof(int));
    cudaMemset(clustered_pnts_mask, 0, point_count*sizeof(char));
    cudaMemset(dist_to_clust, 0, max_degree*thread_block_count*sizeof(float));
    double transfer_time = Timer::Stop(pcie_TH, "PCIe Transfer Time");

    tpb = ( point_count > THREADSPERBLOCK )? THREADSPERBLOCK : point_count;
    compute_degrees<<<grid2D(thread_block_count), tpb>>>((int *)indr_mtrx, (int *)degrees, point_count, max_degree);
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR();

    const char *sizeStr;
    stringstream ss;
    ss << "PointCount=" << (long)point_count;
    sizeStr = strdup(ss.str().c_str());

    if( 0 == cwrank ){
        if( save_clusters ){
            debug_out.open("p");
            for(i=0; i<point_count; i++){
                debug_out << pnts[2*i] << " " << pnts[2*i+1] << endl;
            }
            debug_out.close();
            seeds_out.open("p_seeds");
        }

        cout << "\nInitial ThreadBlockCount: " << thread_block_count;
        cout << " PointCount: " << point_count;
        cout << " Max degree: " << max_degree << "\n" << endl;
        cout.flush();
    }

    max_point_count = point_count;

    tpb = THREADSPERBLOCK;

    if( can_use_texture ){
        distance_matrix = distance_matrix_txt;
    }else{
        distance_matrix = distance_matrix_gmem;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Kernel execution

    int TH = Timer::Start();
    do{
        stringstream ss;
        int winner_node=-1;
        int winner_index=-1;
        bool this_node_participates = true;

        ++iter;

        calculate_participants(point_count, node_count, cwrank, &thread_block_count, &total_thread_block_count, &active_node_count);

        // If there are only a few elements left to cluster, reduce the number of participating nodes (GPUs).
        if( cwrank >= active_node_count ){
            this_node_participates = false;
        }
        comm_update_communicator(cwrank, active_node_count);
        if( !this_node_participates )
            break;
        cwrank = comm_get_rank();

        dim3 grid = grid2D(thread_block_count);

        int Tkernel = Timer::Start();
        ////////////////////////////////////////////////////////////////////////////////////////////////
        ///////// -----------------               Main kernel                ----------------- /////////
        QTC_device<<<grid, tpb>>>((float*)distance_matrix, (char *)Ai_mask, (char *)clustered_pnts_mask,
                                  (int *)indr_mtrx, (int *)cardnl, (int *)ungrpd_pnts_indr,
                                  (float *)dist_to_clust, (int *)degrees, point_count, max_point_count,
                                  max_degree, threshold, cwrank, active_node_count,
                                  total_thread_block_count, matrix_type, can_use_texture);
        ///////// -----------------               Main kernel                ----------------- /////////
        ////////////////////////////////////////////////////////////////////////////////////////////////
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

        int Tsync = Timer::Start();
        comm_barrier();
        t_sync += Timer::Stop(Tsync, "Sync Only");

        int Tcomm = Timer::Start();
        comm_find_winner(&max_card, &winner_node, &winner_index, cwrank, max_point_count+1);
        t_comm += Timer::Stop(Tcomm, "Comm Only");

        if( be_verbose && cwrank == winner_node){ // for non-parallel cases, both "cwrank" and "winner_node" should be zero.
            cout << "[" << cwrank << "] Cluster Cardinality: " << max_card << " (Node: " << cwrank << ", index: " << winner_index << ")" << endl;
        }

        int Ttrim = Timer::Start();
        trim_ungrouped_pnts_indr_array<<<grid2D(1), tpb>>>(winner_index, (int*)ungrpd_pnts_indr, (float*)distance_matrix,
                                          (int *)result, (char *)Ai_mask, (char *)clustered_pnts_mask,
                                          (int *)indr_mtrx, (int *)cardnl, (float *)dist_to_clust, (int *)degrees,
                                          point_count, max_point_count, max_degree, threshold, matrix_type, can_use_texture );
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

    resultDB.AddResult(name+"_Synchron.", sizeStr, "s", t_sync);
    resultDB.AddResult(name+"_Communic.", sizeStr, "s", t_comm);
    resultDB.AddResult(name+"_Kernel", sizeStr, "s", t_krn);
    resultDB.AddResult(name+"_Trimming", sizeStr, "s", t_trim);
    resultDB.AddResult(name+"_Update", sizeStr, "s", t_updt);
    resultDB.AddResult(name+"_Reduction", sizeStr, "s", t_redc);
    resultDB.AddResult(name+"_Algorithm", sizeStr, "s", t);
    resultDB.AddResult(name+"+PCI_Trans.", sizeStr, "s", t+transfer_time);

    pmsFreeHostBuffer(dist_source);
    pmsFreeHostBuffer(indr_mtrx_host);
    if( can_use_texture ){
        cudaFreeArray(distance_matrix_txt);
        cudaUnbindTexture(texDistance);
    }else{
        freeDeviceBuffer(distance_matrix_gmem);
    }

    CHECK_CUDA_ERROR();

    freeDeviceBuffer(indr_mtrx);
    freeDeviceBuffer(Ai_mask);
    freeDeviceBuffer(cardnl);
    freeDeviceBuffer(result);
    pmsFreeHostBuffer(output);

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
allocDeviceBuffer(void** bufferp, unsigned long bytes)
{
    cudaMalloc(bufferp, bytes);
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

