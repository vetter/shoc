#ifndef _KERNELS_COMMON_H_
#define _KERNELS_COMMON_H_

#pragma once
#pragma warning(disable:4996)

#define _USE_MATH_DEFINES
#include <math.h>
//#include <float.h> // float.h should be included by math.h and we need it for FLT_MAX

#include "tuningParameters.h"
#include "qtc_common.h"

// Forward declarations
__global__ void QTC_device( float *dist_matrix, char *Ai_mask, char *clustered_pnts_mask, int *indr_mtrx, int *cluster_cardinalities, int *ungrpd_pnts_indr, float *dist_to_clust, int *degrees, int point_count, int N0, int max_degree, float threshold, int cwrank, int node_rank, int node_count, int total_thread_block_count, bool can_use_texture);

__device__ int generate_candidate_cluster_compact_storage(int seed_point, int degree, char *Ai_mask, float *compact_storage_dist_matrix, char *clustered_pnts_mask, int *indr_mtrx, float *dist_to_clust, int point_count, int N0, int max_degree, int *candidate_cluster, float threshold, bool can_use_texture);

__device__ int generate_candidate_cluster_full_storage(int seed_point, int degree, char *Ai_mask, float *work, char *clustered_pnts_mask, int *indr_mtrx, float *dist_to_clust, int pointCount, int N0, int max_degree, int *candidate_cluster, float threshold, bool can_use_texture);

__device__ int find_closest_point_to_cluster(int seed_point, int latest_point, char *Ai_mask, char *clustered_pnts_mask, float *work, int *indr_mtrx, float *dist_to_clust, int pointCount, int N0, int max_degree, float threshold);

void QTC(const string& name, ResultDatabase &resultDB, OptionParser& op, int matrix_type);


//
// arrange blocks into 2D grid that fits into the GPU ( for powers of two only )
//
inline dim3 grid2D( int nblocks )
{
    int slices = 1;

    if( nblocks < 1 )
        return dim3(1,1);

    while( nblocks/slices > 65535 )
        slices *= 2;
    return dim3( nblocks/slices, slices );
}


inline __device__
int closest_point_reduction(float min_dist, float threshold, int closest_point){
    __shared__ float dist_array[THREADSPERBLOCK];
    __shared__ int point_index_array[THREADSPERBLOCK];

    int tid = threadIdx.x;
    int curThreadCount = blockDim.x*blockDim.y*blockDim.z;

    dist_array[tid] = min_dist;
    point_index_array[tid] = closest_point;

    __syncthreads();

    if(tid == 0 ){
        for(int j=1; j<curThreadCount; j++){
            float dist = dist_array[j];
            // look for a point that is closer, or equally far, but with a smaller index.
            if( (dist < min_dist) || (dist == min_dist && point_index_array[j] < point_index_array[0]) ){
                min_dist = dist;
                point_index_array[0] = point_index_array[j];
            }
        }
        if( min_dist > threshold )
            point_index_array[0] = -1;
    }

    __syncthreads();

    return point_index_array[0];
}


__global__
void reduce_card_device(int *cardinalities, int TB_count){
    int i, max_card = -1, winner_index;

    for(i=0; i<TB_count*2; i+=2){
        if( cardinalities[i] > max_card ){
            max_card = cardinalities[i];
            winner_index = cardinalities[i+1];
        }
    }

    cardinalities[0] = max_card;
    cardinalities[1] = winner_index;

    return;
}




__global__ void
compute_degrees(int *indr_mtrx, int *degrees, int N0, int max_degree){
    int tid, tblock_id, TB_count, offset;
    int local_point_count, curThreadCount;
    int starting_point;

    curThreadCount = blockDim.x*blockDim.y*blockDim.z;
    tid = threadIdx.x;
    tblock_id = (blockIdx.y * gridDim.x + blockIdx.x);
    TB_count = gridDim.y * gridDim.x;
    local_point_count = (N0+TB_count-1)/TB_count;
    starting_point = tblock_id * local_point_count;
    offset =  starting_point*max_degree;
    indr_mtrx = &indr_mtrx[offset];
    degrees = &degrees[starting_point];

    // The last threadblock might end up with less points.
    if( (tblock_id+1)*local_point_count > N0 )
        local_point_count = MAX(0,N0-starting_point);

    for(int i=0; i+tid < local_point_count; i+=curThreadCount){
        int cnt = 0;
        for(int j=0; j < max_degree; j++){
            if( indr_mtrx[(i+tid)*max_degree+j] >= 0 ){
                ++cnt;
            }
        }
        degrees[i+tid] = cnt;
    }
}

__global__ void
update_clustered_pnts_mask(char *clustered_pnts_mask, char *Ai_mask, int N0 ) {
    int tid = threadIdx.x;
    int curThreadCount = blockDim.x*blockDim.y*blockDim.z;

    // If a point is part of the latest winner cluster, then it should be marked as
    // clustered for the future iterations. Otherwise it should be left as it is.
    for(int i = 0; i+tid < N0; i+=curThreadCount){
        clustered_pnts_mask[i+tid] |= Ai_mask[i+tid];
    }
    __syncthreads();
    return;
}


__global__ void
trim_ungrouped_pnts_indr_array(int seed_index, int *ungrpd_pnts_indr, float *dist_matrix, int *result_cluster, char *Ai_mask, char *clustered_pnts_mask, int *indr_mtrx, int *cluster_cardinalities, float *dist_to_clust, int *degrees, int point_count, int N0, int max_degree, float threshold, int matrix_type_mask, bool can_use_texture) {
    int cnt;
    int tid = threadIdx.x;
    int curThreadCount = blockDim.x*blockDim.y*blockDim.z;
    __shared__ int tmp_pnts[THREADSPERBLOCK];

    int degree = degrees[seed_index];
    if( matrix_type_mask & FULL_STORAGE_MATRIX ){
        (void)generate_candidate_cluster_full_storage(seed_index, degree, Ai_mask, dist_matrix, clustered_pnts_mask, indr_mtrx,
                                               dist_to_clust, point_count, N0, max_degree, result_cluster, threshold, can_use_texture);
    }else{
        (void)generate_candidate_cluster_compact_storage(seed_index, degree, Ai_mask, dist_matrix, clustered_pnts_mask, indr_mtrx,
                                               dist_to_clust, point_count, N0, max_degree, result_cluster, threshold, can_use_texture);
    }

    __shared__ int cnt_sh;
    __shared__ bool flag_sh;

    if( 0 == tid ){
        cnt_sh = 0;
        flag_sh = false;
    }
    __syncthreads();

    for(int i = 0; i+tid < point_count; i+=curThreadCount){
        // Have all threads make a coalesced read of contiguous global memory and copy the points assuming they are all good.
        tmp_pnts[tid] = ungrpd_pnts_indr[i+tid];
        int pnt = tmp_pnts[tid];
        // If a point is bad (which should not happen very often), raise a global flag so that thread zero fixes the problem.
        if( 1 == Ai_mask[pnt] ){
            flag_sh = true;
            tmp_pnts[tid] = INVALID_POINT_MARKER;
        }else{
            ungrpd_pnts_indr[cnt_sh+tid] = pnt;
        }

        __syncthreads();

        if( 0 == tid ){
            if( flag_sh ){
                cnt = cnt_sh;
                for(int j = 0; (j < curThreadCount) && (i+j < point_count); j++ ){
                    if( INVALID_POINT_MARKER != tmp_pnts[j] ){
                        ungrpd_pnts_indr[cnt] = tmp_pnts[j];
                        cnt++;
                    }
                }
                cnt_sh = cnt;
            }else{
                cnt_sh += curThreadCount;
            }
            flag_sh  = false;
        }

        __syncthreads();

    }

    return;
}


__global__
void QTC_device( float *dist_matrix, char *Ai_mask, char *clustered_pnts_mask, int *indr_mtrx, int *cluster_cardinalities, int *ungrpd_pnts_indr, float *dist_to_clust, int *degrees, int point_count, int N0, int max_degree, float threshold, int node_rank, int node_count, int total_thread_block_count, int matrix_type_mask, bool can_use_texture) {
    int max_cardinality = -1;
    int max_cardinality_index;
    int i, tblock_id, tid, base_offset;

    tid = threadIdx.x;
    tblock_id = (blockIdx.y * gridDim.x + blockIdx.x);
    Ai_mask = &Ai_mask[tblock_id * N0];
    dist_to_clust = &dist_to_clust[tblock_id * max_degree];
    //base_offset = node_offset+tblock_id;
    base_offset = tblock_id*node_count + node_rank;

    // for i loop of the algorithm.
    // Each thread iterates over all points that the whole thread-block owns
    for(i = base_offset; i < point_count; i+= total_thread_block_count ){
        int cnt;
        int seed_index = ungrpd_pnts_indr[i];
        int degree = degrees[seed_index];
        if( degree <= max_cardinality ) continue;
        if( matrix_type_mask & FULL_STORAGE_MATRIX ){
            cnt = generate_candidate_cluster_full_storage(seed_index, degree, Ai_mask, dist_matrix,
                                                    clustered_pnts_mask, indr_mtrx, dist_to_clust,
                                                    point_count, N0, max_degree, NULL, threshold,
                                                    can_use_texture);
        }else{
            cnt = generate_candidate_cluster_compact_storage( seed_index, degree, Ai_mask, dist_matrix,
                                                    clustered_pnts_mask, indr_mtrx, dist_to_clust,
                                                    point_count, N0, max_degree, NULL, threshold,
                                                    can_use_texture);
        }
        if( cnt > max_cardinality ){
            max_cardinality = cnt;
            max_cardinality_index = seed_index;
        }
    } // for (i

    // since only three elements per block go to the global memory, the offset is:
    int card_offset = (blockIdx.y * gridDim.x + blockIdx.x)*2;
    // only one thread needs to write into the global memory since they all have the same information.
    if( 0 == tid ){
        cluster_cardinalities[card_offset] = max_cardinality;
        cluster_cardinalities[card_offset+1] = max_cardinality_index;
    }

    return;
}

#endif
