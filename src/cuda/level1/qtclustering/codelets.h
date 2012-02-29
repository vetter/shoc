#ifndef _CODELETS_H_
#define _CODELETS_H_

#pragma once
#pragma warning(disable:4996)

#define _USE_MATH_DEFINES
#include <math.h>
//#include <float.h> // float.h should be included by math.h and we need it for FLT_MAX

#include "tuningParameters.h"

#ifdef MIN
# undef MIN
#endif
#define MIN(_X, _Y) ( ((_X) < (_Y)) ? (_X) : (_Y) )

#ifdef MAX
# undef MAX
#endif
#define MAX(_X, _Y) ( ((_X) > (_Y)) ? (_X) : (_Y) )

#define INVALID_POINT_MARKER -42

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
            if( dist < min_dist ){
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



#define COMPUTE_DIAMETER_WITH_POINT( _CAND_PNT_, _CURR_DIST_TO_CLUST_, _I_ ) \
    if( (_CAND_PNT_) < 0 ){\
        break;\
    }\
do{\
    int tmp_index = (_I_)*curThreadCount+tid;\
    if( (_CAND_PNT_) == seed_point ){\
        break;\
    }\
    _CURR_DIST_TO_CLUST_ = dist_to_clust[ tmp_index ];\
    /* if "_CAND_PNT_" is too far away, or already in Ai_mask, or in clustered_points, ignore it. */\
    if( (_CURR_DIST_TO_CLUST_ > threshold) || (0 != Ai_mask[(_CAND_PNT_)]) || (0 != clustered_pnts_mask[(_CAND_PNT_)]) ){ \
        _CAND_PNT_ = seed_point; /* This is so we don't do the lookup again. */\
        break;\
    }\
    dist_to_new_point = threshold+1;\
    /* Find _CAND_PNT_ in the neighborhood of the latest_point.*/\
    for(int j=last_index_checked; j<max_degree; j++){\
        int tmp_pnt = indr_mtrx[ latest_p_off + j ];\
        if( (tmp_pnt > (_CAND_PNT_)) || (tmp_pnt < 0) ){\
            last_index_checked = j;\
            break;\
        }\
        if( tmp_pnt == (_CAND_PNT_) ){\
            dist_to_new_point = dense_dist_matrix[ latest_p_off + j ];\
            break;\
        }\
    }\
\
    /* See if the distance of "_CAND_PNT_" to the "latest_point" is larger */\
    /* than the previous, cached distance of "_CAND_PNT_" to the cluster.  */\
    if(dist_to_new_point > _CURR_DIST_TO_CLUST_){\
        diameter = dist_to_new_point;\
        dist_to_clust[ tmp_index ] = diameter;\
    }else{\
        diameter = _CURR_DIST_TO_CLUST_;\
    }\
\
    /* The point that leads to the cluster with the smallest diameter is the closest point */\
    if( diameter < min_dist ){\
        min_dist = diameter;\
        point_index = (_CAND_PNT_);\
    }\
}while(0)



//#define CHECK_POINT( _CAND_PNT_ )\
//    if( (_CAND_PNT_) < 0 ){\
//        break;\
//    }\

#define FETCH_POINT( _CAND_PNT_ , _I_ )\
{\
    int tmp_index = (_I_)*curThreadCount+tid;\
    if( tmp_index >= max_degree ){\
        break;\
    }\
    _CAND_PNT_ = indr_mtrx[ seed_p_off + tmp_index ];\
    if( (_CAND_PNT_) < 0 ){\
        break;\
    }\
}

inline __device__
int generate_candidate_cluster2(int seed_point, int degree, char *Ai_mask, float *dense_dist_matrix, char *clustered_pnts_mask, int *indr_mtrx, float *dist_to_clust, int point_count, int N0, int max_degree, int *candidate_cluster, float threshold)
{

    bool flag;
    int cnt, latest_point;

    int curThreadCount = blockDim.x*blockDim.y*blockDim.z;
    int tid = threadIdx.x;
    int seed_p_off;

    float curr_dist_to_clust_i;
    float curr_dist_to_clust_0, curr_dist_to_clust_1, curr_dist_to_clust_2, curr_dist_to_clust_3;
    float curr_dist_to_clust_4, curr_dist_to_clust_5, curr_dist_to_clust_6, curr_dist_to_clust_7;
    float curr_dist_to_clust_8, curr_dist_to_clust_9, curr_dist_to_clust_10, curr_dist_to_clust_11;
    int cand_pnt_i=-1;
    int cand_pnt_0=-1, cand_pnt_1=-1, cand_pnt_2=-1, cand_pnt_3=-1;
    int cand_pnt_4=-1, cand_pnt_5=-1, cand_pnt_6=-1, cand_pnt_7=-1;
    int cand_pnt_8=-1, cand_pnt_9=-1, cand_pnt_10=-1, cand_pnt_11=-1;

    // Cleanup the candidate-cluster-mask, Ai_mask
    for(int i=0; i+tid < N0; i+=curThreadCount){
        Ai_mask[i+tid] = 0;
    }

    // Cleanup the "distance cache"
    for(int i=0; i+tid < max_degree; i+=curThreadCount){
        dist_to_clust[i+tid] = 0;
    }

    // Put the seed point in the candidate cluster and mark it as taken in the candidate cluster mask Ai_mask.
    flag = true;
    cnt = 1;
    if( 0 == tid ){
        if( NULL != candidate_cluster )
            candidate_cluster[0] = seed_point;
        Ai_mask[seed_point] = 1;
    }
    __syncthreads();
    seed_p_off = seed_point*max_degree;
    latest_point = seed_point;

    // Prefetch 12 points per thread, into registers, to reduce the memory pressure (and delay) of
    // constantly going to memory to fetch these points inside the while() loop that follows.
    do{
        FETCH_POINT(  cand_pnt_0,  0 );
        FETCH_POINT(  cand_pnt_1,  1 );
        FETCH_POINT(  cand_pnt_2,  2 );
        FETCH_POINT(  cand_pnt_3,  3 );
        FETCH_POINT(  cand_pnt_4,  4 );
        FETCH_POINT(  cand_pnt_5,  5 );
        FETCH_POINT(  cand_pnt_6,  6 );
        FETCH_POINT(  cand_pnt_7,  7 );
        FETCH_POINT(  cand_pnt_8,  8 );
        FETCH_POINT(  cand_pnt_9,  9 );
        FETCH_POINT( cand_pnt_10, 10 );
        FETCH_POINT( cand_pnt_11, 11 );
    }while(0);
    
    // different threads might exit this loop at different times, so let them catch up.
    __syncthreads();

    while( (cnt < point_count) && flag ){
        int min_G_index;
        int point_index = -1;
        float min_dist=3*threshold;
        int last_index_checked = 0;
        float diameter;
        float dist_to_new_point;

        int latest_p_off = latest_point*max_degree;

        do{
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_0,  curr_dist_to_clust_0,  0 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_1,  curr_dist_to_clust_1,  1 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_2,  curr_dist_to_clust_2,  2 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_3,  curr_dist_to_clust_3,  3 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_4,  curr_dist_to_clust_4,  4 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_5,  curr_dist_to_clust_5,  5 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_6,  curr_dist_to_clust_6,  6 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_7,  curr_dist_to_clust_7,  7 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_8,  curr_dist_to_clust_8,  8 );
            COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_9,  curr_dist_to_clust_9,  9 );
            COMPUTE_DIAMETER_WITH_POINT( cand_pnt_10, curr_dist_to_clust_10, 10 );
            COMPUTE_DIAMETER_WITH_POINT( cand_pnt_11, curr_dist_to_clust_11, 11 );
        }while(0);

        // different threads might exit this loop at different times, so let them catch up.
        __syncthreads();

        // The following loop implements the "find point pj s.t. diameter(Ai && pj) is minimum"
        for(int i=12; i*curThreadCount+tid < max_degree; i++){
            FETCH_POINT( cand_pnt_i, i );
            COMPUTE_DIAMETER_WITH_POINT( cand_pnt_i, curr_dist_to_clust_i, i );
        }
        __syncthreads();

        min_G_index = closest_point_reduction(min_dist, threshold, point_index);

        if(min_G_index >= 0 ){
            if( 0 == tid ){
                Ai_mask[min_G_index] = 1;
                if( NULL != candidate_cluster ){
                    candidate_cluster[cnt] = min_G_index;
                }
            }
            latest_point = min_G_index;
            cnt++;
        }else{
            flag = false;
        }

        __syncthreads();
    }
    __syncthreads();

    return cnt;
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


__global__ 
void QTC_device( float *dense_dist_matrix, char *Ai_mask, char *clustered_pnts_mask, int *indr_mtrx, int *cluster_cardinalities, int *ungrpd_pnts_indr, float *dist_to_clust, int *degrees, int point_count, int N0, int max_degree, float threshold, int cwrank, int node_offset, int total_thread_block_count) {
    int max_cardinality = -1;
    int max_cardinality_index;
    int i, tblock_id, tid, base_offset;

    tid = threadIdx.x;
    tblock_id = (blockIdx.y * gridDim.x + blockIdx.x);
    Ai_mask = &Ai_mask[tblock_id * N0];
    dist_to_clust = &dist_to_clust[tblock_id * max_degree];
    base_offset = node_offset+tblock_id;

    // for i loop of the algorithm.
    // Each thread iterates over all points that the whole thread-block owns
    for(i = base_offset; i < point_count; i+= total_thread_block_count ){
        int cnt;
        int seed_index = ungrpd_pnts_indr[i];
        int degree = degrees[seed_index];
        if( degree <= max_cardinality ) continue;
        cnt = generate_candidate_cluster2(seed_index, degree, Ai_mask, dense_dist_matrix, clustered_pnts_mask, indr_mtrx,
                                         dist_to_clust, point_count, N0, max_degree, NULL, threshold);
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


__global__ void
populate_dense_indirection_matrix(int *indr_mtrx, int *degrees, int N0, int max_degree){
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
        for(int j=0; j <max_degree; j++){
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
trim_ungrouped_pnts_indr_array(int seed_index, int *ungrpd_pnts_indr, float *dense_dist_matrix, int *result_cluster, char *Ai_mask, char *clustered_pnts_mask, int *indr_mtrx, int *cluster_cardinalities, float *dist_to_clust, int *degrees, int point_count, int N0, int max_degree, float threshold) {
    int cnt;
    int tid = threadIdx.x;
    int curThreadCount = blockDim.x*blockDim.y*blockDim.z;
    __shared__ int tmp_pnts[THREADSPERBLOCK];

    int degree = degrees[seed_index];
    (void)generate_candidate_cluster2(seed_index, degree, Ai_mask, dense_dist_matrix, clustered_pnts_mask, indr_mtrx,
                                     dist_to_clust, point_count, N0, max_degree, result_cluster, threshold);

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

#endif
