#ifndef _KERNELS_FULL_STORAGE_H_
#define _KERNELS_FULL_STORAGE_H_

inline __device__
int find_closest_point_to_cluster(int seed_point, int latest_point, char *Ai_mask, char *clustered_pnts_mask, float *work, int *indr_mtrx, float *dist_to_clust, int point_count, int N0, int max_degree, float threshold, bool can_use_texture){
    int point_index = -1;
    float min_dist=2*threshold;

    int tid = threadIdx.x;
    int curThreadCount = blockDim.x*blockDim.y*blockDim.z;

    // For each point that is connected to the seed
    for(int i=0; i+tid<max_degree; i+=curThreadCount){
        int pnt2_indx;

        pnt2_indx = indr_mtrx[seed_point*max_degree+i+tid];

        // Negative index means that there are no more points connected to the seed
        if( pnt2_indx >= 0 && 0 == Ai_mask[pnt2_indx] && 0 == clustered_pnts_mask[pnt2_indx] ){
            float dist_to_new_point = 0, diameter = 0;

            float curr_dist_to_clust = dist_to_clust[i+tid];
            if( can_use_texture ){
                dist_to_new_point = tex2D(texDistance, float(latest_point)+0.5f, float(pnt2_indx)+0.5f );
            }else{
                dist_to_new_point = work[pnt2_indx*N0 + latest_point];
            }
            if(dist_to_new_point > curr_dist_to_clust){
                dist_to_clust[i+tid] = dist_to_new_point;
                diameter = dist_to_new_point;
            }else{
                diameter = curr_dist_to_clust;
            }

            // The point that leads to the cluster with the smallest diameter is the closest point
            if( diameter < min_dist ){
                min_dist = diameter;
                point_index = pnt2_indx;
            }
        }
    }

    return closest_point_reduction(min_dist, threshold, point_index);
}


#  define CHECK_POINT_FOR_PROXIMITY( _CAND_PNT_, _I_, _CAND_PNT_DIST_ ) \
do{ \
                /* Negative index means that there are no more points connected to the seed */ \
                if( (_CAND_PNT_) >= 0 ){ \
                    if( 0 != Ai_mask[ (_CAND_PNT_) ] || 0 != clustered_pnts_mask[ (_CAND_PNT_) ] ){ \
                        _CAND_PNT_ = -1; \
                        break; \
                    } \
                    float dist_to_new_point = 0, diameter = 0; \
 \
                    float curr_dist_to_clust = _CAND_PNT_DIST_; \
                    if( can_use_texture ){\
                    dist_to_new_point = tex2D(texDistance, float(latest_point)+0.5f, float( (_CAND_PNT_) )+0.5f ); \
                    }else{\
                    dist_to_new_point = work[ (_CAND_PNT_) * N0 + latest_point]; \
                    }\
                    if(dist_to_new_point > curr_dist_to_clust){ \
                        _CAND_PNT_DIST_ = dist_to_new_point; \
                        diameter = dist_to_new_point; \
                    }else{ \
                        diameter = curr_dist_to_clust; \
                    } \
 \
                    /* The point that leads to the cluster with the smallest diameter is the closest point */ \
                    if( diameter < min_dist ){ \
                        min_dist = diameter; \
                        point_index = (_CAND_PNT_); \
                    } \
                } \
}while(0)

#define FETCH_CAND_PONT( _I_ )  indr_mtrx[seed_point*max_degree +(_I_)]


inline __device__
int generate_candidate_cluster_full_storage(int seed_point, int degree, char *Ai_mask, float *work, char *clustered_pnts_mask, int *indr_mtrx, float *dist_to_clust, int point_count, int N0, int max_degree, int *candidate_cluster, float threshold, bool can_use_texture)
{
    bool flag;
    int cnt, latest_point;

    int curThreadCount = blockDim.x*blockDim.y*blockDim.z;
    int tid = threadIdx.x;

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
    latest_point = seed_point;

    // If we are only computing the cardinality of each cluster, we don't need to store the actual members.
    if( NULL == candidate_cluster ){
        int cand_pnt_1=-1, cand_pnt_2=-1, cand_pnt_3=-1, cand_pnt_4=-1;
        int cand_pnt_5=-1, cand_pnt_6=-1, cand_pnt_7=-1, cand_pnt_8=-1;
        int cand_pnt_9=-1, cand_pnt_10=-1, cand_pnt_11=-1, cand_pnt_12=-1;
        float cp_dist_1=0, cp_dist_2=0, cp_dist_3=0, cp_dist_4=0;
        float cp_dist_5=0, cp_dist_6=0, cp_dist_7=0, cp_dist_8=0;
        float cp_dist_9=0, cp_dist_10=0, cp_dist_11=0, cp_dist_12=0;

        // prefetch 8 points per thread into registers to reduce the memory pressure to the indirection array
        do{
            if( tid>=degree ) break;
            cand_pnt_1 = FETCH_CAND_PONT( (tid) );
            if( curThreadCount+tid>=degree ) break;
            cand_pnt_2 = FETCH_CAND_PONT( (curThreadCount+tid) );
            if( 2*curThreadCount+tid>=degree ) break;
            cand_pnt_3 = FETCH_CAND_PONT( (2*curThreadCount+tid) );
            if( 3*curThreadCount+tid>=degree ) break;
            cand_pnt_4 = FETCH_CAND_PONT( (3*curThreadCount+tid) );
            if( 4*curThreadCount+tid>=degree ) break;
            cand_pnt_5 = FETCH_CAND_PONT( (4*curThreadCount+tid) );
            if( 5*curThreadCount+tid>=degree ) break;
            cand_pnt_6 = FETCH_CAND_PONT( (5*curThreadCount+tid) );
            if( 6*curThreadCount+tid>=degree ) break;
            cand_pnt_7 = FETCH_CAND_PONT( (6*curThreadCount+tid) );
            if( 7*curThreadCount+tid>=degree ) break;
            cand_pnt_8 = FETCH_CAND_PONT( (7*curThreadCount+tid) );

            if( 8*curThreadCount+tid>=degree ) break;
            cand_pnt_9 = FETCH_CAND_PONT( (8*curThreadCount+tid) );
            if( 9*curThreadCount+tid>=degree ) break;
            cand_pnt_10 = FETCH_CAND_PONT( (9*curThreadCount+tid) );
            if( 10*curThreadCount+tid>=degree ) break;
            cand_pnt_11 = FETCH_CAND_PONT( (10*curThreadCount+tid) );
            if( 11*curThreadCount+tid>=degree ) break;
            cand_pnt_12 = FETCH_CAND_PONT( (11*curThreadCount+tid) );

        }while(0);

        __syncthreads();

        while( (cnt < point_count) && flag ){
            int min_G_index;
            int point_index = -1;
            float min_dist=2*threshold;

            CHECK_POINT_FOR_PROXIMITY( cand_pnt_1, (tid), cp_dist_1 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_2, (curThreadCount+tid), cp_dist_2 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_3, (2*curThreadCount+tid), cp_dist_3 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_4, (3*curThreadCount+tid), cp_dist_4 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_5, (4*curThreadCount+tid), cp_dist_5 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_6, (5*curThreadCount+tid), cp_dist_6 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_7, (6*curThreadCount+tid), cp_dist_7 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_8, (7*curThreadCount+tid), cp_dist_8 );

            CHECK_POINT_FOR_PROXIMITY( cand_pnt_9,  (8*curThreadCount+tid),  cp_dist_9  );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_10, (9*curThreadCount+tid),  cp_dist_10 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_11, (10*curThreadCount+tid), cp_dist_11 );
            CHECK_POINT_FOR_PROXIMITY( cand_pnt_12, (11*curThreadCount+tid), cp_dist_12 );

            // For each remaining point that is connected to the seed
            for(int i=12*curThreadCount; i+tid<degree; i+=curThreadCount){
                int cand_pnt;
                cand_pnt = FETCH_CAND_PONT( (i+tid) );
                CHECK_POINT_FOR_PROXIMITY( cand_pnt, (i+tid), dist_to_clust[i+tid] );
            }

            min_G_index = closest_point_reduction(min_dist, threshold, point_index);

            if(min_G_index >= 0 ){
                if( 0 == tid ){
                    Ai_mask[min_G_index] = 1;
                }
                latest_point = min_G_index;
                cnt++;
            }else{
                flag = false;
            }
            __syncthreads();
        }

    }else{ // if we are interested in the actual members of the cluster, put them in the "candidate_cluster[]" array.

        while( (cnt < point_count) && flag ){
            int min_G_index;
            min_G_index = find_closest_point_to_cluster(seed_point, latest_point, Ai_mask, clustered_pnts_mask, work,
                                                        indr_mtrx, dist_to_clust, point_count, N0, max_degree, threshold, can_use_texture);
            if(min_G_index >= 0 ){
                if( 0 == tid ){
                    Ai_mask[min_G_index] = 1;
                    candidate_cluster[cnt] = min_G_index;
                }
                latest_point = min_G_index;
                cnt++;
            }else{
                flag = false;
            }
            __syncthreads();
        }

    }

    __syncthreads();
    return cnt;
}


#endif
