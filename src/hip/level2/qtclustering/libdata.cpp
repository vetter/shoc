#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <map>

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include "qtc_common.h"
#include "libdata.h"
#include "PMSMemMgmt.h"

#define MAX_WIDTH  20.0
#define MAX_HEIGHT 20.0

using namespace std;

// This function reads data from a file in the format:
// %d %d %g
// or in other words:
// integer integer double
int read_BLAST_data(float **rslt_mtrx, int **indr_mtrx, int *max_degree, float threshold, const char *fname, int maxN, int matrix_type_mask){
    FILE *ifp;
    float *dist_mtrx;
    int *index_mtrx;
    int prev_p1=-1, p1, p2;
    int bound = 0, N = 0, D = 0, delta = 0, count=0, max_count=0;
    int scan;
    float dist;

    ifp = fopen(fname, "r");
    if( NULL == ifp ){
        fprintf(stderr,"Cannot open file \"%s\". Exiting.\n",fname);
        exit(-1);
    }

    // Count the number of distinct points.
    N = 0;
    delta = 0;
    scan = fscanf(ifp, "%d %d %g\n",&p1, &p2, &dist);
    while(0 != scan && EOF != scan){
        if( dist < threshold ){
            if( p1 != prev_p1 ){
                prev_p1 = p1;
                N++;
                if( delta > D )
                    D = delta;
                delta = 1;
            }else{
                delta++;
            }
        }
        scan = fscanf(ifp, "%d %d %g\n",&p1, &p2, &dist);
    }
    fseek(ifp, 0, SEEK_SET);

    // Allocate the proper size matrix
    if(maxN>0){
        N = maxN;
    }

    /* new */
    if( matrix_type_mask & FULL_STORAGE_MATRIX ){
        bound = N;
    }else{
        bound = D;
    }
    dist_mtrx = pmsAllocHostBuffer<float>(N*bound);
    index_mtrx = pmsAllocHostBuffer<int>(N*D);

    // Initialize the distances to something huge.
    for(int i=0; i<N; i++){
        for(int j=0; j<D; j++){
            index_mtrx[i*D+j] = -1;
        }
        for(int j=0; j<bound; j++){
            dist_mtrx[i*bound+j] = FLT_MAX;
        }
    }
    /* new */

    /*
    // Initialize the distances to something huge.
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            dist_mtrx[i*N+j] = FLT_MAX;
        }
    }
    */

    // Read in the proper number of elements
    scan = fscanf(ifp, "%d %d %g\n",&p1, &p2, &dist);
    while(0 != scan && EOF != scan && p1 < N){
        int delta=0;
/*
        if( (p2 < N) && (dist < threshold) ){
            float dist2 = dist_mtrx[p2*N+p1];
            if( dist2 < FLT_MAX )
                dist = (dist + dist2)/2.0;
            dist_mtrx[p1*N+p2] = dist;
            dist_mtrx[p2*N+p1] = dist;
        }
*/

        if( (p2 < N) && (dist < threshold) ){
            while(index_mtrx[p1*D+delta] >= 0)
                delta++;
            assert(delta <= D);
            index_mtrx[p1*D+delta] = p2;
            if( matrix_type_mask & FULL_STORAGE_MATRIX ){
                dist_mtrx[p1*N+p2] = dist;
                dist_mtrx[p2*N+p1] = dist;
            }else{
                dist_mtrx[p1*D+delta] = dist;
                delta = 0;
                while(delta < D){
                    int p = index_mtrx[p2*D+delta];
                    if( p < 0 ){
                        index_mtrx[p2*D+delta] = p1;
                        dist_mtrx[p2*D+delta] = dist;
                        break;
                    }
                    if( p == p1 ){ // if p1 is already in p2's proximity table
                        break;
                    }
                    delta++;
                }
            }
        }


        scan = fscanf(ifp, "%d %d %g\n",&p1, &p2, &dist);
    }

    /*
    for(int i=0; i<N; i++){
        int count = 0;
        for(int j=0; j<N; j++){
            if( dist_mtrx[i*N+j] < threshold ){
                count++;
            }else{
                dist_mtrx[i*N+j] = FLT_MAX;
            }
        }
        if( count > max_count ){
            max_count = count;
        }
    }

    *max_degree = max_count;
    */

    *max_degree = D;
    *indr_mtrx = index_mtrx;
    *rslt_mtrx = dist_mtrx;
    return N;
}

static inline float frand(void){
#ifdef _WIN32
    return (float)rand()/RAND_MAX;
#else
    return (float)random()/RAND_MAX;
#endif
}

// This function generates elements as points on a 2D Euclidean plane confined
// in a MAX_WIDTHxMAX_HEIGHT square (20x20 by default).  The elements are not
// uniformly distributed on the plane, but rather appear in clusters of random
// radius and cardinality. The maximum cardinality of a cluster is N/30 where
// N is the total number of data generated.
float *generate_synthetic_data(float **rslt_mtrx, int **indr_mtrx, int *max_degree, float threshold, int N, int matrix_type_mask){
    int count, bound, D=0;
    float *dist_mtrx, *points;
    int *index_mtrx;
    float threshold_sq, min_dim;

    // Create N points in a MAX_WIDTH x MAX_HEIGHT (20x20) space.
    points = (float *)malloc(2*N*sizeof(float));

    min_dim = MIN(MAX_WIDTH,MAX_HEIGHT);

    count = 0;
    while( count < N ){
        int group_cnt;
        float R, cntr_x, cntr_y;

        // Create "group_cnt" points within a circle of radious "R"
        // around center point "(cntr_x, cntr_y)"

        cntr_x = frand()*MAX_WIDTH;
        cntr_y = frand()*MAX_HEIGHT;
        R = frand()*min_dim/2;
#ifdef _WIN32
        group_cnt = rand()%(N/30);
#else
        group_cnt = random()%(N/30);
#endif
        // make sure we don't make more points than we need
        if( group_cnt > (N-count) ){
            group_cnt = N-count;
        }

        while( group_cnt > 0 ){
            float sign, r, x, y, dx, dy;
            sign = (frand()<0.5)?-1.0:1.0;
            r = frand()*R;         // 0 <= r <= R
            dx = (2.0*frand()-1.0)*r;  // -r < dx < r
            dy = sqrtf(r*r-dx*dx)*sign; // y = (r^2-dx^2)^0.5
            x = cntr_x+dx;
            if( x<0 || x>MAX_WIDTH)
                continue;
            y = cntr_y+dy;
            if( y<0 || y>MAX_HEIGHT)
                continue;

            points[2*count]   = x;
            points[2*count+1] = y;

            count++;
            group_cnt--;
        }
    }

    threshold_sq = threshold*threshold;

    // Allocate the proper size matrix
    for(int i=0; i<N; i++){
        int delta = 0;

        float p1_x = points[2*i];
        float p1_y = points[2*i+1];
        for(int j=0; j<N; j++){
            if( j == i ){
                continue;
            }
            float p2_x = points[2*j];
            float p2_y = points[2*j+1];
            float dist_sq = (p1_x-p2_x)*(p1_x-p2_x) + (p1_y-p2_y)*(p1_y-p2_y);
            if( dist_sq < threshold_sq ){
                delta++;
            }
        }
        if( delta > D )
            D = delta;
    }

    if( matrix_type_mask & FULL_STORAGE_MATRIX ){
        bound = N;
    }else{
        bound = D;
    }
    dist_mtrx = pmsAllocHostBuffer<float>(N*bound);
    index_mtrx = pmsAllocHostBuffer<int>(N*D);

    // Initialize the distances to something huge.
    for(int i=0; i<N; i++){
        for(int j=0; j<D; j++){
            index_mtrx[i*D+j] = -1;
        }
        for(int j=0; j<bound; j++){
            dist_mtrx[i*bound+j] = FLT_MAX;
        }
    }

    for(int i=0; i<N; i++){
        int delta = 0;
        float p1_x, p1_y;

        p1_x = points[2*i];
        p1_y = points[2*i+1];
        for(int j=0; j<N; j++){ // This is supposed to be "N", not "Delta"
            float p2_x, p2_y, dist_sq;
            if( j == i ){
                continue;
            }
            p2_x = points[2*j];
            p2_y = points[2*j+1];
            dist_sq = (p1_x-p2_x)*(p1_x-p2_x) + (p1_y-p2_y)*(p1_y-p2_y);
            if( dist_sq < threshold_sq ){
                float dist = (float)sqrt((double)dist_sq);
                index_mtrx[i*D+delta] = j;
                if( matrix_type_mask & FULL_STORAGE_MATRIX ){
                    dist_mtrx[i*N+j] = dist;
                    dist_mtrx[j*N+i] = dist;
                }else{
                    dist_mtrx[i*D+delta] = dist;
                }
                delta++;
            }
        }
    }

    *max_degree = D;
    *rslt_mtrx = dist_mtrx;
    *indr_mtrx = index_mtrx;
    return points;
}
