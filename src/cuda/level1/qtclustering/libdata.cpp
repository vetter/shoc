#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <map>

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include "libdata.h"

#define MAX_WIDTH  20
#define MAX_HEIGHT 20

template<class T2>
void generatePoints(T2 *array, int clusterCount, unsigned long pointCount){
    int cnt=0, ccRoot, ub;
    float dx, dy;

    if(clusterCount == 0 ){
        for (int p = 0; p < pointCount; p++) {
            array[p].x = (float)(rand()/(float)RAND_MAX)*MAX_WIDTH;
            array[p].y = (float)(rand()/(float)RAND_MAX)*MAX_HEIGHT;
        }
    }else{
        // Put ~50% of the points in clusters.
        unsigned long clusterPointCount = (unsigned long)((double)pointCount*0.5);

        ccRoot = (int)sqrt((double)clusterCount);
        ub = clusterCount/ccRoot;
        if( clusterCount%ccRoot )
            ++ub;
        dx = (float)MAX_WIDTH/(2*ub-1);
        dy = (float)MAX_HEIGHT/(2*ccRoot-1);
    
        for (int i = 0; i < ccRoot; i++) {
            // The last row might have fewer clusters
            if( i == ccRoot-1 && clusterCount%ccRoot )
                ub = clusterCount%ub;

            for (int j = 0; j < ub; j++) {

                // Generate the points for cluster (i,j)
                for (int p = 0; p < clusterPointCount/clusterCount; p++) {
                    float sign = (drand48()<0.5)?-1.0:1.0;
                    float R = (float)drand48()*(dx<dy?dx:dy)/2.0;

                    float x = (float)(2*drand48()-1)*R;  /* -R < x < R */
                    float y = (float)sqrt(R*R-x*x)*sign; /* y = (R^2-x^2)^0.5 */

                    array[cnt].x = (float)(2*dx*(float)j + dx/2 + x);
                    array[cnt].y = (float)(2*dy*(float)i + dy/2 + y);
                    cnt++;
                }

            }
        }

        // The remaining ~50% points, will be random
        for (; cnt < pointCount; cnt++) {
            array[cnt].x = (float)drand48()*MAX_WIDTH;
            array[cnt].y = (float)drand48()*MAX_HEIGHT;
        }
    }

    return;
}

using namespace std;

int read_BLAST_data(float **rslt_mtrx, int *max_card, float threshold, const char *fname){
    return read_BLAST_data(rslt_mtrx, max_card, threshold, fname, -1);
}

int read_BLAST_data(float **rslt_mtrx, int *max_card, float threshold, const char *fname, int maxN){
    FILE *ifp;
    float *dist_mtrx;
/*
    map <int, int> pnt2num;
#if defined(VERIFY_DATA)
    set <int> p1_set, p2_set;
#endif
*/
    int prev_p1=-1, p1, p2;
    int N = 0, count=0, max_count=0;
    int scan;
    float dist;
//    string s;

//    ifstream ifs( fname );

#if 0
    // Go over the data to figure out the number of elements,
    // and create the mapping from IDs to array offsets
    while( getline( ifs, s ) ) {
        stringstream ss(s);
        double dist;
        if(!(ss >> p1 >> p2 >> dist)){
            cerr << "Invalid number" << endl;
            return(-1);
        }
        if( p1 != prev_p1 ){
            prev_p1 = p1;
            pnt2num[p1] = N;
            N++;
        }
#if defined(VERIFY_DATA)
        p1_set.insert(p1);
        p2_set.insert(p2);
#endif
    }

#if defined(VERIFY_DATA)
    assert( p1_set.size() == p2_set.size() );
    assert( p1_set.size() == N );
    cout << "Finished reading data. Verifying it." << endl;
    // Verify data
    set <int>::iterator it;
    for ( it=p2_set.begin() ; it != p2_set.end(); it++ )
        assert( p1_set.find(*it) != p1_set.end() );
    cout << "Data has been verified." << endl;
#endif

    ifs.clear();
    ifs.seekg(0,std::ios::beg);
#endif

    ifp = fopen(fname, "r");
    if( NULL == ifp ){
        fprintf(stderr,"Cannot open file \"%s\". Exiting.\n",fname);
        exit(-1);
    }

    // Count the number of distinct points.
    N = 0;
    scan = fscanf(ifp, "%d %d %g\n",&p1, &p2, &dist);
    while(0 != scan && EOF != scan){
        if( p1 != prev_p1 ){
            prev_p1 = p1;
            N++;
        }
        scan = fscanf(ifp, "%d %d %g\n",&p1, &p2, &dist);
    }
    fseek(ifp, 0, SEEK_SET);

    // Allocate the proper size matrix
    if(maxN>0){
        N = maxN;
    }
    allocHostBuffer((void **)&dist_mtrx, N*N*sizeof(float));

    // Initialize the distances to something huge.
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            dist_mtrx[i*N+j] = FLT_MAX;
        }
    }

    // Read in the proper number of elements
    scan = fscanf(ifp, "%d %d %g\n",&p1, &p2, &dist);
    while(0 != scan && EOF != scan && p1 < N){
        if( (p2 < N) && (dist < threshold) ){
            float dist2 = dist_mtrx[p2*N+p1];
            if( dist2 < FLT_MAX )
                dist = (dist + dist2)/2.0;
            dist_mtrx[p1*N+p2] = dist;
            dist_mtrx[p2*N+p1] = dist;
        }

        scan = fscanf(ifp, "%d %d %g\n",&p1, &p2, &dist);
    }

#if 0
    // Read the data into the distance matrix
    while( getline( ifs, s ) ) {
        int i,j;
        double dist;
        stringstream ss(s);
     
        if(!(ss >> p1 >> p2 >> dist)){
            cerr << "Invalid number" << endl;
            return(-1);
        }
        i = pnt2num[p1];
        j = pnt2num[p2];
        if( i < N && j < N ){
            // since A->B might be different than B->A, average them
            if( dist_mtrx[j*N+i] < FLT_MAX )
                dist = (dist_mtrx[j*N+i] + dist)/2.0;
            dist_mtrx[i*N+j] = dist;
            dist_mtrx[j*N+i] = dist;
        }
    }
#endif

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

    *max_card = max_count;

    *rslt_mtrx = dist_mtrx;
    return N;
}

float *
#if defined(N_SQUARE)
fake_BLAST_data(float **rslt_mtrx, int *max_degree, float threshold, int N){
#else
fake_BLAST_data(float **rslt_mtrx, int **indr_mtrx, int *max_degree, float threshold, int N){
#endif
    int count, D=0;
    float *dist_mtrx, *points;
#if !defined(N_SQUARE)
    int *index_mtrx;
#endif
    float threshold_sq;

    //FILE *ofp = fopen("points.dat","w");

    // Create N points in a MAX_WIDTH x MAX_HEIGHT (20x20) space.
    points = (float *)malloc(2*N*sizeof(float));

    count = 0;
    while( count < N ){
        int group_cnt;
        float R, cntr_x, cntr_y;

        cntr_x = (float)drand48()*MAX_WIDTH;
        cntr_y = (float)drand48()*MAX_HEIGHT;
        R = (float)drand48()*MAX_WIDTH/2;
        group_cnt = random()%(N/30);
        if( group_cnt > (N-count) )
            group_cnt = N-count;

        while( group_cnt > 0 ){
            float sign, r, x, y, dx, dy;
            sign = (drand48()<0.5)?-1.0:1.0;
            r = (float)drand48()*R;         // 0 <= r <= R
            dx = (float)(2*drand48()-1)*r;  // -r < dx < r
            dy = (float)sqrt(r*r-dx*dx)*sign; // y = (r^2-dx^2)^0.5
            x = cntr_x+dx;
            if( x<0 || x>MAX_WIDTH)
                continue;
            y = cntr_y+dy;
            if( y<0 || y>MAX_HEIGHT)
                continue;

            points[2*count]   = x;
            points[2*count+1] = y;

            //if( NULL != ofp )
            //    fprintf(ofp, "%f %f\n",x,y);

            count++;
            group_cnt--;
        }
    }

    //fclose(ofp);

    threshold_sq = threshold*threshold;

    // Allocate the proper size matrix
#if defined(N_SQUARE)
    allocHostBuffer((void **)&dist_mtrx, N*N*sizeof(float));
#else
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
    allocHostBuffer((void **)&dist_mtrx, N*D*sizeof(float));
    allocHostBuffer((void **)&index_mtrx, N*D*sizeof(int));
#endif

    // Initialize the distances to something huge.
    for(int i=0; i<N; i++){
#if defined(N_SQUARE)
        for(int j=0; j<N; j++){
            dist_mtrx[i*N+j] = FLT_MAX;
#else
        for(int j=0; j<D; j++){
            index_mtrx[i*D+j] = -1;
            dist_mtrx[i*D+j] = FLT_MAX;
#endif
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
#if defined(N_SQUARE)
                dist_mtrx[i*N+j] = dist;
                dist_mtrx[j*N+i] = dist;
#else
                index_mtrx[i*D+delta] = j;
                dist_mtrx[i*D+delta] = dist;
#endif
                delta++;
            }
        }
#if defined(N_SQUARE)
        if( delta > D )
            D = delta;
#endif
    }

    //free(points);

   *max_degree = D;
   *rslt_mtrx = dist_mtrx;
#if !defined(N_SQUARE)
    *indr_mtrx = index_mtrx;
#endif
    return points;
}
