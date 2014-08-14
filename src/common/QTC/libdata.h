#ifndef _LIBDATA_H_
#define _LIBDATA_H_
#include <math.h>

float *generate_synthetic_data(float **rslt_mtrx, 
                                int **indr_mtrx, 
                                int *max_degree, 
                                float threshold, 
                                int N, 
                                bool useFullLayout);
int read_BLAST_data(float **rslt_mtrx, 
                        int **indr_mtrx, 
                        int *max_degree, 
                        float threshold, 
                        const char *fname, 
                        int maxN, 
                        bool useFullLayout);

#endif
