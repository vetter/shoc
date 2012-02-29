#ifndef _LIBDATA_H_
#define _LIBDATA_H_
#include <math.h>
#include "support.h"
#include "qtclib.h"

#if defined(N_SQUARE)
float *fake_BLAST_data(float **rslt_mtrx, int *max_degree, float threshold, int N);
#else
float *fake_BLAST_data(float **rslt_mtrx, int **index_mtrx, int *max_degree, float threshold, int N);
#endif
int read_BLAST_data(float **dist_mtrx, int *max_degree, float threshold, const char *fname);
int read_BLAST_data(float **dist_mtrx, int *max_degree, float threshold, const char *fname, int maxN);

template <class T2>
void generatePoints(T2 *array, int clusterCount, unsigned long pointCount);
#endif
