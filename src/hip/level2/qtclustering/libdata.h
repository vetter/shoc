#ifndef _LIBDATA_H_
#define _LIBDATA_H_
#include <math.h>
#include "support.h"
#include "qtclib.h"

float *generate_synthetic_data(float **rslt_mtrx, int **indr_mtrx, int *max_degree, float threshold, int N, int type);
int read_BLAST_data(float **rslt_mtrx, int **indr_mtrx, int *max_degree, float threshold, const char *fname, int maxN, int matrix_type_mask);

#endif
