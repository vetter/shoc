#ifndef SPMV_H_
#define SPMV_H_

// Block size
static const int BLOCK_SIZE = 128;
static const int WARP_SIZE = 32;

enum kernelType{CSR_SCALAR, CSR_VECTOR, ELLPACKR};

#endif // SPMV_H_
