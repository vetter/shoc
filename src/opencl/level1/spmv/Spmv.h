#ifndef SPMV_H_
#define SPMV_H_

// Default Block size -- note this may be adjusted
// at runtime if it's not compatible with the device's
// capabilities
static const int BLOCK_SIZE = 128; 

// Number of work items to use per row of the matrix
// If you change this value, also change the setting in
// the kernel code, spmv.cl
static const int VECTOR_SIZE = 32;

#endif // SPMV_H_
