extern "C" void cuda_allocate(float *A, float *B, float *C, int nItems)
{
    cudaMallocHost( (void **) &A, (sizeof(float)*nItems) );
    cudaMallocHost( (void **) &B, (sizeof(float)*nItems) );
    cudaMallocHost( (void **) &C, (sizeof(float)*nItems) );
    //A = malloc(sizeof(float)*nItems);
    //B = malloc(sizeof(float)*nItems);
    //C = malloc(sizeof(float)*nItems);
}

