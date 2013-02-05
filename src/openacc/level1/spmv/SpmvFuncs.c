#include <stdio.h>
#include "Spmv.h"
#include "CTimer.h"

void spmv_csr_scalar_kernel_float (const void * vval,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, const int nItems, void * vvec, void * vout,
                       const int niters, double * iTransferTime,
                       double * oTransferTime, double * kernelTime)
{
	// Setup thread configuration
    int nBlocksScalar = (int) ceil((float) dim / BLOCK_SIZE);
	int timerHandle = Timer_Start();
	float * restrict val = (float *)vval;
	float * out = (float *)vout;
	float * vec = (float *)vvec;
	#pragma acc data copyin(val[0:nItems], cols[0:nItems], rowDelimiters[0:(dim+1)], vec[0:dim]) copyout(out[0:dim])
    {
		*iTransferTime = Timer_Stop(timerHandle, "");
		timerHandle = Timer_Start();
    	for( int k=0; k<niters; k++ ) {
			#pragma acc kernels loop gang(nBlocksScalar) vector(BLOCK_SIZE) present(val[0:nItems], cols[0:nItems], rowDelimiters[0:(dim+1)], out[0:dim])
        	for (int i=0; i<dim; i++) 
        	{
            	float t = 0; 
				int start = rowDelimiters[i];
				int end = rowDelimiters[i+1];
            	for (int j = start; j < end; j++)
            	{
                	int col = cols[j]; 
                	t += val[j] * vec[col];
            	}    
            	out[i] = t; 
        	}
    	}
		*kernelTime = Timer_Stop(timerHandle, "");
		timerHandle = Timer_Start();
    }
	*oTransferTime = Timer_Stop(timerHandle, "");
}

void spmv_csr_scalar_kernel_double (const void * vval,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, const int nItems, void * vvec, void * vout,
                       const int niters, double * iTransferTime,
                       double * oTransferTime, double * kernelTime)
{
	// Setup thread configuration
    int nBlocksScalar = (int) ceil((float) dim / BLOCK_SIZE);
	int timerHandle = Timer_Start();
	double * restrict val = (double *)vval;
	double * out = (double *)vout;
	double * vec = (double *)vvec;
	#pragma acc data copyin(val[0:nItems], cols[0:nItems], rowDelimiters[0:(dim+1)], vec[0:dim]) copyout(out[0:dim])
    {
		*iTransferTime = Timer_Stop(timerHandle, "");
		timerHandle = Timer_Start();
    	for( int k=0; k<niters; k++ ) {
			#pragma acc kernels loop gang(nBlocksScalar) vector(BLOCK_SIZE) present(val[0:nItems], cols[0:nItems], rowDelimiters[0:(dim+1)], out[0:dim])
        	for (int i=0; i<dim; i++) 
        	{
            	double t = 0; 
				int start = rowDelimiters[i];
				int end = rowDelimiters[i+1];
            	for (int j = start; j < end; j++)
            	{
                	int col = cols[j]; 
                	t += val[j] * vec[col];
            	}    
            	out[i] = t; 
        	}
    	}
		*kernelTime = Timer_Stop(timerHandle, "");
		timerHandle = Timer_Start();
    }
	*oTransferTime = Timer_Stop(timerHandle, "");
}

//TODO: below is bogus code; need to be completely rewritten
void spmv_csr_vector_kernel_float (const void * vval,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, const int nItems, void * vvec, void * vout,
                       const int niters, double * iTransferTime,
                       double * oTransferTime, double * kernelTime)
{
	// Setup thread configuration
    int nBlocksVector = (int) ceil(dim /
                  (float)(BLOCK_SIZE / WARP_SIZE));
	int timerHandle = Timer_Start();
	float * restrict val = (float *)vval;
	float * out = (float *)vout;
	float * vec = (float *)vvec;
	#pragma acc data copyin(val[0:nItems], cols[0:nItems], rowDelimiters[0:(dim+1)], vec[0:dim]) copyout(out[0:dim])
    {
		*iTransferTime = Timer_Stop(timerHandle, "");
		timerHandle = Timer_Start();
    	for( int k=0; k<niters; k++ ) {
			#pragma acc kernels loop gang(nBlocksVector) vector(BLOCK_SIZE) present(val[0:nItems], cols[0:nItems], rowDelimiters[0:(dim+1)], out[0:dim])
        	for (int i=0; i<dim; i++) 
        	{
            	float t = 0; 
				int start = rowDelimiters[i];
				int end = rowDelimiters[i+1];
            	for (int j = start; j < end; j++)
            	{
                	int col = cols[j]; 
                	t += val[j] * vec[col];
            	}    
            	out[i] = t; 
        	}
    	}
		*kernelTime = Timer_Stop(timerHandle, "");
		timerHandle = Timer_Start();
    }
	*oTransferTime = Timer_Stop(timerHandle, "");
}

//TODO: below is bogus code; need to be completely rewritten
void spmv_csr_vector_kernel_double (const void * vval,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, const int nItems, void * vvec, void * vout,
                       const int niters, double * iTransferTime,
                       double * oTransferTime, double * kernelTime)
{
	// Setup thread configuration
    int nBlocksVector = (int) ceil(dim /
                  (float)(BLOCK_SIZE / WARP_SIZE));
	int timerHandle = Timer_Start();
	double * restrict val = (double *)vval;
	double * out = (double *)vout;
	double * vec = (double *)vvec;
	#pragma acc data copyin(val[0:nItems], cols[0:nItems], rowDelimiters[0:(dim+1)], vec[0:dim]) copyout(out[0:dim])
    {
		*iTransferTime = Timer_Stop(timerHandle, "");
		timerHandle = Timer_Start();
    	for( int k=0; k<niters; k++ ) {
			#pragma acc kernels loop gang(nBlocksVector) vector(BLOCK_SIZE) present(val[0:nItems], cols[0:nItems], rowDelimiters[0:(dim+1)], out[0:dim])
        	for (int i=0; i<dim; i++) 
        	{
            	double t = 0; 
				int start = rowDelimiters[i];
				int end = rowDelimiters[i+1];
            	for (int j = start; j < end; j++)
            	{
                	int col = cols[j]; 
                	t += val[j] * vec[col];
            	}    
            	out[i] = t; 
        	}
    	}
		*kernelTime = Timer_Stop(timerHandle, "");
		timerHandle = Timer_Start();
    }
	*oTransferTime = Timer_Stop(timerHandle, "");
}

void spmv_ellpackr_kernel_float (const void * vval,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, const int nItems, void * vvec, void * vout,
                     const int niters, double * iTransferTime,
                     double * oTransferTime, double * kernelTime)
{
}

void spmv_ellpackr_kernel_double (const void * vval,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, const int nItems, void * vvec, void * vout,
                     const int niters, double * iTransferTime,
                     double * oTransferTime, double * kernelTime)
{
}

void zero_float (void * va, const int size)
{
}

void zero_double (void * va, const int size)
{
}
