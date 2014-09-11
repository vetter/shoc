// Code derived from work done by the authors quoted in the original header below:

//
// (c) January 24, 2008 Vasily Volkov @ UC Berkeley
//
// Other credits:
// - Paul Leventis @ Altera Corp. for prefetching and -maxrregcount techniques
// - many thanks to Wladimir J. van der Laan @ the University of Groningen
// for his cubin disassembler (http://www.cs.rug.nl/~wladimir/decuda/)
//
//

#ifdef SINGLE_PRECISION
#define FPTYPE float
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#endif

#define SAXPY( _A_, _BS_ , _C_) do{ \
	_C_[0] += _A_ * _BS_[0]; \
	_C_[1] += _A_ * _BS_[1]; \
	_C_[2] += _A_ * _BS_[2]; \
	_C_[3] += _A_ * _BS_[3]; \
	_C_[4] += _A_ * _BS_[4]; \
	_C_[5] += _A_ * _BS_[5]; \
	_C_[6] += _A_ * _BS_[6]; \
	_C_[7] += _A_ * _BS_[7]; \
	_C_[8] += _A_ * _BS_[8]; \
	_C_[9] += _A_ * _BS_[9]; \
	_C_[10] += _A_ * _BS_[10]; \
	_C_[11] += _A_ * _BS_[11]; \
	_C_[12] += _A_ * _BS_[12]; \
	_C_[13] += _A_ * _BS_[13]; \
	_C_[14] += _A_ * _BS_[14]; \
	_C_[15] += _A_ * _BS_[15]; \
    }while(0)

__kernel void sgemmNT( __global const FPTYPE *A, int lda,
                       __global const FPTYPE *B, int ldb,
                       __global FPTYPE *C, int ldc, int k,
                       FPTYPE alpha, FPTYPE beta )
{
	const int inx = get_local_id(0);
	const int iny = get_local_id(1);
	const int ibx = get_group_id(0) * 64;
	const int iby = get_group_id(1) * 16;
	const int id  = inx + iny*16;

        int i, counter = 0;

	A += ibx + id;
	B += iby + inx + (iny*ldb);
	C += ibx + id  + (iby*ldc );

	FPTYPE a[4];
	for(i=0; i<4; ++i){ a[i] = A[i*lda]; }
	__private FPTYPE b;
	b = B[0];

	A += 4*lda;
	B += 4*ldb;
        counter+= 4*ldb;

	__local FPTYPE bs[4][16];
	FPTYPE c[16];
        for(i=0; i<16; ++i){
            c[i] = 0.0;
        }

	do
	{
	        __private FPTYPE as[4];
		for(i=0; i<4; ++i){ as[i] = a[i]; }

		bs[iny][inx] = b;
  		barrier(CLK_LOCAL_MEM_FENCE);

		a[0] = A[0*lda];
		a[1] = A[1*lda];
		a[2] = A[2*lda];
		a[3] = A[3*lda];
		b    = B[0];

		SAXPY( as[0], bs[0], c );
		SAXPY( as[1], bs[1], c );
		SAXPY( as[2], bs[2], c );
		SAXPY( as[3], bs[3], c );

		A += 4*lda;
		B += 4*ldb;
                counter += 4*ldb;
  		barrier(CLK_LOCAL_MEM_FENCE);

	} while( counter < k*ldb );

	bs[iny][inx] = b;
	barrier(CLK_LOCAL_MEM_FENCE);

	SAXPY( a[0], bs[0], c );
	SAXPY( a[1], bs[1], c );
	SAXPY( a[2], bs[2], c );
	SAXPY( a[3], bs[3], c );

	for( int i = 0; i < 16; i++, C += ldc ){
		C[0] = alpha*c[i] + beta*C[0];
        }
}


__kernel void sgemmNN( __global const FPTYPE *A, int lda,
                       __global const FPTYPE *B, int ldb,
                       __global FPTYPE *C, int ldc, int k,
                       FPTYPE alpha, FPTYPE beta )
{
	const int inx = get_local_id(0);
	const int iny = get_local_id(1);
	const int ibx = get_group_id(0) * 64;
	const int iby = get_group_id(1) * 16;
	const int id = inx + iny*16;

        int i, j, ii, counter=0;

	A += ibx + id;

	B += inx + (iby+iny) * ldb;

	C += ibx + id  + (iby*ldc);

	FPTYPE c[16];
        for(i=0; i<16; ++i){
            c[i] = 0.0;
	}

       	__local FPTYPE bs[16][17];

	do
	{
		__private FPTYPE a[4];
		for(ii=0; ii<4; ++ii) { a[ii] = A[ii*lda]; }

		bs[inx][iny]    = B[0*ldb];
		bs[inx][iny+4]  = B[4*ldb];
		bs[inx][iny+8]  = B[8*ldb];
		bs[inx][iny+12] = B[12*ldb];
		barrier(CLK_LOCAL_MEM_FENCE);

		A += 4*lda;

		SAXPY( a[0], bs[0], c );	a[0] = A[0*lda];
		SAXPY( a[1], bs[1], c );	a[1] = A[1*lda];
		SAXPY( a[2], bs[2], c );	a[2] = A[2*lda];
		SAXPY( a[3], bs[3], c );	a[3] = A[3*lda];

		A += 4*lda;
		SAXPY( a[0], bs[4], c );	a[0] = A[0*lda];
		SAXPY( a[1], bs[5], c );	a[1] = A[1*lda];
		SAXPY( a[2], bs[6], c );	a[2] = A[2*lda];
		SAXPY( a[3], bs[7], c );	a[3] = A[3*lda];

		A += 4*lda;
		SAXPY( a[0], bs[8], c );	a[0] = A[0*lda];
		SAXPY( a[1], bs[9], c );	a[1] = A[1*lda];
		SAXPY( a[2], bs[10], c );	a[2] = A[2*lda];
		SAXPY( a[3], bs[11], c );	a[3] = A[3*lda];

		A += 4*lda;
		SAXPY( a[0], bs[12], c );
		SAXPY( a[1], bs[13], c );
		SAXPY( a[2], bs[14], c );
		SAXPY( a[3], bs[15], c );

		B += 16;
	        counter += 16;
		barrier(CLK_LOCAL_MEM_FENCE);
	} while( counter < k );

	for( int i = 0; i < 16; i++, C += ldc ){
		C[0] = alpha*c[i] + beta*C[0];
	}

}

