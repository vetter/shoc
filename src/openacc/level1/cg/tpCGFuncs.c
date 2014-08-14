//!-------------------------------------------------------------------------!
//!                                                                         !
//!        N  A  S     P A R A L L E L     B E N C H M A R K S  3.3         !
//!                                                                         !
//!                                   C G                                   !
//!                                                                         !
//!-------------------------------------------------------------------------!
//!                                                                         !
//!    This benchmark is part of the NAS Parallel Benchmark 3.3 suite.      !
//!    It is described in NAS Technical Reports 95-020 and 02-007           !
//!                                                                         !
//!    Permission to use, copy, distribute and modify this software         !
//!    for any purpose with or without fee is hereby granted.  We           !
//!    request, however, that all derived work reference the NAS            !
//!    Parallel Benchmarks 3.3. This software is provided "as is"           !
//!    without express or implied warranty.                                 !
//!                                                                         !
//!    Information on NPB 3.3, including the technical report, the          !
//!    original specifications, source code, results and information        !
//!    on how to submit new results, is available at:                       !
//!                                                                         !
//!           http://www.nas.nasa.gov/Software/NPB/                         !
//!                                                                         !
//!    Send comments or suggestions to  npb@nas.nasa.gov                    !
//!                                                                         !
//!          NAS Parallel Benchmarks Group                                  !
//!          NASA Ames Research Center                                      !
//!          Mail Stop: T27A-1                                              !
//!          Moffett Field, CA   94035-1000                                 !
//!                                                                         !
//!          E-mail:  npb@nas.nasa.gov                                      !
//!          Fax:     (650) 604-3957                                        !
//!                                                                         !
//!-------------------------------------------------------------------------!
//
//
//c---------------------------------------------------------------------
//c
//c Authors: M. Yarrow
//c          C. Kuszmaul
//c          R. F. Van der Wijngaart
//c          H. Jin
//c
//c---------------------------------------------------------------------
//
//
//c---------------------------------------------------------------------
//c---------------------------------------------------------------------
//      program cg
//c---------------------------------------------------------------------
//c---------------------------------------------------------------------


//      implicit none
//
//      include 'mpinpb.h'
//      include 'timing.h'
//      integer status(MPI_STATUS_SIZE), request, ierr
//
//      include 'npbparams.h'

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "randdp.h"
#include "timers.h"
#include "print_results.h"
#include "npbparams.h"
#include "timing.h"
#include "mpinpb.h"

//c---------------------------------------------------------------------
//c  num_procs must be a power of 2, and num_procs=num_proc_cols*num_proc_rows.
//c  num_proc_cols and num_proc_cols are to be found in npbparams.h.
//c  When num_procs is not square, then num_proc_cols must be = 2*num_proc_rows.
//c---------------------------------------------------------------------

#define NPBVERSION "3.3.1"
#define NNODES_COMPILED   1

#define max(x,y)    ((x) > (y) ? (x) : (y))

#define NUM_PROCS NUM_PROC_COLS * NUM_PROC_ROWS
#define NZ NA*(NONZER+1)/NUM_PROCS*(NONZER+1)+NONZER + NA*(NONZER+2+NUM_PROCS/256)/NUM_PROC_COLS

MPI_Request request;
MPI_Status status;

int NUM_PROC_ROWS, NUM_PROC_COLS;
int NA;
int NONZER;
int NITER;
double SHIFT;
double RCOND;

//      common / partit_size  /
int naa, nzz;
int npcols, nprows;
int proc_col, proc_row;
int firstrow;
int lastrow;
int firstcol;
int lastcol;
int exch_proc;
int exch_recv_length;
int send_start;
int send_len;

int* colidx;
int* rowstr;
int* iv;
int* arow;
int* acol;

double* v;
double* aelt;
double* a;
double* x;
double* z;
double* p;
double* q;
double* r;
double* w;

double amult, tran;

int l2npcols;
int* reduce_exch_proc;
int* reduce_send_starts;
int* reduce_send_lengths;
int* reduce_recv_starts;
int* reduce_recv_lengths;

double zeta;
double rnorm;
double norm_temp1[2+1];
double norm_temp2[2+1];

double t, tmax, mflops;
char Class;
int verified;
double zeta_verify_value, epsilon, err;

void conj_grad (int *colidx, int* rowstr, double *x, double *z, double *a, double *p, double *q, double *r,
    double *w, double *rnorm, int l2npcols,
    int *reduce_exch_proc,
    int *reduce_send_starts,
    int *reduce_send_lengths,
    int *reduce_recv_starts,
    int *reduce_recv_lengths );
void sparse(double *a, int *colidx, int *rowstr, int n, int *arow, int *acol, double *aelt,
				int firstrow, int lastrow,
				double *x, int *mark, int *nzloc, int nnza );
int icnvrt(double x, int ipwr2);
void sprnvc(int n, int nz, double *v, int *iv, int *nzloc, int *mark );
void vecset(int n, double* v, int *iv, int *nzv, int i, double val);

void makea(int n, int nz, double *a, int *colidx, int *rowstr, int nonzer,
						int firstrow, int lastrow, int firstcol, int lastcol,
						double rcond, int *arow, int* acol, double* aelt, double* v, int *iv, double shift);

void setup_proc_info(int num_procs, int num_proc_rows, int num_proc_cols );
void setup_arrays();
int ilog2(int i);
int ipow2(int i);

int ilog2(int i)
{
    int log2;
    int exp2 = 1;
    if (i <= 0) return(-1);

    for (log2 = 0; log2 < 30; log2++) {
        if (exp2 == i) return(log2);
        if (exp2 > i) break;
        exp2 *= 2;
    }
    return(-1);
}

int ipow2(int i)
{
    int pow2 = 1;
    if (i < 0) return(-1);
    if (i == 0) return(1);
    while(i--) pow2 *= 2;
    return(pow2);
}

void setup_arrays() {
    colidx = (int*) calloc(NZ+1, sizeof(int));
    rowstr = (int*) calloc(NA+1+1, sizeof(int));
    iv = (int*) calloc(2*NA+1+1, sizeof(int));
    arow = (int*) calloc(NZ+1, sizeof(int));
    acol = (int*) calloc(NZ+1, sizeof(int));

    v = (double*) calloc(NA+1+1, sizeof(double));
    aelt = (double*) calloc(NZ+1, sizeof(double));
    a = (double*) calloc(NZ+1, sizeof(double));
    x = (double*) calloc(NA/NUM_PROC_ROWS+2+1, sizeof(double));
    z = (double*) calloc(NA/NUM_PROC_ROWS+2+1, sizeof(double));
    p = (double*) calloc(NA/NUM_PROC_ROWS+2+1, sizeof(double));
    q = (double*) calloc(NA/NUM_PROC_ROWS+2+1, sizeof(double));
    r = (double*) calloc(NA/NUM_PROC_ROWS+2+1, sizeof(double));
    w = (double*) calloc(NA/NUM_PROC_ROWS+2+1, sizeof(double));

    reduce_exch_proc = (int*) calloc(NUM_PROC_COLS+1, sizeof(int));
    reduce_send_starts = (int*) calloc(NUM_PROC_COLS+1, sizeof(int));
    reduce_send_lengths = (int*) calloc(NUM_PROC_COLS+1, sizeof(int));
    reduce_recv_starts = (int*) calloc(NUM_PROC_COLS+1, sizeof(int));
    reduce_recv_lengths = (int*) calloc(NUM_PROC_COLS+1, sizeof(int));
}

void setup_proc_info(int num_procs, int num_proc_rows, int num_proc_cols )
{
      int i, ierr;
      int log2nprocs;

      if( nprocs != num_procs ) {
    	  printf("Number of processes is wrongly set!\n");
    	  exit(EXIT_FAILURE);
      }

	  for (i = num_proc_cols; i>0; i=i/2){
		  if( i != 1 && i/2*2 != i ){
			if ( me == root ) printf("ERROR: num_proc_cols is not a power of two!\n");
			exit(EXIT_FAILURE);
		  }
	  }

	  for (i = num_proc_rows; i>0; i=i/2){
		  if( i != 1 && i/2*2 != i ){
			if ( me == root ) printf("ERROR: num_proc_cols is not a power of two!\n");
			exit(EXIT_FAILURE);
		  }
	  }

	  log2nprocs = 0;
	  for (i = nprocs; i/2>0; i=i/2){
		  if( i != 1 && i/2*2 != i ){
			if ( me == root ) printf("ERROR: num_proc_cols is not a power of two!\n");
			exit(EXIT_FAILURE);
		  }
		  log2nprocs++;
	  }

	  printf("nprocs, log2nprocs : %d, %d\n", nprocs,log2nprocs);
      npcols = num_proc_cols;
      nprows = num_proc_rows;
}



void setup_submatrix_info( int *l2npcols, //reference
     int *reduce_exch_proc,
     int *reduce_send_starts,
     int *reduce_send_lengths,
     int *reduce_recv_starts,
     int *reduce_recv_lengths )
{
      int col_size, row_size;
      int i, j;
      int div_factor;

      proc_row = me / npcols;
      proc_col = me - proc_row*npcols;

//c  If naa evenly divisible by npcols, then it is evenly divisible
//c  by nprows

      if (naa/npcols*npcols == naa ){
          col_size = naa/npcols;
          firstcol = proc_col*col_size + 1;
          lastcol  = firstcol - 1 + col_size;
          row_size = naa/nprows;
          firstrow = proc_row*row_size + 1;
          lastrow  = firstrow - 1 + row_size;
      }

//c  If naa not evenly divisible by npcols, then first subdivide for nprows
//c  and then, if npcols not equal to nprows (i.e., not a sq number of procs),
//c  get col subdivisions by dividing by 2 each row subdivision.

      else {
    	  if( proc_row < naa - naa/nprows*nprows){
              row_size = naa/nprows+ 1;
              firstrow = proc_row*row_size + 1;
              lastrow  = firstrow - 1 + row_size;
          }
          else {
              row_size = naa/nprows;
              firstrow = (naa - naa/nprows*nprows)*(row_size+1) + (proc_row-(naa-naa/nprows*nprows))*row_size + 1;
              lastrow  = firstrow - 1 + row_size;
          }
          if( npcols == nprows ){
              if( proc_col < naa - naa/npcols*npcols ){
                  col_size = naa/npcols+ 1;
                  firstcol = proc_col*col_size + 1;
                  lastcol  = firstcol - 1 + col_size;
              }
              else {
                  col_size = naa/npcols;
                  firstcol = (naa - naa/npcols*npcols)*(col_size+1) + (proc_col-(naa-naa/npcols*npcols))*col_size + 1;
                  lastcol  = firstcol - 1 + col_size;
              }
          }
          else {
        	  if ((proc_col/2)< naa - naa/(npcols/2)*(npcols/2)){
                  col_size = naa/(npcols/2) + 1;
                  firstcol = (proc_col/2)*col_size + 1;
                  lastcol  = firstcol - 1 + col_size;
			  }
              else{
                  col_size = naa/(npcols/2);
                  firstcol = (naa - naa/(npcols/2)*(npcols/2))*(col_size+1)+ ((proc_col/2)-(naa-naa/(npcols/2)*(npcols/2)))*col_size + 1;
                  lastcol  = firstcol - 1 + col_size;
              }
			  printf("*,*: %d, %d, %d\n", col_size,firstcol,lastcol);
              if (me%2 == 0 )
                  lastcol  = firstcol - 1 + (col_size-1)/2 + 1;
              else {
                  firstcol = firstcol + (col_size-1)/2 + 1;
                  lastcol  = firstcol - 1 + col_size/2;
                  printf("*,*: %d, %d\n", firstcol, lastcol);
              }
          }
		}

      if( npcols == nprows ){
          send_start = 1;
          send_len   = lastrow - firstrow + 1;
      }
      else {
          if( me%2 == 0 ){
              send_start = 1;
              send_len   = (1 + lastrow-firstrow+1)/2;
          }
          else {
              send_start = (1 + lastrow-firstrow+1)/2 + 1;
              send_len   = (lastrow-firstrow+1)/2;
          }
      }

//Transpose exchange processor
      if (npcols == nprows )
          exch_proc = (me%nprows) *nprows + me/nprows;
      else
          exch_proc = 2*(((me/2)%nprows )*nprows + me/2/nprows) + me%2;

      i = npcols / 2;
      *l2npcols = 0;
      while (i>0) {
         *l2npcols = *l2npcols + 1;
         i = i / 2;
      }

//Set up the reduce phase schedules...
      div_factor = npcols;
      for (i = 1; i <= *l2npcols; i++){
         j = (proc_col+div_factor/2) % div_factor + proc_col / div_factor * div_factor;
         reduce_exch_proc[i] = proc_row*npcols + j;
         div_factor = div_factor / 2;
      }

      for (i = *l2npcols; i >=1; i--){
		if (nprows == npcols ) {
		   reduce_send_starts[i]  = send_start;
		   reduce_send_lengths[i] = send_len;
		   reduce_recv_lengths[i] = lastrow - firstrow + 1;
		}
		else {
		   reduce_recv_lengths[i] = send_len;
		   if (i == *l2npcols) {
			  reduce_send_lengths[i] = lastrow-firstrow+1 - send_len;
			  if (me/2*2 == me)
				 reduce_send_starts[i] = send_start + send_len;
			  else
				 reduce_send_starts[i] = 1;
		   }
		   else {
			  reduce_send_lengths[i] = send_len;
			  reduce_send_starts[i]  = send_start;
		   }
		}
		reduce_recv_starts[i] = send_start;
      }
      exch_recv_length = lastcol - firstcol + 1;
}


void makea(int n, int nz, double *a, int *colidx, int *rowstr, int nonzer,
						int firstrow, int lastrow, int firstcol, int lastcol,
						double rcond, int *arow, int* acol, double* aelt, double* v, int *iv, double shift)
{
	int i, nnza, iouter, ivelt, ivelt1, irow, nzv, jcol;

//nonzer is approximately  (int(sqrt(nnza /n)));

	double  size, ratio, scale;
	size = 1.0;
	ratio = pow(rcond, (1.0 / (double)(n)));
	nnza = 0;

//Initialize iv(n+1 .. 2n) to zero.
//Used by sprnvc to mark nonzero positions

	for (i=1; i<=n; i++) iv[n+i] = 0;

	for (iouter = 1; iouter <= n;iouter++){
         nzv = nonzer;
         sprnvc( n, nzv, v, colidx, (int*)&iv[1], (int*)&iv[n+1]);
         vecset( n, v, colidx, &nzv, iouter, 0.5 );

         for (ivelt = 1; ivelt <= nzv; ivelt++){
              jcol = colidx[ivelt];
              if (jcol>=firstcol && jcol<=lastcol) {
                 scale = size * v[ivelt];
                 for (ivelt1 = 1; ivelt1 <= nzv; ivelt1++) {
                    irow = colidx[ivelt1];
                    if (irow>=firstrow && irow<=lastrow) {
                       nnza = nnza + 1;
                       if (nnza > nz) {
                    	   printf("Space for matrix elements exceeded in makea\n");
                    	   printf("nnza, nzmax = %d, %d\n", nnza, nz);
                    	   printf("iouter = %d\n",iouter);
                    	   return;
                       }
                       acol[nnza] = jcol;
                       arow[nnza] = irow;
                       aelt[nnza] = v[ivelt1] * scale;
                    }
                 }
              }
         }
         size = size * ratio;
	}

//c       ... add the identity * rcond to the generated matrix to bound
//c           the smallest eigenvalue from below by rcond

	for (i = firstrow; i<=lastrow; i++){
	   if (i>=firstcol && i<=lastcol) {
		  iouter = n + i;
		  nnza = nnza + 1;
		  if (nnza > nz) {
			printf("Space for matrix elements exceeded in makea\n");
			printf("nnza, nzmax = %d, %d\n", nnza, nz);
			printf("iouter = %d\n",iouter);
			return;
		  }
		  acol[nnza] = i;
		  arow[nnza] = i;
		  aelt[nnza] = rcond - shift;
	   }
	}

//c       ... make the sparse matrix from list of elements with duplicates
//c           (v and iv are used as  workspace)
	sparse(a, colidx, rowstr, n, arow, acol, aelt,
				firstrow, lastrow,v, (int*)&iv[1], (int*)&iv[n+1], nnza);
}

void cgDouble(int _NA, int _NONZER, int _NITER, double _SHIFT, double _RCOND, double* gflops)
{
	int i, j, k, it;
    int ierr, fstatus;

    NA = _NA;
    NONZER = _NONZER;
    NITER = _NITER;
    SHIFT = _SHIFT;
    RCOND = _RCOND;

    //subroutine initialize_mpi
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    root = 0;

    NUM_PROC_COLS = NUM_PROC_ROWS = ilog2(nprocs)/2;
    if (NUM_PROC_COLS+NUM_PROC_ROWS != ilog2(nprocs)) NUM_PROC_COLS += 1;
    NUM_PROC_COLS = ipow2(NUM_PROC_COLS); NUM_PROC_ROWS = ipow2(NUM_PROC_ROWS);

//Set up mpi initialization and number of proc testing
      //call initialize_mpi

	if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10) {
	  Class = 'S';
	  zeta_verify_value = 8.5971775078648;
	} else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12) {
	  Class = 'W';
	  zeta_verify_value = 10.362595087124;
	} else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20) {
	  Class = 'A';
	  zeta_verify_value = 17.130235054029;
	} else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60) {
	  Class = 'B';
	  zeta_verify_value = 22.712745482631;
	} else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110) {
	  Class = 'C';
	  zeta_verify_value = 28.973605592845;
	} else if (NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500) {
	  Class = 'D';
	  zeta_verify_value = 52.514532105794;
	} else if (NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500) {
	  Class = 'E';
	  zeta_verify_value = 77.522164599383;
	} else {
	  Class = 'U';
	}

	dp_type = MPI_DOUBLE;

	naa = NA;
	nzz = NZ;

    setup_arrays();

//Set up processor info, such as whether sq num of procs, etc
    setup_proc_info( NUM_PROCS, NUM_PROC_ROWS, NUM_PROC_COLS);

//Set up partition's submatrix info: firstcol, lastcol, firstrow, lastrow
    setup_submatrix_info(&l2npcols,
    		  reduce_exch_proc,
    		  reduce_send_starts,
    		  reduce_send_lengths,
    		  reduce_recv_starts,
    		  reduce_recv_lengths);

    for (i = 1; i <= t_last; i++) timer_clear(i);

//Inialize random number generator
	tran    = 314159265.0;
	amult   = 1220703125.0;
	zeta    = randlc(&tran, amult);

//Set up partition's sparse random matrix for given class size
	makea(naa, nzz, a, colidx, rowstr, NONZER,
			  firstrow, lastrow, firstcol, lastcol,
			  RCOND, arow, acol, aelt, v, iv, SHIFT);

	for (j=1; j<=lastrow-firstrow+1; j++){
		for (k=rowstr[j]; k<rowstr[j+1]; k++){
			colidx[k] = colidx[k] - firstcol + 1;
		}
	}

	for (i=1; i<=NA/NUM_PROC_ROWS+1; i++){
		x[i] = 1.0;
	}
	zeta  = 0.0;

	//warming up
    for (it=1; it <= 1; it++){
    	conj_grad ( colidx, rowstr, x, z, a, p, q, r, w, &rnorm,
					 l2npcols,
					 reduce_exch_proc,
					 reduce_send_starts,
					 reduce_send_lengths,
					 reduce_recv_starts,
					 reduce_recv_lengths );

		norm_temp1[1] = 0.0;
		norm_temp1[2] = 0.0;

		for (j=1; j<=lastcol-firstcol+1; j++){
			norm_temp1[1] = norm_temp1[1] + x[j]*z[j];
			norm_temp1[2] = norm_temp1[2] + z[j]*z[j];
		}

		for (i=1; i<=l2npcols; i++){
			MPI_Irecv((double*)&norm_temp2[1], 2, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
			MPI_Send((double*)&norm_temp1[1], 2, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
			MPI_Wait(&request, &status);
			norm_temp1[1] = norm_temp1[1] + norm_temp2[1];
			norm_temp1[2] = norm_temp1[2] + norm_temp2[2];
		}
		norm_temp1[2] = 1.0 / sqrt( norm_temp1[2]);
		//Normalize z to obtain x
		for (j=1; j<=lastcol-firstcol+1; j++){
			x[j] = norm_temp1[2]*z[j];
		}
    }


//set starting vector to (1, 1, .... 1)
//NOTE: a questionable limit on size:  should this be na/num_proc_cols+1 ?

    for (i=1; i<=NA/NUM_PROC_ROWS+1; i++){
		x[i] = 1.0;
    }
	zeta  = 0.0;

//Synchronize and start timing
	for (i=1; i<=t_last; i++){
		timer_clear(i);
	}
	timer_clear(1);
	timer_start(1);

//Main Iteration for inverse power method
	for (it=1; it <= NITER; it++){
        /*
		conj_grad ( colidx, rowstr, x, z, a, p, q, r, w, &rnorm,
							 l2npcols,
							 reduce_exch_proc,
							 reduce_send_starts,
							 reduce_send_lengths,
							 reduce_recv_starts,
							 reduce_recv_lengths );
                             */
{
//Floaging point arrays here are named as in NPB1 spec discussion of CG algorithm
//  integer status(MPI_STATUS_SIZE ), request
	int i, j, k, ierr;
	int cgit, cgitmax;
	double d, sum, rho, rho0, alpha, beta;
	cgitmax = 25;

//Initialize the CG algorithm:

#pragma acc parallel loop copyin(x[0:NA/NUM_PROC_ROWS+2+1]) copyout(q[0:NA/NUM_PROC_ROWS+2+1],z[0:NA/NUM_PROC_ROWS+2+1],r[0:NA/NUM_PROC_ROWS+2+1],p[0:NA/NUM_PROC_ROWS+2+1],w[0:NA/NUM_PROC_ROWS+2+1])
	for (j=1; j<=naa/nprows+1; j++){
		q[j] = 0.0;
		z[j] = 0.0;
		r[j] = x[j];
		p[j] = r[j];
		w[j] = 0.0;
	}

//rho = r.r
//Now, obtain the norm of r: First, sum squares of r elements locally...

	sum = 0.0;
#pragma acc parallel loop copyin(r[0:NA/NUM_PROC_ROWS+2+1]) reduction(+:sum)
	for (j=1; j<=lastcol-firstcol+1; j++){
		sum = sum + r[j]*r[j];
	}

//Exchange and sum with procs identified in reduce_exch_proc
//(This is equivalent to mpi_allreduce.)
//Sum the partial sums of rho, leaving rho on all processors

      for (i=1; i<=l2npcols; i++){
         MPI_Irecv(&rho, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
         MPI_Send(&sum, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
         MPI_Wait(&request, &status);
         sum = sum + rho;
      }
      rho = sum;

//The conj grad iteration loop
      for (cgit=1; cgit <= cgitmax; cgit++){
//q = A.p
//The partition submatrix-vector multiply: use workspace w

#pragma acc parallel loop gang vector copyin(rowstr[0:NA+1+1], a[0:NZ+1], p[0:NA/NUM_PROC_ROWS+2+1], colidx[0:NZ+1]) copyout(w[0:NA/NUM_PROC_ROWS+2+1]) private(sum)
		for (j=1; j<=lastrow-firstrow+1; j++){
			sum = 0.0;
			for (k=rowstr[j]; k<=rowstr[j+1]-1; k++){
				sum = sum + a[k]*p[colidx[k]];
			}
			w[j] = sum;
		}

//Sum the partition submatrix-vec A.p's across rows
//Exchange and sum piece of w with procs identified in reduce_exch_proc

		for (i=l2npcols; i>=1; i--){
			MPI_Irecv((double*)&q[reduce_recv_starts[i]], reduce_recv_lengths[i], dp_type, reduce_exch_proc[i],
						i, MPI_COMM_WORLD, &request);
			MPI_Send((double*)&w[reduce_send_starts[i]], reduce_send_lengths[i], dp_type, reduce_exch_proc[i],
						i, MPI_COMM_WORLD);
			MPI_Wait(&request, &status);
			for (j=send_start; j<=send_start + reduce_recv_lengths[i] - 1; j++){
				w[j] = w[j] + q[j];
			}
		}

//Exchange piece of q with transpose processor:

         if(l2npcols != 0 ) {
            MPI_Irecv((double*)&q[1], exch_recv_length, dp_type, exch_proc, 1, MPI_COMM_WORLD, &request);
            MPI_Send((double*)&w[send_start], send_len, dp_type, exch_proc, 1, MPI_COMM_WORLD);
            MPI_Wait(&request, &status);
         }
         else {
#pragma acc parallel loop copyin(w[0:NA/NUM_PROC_ROWS+2+1]) copyout(q[0:NA/NUM_PROC_ROWS+2+1])
            for (j=1; j<= exch_recv_length; j++){
               q[j] = w[j];
            }
         }

//Clear w for reuse...
#pragma acc parallel loop copyout(w[0:NA/NUM_PROC_ROWS+2+1])
		 for (j=1; j<= max( lastrow-firstrow+1, lastcol-firstcol+1 ); j++){
            w[j] = 0.0;
		 }

//Obtain p.q

         sum = 0.0;
#pragma acc parallel loop copyin(p[0:NA/NUM_PROC_ROWS+2+1], q[0:NA/NUM_PROC_ROWS+2+1]) reduction(+:sum)
         for (j=1; j<= lastcol-firstcol+1; j++){
            sum = sum + p[j]*q[j];
         }

//Obtain d with a sum-reduce

         for (i=1; i<=l2npcols; i++){
            MPI_Irecv(&d, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
            MPI_Send(&sum, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
            MPI_Wait(&request, &status);
            sum = sum + d;
         }
         d = sum;

//Obtain alpha = rho / (p.q)
         alpha = rho / d;

//Save a temporary of rho
         rho0 = rho;

//Obtain z = z + alpha*p
//and    r = r - alpha*q

#pragma acc parallel loop copy(r[0:NA/NUM_PROC_ROWS+2+1], z[0:NA/NUM_PROC_ROWS+2+1]) copyin(p[0:NA/NUM_PROC_ROWS+2+1], q[0:NA/NUM_PROC_ROWS+2+1])
         for (j=1; j<=lastcol-firstcol+1; j++){
            z[j] = z[j] + alpha*p[j];
            r[j] = r[j] - alpha*q[j];
         }

//rho = r.r
//Now, obtain the norm of r: First, sum squares of r elements locally...

         sum = 0.0;
#pragma acc parallel loop copyin(r[0:NA/NUM_PROC_ROWS+2+1]) reduction(+:sum)
         for (j=1; j<=lastcol-firstcol+1; j++){
            sum = sum + r[j]*r[j];
         }

//Obtain rho with a sum-reduce
         for (i=1; i<=l2npcols; i++){
            MPI_Irecv(&rho, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
            MPI_Send(&sum, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
            MPI_Wait(&request, &status);
            sum = sum + rho;
         }
         rho = sum;

//Obtain beta:
         beta = rho / rho0;

//p = r + beta*p
#pragma acc parallel loop copyin(r[0:NA/NUM_PROC_ROWS+2+1]) copy(p[0:NA/NUM_PROC_ROWS+2+1])
         for (j=1; j<=lastcol-firstcol+1; j++){
            p[j] = r[j] + beta*p[j];
         }

      }//endif enddo                             ! end of do cgit=1,cgitmax


//Compute residual norm explicitly:  ||r|| = ||x - A.z||
//First, form A.z
//The partition submatrix-vector multiply

#pragma acc parallel loop gang vector copyin(rowstr[0:NA+1+1], a[0:NZ+1], z[0:NA/NUM_PROC_ROWS+2+1], colidx[0:NZ+1]) copyout(w[0:NA/NUM_PROC_ROWS+2+1]) private(sum)
      for (j=1; j<=lastrow-firstrow+1; j++){
         sum = 0.0;
         for (k=rowstr[j]; k<=rowstr[j+1]-1; k++){
            sum = sum + a[k]*z[colidx[k]];
         }
         w[j] = sum;
      }

//Sum the partition submatrix-vec A.z's across rows

      for (i=l2npcols; i>=1; i--){
         MPI_Irecv((double*)&r[reduce_recv_starts[i]], reduce_recv_lengths[i], dp_type, reduce_exch_proc[i],
						 i, MPI_COMM_WORLD, &request);
         MPI_Send((double*)&w[reduce_send_starts[i]], reduce_send_lengths[i], dp_type, reduce_exch_proc[i],
						 i, MPI_COMM_WORLD);
         MPI_Wait(&request, &status);

         for (j=send_start; j<=send_start + reduce_recv_lengths[i] - 1; j++){
            w[j] = w[j] + r[j];
         }
      }

//Exchange piece of q with transpose processor:
      if( l2npcols != 0 ) {
         MPI_Irecv((double*)&r[1], exch_recv_length, dp_type, exch_proc, 1, MPI_COMM_WORLD, &request);
         MPI_Send((double*)&w[send_start], send_len, dp_type, exch_proc, 1, MPI_COMM_WORLD);
         MPI_Wait(&request, &status);
      }
      else {
#pragma acc parallel loop copyin(w[0:NA/NUM_PROC_ROWS+2+1]) copyout(r[0:NA/NUM_PROC_ROWS+2+1])
         for (j=1; j<=exch_recv_length; j++)
            r[j] = w[j];
      }

//At this point, r contains A.z
         sum = 0.0;
#pragma acc parallel loop copyin(x[0:NA/NUM_PROC_ROWS+2+1],r[0:NA/NUM_PROC_ROWS+2+1]) reduction(+:sum)
         for (j=1; j<=lastcol-firstcol+1; j++){
            d = x[j] - r[j];
            sum = sum + d*d;
         }

//Obtain d with a sum-reduce
      for (i=1; i<=l2npcols; i++){
         MPI_Irecv(&d, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
         MPI_Send(&sum, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
         MPI_Wait(&request, &status);
         sum = sum + d;
      }
      d = sum;

      if( me == root ) rnorm = sqrt( d );
}

		norm_temp1[1] = 0.0;
		norm_temp1[2] = 0.0;
		for (j=1; j<=lastcol-firstcol+1; j++){
			norm_temp1[1] = norm_temp1[1] + x[j]*z[j];
			norm_temp1[2] = norm_temp1[2] + z[j]*z[j];
		}
		for (i=1; i<=l2npcols; i++){
            MPI_Irecv((double*)&norm_temp2[1], 2, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
            MPI_Send((double*)&norm_temp1[1], 2, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
            MPI_Wait(&request, &status);
            norm_temp1[1] = norm_temp1[1] + norm_temp2[1];
            norm_temp1[2] = norm_temp1[2] + norm_temp2[2];
		}
		norm_temp1[2] = 1.0 / sqrt( norm_temp1[2]);

		if(me == root){
			zeta = SHIFT + 1.0 / norm_temp1[1];
			if (it == 1)
				  printf("\n   iteration           ||r||                 zeta\n");
			printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);
		}
//  Normalize z to obtain x
		for (j=1; j<=lastcol-firstcol+1; j++){
			x[j] = norm_temp1[2]*z[j];
		}
	}//end of main iteration

	timer_stop(1);

//End of timed section
	t = timer_read(1);
	MPI_Reduce(&t, &tmax, 1, dp_type, MPI_MAX,root, MPI_COMM_WORLD);

      if( me == root ){
    	  printf(" Benchmark completed\n");
    	  epsilon = 1.0e-10;
    	  if (Class != 'U') {
    	    err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
    	    if (err <= epsilon) {
    	      verified = 1;
    	      printf(" VERIFICATION SUCCESSFUL\n");
    	      printf(" Zeta is    %20.13E\n", zeta);
    	      printf(" Error is   %20.13E\n", err);
    	    } else {
    	      verified = 0;
    	      printf(" VERIFICATION FAILED\n");
    	      printf(" Zeta                %20.13E\n", zeta);
    	      printf(" The correct zeta is %20.13E\n", zeta_verify_value);
    	    }
    	  } else {
    	    verified = 0;
    	    printf(" Problem size unknown\n");
    	    printf(" NO VERIFICATION PERFORMED\n");
    	  }

         if( tmax != 0.0 ) {
            mflops = (double)( 2*NITER*NA )
							* ( 3.+(double)( NONZER*(NONZER+1) )
							+ 25.*(5.+(double)( NONZER*(NONZER+1) ))
							+ 3. ) / tmax / 1000000.0;
         }else{
            mflops = 0.0;
         }

         c_print_results("CG", Class, NA, 0, 0,
							 NITER, NNODES_COMPILED, nprocs, tmax,
							 mflops, "          floating point",
							 verified, NPBVERSION);
         *gflops = mflops;

      }

}//END MAIN


void conj_grad (int *colidx, int* rowstr, double *x, double *z, double *a, double *p, double *q, double *r,
    double *w, double *rnorm, int l2npcols,
    int *reduce_exch_proc,
    int *reduce_send_starts,
    int *reduce_send_lengths,
    int *reduce_recv_starts,
    int *reduce_recv_lengths )
{
//Floaging point arrays here are named as in NPB1 spec discussion of CG algorithm
//  integer status(MPI_STATUS_SIZE ), request
	int i, j, k, ierr;
	int cgit, cgitmax;
	double d, sum, rho, rho0, alpha, beta;
	cgitmax = 25;

//Initialize the CG algorithm:

	for (j=1; j<=naa/nprows+1; j++){
		q[j] = 0.0;
		z[j] = 0.0;
		r[j] = x[j];
		p[j] = r[j];
		w[j] = 0.0;
	}

//rho = r.r
//Now, obtain the norm of r: First, sum squares of r elements locally...

	sum = 0.0;
	for (j=1; j<=lastcol-firstcol+1; j++){
		sum = sum + r[j]*r[j];
	}

//Exchange and sum with procs identified in reduce_exch_proc
//(This is equivalent to mpi_allreduce.)
//Sum the partial sums of rho, leaving rho on all processors

      for (i=1; i<=l2npcols; i++){
         MPI_Irecv(&rho, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
         MPI_Send(&sum, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
         MPI_Wait(&request, &status);
         sum = sum + rho;
      }
      rho = sum;

//The conj grad iteration loop
      for (cgit=1; cgit <= cgitmax; cgit++){
//q = A.p
//The partition submatrix-vector multiply: use workspace w


		for (j=1; j<=lastrow-firstrow+1; j++){
			sum = 0.0;
			for (k=rowstr[j]; k<=rowstr[j+1]-1; k++){
				sum = sum + a[k]*p[colidx[k]];
			}
			w[j] = sum;
		}

//Sum the partition submatrix-vec A.p's across rows
//Exchange and sum piece of w with procs identified in reduce_exch_proc


		for (i=l2npcols; i>=1; i--){
			MPI_Irecv((double*)&q[reduce_recv_starts[i]], reduce_recv_lengths[i], dp_type, reduce_exch_proc[i],
						i, MPI_COMM_WORLD, &request);
			MPI_Send((double*)&w[reduce_send_starts[i]], reduce_send_lengths[i], dp_type, reduce_exch_proc[i],
						i, MPI_COMM_WORLD);
			MPI_Wait(&request, &status);
			for (j=send_start; j<=send_start + reduce_recv_lengths[i] - 1; j++){
				w[j] = w[j] + q[j];
			}
		}

//Exchange piece of q with transpose processor:

         if(l2npcols != 0 ) {
            MPI_Irecv((double*)&q[1], exch_recv_length, dp_type, exch_proc, 1, MPI_COMM_WORLD, &request);
            MPI_Send((double*)&w[send_start], send_len, dp_type, exch_proc, 1, MPI_COMM_WORLD);
            MPI_Wait(&request, &status);
         }
         else {
            for (j=1; j<= exch_recv_length; j++){
               q[j] = w[j];
            }
         }

//Clear w for reuse...
		 for (j=1; j<= max( lastrow-firstrow+1, lastcol-firstcol+1 ); j++){
            w[j] = 0.0;
		 }

//Obtain p.q

         sum = 0.0;
         for (j=1; j<= lastcol-firstcol+1; j++){
            sum = sum + p[j]*q[j];
         }

//Obtain d with a sum-reduce

         for (i=1; i<=l2npcols; i++){
            MPI_Irecv(&d, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
            MPI_Send(&sum, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
            MPI_Wait(&request, &status);
            sum = sum + d;
         }
         d = sum;

//Obtain alpha = rho / (p.q)
         alpha = rho / d;

//Save a temporary of rho
         rho0 = rho;

//Obtain z = z + alpha*p
//and    r = r - alpha*q

         for (j=1; j<=lastcol-firstcol+1; j++){
            z[j] = z[j] + alpha*p[j];
            r[j] = r[j] - alpha*q[j];
         }

//rho = r.r
//Now, obtain the norm of r: First, sum squares of r elements locally...

         sum = 0.0;
         for (j=1; j<=lastcol-firstcol+1; j++){
            sum = sum + r[j]*r[j];
         }

//Obtain rho with a sum-reduce
         for (i=1; i<=l2npcols; i++){
            MPI_Irecv(&rho, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
            MPI_Send(&sum, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
            MPI_Wait(&request, &status);
            sum = sum + rho;
         }
         rho = sum;

//Obtain beta:
         beta = rho / rho0;

//p = r + beta*p
         for (j=1; j<=lastcol-firstcol+1; j++){
            p[j] = r[j] + beta*p[j];
         }

      }//endif enddo                             ! end of do cgit=1,cgitmax


//Compute residual norm explicitly:  ||r|| = ||x - A.z||
//First, form A.z
//The partition submatrix-vector multiply

      for (j=1; j<=lastrow-firstrow+1; j++){
         sum = 0.0;
         for (k=rowstr[j]; k<=rowstr[j+1]-1; k++){
            sum = sum + a[k]*z[colidx[k]];
         }
         w[j] = sum;
      }

//Sum the partition submatrix-vec A.z's across rows

      for (i=l2npcols; i>=1; i--){
         MPI_Irecv((double*)&r[reduce_recv_starts[i]], reduce_recv_lengths[i], dp_type, reduce_exch_proc[i],
						 i, MPI_COMM_WORLD, &request);
         MPI_Send((double*)&w[reduce_send_starts[i]], reduce_send_lengths[i], dp_type, reduce_exch_proc[i],
						 i, MPI_COMM_WORLD);
         MPI_Wait(&request, &status);

         for (j=send_start; j<=send_start + reduce_recv_lengths[i] - 1; j++){
            w[j] = w[j] + r[j];
         }
      }

//Exchange piece of q with transpose processor:
      if( l2npcols != 0 ) {
         MPI_Irecv((double*)&r[1], exch_recv_length, dp_type, exch_proc, 1, MPI_COMM_WORLD, &request);
         MPI_Send((double*)&w[send_start], send_len, dp_type, exch_proc, 1, MPI_COMM_WORLD);
         MPI_Wait(&request, &status);
      }
      else {
         for (j=1; j<=exch_recv_length; j++)
            r[j] = w[j];
      }

//At this point, r contains A.z
         sum = 0.0;
         for (j=1; j<=lastcol-firstcol+1; j++){
            d = x[j] - r[j];
            sum = sum + d*d;
         }

//Obtain d with a sum-reduce
      for (i=1; i<=l2npcols; i++){
         MPI_Irecv(&d, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD, &request);
         MPI_Send(&sum, 1, dp_type, reduce_exch_proc[i], i, MPI_COMM_WORLD);
         MPI_Wait(&request, &status);
         sum = sum + d;
      }
      d = sum;

      if( me == root ) *rnorm = sqrt( d );

}


void sparse(double *a, int *colidx, int *rowstr, int n, int *arow, int *acol, double *aelt,
				int firstrow, int lastrow,
				double *x, int *mark, int *nzloc, int nnza )
{
//rows range from firstrow to lastrow
//the rowstr pointers are defined for nrows = lastrow-firstrow+1 values

      int nrows;
//generate a sparse matrix from a list of
//[col, row, element] tri

      int i, j, jajp1, nza, k, nzrow;
      double xi;

      nrows = lastrow - firstrow + 1;

      for (j=1; j<=n; j++){
         rowstr[j] = 0;
         mark[j] = 0;
      }
      rowstr[n+1] = 0;

      for (nza=1; nza<=nnza; nza++){
         j = (arow[nza] - firstrow + 1) + 1;
         rowstr[j] = rowstr[j] + 1;
      }

      rowstr[1] = 1;
      for (j=2; j<=nrows+1; j++)
         rowstr[j] = rowstr[j] + rowstr[j-1];

//do a bucket sort of the triples on the row index

      for (nza=1; nza<=nnza; nza++){
         j = arow[nza] - firstrow + 1;
         k = rowstr[j];
         a[k] = aelt[nza];
         colidx[k] = acol[nza];
         rowstr[j] = rowstr[j] + 1;
      }

//rowstr(j) now points to the first element of row j+1

      for (j=nrows; j>=1; j--){
          rowstr[j+1] = rowstr[j];
      }
      rowstr[1] = 1;

//generate the actual output rows by adding elements

      nza = 0;
      for (i=1; i<=n; i++){
          x[i]    = 0.0;
          mark[i] = 0;
      }

      jajp1 = rowstr[1];
      for (j=1; j<=nrows; j++) {
         nzrow = 0;

//...loop over the jth row of a
         for (k=jajp1; k<=rowstr[j+1]-1; k++){
            i = colidx[k];
            x[i] = x[i] + a[k];
            if ((!mark[i]) && (x[i] != 0.0)) {
             mark[i] = 1;
             nzrow = nzrow + 1;
             nzloc[nzrow] = i;
            }
         }

//extract the nonzeros of this row
         for (k=1; k <= nzrow; k++){
            i = nzloc[k];
            mark[i] = 0;
            xi = x[i];
            x[i] = 0.0;
            if (xi != 0.0) {
             nza = nza + 1;
             a[nza] = xi;
             colidx[nza] = i;
            }
         }
         jajp1 = rowstr[j+1];
         rowstr[j+1] = nza + rowstr[1];
      } //end for
}

/*
 *
	generate a sparse n-vector (v, iv)
	having nzv nonzeros

	mark(i) is set to 1 if position i is nonzero.
	mark is all zero on entry and is reset to all zero before exit
	this corrects a performance bug found by John G. Lewis, caused by
	reinitialization of mark on every one of the n calls to sprnvc
 *
 */

int icnvrt(double x, int ipwr2)
{
	return (int)(ipwr2 * x);
}

void sprnvc(int n, int nz, double *v, int* iv, int* nzloc, int* mark )
{
    int nn1;
    int	nzrow, nzv, ii, i;
    double vecelt, vecloc;

    nzv = 0;
    nzrow = 0;
    nn1 = 1;

    while (nn1 < n){
    	nn1 = 2*nn1;
    }

	while (nzv < nz){
		vecelt = randlc(&tran, amult);
		vecloc = randlc(&tran, amult);

		i = icnvrt(vecloc, nn1) + 1;
		if (i > n) continue;
		if (mark[i] == 0) {
			mark[i] = 1;
			nzrow = nzrow + 1;
			nzloc[nzrow] = i;
			nzv = nzv + 1;
			v[nzv] = vecelt;
			iv[nzv] = i;
		}
	}//end while
	for (ii=1; ii<=nzrow; ii++){
		i = nzloc[ii];
		mark[i] = 0;
	}
}

void vecset(int n, double *v, int *iv, int *nzv, int i, double val)
{
      int set = 0;
      int k;
      for (k=1; k<=*nzv; k++){
         if (iv[k] == i) {
            v[k] = val;
            set  = 1;
         }
      }
      if (!set) {
         *nzv     = *nzv + 1;
         v[*nzv]  = val;
         iv[*nzv] = i;
      }
}

