//---------------------------------------------------------------------
//      program EMBAR
//---------------------------------------------------------------------
//
//   This is the MPI version of the APP Benchmark 1,
//   the "embarassingly parallel" benchmark.
//
//
//   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
//   numbers.  MK is the Log_2 of the size of each batch of uniform random
//   numbers.  MK can be set for convenience on a given system, since it does
//   not affect the results.

#define NPBVERSION "3.3.1"
#define GPUS_PER_NODE   1
//#define MAX_GANG        32768
#define MAX_GANG        32

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "randdp.h"
#include "timers.h"
#include "print_results.h"
#include <openacc.h>
#include "mpinpb.h"

//#define MK        16
#define MK        12
/*
#define MM        (M - MK)
#define NN        (1 << MM)
*/
#define NK        (1 << MK)
#define NQ        10
#define EPSILON   1.0e-8
#define A         1220703125.0
#define S         271828183.0

#define t_total   0
#define t_gpairs  1
#define t_randn   2
#define t_rcomm   3
#define t_last    4

#define max(x,y)    ((x) > (y) ? (x) : (y))

/* common/storage/ */
static double x[2*NK];
static double q[NQ];
static double q0, q1, q2, q3, q4, q5, q6, q7, q8, q9;

int epDouble(const char* CLASSX, int M, double* gflops)
{
  double Mops, t1, t2, t3, t4, x1, x2;
  double  sx, sy, tm, an, tt, gc, dum[3];
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  int np, node, no_nodes; 
  int i, ik, kk, l, k, nit, ierrcode, no_large_nodes;
  int _k, _np, _nnp;
  int np_add, k_offset, j;
  int verified;
  char size[16];

  int MM = (M - MK);
  int NN = (1 << MM);

  int acc_no_devs = acc_get_num_devices(acc_device_default);
  acc_set_device_num(acc_no_devs % GPUS_PER_NODE, acc_device_default);

  MPI_Comm_rank(MPI_COMM_WORLD, &node);
  MPI_Comm_size(MPI_COMM_WORLD, &no_nodes);

  root = 0;

  dp_type = MPI_DOUBLE;

  if (node == root)  {

    // Because the size of the problem is too large to store in a 32-bit
    // integer for some classes, we put it into a string (for printing).
    // Have to strip off the decimal point put in there by the floating
    // point print statement (internal file)
    printf("\n NAS Parallel Benchmarks (NPB3.3-MPI-C) - EP Benchmark\n\n");
    sprintf(size, "%15.0lf", pow(2.0, M+1));
    j = 14;
    if (size[j] == '.') j = j - 1;
    size[j+1] = '\0';
    printf(" Number of random numbers generated: %15s\n", size);
    printf(" Number of active processes:           %13d\n", no_nodes);
  }

  verified = 0;

  // Compute the number of "batches" of random number pairs generated 
  // per processor. Adjust if the number of processors does not evenly 
  // divide the total number

  np = NN / no_nodes;
  no_large_nodes = NN % no_nodes;
  if (node < no_large_nodes) {
    np_add = 1;
  } else {
    np_add = 0;
  }
  np = np + np_add;

  if (np == 0) {
    fprintf(stderr, "Too many nodes:%6d%6d\n", no_nodes, NN);
    ierrcode = 1;
    MPI_Abort(MPI_COMM_WORLD, ierrcode);
    exit(EXIT_FAILURE);
  }

  // Call the random number generator functions and initialize
  // the x-array to reduce the effects of paging on the timings.
  // Also, all mathematical functions that are used. Make
  // sure these initializations cannot be eliminated as dead code.

  vranlc(0, &dum[0], dum[1], &dum[2]);
  dum[0] = randlc(&dum[1], dum[2]);
  for (i = 0; i < 2*NK; i++) {
    x[i] = -1.0e99;
  }
  Mops = log(sqrt(fabs(max(1.0, 1.0))));

  //---------------------------------------------------------------------
  // Synchronize before placing time stamp
  //---------------------------------------------------------------------
  for (i = 0; i < t_last; i++) {
    timer_clear(i);
  }
  timer_start(t_total);

  t1 = A;
  vranlc(0, &t1, A, x);

  // Compute AN = A ^ (2 * NK) (mod 2^46).

  t1 = A;

  for (i = 0; i < MK + 1; i++) {
    t2 = randlc(&t1, t1);
  }

  an = t1;
  tt = S;
  gc = 0.0;
  sx = 0.0;
  sy = 0.0;

  for (i = 0; i < NQ; i++) {
    q[i] = 0.0;
  }
  q0 = q1 = q2 = q3 = q4 = q5 = q6 = q7 = q8 = q9 = 0.0;

  // Each instance of this loop may be performed independently. We compute
  // the k offsets separately to take into account the fact that some nodes
  // have more numbers to generate than others

  if (np_add == 1) {
    k_offset = node * np -1;
  } else {
    k_offset = no_large_nodes*(np+1) + (node-no_large_nodes)*np -1;
  }

  _np = np;
  while (_np > MAX_GANG * 128) {
    _np >>= 1;
  }
  _nnp = np / _np;
  if (np % _np != 0) printf("Unrolling failed! np(%d) % _np(%d) != 0\n", np, _np);

#pragma acc parallel loop gang vector private(x, t1, t2, t3, t4, x1, x2, kk, i, ik, l, k, k_offset) reduction(+:sx, sy)
  for (_k = 0; _k < _np; _k++)
  for (k = 0; k < _nnp; k++) {
    kk = k_offset + k + 1 + (_k * _nnp);
    t1 = S;
    t2 = an;

    // Find starting seed t1 for this kk.

    for (i = 1; i <= 100; i++) {
      ik = kk / 2;
      if (2 * ik != kk) {
          //t3 = randlc(&t1, t2);
          const double _r23 = 1.1920928955078125e-07;
          const double _r46 = _r23 * _r23;
          const double _t23 = 8.388608e+06;
          const double _t46 = _t23 * _t23;

          double _t1, _t2, _t3, _t4, _a1, _a2, _x1, _x2, _z;
          double _r;

          _t1 = _r23 * t2;
          _a1 = (int) _t1;
          _a2 = t2 - _t23 * _a1;

          _t1 = _r23 * t1;
          _x1 = (int) _t1;
          _x2 = t1 - _t23 * _x1;
          _t1 = _a1 * _x2 + _a2 * _x1;
          _t2 = (int) (_r23 * _t1);
          _z = _t1 - _t23 * _t2;
          _t3 = _t23 * _z + _a2 * _x2;
          _t4 = (int) (_r46 * _t3);
          t1= _t3 - _t46 * _t4;
          _r = _r46 * t1;

          t3 = _r;
      }
      if (ik == 0) break;
      {
          //t3 = randlc(&t2, t2);
          const double _r23 = 1.1920928955078125e-07;
          const double _r46 = _r23 * _r23;
          const double _t23 = 8.388608e+06;
          const double _t46 = _t23 * _t23;

          double _t1, _t2, _t3, _t4, _a1, _a2, _x1, _x2, _z;
          double _r;

          _t1 = _r23 * t2;
          _a1 = (int) _t1;
          _a2 = t2 - _t23 * _a1;

          _t1 = _r23 * t2;
          _x1 = (int) _t1;
          _x2 = t2 - _t23 * _x1;
          _t1 = _a1 * _x2 + _a2 * _x1;
          _t2 = (int) (_r23 * _t1);
          _z = _t1 - _t23 * _t2;
          _t3 = _t23 * _z + _a2 * _x2;
          _t4 = (int) (_r46 * _t3);
          t2 = _t3 - _t46 * _t4;
          _r = _r46 * t2;

          t3 = _r;
      }
      kk = ik;
    }

    // Compute uniform pseudorandom numbers.
    {
      //vranlc(2 * NK, &t1, A, x);
      const double _r23 = 1.1920928955078125e-07;
      const double _r46 = _r23 * _r23;
      const double _t23 = 8.388608e+06;
      const double _t46 = _t23 * _t23;

      double _t1, _t2, _t3, _t4, _a1, _a2, _x1, _x2, _z;

      int _i;

      _t1 = _r23 * A;
      _a1 = (int) _t1;
      _a2 = A - _t23 * _a1;

      for ( _i = 0; _i < 2 * NK; _i++ ) {
        _t1 = _r23 * t1;
        _x1 = (int) _t1;
        _x2 = t1 - _t23 * _x1;
        _t1 = _a1 * _x2 + _a2 * _x1;
        _t2 = (int) (_r23 * _t1);
        _z = _t1 - _t23 * _t2;
        _t3 = _t23 * _z + _a2 * _x2;
        _t4 = (int) (_r46 * _t3) ;
        t1 = _t3 - _t46 * _t4;
        x[_i] = _r46 * t1;
      }
    }

    // Compute Gaussian deviates by acceptance-rejection method and 
    // tally counts in concentric square annuli.  This loop is not 
    // vectorizable. 
    for (i = 0; i < NK; i++) {
      x1 = 2.0 * x[2*i] - 1.0;
      x2 = 2.0 * x[2*i+1] - 1.0;
      t1 = x1 * x1 + x2 * x2;
      if (t1 <= 1.0) {
        t2   = sqrt(-2.0 * log(t1) / t1);
        t3   = (x1 * t2);
        t4   = (x2 * t2);
        l    = max(fabs(t3), fabs(t4));
        //q[l] = q[l] + 1.0;
        if (l == 0) q0 += 1.0;
        else if (l == 1) q1 += 1.0;
        else if (l == 2) q2 += 1.0;
        else if (l == 3) q3 += 1.0;
        else if (l == 4) q4 += 1.0;
        else if (l == 5) q5 += 1.0;
        else if (l == 6) q6 += 1.0;
        else if (l == 7) q7 += 1.0;
        else if (l == 8) q8 += 1.0;
        else q9 += 1.0;
        sx   = sx + t3;
        sy   = sy + t4;
      }
    }

  }

  MPI_Allreduce(&sx, x, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  sx = x[0];
  MPI_Allreduce(&sy, x, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  sy = x[0];

  /*
  MPI_Allreduce(q, x, NQ, dp_type, MPI_SUM, MPI_COMM_WORLD);
  for (i = 0; i < NQ; i++) {
    q[i] = x[i];
  }
  */

  MPI_Allreduce(&q0, q + 0, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q1, q + 1, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q2, q + 2, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q3, q + 3, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q4, q + 4, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q5, q + 5, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q6, q + 6, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q7, q + 7, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q8, q + 8, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q9, q + 9, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);

  for (i = 0; i < NQ; i++) {
    gc = gc + q[i];
  }

  timer_stop(t_total);
  tm = timer_read(t_total);

  MPI_Allreduce(&tm, x, 1, dp_type, MPI_MAX, MPI_COMM_WORLD);
  tm = x[0];

  if (node == root) {
    nit = 0;
    verified = 1;
    if (M == 24) {
      sx_verify_value = -3.247834652034740e+3;
      sy_verify_value = -6.958407078382297e+3;
    } else if (M == 25) {
      sx_verify_value = -2.863319731645753e+3;
      sy_verify_value = -6.320053679109499e+3;
    } else if (M == 28) {
      sx_verify_value = -4.295875165629892e+3;
      sy_verify_value = -1.580732573678431e+4;
    } else if (M == 30) {
      sx_verify_value =  4.033815542441498e+4;
      sy_verify_value = -2.660669192809235e+4;
    } else if (M == 32) {
      sx_verify_value =  4.764367927995374e+4;
      sy_verify_value = -8.084072988043731e+4;
    } else if (M == 36) {
      sx_verify_value =  1.982481200946593e+5;
      sy_verify_value = -1.020596636361769e+5;
    } else if (M == 40) {
      sx_verify_value = -5.319717441530e+05;
      sy_verify_value = -3.688834557731e+05;
    } else {
      verified = 0;
    }
    if (verified) {
      sx_err = fabs((sx - sx_verify_value)/sx_verify_value);
      sy_err = fabs((sy - sy_verify_value)/sy_verify_value);
      verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
    }
    Mops = pow(2.0, M+1)/tm/1000000.0;

    printf("EP Benchmark Results:\n\n"
        "CPU Time =%10.4lf\n"
        "N = 2^%5d\n"
        "No. Gaussian Pairs =%14.0lf\n"
        "Sums = %25.15lE%25.15lE\n"
        "Counts:\n",
        tm, M, gc, sx, sy);
    for (i = 0; i < NQ; i++) {
      printf("%3d%14.0lf\n", i, q[i]);
    }

    c_print_results("EP", CLASSX[0], M+1, 0, 0, nit, no_nodes, 
        no_nodes, tm, Mops, 
        "Random numbers generated", 
        verified, NPBVERSION);

  }
  *gflops = Mops;

  return 0;
}

