#ifndef GPU_GLOBAL_H
#define GPU_GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "cudacommon.h"

#define BLOCK_SIZE   64
#define BLOCK_SIZE2  (2*BLOCK_SIZE)

//alleviate aliasing issues
#define RESTRICT __restrict__

//replace divisions by multiplication with the reciprocal
#define REPLACE_DIV_WITH_RCP 1

#if REPLACE_DIV_WITH_RCP
template <class T1, class T2>
__device__ __forceinline__ T1 DIV(T1 x, T2 y)
{
   return x * (1.0f / y);
}
#else
template <class T1, class T2>
__device__ __forceinline__ T1 DIV(T1 x, T2 y)
{
   return x / y;
}
#endif

// Choose correct intrinsics based on precision
// POW
template<class T>
__device__ __forceinline__ T POW (T in, T in2);

template<>
__device__ __forceinline__ double POW<double>(double in, double in2)
{
    return pow(in, in2);
}

template<>
__device__ __forceinline__ float POW<float>(float in, float in2)
{
    return powf(in, in2);
}
// EXP
template<class T>
__device__ __forceinline__ T EXP(T in);

template<>
__device__ __forceinline__ double EXP<double>(double in)
{
    return exp(in);
}

template<>
__device__ __forceinline__ float EXP<float>(float in)
{
    return expf(in);
}

// EXP10
template<class T>
__device__ __forceinline__ T EXP10(T in);

template<>
__device__ __forceinline__ double EXP10<double>(double in)
{
    return exp10(in);
}

template<>
__device__ __forceinline__ float EXP10<float>(float in)
{
    return exp10f(in);
}

// EXP2
template<class T>
__device__ __forceinline__ T EXP2(T in);

template<>
__device__ __forceinline__ double EXP2<double>(double in)
{
    return exp2(in);
}

template<>
__device__ __forceinline__ float EXP2<float>(float in)
{
    return exp2f(in);
}

// FMAX
template<class T>
__device__ __forceinline__ T MAX(T in, T in2);

template<>
__device__ __forceinline__ double MAX<double>(double in, double in2)
{
    return fmax(in, in2);
}

template<>
__device__ __forceinline__ float MAX<float>(float in, float in2)
{
    return fmaxf(in, in2);
}

// FMIN
template<class T>
__device__ __forceinline__ T MIN(T in, T in2);

template<>
__device__ __forceinline__ double MIN<double>(double in, double in2)
{
    return fmin(in, in2);
}

template<>
__device__ __forceinline__ float MIN<float>(float in, float in2)
{
    return fminf(in, in2);
}

// LOG
template<class T>
__device__ __forceinline__ T LOG(T in);

template<>
__device__ __forceinline__ double LOG<double>(double in)
{
    return log(in);
}

template<>
__device__ __forceinline__ float LOG<float>(float in)
{
    return logf(in);
}

// LOG10
template<class T>
__device__ __forceinline__ T LOG10(T in);

template<>
__device__ __forceinline__ double LOG10<double>(double in)
{
    return log10(in);
}

template<>
__device__ __forceinline__ float LOG10<float>(float in)
{
    return log10f(in);
}

//Kernel indexing macros
#define N_GP (blockDim.x * gridDim.x) // number of grid points
#define thread_num (threadIdx.x + (blockIdx.x * blockDim.x))
#define idx2(p,z) (p[(((z)-1)*(N_GP)) + thread_num])
#define idx(x, y) ((x)[(y)-1])

#define C(q)     idx2(C, q)
#define Y(q)     idx2(Y, q)
#define RF(q)    idx2(RF, q)
#define EG(q)    idx2(EG, q)
#define RB(q)    idx2(RB, q)
#define RKLOW(q) idx2(RKLOW, q)
#define ROP(q)   idx(ROP, q)
#define WDOT(q)  idx2(WDOT, q)
#define RKF(q)   idx2(RKF, q)
#define RKR(q)   idx2(RKR, q)
#define A_DIM    (11)
#define A(b, c)  idx2(A, (((b)*A_DIM)+c) )

// Size macros
// This is the number of floats/doubles per thread for each var

#define C_SIZE               (22)
#define RF_SIZE             (206)
#define RB_SIZE             (206)
#define WDOT_SIZE            (22)
#define RKLOW_SIZE           (21)
#define Y_SIZE               (22)
#define A_SIZE    (A_DIM * A_DIM)
#define EG_SIZE              (32)

// Launch Bounds Macros

//#if defined(USE_LAUNCH_BOUNDS)
//
//#if __CUDA_ARCH__ >= 200  // Fermi -- GF100
//
//#define GR_BASE_THRD    (BLOCK_SIZE2)
//#define GR_BASE_BLK     (4)
//#define QSSA2_THRD      (BLOCK_SIZE2)
//#define QSSA2_BLK       (7)
//#define QSSAB_THRD      (BLOCK_SIZE)
//#define QSSAB_BLK       (8)
//#define QSSA_THRD       (BLOCK_SIZE2)
//#define QSSA_BLK        (4)
//#define RATT10_THRD     (BLOCK_SIZE2)
//#define RATT10_BLK      (8)
//#define RATT2_THRD      (BLOCK_SIZE2)
//#define RATT2_BLK       (5)
//#define RATT3_THRD      (BLOCK_SIZE2)
//#define RATT3_BLK       (4)
//#define RATT4_THRD      (BLOCK_SIZE2)
//#define RATT4_BLK       (4)
//#define RATT5_THRD      (BLOCK_SIZE2)
//#define RATT5_BLK       (4)
//#define RATT6_THRD      (BLOCK_SIZE2)
//#define RATT6_BLK       (5)
//#define RATT7_THRD      (BLOCK_SIZE2)
//#define RATT7_BLK       (5)
//#define RATT8_THRD      (BLOCK_SIZE2)
//#define RATT8_BLK       (5)
//#define RATT9_THRD      (BLOCK_SIZE2)
//#define RATT9_BLK       (4)
//#define RATT_THRD       (BLOCK_SIZE2)
//#define RATT_BLK        (8)
//#define RATX2_THRD      (BLOCK_SIZE2)
//#define RATX2_BLK       (6)
//#define RATX4_THRD      (BLOCK_SIZE2)
//#define RATX4_BLK       (5)
//#define RATXB_THRD      (BLOCK_SIZE)
//#define RATXB_BLK       (8)
//#define RATX_THRD       (BLOCK_SIZE)
//#define RATX_BLK        (8)
//#define RDSMH_THRD      (BLOCK_SIZE2)
//#define RDSMH_BLK       (4)
//#define RDWDOT10_THRD   (BLOCK_SIZE2)
//#define RDWDOT10_BLK    (5)
//#define RDWDOT2_THRD    (BLOCK_SIZE2)
//#define RDWDOT2_BLK     (7)
//#define RDWDOT3_THRD    (BLOCK_SIZE2)
//#define RDWDOT3_BLK     (8)
//#define RDWDOT6_THRD    (BLOCK_SIZE2)
//#define RDWDOT6_BLK     (8)
//#define RDWDOT7_THRD    (BLOCK_SIZE2)
//#define RDWDOT7_BLK     (5)
//#define RDWDOT8_THRD    (BLOCK_SIZE2)
//#define RDWDOT8_BLK     (7)
//#define RDWDOT9_THRD    (BLOCK_SIZE2)
//#define RDWDOT9_BLK     (8)
//#define RDWDOT_THRD     (BLOCK_SIZE2)
//#define RDWDOT_BLK      (8)
//
//#else // Tesla -- GT200
//
//#define GR_BASE_THRD    (BLOCK_SIZE2)
//#define GR_BASE_BLK     (3)
//#define QSSA2_THRD      (BLOCK_SIZE2)
//#define QSSA2_BLK       (2)
//#define QSSAB_THRD      (BLOCK_SIZE)
//#define QSSAB_BLK       (3)
//#define QSSA_THRD       (BLOCK_SIZE2)
//#define QSSA_BLK        (2)
//#define RATT10_THRD     (BLOCK_SIZE2)
//#define RATT10_BLK      (6)
//#define RATT2_THRD      (BLOCK_SIZE2)
//#define RATT2_BLK       (2)
//#define RATT3_THRD      (BLOCK_SIZE2)
//#define RATT3_BLK       (2)
//#define RATT4_THRD      (BLOCK_SIZE2)
//#define RATT4_BLK       (2)
//#define RATT5_THRD      (BLOCK_SIZE2)
//#define RATT5_BLK       (2)
//#define RATT6_THRD      (BLOCK_SIZE2)
//#define RATT6_BLK       (2)
//#define RATT7_THRD      (BLOCK_SIZE2)
//#define RATT7_BLK       (3)
//#define RATT8_THRD      (BLOCK_SIZE2)
//#define RATT8_BLK       (3)
//#define RATT9_THRD      (BLOCK_SIZE2)
//#define RATT9_BLK       (3)
//#define RATT_THRD       (BLOCK_SIZE2)
//#define RATT_BLK        (6)
//#define RATX2_THRD      (BLOCK_SIZE2)
//#define RATX2_BLK       (4)
//#define RATX4_THRD      (BLOCK_SIZE2)
//#define RATX4_BLK       (3)
//#define RATXB_THRD      (BLOCK_SIZE)
//#define RATXB_BLK       (6)
//#define RATX_THRD       (BLOCK_SIZE)
//#define RATX_BLK        (6)
//#define RDSMH_THRD      (BLOCK_SIZE2)
//#define RDSMH_BLK       (3)
//#define RDWDOT10_THRD   (BLOCK_SIZE2)
//#define RDWDOT10_BLK    (5)
//#define RDWDOT2_THRD    (BLOCK_SIZE2)
//#define RDWDOT2_BLK     (4)
//#define RDWDOT3_THRD    (BLOCK_SIZE2)
//#define RDWDOT3_BLK     (5)
//#define RDWDOT6_THRD    (BLOCK_SIZE2)
//#define RDWDOT6_BLK     (4)
//#define RDWDOT7_THRD    (BLOCK_SIZE2)
//#define RDWDOT7_BLK     (2)
//#define RDWDOT8_THRD    (BLOCK_SIZE2)
//#define RDWDOT8_BLK     (3)
//#define RDWDOT9_THRD    (BLOCK_SIZE2)
//#define RDWDOT9_BLK     (2)
//#define RDWDOT_THRD     (BLOCK_SIZE2)
//#define RDWDOT_BLK      (3)
//
//#endif /* __CUDA_ARCH__ */
//
//#define LAUNCH_BOUNDS(maxThrd, minBlk) __launch_bounds__(maxThrd, minBlk)
//
//#else

#define LAUNCH_BOUNDS(maxThrd, minBlk)

//#endif /* USE_LAUNCH_BOUNDS */

#endif
