#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.65])
AC_INIT([SHOC], [1.1.5], [shoc-help@elist.ornl.gov])
AC_CONFIG_AUX_DIR([build-aux])
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([src/common/Timer.h])
AC_CONFIG_HEADERS([config/config.h])

# SHOC is a collection of C++ programs
AC_LANG([C++])

# Because of the nature of SHOC (a collection of benchmark programs),
# autoconf's default prefix (/usr/local) is not appropriate. 
# Instead, we set the default installation location to be the build 
# location.
AC_PREFIX_DEFAULT(`pwd`)


# Checks for fundamental programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_RANLIB
AC_CHECK_TOOL([AR], [ar], [:])
AC_PATH_PROG([PERL], [perl])
AC_PATH_PROG([UNZIP], [unzip])


# check whether special NVCXXFLAGS were specified
# if not, set default flags based on whehter we are using g++ and whether our
# C++ compiler supports -g.
# Logic based on the logic used in the AC_PROG_CXX macro.
ac_save_NVCXXFLAGS=${NVCXXFLAGS}
ac_test_NVCXXFLAGS=${NVCXXFLAGS+set}
if test "$ac_test_NVCXXFLAGS" = set
then
    NVCXXFLAGS=$ac_save_NVCXXFLAGS
elif test $ac_cv_prog_cxx_g = yes
then
    if test "$GXX" = yes; then
        NVCXXFLAGS="-g -O2"
    else
        NVCXXFLAGS="-g"
    fi
else
    if test "$GXX" = yes
    then
        NVCXXFLAGS="-O2"
    else
        NVCXXFLAGS=
    fi
fi




# on Linux, we expect that the clock_gettime function will be available,
# but it seems to require the rt library
AS_CASE([$host],
    [*-*-linux*], [LIBS="$LIBS -lrt"],
    [*], []
)


# Checks for header files.
AC_FUNC_ALLOCA
AC_CHECK_HEADERS([float.h stdlib.h string.h sys/time.h sys/timeb.h time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_C_RESTRICT
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_CHECK_FUNCS([clock_gettime floor gethostname gettimeofday localtime_r memset pow sqrt strdup])


AS_IF([test "x$ac_cv_func_clock_gettime" = xyes],
    [ AC_MSG_CHECKING([whether clock_gettime() supports CLOCK_PROCESS_CPUTIME_ID timer])
      AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])],
                    [AC_MSG_RESULT([yes])
                     AC_DEFINE([HAVE_CLOCK_PROCESS_CPUTIME_ID],[1],[Define to 1 if you have the CLOCK_PROCESS_CPUTIME_ID timer for the 'clock_gettime' function])
                    ],
                    [AC_MSG_RESULT([no])]
      )
    ]
)    



# OpenCL support
AC_ARG_WITH([opencl],
    [AS_HELP_STRING([--with-opencl],
        [build OpenCL versions of SHOC benchmark programs @<:@default=check@:>@])],
    [],
    [with_opencl=check])

# detect/verify we have a working OpenCL header and library
AS_IF([test "x$with_opencl" != xno],
    [
        # check for usable OpenCL header
        # note C++ binding uses opencl.h, not just cl.h
        AC_MSG_NOTICE([checking for usable OpenCL opencl.h header])
        AC_CHECK_HEADER([OpenCL/opencl.h],
            [OPENCL_HEADER='"OpenCL/opencl.h"'],
            [AC_CHECK_HEADER([CL/opencl.h],
                [OPENCL_HEADER='"CL/opencl.h"'],
                [if test "x$with_opencl" != xcheck; then
                    AC_MSG_FAILURE([OpenCL support was requested, but no usable OpenCL header was found])
                 fi
                ]
             )
            ]
        )

        AS_IF([test "x$OPENCL_HEADER" != x],
            [
            # we have a header, check for usable OpenCL library
            # we cannot use AC_CHECK_LIB because we want to check
            # using the OS X OpenCL framework, and AC_CHECK_LIB does not
            # know how to use the framework linker syntax.
            # (we also do not want the OpenCL libs to be added to the
            # LIBS variable, and AC_CHECK_LIB does that if it succeeds)
            AC_MSG_CHECKING([for usable OpenCL library])
            savedLIBS=$LIBS
            OPENCL_LIBS="-framework OpenCL"
            LIBS="$OPENCL_LIBS $savedLIBS"
            AC_LINK_IFELSE( [AC_LANG_PROGRAM([@%:@include $OPENCL_HEADER],[clGetPlatformIDs(0,0,0);])],
                [AC_MSG_RESULT($OPENCL_LIBS)],
                [OPENCL_LIBS="-lOpenCL"
                 LIBS="$OPENCL_LIBS $savedLIBS"
                 AC_LINK_IFELSE( [AC_LANG_PROGRAM([@%:@include $OPENCL_HEADER],[clGetPlatformIDs(0,0,0);])],
                    [AC_MSG_RESULT($OPENCL_LIBS)],
                    [AC_MSG_RESULT([no])
                     if test "x$with_opencl" != xcheck; then
                         AC_MSG_FAILURE([OpenCL support was requested, but no usable OpenCL library was fund])
                     fi
                    ]
                 )
                ]
            )
            LIBS=$savedLIBS
            ]
        )

        AS_IF([test "x$OPENCL_HEADER" != "x" -a "x$OPENCL_LIBS" != "x"],
            [with_opencl=yes
             MPI_SUBDIRS="$MPI_SUBDIRS opencl"
             AC_SUBST([OPENCL_LIBS])
             ],
            [AC_MSG_NOTICE([no usable OpenCL installation found])
             with_opencl=no
            ])
    ])
AM_CONDITIONAL([BUILD_OPENCL], [test "x$with_opencl" = "xyes"])


# CUDA support
AC_ARG_WITH([cuda],
    [AS_HELP_STRING([--with-cuda],
        [build CUDA versions of SHOC benchmark programs @<:@default=check@:>@])],
    [],
    [with_cuda=check])

USE_CUDA=no
AS_IF([test "x$with_cuda" != xno],
    [
        # check for the nvcc program
        AC_PATH_PROG([NVCC],[nvcc],[])

        AS_IF([test "x$NVCC" != x],
            [ savedCXX=$CXX
              CXX=$NVCC
              savedLIBS=$LIBS
              savedCPPFLAGS=$CPPFLAGS
              savedCXXFLAGS=$CXXFLAGS
              savedLDFLAGS=$LDFLAGS
              CXXFLAGS=$NVCXXFLAGS
              LDFLAGS=""

              # find cuda include path based on location of nvcc command
              cudabindir=`AS_DIRNAME([$NVCC])`
              CUDA_INCDIR=`AS_DIRNAME([$cudabindir])`/include
              CPPFLAGS="$CPPFLAGS -I$CUDA_INCDIR"

              AC_CHECK_HEADERS([cuda.h cuda_runtime.h],[],
                [NVCC=""]
              )
              AC_CHECK_LIB([cublas], [cublasInit],[],
                [NVCC=""]
              )
              AC_CHECK_LIB([cufft], [cufftPlan1d], [],
                [NVCC=""]
              )

              CPPFLAGS=$savedCPPFLAGS
              LIBS=$savedLIBS
              CXX=$savedCXX
              CXXFLAGS=$savedCXXFLAGS
              LDFLAGS=$savedLDFLAGS
            ],
            [ if test "x$with_cuda" != xcheck; then
                AC_MSG_FAILURE([CUDA support was requested, but no usable CUDA installation found])
              fi
              NVCC=""
            ]
        )

        AS_IF([test "x$NVCC" != x],
            [with_cuda=yes
             USE_CUDA=yes
             MPI_SUBDIRS="$MPI_SUBDIRS cuda"
             AC_SUBST([CUDA_INCDIR])],
            [AC_MSG_NOTICE([no usable CUDA installation found])
             with_cuda=no
            ])

        AS_IF([test "x$with_cuda" = xyes],
            [AS_IF([test "x$CUDA_CPPFLAGS" = x],
                [dnl generate CUDA code for broad spectrum of devices
                 dnl Note: cc 13 for Tesla
                 dnl Note: cc 20 for Fermi
                 dnl Note: cc 30 for Kepler
                 CUDA_CPPFLAGS=["-gencode=arch=compute_12,code=sm_12 \
 -gencode=arch=compute_13,code=sm_13 \
 -gencode=arch=compute_20,code=sm_20 \
 -gencode=arch=compute_20,code=compute_20"]
                 AC_MSG_NOTICE([using CUDA_CPPFLAGS=$CUDA_CPPFLAGS])
                 AC_MSG_WARN([*** SHOC programs may fail if these -gencode values do not apply to your GPU.  If they don't apply, or if you want to shorten SHOC's compile time, determine the -gencode options that apply for your GPU and re-configure SHOC explicitly specifying only those options in CUDA_CPPFLAGS during the configure. ***])
                ]
             )
             AC_SUBST([CUDA_CPPFLAGS])
            ]
        )
    ]
)
AM_CONDITIONAL([BUILD_CUDA], [test "x$with_cuda" = "xyes"])

# build documentation? by default, just copy existing documentation
AC_ARG_ENABLE([builddoc],
    [AS_HELP_STRING([--enable-builddoc],
        [build SHOC documentation; requires pdflatex, bibtex, and latexmk @<:@default=disabled@:>@])],
    [],
    [enable_builddoc=no]
)

COND_BUILDDOC=no
AS_IF([test "x$enable_builddoc" = xyes],
    [
        AC_PATH_PROG([PDFLATEX], [pdflatex], [])

        AS_IF([test "x$PDFLATEX" != x],
            [
                AC_PATH_PROG([BIBTEX], [bibtex], [])

                AS_IF([test "x$BIBTEX" != x],
                    [
                        AC_PATH_PROG([LATEXMK], [latexmk], [])
                        AS_IF([test "x$LATEXMK" != x],
                            [
                                # we have been asked to build documentation,
                                # we have pdflatex, bibtex, and latexmk
                                COND_BUILDDOC=yes
                            ]
                        )
                    ]
                )
            ]
        )

        AS_IF([test "x$COND_BUILDDOC" != xyes],
            [
                AC_MSG_FAILURE([Documentation build requested, but required tools (pdflatex, bibtex, latexmk) were not all found])
            ]
        )
    ]
)
AM_CONDITIONAL([BUILD_DOC], [test "x$COND_BUILDDOC" = "xyes"])

# Unzip and move data by default, just copy existing documentation
AC_ARG_ENABLE([dataunzip],
    [AS_HELP_STRING([--enable-data-unzip],
        [Unzip and move SHOC data; requires unzip and cp @<:@default=enabled@:>@])],
    [],
    [enable_dataunzip=check]
)

COND_DATAUNZIP=no
AS_IF([test "x$enable_dataunzip" != xno],
    [
        #Just test to make sure unzip exists
        AS_IF([test "$UNZIP" != ''],
            [
                        # we have been asked to unzip and move data,
                        # we have unzip 
                        COND_DATAUNZIP=yes
                    ]
                )

        AS_IF([test "x$COND_DATAUNZIP" != xyes],
            [
                AC_MSG_FAILURE([Data unzip requested, but required tools (unzip) were not all found])
            ]
        )
    ]
)
AM_CONDITIONAL([DATA_UNZIP], [test "x$COND_DATAUNZIP" = "xyes"])

# build stability tests? by default, enable if CUDA is enabled
AC_ARG_ENABLE([stability],
    [AS_HELP_STRING([--disable-stability],
        [build SHOC stability test; requires CUDA @<:@default=enabled@:>@])],
    [],
    [enable_stability=check]
)

COND_STABILITY=no
AS_IF([test "x$enable_stability" = xyes],
    [AS_IF([test "x$with_cuda" != xno],
        [COND_STABILITY=yes],
        [ AC_MSG_FAILURE([CUDA-based stability test enabled but no usable CUDA installation found])
          with_stability=no
        ]
    )
    ],
    [AS_IF([test "x$enable_stability" != xno],
        [AS_IF([test "x$with_cuda" != xno],
            [COND_STABILITY=yes],
        [
          with_stability=no
        ]
        )]
    )]
    
)
AM_CONDITIONAL([BUILD_STABILITY], [test "x$COND_STABILITY" = "xyes"])


# MPI
AC_ARG_WITH([mpi],
    [AS_HELP_STRING([--with-mpi],
        [build MPI-based parallel versions of SHOC benchmarks @<:@default=check@:>@])],
    [],
    [with_mpi=check]
)

USE_MPI=no
AS_IF([test "x$with_mpi" != xno],
    [ 
        AC_PATH_PROGS([MPICXX], [[$MPICXX] mpicxx mpic++ mpiCC])
        savedCXX=$CXX
        CXX=$MPICXX
        AS_IF([test "x$MPICXX" != "x"],
            [
                AC_MSG_CHECKING([whether we can compile an MPI program using $CXX])
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([@%:@include "mpi.h"],[MPI_Init(0,0);])],
                    [ AC_MSG_RESULT([yes])],
                    [ AC_MSG_RESULT([no])
                      MPICXX=""
                    ]
                )
            ]
        )

        AS_IF([test "x$MPICXX" != "x"],
            [
                AC_MSG_CHECKING([whether we can link an MPI program using $CXX])
                AC_LINK_IFELSE([AC_LANG_PROGRAM([@%:@include "mpi.h"],[MPI_Init(0,0);])],
                    [
                        AC_MSG_RESULT([yes])
                    ],
                    [
                        AC_MSG_RESULT([no])
                        MPICXX=""
                    ]
                )
            ]
        )
        CXX=$savedCXX

        AS_IF([test "x$MPICXX" != "x"],
            [ AS_IF([test "x$with_opencl" = xyes -o "x$with_cuda" = xyes],
                  [ with_mpi=yes
                    USE_MPI=yes
                  ],
                  [ AC_MSG_FAILURE([MPI support was requested, but is only meaningful if OpenCL and/or CUDA support is available])
                    with_mpi=no
                  ]
              )
            ],
            [ AS_IF([test "x$with_mpi" != xcheck],
                    [ AC_MSG_FAILURE([--with-mpi specified, but no usable MPI installation found.])
                      with_mpi=no
                    ]
              )
            ]
        )
    ]
)
AM_CONDITIONAL([BUILD_MPI], [test "x$with_mpi" = "xyes"])

AC_SUBST([USE_MPI])
AC_SUBST([MPICXX])
AC_SUBST([USE_CUDA])
AC_SUBST([MPI_SUBDIRS])
AC_SUBST([NVCXXFLAGS])

AC_CONFIG_FILES([Makefile
                 config/common.mk
                 config/config.mk
                 config/dirtargets.mk
                 config/targets.mk
                 doc/Makefile
                 src/Makefile
                 src/common/Makefile
                 src/cuda/Makefile
                 src/cuda/level0/Makefile
                 src/cuda/level0/epmpi/Makefile
                 src/cuda/level1/Makefile
                 src/cuda/level1/bfs/Makefile
                 src/cuda/level1/bfs/epmpi/Makefile
                 src/cuda/level1/fft/Makefile
                 src/cuda/level1/fft/epmpi/Makefile
                 src/cuda/level1/md/Makefile
                 src/cuda/level1/md/epmpi/Makefile
                 src/cuda/level1/md5hash/Makefile
                 src/cuda/level1/md5hash/epmpi/Makefile
                 src/cuda/level1/neuralnet/Makefile
                 src/cuda/level1/neuralnet/epmpi/Makefile
                 src/cuda/level1/reduction/Makefile
                 src/cuda/level1/reduction/epmpi/Makefile
                 src/cuda/level1/reduction/tpmpi/Makefile
                 src/cuda/level1/scan/Makefile
                 src/cuda/level1/scan/epmpi/Makefile
                 src/cuda/level1/scan/tpmpi/Makefile
                 src/cuda/level1/gemm/Makefile
                 src/cuda/level1/gemm/epmpi/Makefile
                 src/cuda/level1/sort/Makefile
                 src/cuda/level1/sort/epmpi/Makefile
                 src/cuda/level1/spmv/Makefile
                 src/cuda/level1/spmv/epmpi/Makefile
                 src/cuda/level1/stencil2d/Makefile
                 src/cuda/level1/stencil2d/tpmpi/Makefile
                 src/cuda/level1/triad/Makefile
                 src/cuda/level1/triad/epmpi/Makefile
                 src/cuda/level2/Makefile
                 src/cuda/level2/qtclustering/Makefile
                 src/cuda/level2/qtclustering/tpmpi/Makefile
                 src/cuda/level2/s3d/Makefile
                 src/cuda/level2/s3d/epmpi/Makefile
                 src/mpi/Makefile
                 src/mpi/common/Makefile
                 src/mpi/contention/Makefile
                 src/mpi/contention/cuda/Makefile
                 src/mpi/contention/opencl/Makefile
                 src/mpi/contention-mt/Makefile
                 src/mpi/contention-mt/cuda/Makefile
                 src/mpi/contention-mt/opencl/Makefile
                 src/opencl/Makefile
                 src/opencl/common/Makefile
                 src/opencl/level0/Makefile
                 src/opencl/level0/epmpi/Makefile
                 src/opencl/level1/Makefile
                 src/opencl/level1/bfs/Makefile
                 src/opencl/level1/bfs/epmpi/Makefile
                 src/opencl/level1/fft/Makefile
                 src/opencl/level1/fft/epmpi/Makefile
                 src/opencl/level1/md/Makefile
                 src/opencl/level1/md/epmpi/Makefile
                 src/opencl/level1/md5hash/Makefile
                 src/opencl/level1/md5hash/epmpi/Makefile
                 src/opencl/level1/reduction/Makefile
                 src/opencl/level1/reduction/epmpi/Makefile
                 src/opencl/level1/reduction/tpmpi/Makefile
                 src/opencl/level1/scan/Makefile
                 src/opencl/level1/scan/epmpi/Makefile
                 src/opencl/level1/scan/tpmpi/Makefile
                 src/opencl/level1/gemm/Makefile
                 src/opencl/level1/gemm/epmpi/Makefile
                 src/opencl/level1/sort/Makefile
                 src/opencl/level1/sort/epmpi/Makefile
                 src/opencl/level1/spmv/Makefile
                 src/opencl/level1/spmv/epmpi/Makefile
                 src/opencl/level1/stencil2d/Makefile
                 src/opencl/level1/stencil2d/tpmpi/Makefile
                 src/opencl/level1/triad/Makefile
                 src/opencl/level1/triad/epmpi/Makefile
                 src/opencl/level2/Makefile
                 src/opencl/level2/s3d/Makefile
                 src/opencl/level2/s3d/epmpi/Makefile
                 src/stability/Makefile
                 src/stability/epmpi/Makefile
                 tools/Makefile
                 ])
AC_OUTPUT
