include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# Which compiler to use
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(MPI_CPPFLAGS) $(OCL_CPPFLAGS) -I${top_srcdir}/src/mpi/common
L1LIBS = -lSHOCCommonOpenCL -lSHOCCommon $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
epopencldir = $(bindir)/EP/OpenCL

# what programs should be installed in the destination?
epopencl_PROGRAMS = S3D

# how to build those programs
S3D_SOURCES = S3D.cpp \
gr_base_cl.cpp \
qssa_cl.cpp \
qssab_cl.cpp \
qssa2_cl.cpp \
ratt_cl.cpp \
ratt2_cl.cpp \
ratt3_cl.cpp \
ratt4_cl.cpp \
ratt5_cl.cpp \
ratt6_cl.cpp \
ratt7_cl.cpp \
ratt8_cl.cpp \
ratt9_cl.cpp \
ratt10_cl.cpp \
ratx_cl.cpp \
ratxb_cl.cpp \
ratx2_cl.cpp \
ratx4_cl.cpp \
rdsmh_cl.cpp \
rdwdot_cl.cpp \
rdwdot2_cl.cpp \
rdwdot3_cl.cpp \
rdwdot6_cl.cpp \
rdwdot7_cl.cpp \
rdwdot8_cl.cpp \
rdwdot9_cl.cpp \
rdwdot10_cl.cpp \
main.cpp
S3D_LDADD   = $(L1LIBS)

