if BUILD_MPI
    MAYBE_MPI = epmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(OCL_CPPFLAGS)
L1LIBS = -lSHOCCommonOpenCL -lSHOCCommon $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
serialopencldir = $(bindir)/Serial/OpenCL

# what programs should be installed in the destination?
serialopencl_PROGRAMS = S3D

# how to build those programs
S3D_SOURCES = S3D.cpp \
main.cpp \
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
rdwdot10_cl.cpp
S3D_LDADD   = $(L1LIBS)

