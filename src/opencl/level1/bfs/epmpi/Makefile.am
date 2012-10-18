include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# which compiler to use
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(MPI_CPPFLAGS) $(OCL_CPPFLAGS) -I${top_srcdir}/src/mpi/common
L1LIBS = $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
epopencldir = $(bindir)/EP/OpenCL

# what programs should be installed in the destination?
epopencl_PROGRAMS = BFS

# how to build those programs
BFS_SOURCES	= BFS.cpp \
				bfs_uiuc_spill_cl.cpp \
				bfs_iiit_cl.cpp \
				Graph.cpp \
				main.cpp
BFS_LDADD	= $(L1LIBS)

