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
L1LIBS = $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
serialopencldir = $(bindir)/Serial/OpenCL

# what programs should be installed in the destination?
serialopencl_PROGRAMS = BFS

# how to build those programs
BFS_SOURCES	= BFS.cpp \
				bfs_uiuc_spill_cl.cpp \
				bfs_iiit_cl.cpp \
				Graph.cpp \
				main.cpp
BFS_LDADD	= $(L1LIBS)

