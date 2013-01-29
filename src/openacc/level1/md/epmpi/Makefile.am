include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# which compiler to use
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OPENACC_LDFLAGS)
AM_CPPFLAGS = ${MPI_CPPFLAGS} -I${top_srcdir}/src/mpi/common
AM_CFLAGS	= $(OPENACC_CFLAGS)

# what is the destination for programs built from this directory?
epopenaccdir = $(bindir)/EP/OpenACC

# what programs should be installed in the destination?
epopenacc_PROGRAMS = MD

# how to build those programs
MD_SOURCES = MD.cpp \
				MDFuncs.c \
                main.cpp
MD_LDADD   = -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

