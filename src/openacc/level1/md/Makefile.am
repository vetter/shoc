if BUILD_MPI
    MAYBE_MPI = epmpi tpmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(OPENACC_LDFLAGS)
AM_CFLAGS	= $(OPENACC_CFLAGS)

# what is the destination for programs built from this directory?
serialopenaccdir = $(bindir)/Serial/OpenACC

# what programs should be installed in the destination?
serialopenacc_PROGRAMS = MD

# how to build those programs
MD_SOURCES = MD.cpp \
				MDFuncs.c \
                main.cpp
MD_LDADD   = -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

