if BUILD_MPI
    MAYBE_MPI = 
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common:$(srcdir)/../../../common/QTC

AM_LDFLAGS = $(OPENACC_LDFLAGS)
AM_CFLAGS	= $(OPENACC_CFLAGS)

# what is the destination for programs built from this directory?
serialopenaccdir = $(bindir)/Serial/OpenACC

# what programs should be installed in the destination?
serialopenacc_PROGRAMS = QTC

# how to build those programs
QTC_SOURCES = QTC.cpp \
				QTCFuncs.c \
				qtclib.cpp \
				libdata.cpp \
                main.cpp
QTC_LDADD   = -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

