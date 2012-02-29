SHOC_ROOT=../../..
include $(SHOC_ROOT)/config/config.mk

LIBS+=$(OCL_LIBS)
CPPFLAGS+=$(OCL_CPPFLAGS)

OBJ= \
 OpenCLDeviceInfo.o \
 OpenCLPlatform.o \
 OpenCLNodePlatformContainer.o \
 Event.o


ARCHIVE=$(SHOC_ROOT)/lib/libSHOCCommonOpenCL.a

all: $(ARCHIVE)

$(ARCHIVE): $(OBJ)
	$(AR) $(ARFLAGS) $@ $(OBJ)

clean:
	rm -f $(OBJ) $(ARCHIVE)

distclean: clean
	rm -f $(DEP) *~ .buildflags Makefile


include $(SHOC_ROOT)/config/targets.mk

