SHOC_ROOT=../..
include $(SHOC_ROOT)/config/config.mk


OBJ= \
 CTimer.o \
 ResultDatabase.o \
 OptionParser.o \
 Option.o \
 Timer.o \
 ProgressBar.o \
 InvalidArgValue.o


ARCHIVE=$(SHOC_ROOT)/lib/libSHOCCommon.a

all: $(ARCHIVE)

$(ARCHIVE): $(OBJ)
	$(AR) $(ARFLAGS) $@ $(OBJ)

clean:
	rm -f $(OBJ) $(ARCHIVE)

distclean: clean
	rm -f $(DEP) *~ .buildflags

include $(SHOC_ROOT)/config/targets.mk
