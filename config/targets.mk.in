#.SUFFIXES: .C .cpp .c .o .d .cu .cl .prog .mpiprog _mpi.o _mpi.d .oclmpiprog _ocl.o _ocl.d _cl.cpp _cuda.o _cuda.d 

#%.o: %.cpp
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

#%_mpi.o: %.cpp
#	$(MPICXX) -DPARALLEL $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

#%_ocl.o: %.cpp
#	$(CXX) $(OCL_CPPFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(OCL_CXXFLAGS) -c $< -o $@

#%_cuda.o: %.cu
#	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -c $< -o $@

#%_cuda.o: %.cpp
#	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -c $< -o $@

%_cl.cpp: %.cl
	echo "const char *cl_source_$* =" >  $@
	tr -d "\r" < $< | sed 's/\\/\\\\/g' | sed 's/\"/\\\"/g' | sed 's,^\(.*\)$$,\"\1\\n\",' >> $@
	echo ";" >> $@

%.o: %.cu
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -c $<

#%.d: %.cpp
#	@echo "updating $@"; set -e; ${RM} $@; \
#	$(CXX) -M $(CPPFLAGS) $< > $@.tmp; \
#    sed 's,\($*\)\.o[ :]*,\1.o $@ : .buildflags ,g' < $@.tmp > $@; \
#    ${RM} $@.tmp
#
#%_mpi.d: %.cpp
#	@echo "updating $@"; set -e; ${RM} $@; \
#	$(MPICXX) -M -DPARALLEL $(CPPFLAGS) $< > $@.tmp; \
#    sed 's,\($*\)\.o[ :]*,\1_mpi.o $@ : .buildflags ,g' < $@.tmp > $@; \
#    ${RM} $@.tmp
#
#%_ocl.d: %.cpp
#	@echo "updating $@"; set -e; ${RM} $@; \
#	$(CXX) -M $(OCL_CPPFLAGS) $(CPPFLAGS) $< > $@.tmp; \
#    sed 's,\($*\)\.o[ :]*,\1_ocl.o $@ : .buildflags ,g' < $@.tmp > $@; \
#    ${RM} $@.tmp
#
#%_cuda.d: %.cpp
#	@echo "updating $@"; set -e; ${RM} $@; \
#	$(NVCC) -M ${CUDA_CPPFLAGS} $(CPPFLAGS) $< > $@.tmp; \
#    sed 's,\($*\)\.o[ :]*,\1.o $@ : .buildflags ,g' < $@.tmp > $@; \
#    ${RM} $@.tmp
#
#%_cuda.d: %.cu
#	@echo "updating $@"; set -e; ${RM} $@; \
#	$(NVCC) -M ${CUDA_CPPFLAGS} $(CPPFLAGS) $< > $@.tmp; \
#    sed 's,\($*\)\.o[ :]*,\1.o $@ : .buildflags ,g' < $@.tmp > $@; \
#    ${RM} $@.tmp
#
#%.d: %.cu
#	@echo "updating $@"; set -e; ${RM} $@; \
#	$(NVCC) -M ${CUDA_CPPFLAGS} $(CPPFLAGS) $< > $@.tmp; \
#    sed 's,\($*\)\.o[ :]*,\1.o $@ : .buildflags ,g' < $@.tmp > $@; \
#    ${RM} $@.tmp
#
#.buildflags: Makefile
#	@echo "CXX=$(CXX)"               >  .buildflagstmp
#	@echo "CC=$(CXX)"                >> .buildflagstmp
#	@echo "NVCC=$(NVCC)"             >> .buildflagstmp
#	@echo "AR=$(AR)"                 >> .buildflagstmp
#	@echo "LD=$(LD)"                 >> .buildflagstmp
#	@echo "CPPFLAGS=$(CPPFLAGS)"     >> .buildflagstmp
#	@echo "CFLAGS=$(CFLAGS)"         >> .buildflagstmp
#	@echo "CXXFLAGS=$(CXXFLAGS)"     >> .buildflagstmp
#	@echo "NVCXXFLAGS=$(NVCXXFLAGS)" >> .buildflagstmp
#	@echo "LDFLAGS=$(LDFLAGS)"       >> .buildflagstmp
#	@echo "ARFLAGS=$(ARFLAGS)"       >> .buildflagstmp
#	@echo "LIBS=$(LIBS)"             >> .buildflagstmp
#	@echo "uname=" `uname -a`        >> .buildflagstmp
#	@if (test -f .buildflags); then \
#	  if (test -z "`diff .buildflagstmp .buildflags`"); then \
#	    echo "Build flags were the same."; rm -f .buildflagstmp; \
#	  else \
#	    echo "Build flags changed."; mv .buildflagstmp .buildflags; \
#	  fi \
#        else \
#	  echo "Creating build flags."; mv .buildflagstmp .buildflags; \
#	fi
#
#DEP:=$(OBJ:.o=.d) $(ALLOBJ:.o=.d)
#include $(DEP)
