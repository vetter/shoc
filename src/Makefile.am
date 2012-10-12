
if BUILD_CUDA
    MAYBE_CUDA = cuda
endif

if BUILD_OPENCL
    MAYBE_OPENCL = opencl
endif

if BUILD_STABILITY
    MAYBE_STABILITY = stability
endif

if BUILD_MPI
    MAYBE_MPI = mpi
endif

SUBDIRS=common $(MAYBE_OPENCL) $(MAYBE_CUDA) $(MAYBE_MPI) $(MAYBE_STABILITY)

