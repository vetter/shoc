
if BUILD_CUDA
    MAYBE_GPU = cuda
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

if BUILD_HIP
if BUILD_CUDA
MAYBE_GPU = hip
else
MAYBE_GPU = hip
endif
endif

SUBDIRS=common $(MAYBE_OPENCL) $(MAYBE_GPU) $(MAYBE_MPI) #$(MAYBE_STABILITY)

