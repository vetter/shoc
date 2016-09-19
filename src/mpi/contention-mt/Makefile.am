
if BUILD_CUDA
    MAYBE_GPU = cuda
endif

if BUILD_OPENCL
    MAYBE_OPENCL = opencl
endif

if BUILD_HIP
if BUILD_CUDA
MAYBE_GPU = hip
else
MAYBE_GPU = hip
endif
endif

SUBDIRS=$(MAYBE_OPENCL) $(MAYBE_GPU)

