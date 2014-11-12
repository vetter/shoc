#!/bin/sh

# We do not use nvcc to link CUDA programs, because we may be linking
# against MPI libraries also, and prefer to allow the MPI compiler
# drivers to handle the link.
#
# This requires us to determine which libraries are needed to link CUDA
# programs.   We use nvcc -dryrun to determine which libraries are 
# needed to link CUDA programs.  Prior to the release of CUDA version 6.0,
# the output of nvcc -dryrun included a line of the form LIBRARIES=...
# that indicated all libraries needed to link as -llib flags.  
# The nvcc distributed with CUDA 6.0 no longer lists the libraries 
# in the LIBRARIES line itself, but only as part of the actual
# command that would have been executed to link the executable.
#
# For CUDA < 6.0, we just use the output of the LIBRARIES line.
# For CUDA 6.0, we determine the libraries to use by:
# 
#   Running nvcc -dryrun and saving the LIBRARIES line from the output.
#   Re-running nvcc -dryrun and parsing the link line to remove 
#     everything before the LIBRARIES contents and possibly a -Wl,--end-group
#     specification.
#   
if [ "$#" -ne 1 ]
then
   echo "Usage: $0 <nvcc>" >&2
   echo "  where <nvcc> is the filename or path to the nvcc executable to use." >&2
   exit 1
fi
NVCC=$1
#echo "Using NVCC=$NVCC"

cudart_flag_supported=0
$NVCC -dryrun -cudart shared bogus.cu > /dev/null 2>&1
if [ $? -eq 0 ]
then
    cudart_flag_supported=1  
fi
#echo "cudart_flag_supported=$cudart_flag_supported"

libspec=`$NVCC -dryrun bogus.cu 2>&1 | grep LIBRARIES | sed 's/^.*LIBRARIES=//'`
#echo "libspec=$libspec"
if [ $cudart_flag_supported -eq 1 ]
then
    cudalibs=`$NVCC -dryrun bogus.cu 2>&1 | tail -1 | sed "s#^.*-o \"a.out\"##" | sed 's#"[a-zA-Z0-9/_-]*\.o"##g' | sed 's/-Wl,--start-group//' | sed 's/-Wl,--end-group//'`
else
    cudalibs=$libspec
fi

echo $cudalibs

