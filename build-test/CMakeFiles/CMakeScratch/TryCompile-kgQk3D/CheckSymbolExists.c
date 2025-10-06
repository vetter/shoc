/* */
#include <OpenCL_INCLUDE_DIR-NOTFOUND/CL/cl.h>

int main(int argc, char** argv)
{
  (void)argv;
#ifndef CL_VERSION_2_2
  return ((int*)(&CL_VERSION_2_2))[argc];
#else
  (void)argc;
  return 0;
#endif
}
