#ifndef __PRINT_RESULTS_H__
#define __PRINT_RESULTS_H__

void c_print_results( char   *name,
                      char   classX,
                      int    n1, 
                      int    n2,
                      int    n3,
                      int    niter,
                      int    nprocs_compiled,
                      int    nprocs_total,
                      double t,
                      double mops,
                      char   *optype,
                      int    passed_verification,
                      char   *npbversion);

#endif //__PRINT_RESULTS_H__
