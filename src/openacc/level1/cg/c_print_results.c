/*****************************************************************/
/******     C  _  P  R  I  N  T  _  R  E  S  U  L  T  S     ******/
/*****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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
                      char   *npbversion)
{
    char size[16];
    int j;

    printf( "\n\n %s Benchmark Completed.\n", name ); 

    printf( " Class           =                        %c\n", classX );

    if( n3 == 0 ) {
      if ( ( name[0] == 'E' ) && ( name[1] == 'P' ) ) {
        sprintf( size, "%15.0lf", pow(2.0, n1) );
        j = 14;
        if ( size[j] == '.' ) {
          size[j] = ' '; 
          j--;
        }
        size[j+1] = '\0';
        printf( " Size            =          %15s\n", size );
      } else {
        long nn = n1;
        if ( n2 != 0 ) nn *= n2;
        printf( " Size            =             %12ld\n", nn );   /* as in IS */
      }
    }
    else
        printf( " Size            =            %3dx %3dx %3d\n", n1,n2,n3 );

    printf( " Iterations      =             %12d\n", niter );
 
    printf( " Time in seconds =             %12.2f\n", t );

    printf( " Total processes =             %12d\n", nprocs_total );

    if ( nprocs_compiled != 0 )
        printf( " Compiled procs  =             %12d\n", nprocs_compiled );

    printf( " Mop/s total     =             %12.2f\n", mops );

    printf( " Mop/s/process   =             %12.2f\n", mops/((float) nprocs_total) );

    printf( " Operation type  = %24s\n", optype);

    if( passed_verification )
        printf( " Verification    =               SUCCESSFUL\n" );
    else
        printf( " Verification    =             UNSUCCESSFUL\n" );

    printf( " Version         =             %12s\n", npbversion );

    printf( "\n--------------------------------------\n"
            " OpenACC version by http://ft.ornl.gov\n"
            " based on the MPI-C version (http://aces.snu.ac.kr)\n"
            "--------------------------------------\n\n");
}
 
