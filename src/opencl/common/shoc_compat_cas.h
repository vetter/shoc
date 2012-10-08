#ifndef SHOC_COMPAT_CAS_H
#define SHOC_COMPAT_CAS_H

#if defined( __PGI )

#if defined( __x86_64 )
inline
int
__sync_val_compare_and_swap( volatile int* destptr, int testval, int newval )
{
    int retval;
    __asm__ __volatile__ ("lock; cmpxchg %2, %1"
        : "=a" (retval),
          "+m" (*destptr)
        : "r" (newval),
          "0" (testval)
        : "memory");
    return retval;
}
#else /* defined(__x86_64) */
#  error "atomic compare-and-swap on this processor architecture not yet supported"
#endif /* defined(__x86_64) */
#endif // defined( PGI )


#endif // SHOC_COMPAT_CAS_H
