#ifndef _QTC_COMMON_H_
#define _QTC_COMMON_H_

//#define SPARSE_DISTANCE_MATRIX 0x1
//#define DENSE_DISTANCE_MATRIX  0x2 

#define GLOBAL_MEMORY 0x0
#define TEXTUR_MEMORY 0x1
#define DENSE_MATRIX  0x00
#define SPARS_MATRIX  0x10

#ifdef MIN
# undef MIN
#endif
#define MIN(_X, _Y) ( ((_X) < (_Y)) ? (_X) : (_Y) )
    
#ifdef MAX
# undef MAX
#endif
#define MAX(_X, _Y) ( ((_X) > (_Y)) ? (_X) : (_Y) )
                                               
#define INVALID_POINT_MARKER -42

#endif
