#ifndef _QTC_COMMON_H_
#define _QTC_COMMON_H_

#define GLOBAL_MEMORY 0x0
#define TEXTUR_MEMORY 0x1
#define COMPACT_STORAGE_MATRIX 0x00
#define FULL_STORAGE_MATRIX    0x10

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
