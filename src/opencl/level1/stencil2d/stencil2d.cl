
// define types based on compiler "command line"
#if defined(SINGLE_PRECISION)
#define VALTYPE float
#elif defined(K_DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define VALTYPE double
#elif defined(AMD_DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define VALTYPE double
#else
#error No precision defined.
#endif


inline
int
ToGlobalRow( int gidRow, int lidRow )
{
    // assumes coordinates and dimensions are logical (without halo)
    // returns logical global row (without halo)
    return gidRow*LROWS + lidRow;
}

inline
int
ToGlobalCol( int gidCol, int lidCol )
{
    // assumes coordinates and dimensions are logical (without halo)
    // returns logical global column (without halo)
    return gidCol*LCOLS + lidCol;
}


inline
int
ToFlatHaloedIdx( int row, int col, int rowPitch )
{
    // assumes input coordinates and dimensions are logical (without halo)
    // and a halo of width 1
    return (row + 1)*(rowPitch + 2) + (col + 1);
}


inline
int
ToFlatIdx( int row, int col, int pitch )
{
    return row * pitch + col;
}


__kernel
void
CopyRect( __global VALTYPE* dest,
            int doffset,
            int dpitch,
            __global VALTYPE* src,
            int soffset,
            int spitch,
            int width,
            int height )
{
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int gsz = get_global_size(0);
    int lsz = get_local_size(0);
    int grow = gid * lsz + lid;

    if( grow < height )
    {
        for( int c = 0; c < width; c++ )
        {
            (dest + doffset)[ToFlatIdx(grow,c,dpitch)] = (src + soffset)[ToFlatIdx(grow,c,spitch)];
        }
    }
}





__kernel 
void 
StencilKernel( __global VALTYPE* data, 
                __global VALTYPE* newData,
                int pad,
                VALTYPE wCenter,
                VALTYPE wCardinal,
                VALTYPE wDiagonal )
{
    // determine our location in the OpenCL coordinate system
    // To match with the row-major ordering used to store the 2D
    // array in both the host and on the device, we use:
    //   dimension 0 == rows,
    //   dimension 1 == columns
    int gidRow = get_group_id(0);
    int gidCol = get_group_id(1);
    int gszRow = get_num_groups(0);
    int gszCol = get_num_groups(1);
    int lidRow = get_local_id(0);
    int lidCol = get_local_id(1);

    // determine our logical global data coordinates (without halo)
    int gRow = ToGlobalRow( gidRow, lidRow );
    int gCol = ToGlobalCol( gidCol, lidCol );

    // determine pitch of rows (without halo)
    int nCols = gszCol * LCOLS + 2;     // num columns including halo
    int nPaddedCols = nCols + (((nCols % pad) == 0) ? 0 : (pad - (nCols % pad)));
    int gRowWidth = nPaddedCols - 2;    // remove the halo

    // determine our coodinate in the flattened data (with halo)
    int gidx = ToFlatHaloedIdx( gRow, gCol, gRowWidth );

    // copy my global data item to a shared local buffer
    // (i.e., it is same size as our local block but with halo of width 1)
    __local VALTYPE sh[LROWS+2][LCOLS+2];
    sh[lidRow+1][lidCol+1] = data[gidx];

    // copy halo data into shared local buffer
    // We follow the approach of Micikevicius (NVIDIA) from the
    // GPGPU-2 Workshop, 3/8/2009.
    // We leave many threads idle while those along two of the edges
    // copy the boundary data for all four edges. This seems to be
    // a performance win even with the idle threads because it 
    // limits the branching logic.
    if( lidRow == 0 )
    {
        sh[0][lidCol+1] = data[ToFlatHaloedIdx(gRow-1, gCol, gRowWidth)];
        sh[LROWS+1][lidCol+1] = data[ToFlatHaloedIdx(gRow+LROWS, gCol, gRowWidth)];
    }
    if( lidCol == 0 )
    {
        sh[lidRow+1][0] = data[ToFlatHaloedIdx(gRow, gCol-1, gRowWidth)];
        sh[lidRow+1][LCOLS+1] = data[ToFlatHaloedIdx(gRow, gCol+LCOLS, gRowWidth)];
    }
    if( (lidRow == 0) && (lidCol == 0) )
    {
        // since we are doing 9-pt stencil, we have to copy corner elements.
        // Note: stencil used by Micikevicius did not use 'diagonals' - 
        // in 2D, it would be a 5-pt stencil.  But these loads are costly.
        sh[0][0] = data[ToFlatHaloedIdx(gRow-1,gCol-1,gRowWidth)];
        sh[LROWS+1][0] = data[ToFlatHaloedIdx(gRow+LROWS,gCol-1, gRowWidth)];
        sh[0][LCOLS+1] = data[ToFlatHaloedIdx(gRow-1,gCol+LCOLS,gRowWidth)];
        sh[LROWS+1][LCOLS+1] = data[ToFlatHaloedIdx(gRow+LROWS,gCol+LCOLS, gRowWidth)];
    }

    // let all those loads finish
    barrier( CLK_LOCAL_MEM_FENCE );

    // do my part of the smoothing operation
    VALTYPE centerValue = sh[lidRow+1][lidCol+1];
    VALTYPE cardinalValueSum = sh[lidRow][lidCol+1] +
                        sh[lidRow+2][lidCol+1] +
                        sh[lidRow+1][lidCol] +
                        sh[lidRow+1][lidCol+2];
    VALTYPE diagonalValueSum = sh[lidRow][lidCol] +
                        sh[lidRow][lidCol+2] +
                        sh[lidRow+2][lidCol] +
                        sh[lidRow+2][lidCol+2];
    newData[gidx] = wCenter * centerValue +
            wCardinal * cardinalValueSum + 
            wDiagonal * diagonalValueSum;
}

