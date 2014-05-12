#ifndef SPMV_UTIL_H_
#define SPMV_UTIL_H_

#include <cassert>
#include <iostream>
#include <fstream>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "PMSMemMgmt.h"


// Constants

// threshold for error in GPU results
static const double MAX_RELATIVE_ERROR = .02;

// alignment factor in terms of number of floats, used to enforce
// memory coalescing
static const int PAD_FACTOR = 16;

// size of atts buffer
static const int TEMP_BUFFER_SIZE = 1024;

// length of array for reading fields of mtx header
static const int FIELD_LENGTH = 128;

// If using a matrix market pattern, assign values from 0-MAX_RANDOM_VAL
static const float MAX_RANDOM_VAL = 10.0f;

struct Coordinate {
    int x;
    int y;
    float val;
};

inline int intcmp(const void *v1, const void *v2);
inline int coordcmp(const void *v1, const void *v2);
template <typename floatType>
void readMatrix(char *filename, floatType **val_ptr, int **cols_ptr,
                int **rowDelimiters_ptr, int *n, int *size);
template <typename floatType>
void fill(floatType *A, const int n, const float maxi);
void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim);
template <typename floatType>
void printSparse(floatType *A, int n, int dim, int *cols, int *rowDelimiters);
template <typename floatType>
void convertToColMajor(floatType *A, int *cols, int dim, int *rowDelimiters,
                       floatType *newA, int *newcols, int *rl, int maxrl,
                       bool padded);
template <typename floatType>
void convertToPadded(floatType *A, int *cols, int dim, int *rowDelimiters,
                     floatType **newA_ptr, int **newcols_ptr, int *newIndices,
                     int *newSize);


// ****************************************************************************
// Function: readMatrix
//
// Purpose:
//   Reads a sparse matrix from a file of Matrix Market format
//   Returns the data structures for the CSR format
//
// Arguments:
//   filename: c string with the name of the file to be opened
//   val_ptr: input - pointer to uninitialized pointer
//            output - pointer to array holding the non-zero values
//                     for the  matrix
//   cols_ptr: input - pointer to uninitialized pointer
//             output - pointer to array of column indices for each
//                      element of the sparse matrix
//   rowDelimiters: input - pointer to uninitialized pointer
//                  output - pointer to array holding
//                           indices to rows of the matrix
//   n: input - pointer to uninitialized int
//      output - pointer to an int holding the number of non-zero
//               elements in the matrix
//   size: input - pointer to uninitialized int
//         output - pointer to an int holding the number of rows in
//                  the matrix
//
// Programmer: Lukasz Wesolowski
// Creation: July 2, 2010
// Returns:  nothing directly
//           allocates and returns *val_ptr, *cols_ptr, and
//           *rowDelimiters_ptr indirectly
//           returns n and size indirectly through pointers
// ****************************************************************************
template <typename floatType>
void readMatrix(char *filename, floatType **val_ptr, int **cols_ptr,
                int **rowDelimiters_ptr, int *n, int *size)
{
    std::string line;
    char id[FIELD_LENGTH];
    char object[FIELD_LENGTH];
    char format[FIELD_LENGTH];
    char field[FIELD_LENGTH];
    char symmetry[FIELD_LENGTH];

    std::ifstream mfs( filename );
    if( !mfs.good() )
    {
        std::cerr << "Error: unable to open matrix file " << filename << std::endl;
        exit( 1 );
    }

    int symmetric = 0;
    int pattern = 0;

    int nRows, nCols, nElements;

    struct Coordinate *coords;

    // read matrix header
    if( getline( mfs, line ).eof() )
    {
        std::cerr << "Error: file " << filename << " does not store a matrix" << std::endl;
        exit( 1 );
    }

    sscanf(line.c_str(), "%s %s %s %s %s", id, object, format, field, symmetry);

    if (strcmp(object, "matrix") != 0)
    {
        fprintf(stderr, "Error: file %s does not store a matrix\n", filename);
        exit(1);
    }

    if (strcmp(format, "coordinate") != 0)
    {
        fprintf(stderr, "Error: matrix representation is dense\n");
        exit(1);
    }

    if (strcmp(field, "pattern") == 0)
    {
        pattern = 1;
    }

    if (strcmp(symmetry, "symmetric") == 0)
    {
        symmetric = 1;
    }

    while (!getline( mfs, line ).eof() )
    {
        if (line[0] != '%')
        {
            break;
        }
    }

    // read the matrix size and number of non-zero elements
    sscanf(line.c_str(), "%d %d %d", &nRows, &nCols, &nElements);

    int valSize = nElements * sizeof(struct Coordinate);
    if (symmetric)
    {
        valSize*=2;
    }
    coords = new Coordinate[valSize];

    int index = 0;
    while (!getline( mfs, line ).eof() )
    {
        if (pattern)
        {
            sscanf(line.c_str(), "%d %d", &coords[index].x, &coords[index].y);
            // assign a random value
            coords[index].val = ((floatType) MAX_RANDOM_VAL *
                                 (rand() / (RAND_MAX + 1.0)));
        }
        else
        {
            // read the value from file
            sscanf(line.c_str(), "%d %d %f", &coords[index].x, &coords[index].y,
                   &coords[index].val);
        }

        // convert into index-0-as-start representation
        coords[index].x--;
        coords[index].y--;
        index++;

        // add the mirror element if not on main diagonal
        if (symmetric && coords[index-1].x != coords[index-1].y)
        {
            coords[index].x = coords[index-1].y;
            coords[index].y = coords[index-1].x;
            coords[index].val = coords[index-1].val;
            index++;
        }
    }

    nElements = index;
    // sort the elements
    qsort(coords, nElements, sizeof(struct Coordinate), coordcmp);

    // create CSR data structures
    *n = nElements;
    *size = nRows;
    *val_ptr = pmsAllocHostBuffer<floatType>( nElements );
    *cols_ptr = pmsAllocHostBuffer<int>( nElements );
    *rowDelimiters_ptr = pmsAllocHostBuffer<int>( nRows + 1 );

    floatType *val = *val_ptr;
    int *cols = *cols_ptr;
    int *rowDelimiters = *rowDelimiters_ptr;

    rowDelimiters[0] = 0;
    rowDelimiters[nRows] = nElements;
    int r=0;
    for (int i=0; i<nElements; i++)
    {
        while (coords[i].x != r)
        {
            rowDelimiters[++r] = i;
        }
        val[i] = coords[i].val;
        cols[i] = coords[i].y;
    }

    r = 0;

    delete[] coords;
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Simple routine to initialize input array
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxi: specifies range of random values
//
// Programmer: Lukasz Wesolowski
// Creation: June 21, 2010
// Returns:  nothing
//
// ****************************************************************************
template <typename floatType>
void fill(floatType *A, const int n, const float maxi)
{
    for (int j = 0; j < n; j++)
    {
        A[j] = ((floatType) maxi * (rand() / (RAND_MAX + 1.0f)));
    }
}

// ****************************************************************************
// Function initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix, A.  The function encodes these positions in compressed sparse
//   row format.
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows of A;
//                  last element is the index one past the last element of A
//   n:             number of nonzero elements in A
//   dim:           number of rows/columns in A
//
// Programmer: Kyle Spafford
// Creation: July 28, 2010
// Returns: nothing
//
// ****************************************************************************
void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim)
{
    int nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)n / ((double)dim * (double)dim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
    for (int i = 0; i < dim; i++)
    {
        rowDelimiters[i] = nnzAssigned;
        for (int j = 0; j < dim; j++)
        {
            int numEntriesLeft = (dim * dim) - ((i * dim) + j);
            int needToAssign   = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    rowDelimiters[dim] = n;
    assert(nnzAssigned == n);
}

// ****************************************************************************
// Function printSparse
//
// Purpose:
//   Prints a sparse matrix in dense form for debugging purposes
//
// Arguments:
//   A: array holding the non-zero values for the matrix
//   n: number of elements in A
//   dim: number of rows/columns in the matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of A;
//               last element is the index one past the last element of A
//
// Programmer: Lukasz Wesolowski
// Creation: June 22, 2010
// Returns: nothing
//
// ****************************************************************************
template <typename floatType>
void printSparse(floatType *A, int n, int dim, int *cols, int *rowDelimiters)
{

    int colIndex;
    int zero = 0;

    for (int i=0; i<dim; i++)
    {
        colIndex = 0;
        for (int j=rowDelimiters[i]; j<rowDelimiters[i+1]; j++)
        {
            while (colIndex++ < cols[j])
            {
                printf("%7d ", zero);
            }
            printf("%1.1e ", A[j]);;
        }
        while (colIndex++ < dim)
        {
            printf("%7d ", zero);
        }
        printf("\n");
    }

}

// ****************************************************************************
// Function: convertToColMajor
//
// Purpose:
//   Converts a sparse matrix representation whose data structures are
//   in row-major format into column-major format.
//
// Arguments:
//   A: array holding the non-zero values for the matrix in
//      row-major format
//   cols: array of column indices of the sparse matrix in
//         row-major format
//   dim: number of rows/columns in the matrix
//   rowDelimiters: array holding indices in A to rows of the sparse matrix
//   newA: input - buffer of size dim * maxrl
//         output - A in ELLPACK-R format
//   newcols: input - buffer of same size as newA
//            output - cols in ELLPACK-R format
//   rl: array storing length of every row of A
//   maxrl: maximum number of non-zero elements per row in A
//   padded: indicates whether newA should be padded so that the
//           number of rows divides PAD_FACTOR
//
// Programmer: Lukasz Wesolowski
// Creation: June 29, 2010
// Returns:
//   nothing directly
//   newA and newcols indirectly through pointers
// ****************************************************************************
template <typename floatType>
void convertToColMajor(floatType *A, int *cols, int dim, int *rowDelimiters,
                       floatType *newA, int *newcols, int *rl, int maxrl,
                       bool padded)
{
    int pad = 0;
    if (padded && dim % PAD_FACTOR != 0)
    {
        pad = PAD_FACTOR - dim % PAD_FACTOR;
    }

    int newIndex = 0;
    for (int j=0; j<maxrl; j++)
    {
        for (int i=0; i<dim; i++)
        {
            if (rowDelimiters[i] + j < rowDelimiters[i+1])
            {
                newA[newIndex] = A[rowDelimiters[i]+j];
                newcols[newIndex] = cols[rowDelimiters[i]+j];
            }
            else
            {
                newA[newIndex] = 0;
            }
            newIndex++;
        }
        if (padded)
        {
            for (int p=0; p<pad; p++)
            {
                newA[newIndex] = 0;
                newIndex++;
            }
        }
    }
}

// ****************************************************************************
// Function: convertToPadded
//
// Purpose: pads a CSR matrix with zeros so that each line of values
//          for the matrix is aligned to PAD_FACTOR*4 bytes
//
// Arguments:
//   A: array holding the non-zero values for the matrix
//   cols: array of column indices of the sparse matrix
//   dim: number of rows/columns in the matrix
//   rowDelimiters: array holding indices in A to rows of the sparse matrix
//   newA_ptr: input - pointer to an uninitialized pointer
//             output - pointer to padded A
//   newcols_ptr: input - pointer to an uninitialized pointer
//                output - pointer to padded cols
//   newIndices: input - buffer of size dim + 1
//               output - array holding indices in newA to rows of the
//                        sparse matrix
//   newSize: input - pointer to uninitialized int
//            output - pointer to the size of A
//
// Programmer: Lukasz Wesolowski
// Creation: July 8, 2010
// Returns:
//   nothing directly
//   allocates and returns *newA_ptr and *newcols_ptr indirectly
//   returns newIndices and newSize indirectly through pointers
// ****************************************************************************
template <typename floatType>
void convertToPadded(floatType *A, int *cols, int dim, int *rowDelimiters,
                     floatType **newA_ptr, int **newcols_ptr, int *newIndices,
                     int *newSize)
{

    // determine total padded size and new row indices
    int paddedSize = 0;
    int rowSize;

    for (int i=0; i<dim; i++)
    {
        newIndices[i] = paddedSize;
        rowSize = rowDelimiters[i+1] - rowDelimiters[i];
        if (rowSize % PAD_FACTOR != 0)
        {
            rowSize += PAD_FACTOR - rowSize % PAD_FACTOR;
        }
        paddedSize += rowSize;
    }
    *newSize = paddedSize;
    newIndices[dim] = paddedSize;

    *newA_ptr = pmsAllocHostBuffer<floatType>( paddedSize );
    *newcols_ptr = pmsAllocHostBuffer<int>( paddedSize );

    floatType *newA = *newA_ptr;
    int *newcols = *newcols_ptr;

    memset(newA, 0, paddedSize * sizeof(floatType));
    memset(newcols, 0, paddedSize*sizeof(int));

    // fill newA and newcols
    for (int i=0; i<dim; i++)
    {
        for (int j=rowDelimiters[i], k=newIndices[i]; j<rowDelimiters[i+1];
             j++, k++)
        {
            newA[k] = A[j];
            newcols[k] = cols[j];
        }
    }

}

// comparison functions used for qsort

inline int intcmp(const void *v1, const void *v2)
{
    return (*(int *)v1 - *(int *)v2);
}


inline int coordcmp(const void *v1, const void *v2)
{
    struct Coordinate *c1 = (struct Coordinate *) v1;
    struct Coordinate *c2 = (struct Coordinate *) v2;

    if (c1->x != c2->x)
    {
        return (c1->x - c2->x);
    }
    else
    {
        return (c1->y - c2->y);
    }
}

#endif // SPMV_UTIL_H_
