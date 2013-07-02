// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Jeffrey Vetter <vetter@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011-2013, UT-Battelle, LLC
// Copyright (c) 2013, Intel Corporation
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//   
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Oak Ridge National Laboratory, nor UT-Battelle, LLC, 
//    nor the names of its contributors may be used to endorse or promote 
//    products derived from this software without specific prior written 
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
// THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <string>
#include <cassert>
#include "CommonMICStencilFactory.h"
#include "InvalidArgValue.h"


template<class T>
void 
CommonMICStencilFactory<T>::CheckOptions( const OptionParser& opts ) const
{
    // let base class check its options first
    StencilFactory<T>::CheckOptions( opts );

    // check our options
    std::vector<long long> shDims = opts.getOptionVecInt( "lsize" );
    if( shDims.size() != 2 )
    {
        throw InvalidArgValue( "lsize must have two dimensions" );
    }
    if( (shDims[0] <= 0) || (shDims[1] <= 0) )
    {
        throw InvalidArgValue( "all lsize values must be positive" );
    }

    std::vector<long long> arrayDims = opts.getOptionVecInt( "customSize" );
    assert( arrayDims.size() == 2 );

    // If both of these are zero, we're using a non-custom size, skip this test
    if (arrayDims[0] == 0 && arrayDims[0] == 0)
    {
        return;
    }

    size_t gRows = (size_t)arrayDims[0];
    size_t gCols = (size_t)arrayDims[1];
    size_t lRows = (size_t)shDims[0];
    size_t lCols = (size_t)shDims[1];

    // verify that local dimensions evenly divide global dimensions
    if( ((gRows % lRows) != 0) || (lRows > gRows) )
    {
        throw InvalidArgValue( "number of rows must be even multiple of lsize rows" );
    }
    if( ((gCols % lCols) != 0) || (lCols > gCols) )
    {
        throw InvalidArgValue( "number of columns must be even multiple of lsize columns" );
    }

    // TODO ensure local dims are smaller than CUDA implementation limits
}

template<class T>
void
CommonMICStencilFactory<T>::ExtractOptions( const OptionParser& options,
                                            T& wCenter,
                                            T& wCardinal,
                                            T& wDiagonal,
                                            std::vector<long long>& devices )
{
    // let base class extract its options
    StencilFactory<T>::ExtractOptions( options, wCenter, wCardinal, wDiagonal );

    // extract our options
    // with hardcoded lsize, we no longer have any to extract

    // determine which device to use
    // We would really prefer this to be done in main() but 
    // since BuildStencil is a virtual function, we cannot change its
    // signature, and OptionParser provides no way to override an
    // option's value after it is set during parsing.
    devices = options.getOptionVecInt("device");
}


