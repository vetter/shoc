#ifndef STENCILFACTORY_H
#define STENCILFACTORY_H

#include <map>
#include "OptionParser.h"
#include "Stencil.h"

// ****************************************************************************
// Class:  StencilFactory
//
// Purpose:
//   Class to generate stencils.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class StencilFactory
{
public:
    typedef std::map<std::string, StencilFactory*> FactoryMap;

private:
    // map of class name to a StencilFactory object
    // would be much easier if C++ classes were first class objects
    // so that we could programmatically construct a class name and
    // then create an instance of that class
    static FactoryMap* factoryMap;

    std::string sname;

protected:
    void ExtractOptions( const OptionParser& options,
                        T& wCenter,
                        T& wCardinal,
                        T& wDiagonal );

public:
    StencilFactory( std::string _sname )
      : sname( _sname )
    {
        // nothing else to do
    }
    virtual ~StencilFactory( void ) { }

    std::string GetStencilName( void ) { return sname; }

    virtual Stencil<T>* BuildStencil( const OptionParser& options ) = 0;
    virtual void CheckOptions( const OptionParser& options ) const = 0;

    static std::vector<long long> GetStandardProblemSize( int sizeClass );
};

#endif // STENCILFACTORY_H
