#ifndef MATRIX2DFILESUPPORT_H
#define MATRIX2DFILESUPPORT_H

template<class T>
std::string
GetMatrixFileName( std::string baseName )
{
    // nothing to do - this should never be instantiated
    assert( false );
    return "";
}

template<>
std::string
GetMatrixFileName<float>( std::string baseName )
{
    return baseName + "-sp.dat";
}

template<>
std::string
GetMatrixFileName<double>( std::string baseName )
{
    return baseName + "-dp.dat";
}


 template<class T>
bool
SaveMatrixToFile( const Matrix2D<T>& m, std::string fileName )
{
    bool ok = true;

    std::ofstream ofs( fileName.c_str(), ios::out | ios::binary );
    if( ofs.is_open() )
    {
        ok = m.WriteTo( ofs );
        ofs.close();
    }
    else
    {
        std::cerr << "Unable to write matrix to file \'" << fileName << "\'" << std::endl;
        ok = false;
    }
    return ok;
}


template<class T>
bool
ReadMatrixFromFile( Matrix2D<T>& m, std::string fileName )
{
    bool ok = true;

    std::ifstream ifs( fileName.c_str(), ios::in | ios::binary );
    if( ifs.is_open() )
    {
        ok = m.ReadFrom( ifs );
        ifs.close();
    }
    else
    {
        std::cerr << "Unable to read matrix from file \'" << fileName << "\'" << std::endl;
        ok = false;
    }
    return ok;
}

#endif // MATRIX2DFILESUPPORT_H
