

void
ReduceDoubles( void* ivdata, unsigned int nItems, void* ovres )
{
    double sum = 0.0;
    double* idata = (double*)ivdata;
    double* ores = (double*)ovres;

#pragma acc data pcopyin(idata[0:nItems])
#pragma acc parallel reduction( +:sum )
    for( unsigned int i = 0; i < nItems; i++ )
    {
        sum += idata[i];
    }
    *ores = sum;
}


void
ReduceFloats( void* ivdata, unsigned int nItems, void* ovres )
{
    float sum = 0.0;
    float* idata = (float*)ivdata;
    float* ores = (float*)ovres;

#pragma acc data pcopyin(idata[0:nItems])
#pragma acc parallel reduction( +:sum )
    for( unsigned int i = 0; i < nItems; i++ )
    {
        sum += idata[i];
    }
    *ores = sum;
}


