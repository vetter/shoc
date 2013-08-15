#ifndef PMSMEMMGR_H
#define PMSMEMMGR_H

template<typename T>
class PMSMemMgr
{
public:
    virtual T* AllocHostBuffer( size_t nItems ) = 0;
    virtual void ReleaseHostBuffer( T* buf ) = 0;
};


template<typename T>
class DefaultPMSMemMgr : public PMSMemMgr<T>
{
public:
    virtual T* AllocHostBuffer( size_t nItems )
    {
        return new T[nItems];
    }

    virtual void ReleaseHostBuffer( T* buf )
    {
        delete[] buf;
    }
};

#endif // PMSMEMMGR_H
