#ifdef SINGLE_PRECISION
#define POSVECTYPE float4
#define FORCEVECTYPE float4
#define FPTYPE float
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define POSVECTYPE double4
#define FORCEVECTYPE double4
#define FPTYPE double
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define POSVECTYPE double4
#define FORCEVECTYPE double4
#define FPTYPE double
#endif

__kernel void compute_lj_force(__global FORCEVECTYPE *force,
                               __global POSVECTYPE *position,
                               const int neighCount,
                               __global int* neighList,
                               const FPTYPE cutsq,
                               const FPTYPE lj1,
                               const FPTYPE lj2,
                               const int inum)
{
    uint idx = get_global_id(0);

    POSVECTYPE ipos = position[idx];
    FORCEVECTYPE f = {0.0f, 0.0f, 0.0f, 0.0f};

    int j = 0;
    while (j < neighCount)
    {
        int jidx = neighList[j*inum + idx];

        // Uncoalesced read
        POSVECTYPE jpos = position[jidx];

        // Calculate distance
        FPTYPE delx = ipos.x - jpos.x;
        FPTYPE dely = ipos.y - jpos.y;
        FPTYPE delz = ipos.z - jpos.z;
        FPTYPE r2inv = delx*delx + dely*dely + delz*delz;

        // If distance is less than cutoff, calculate force
        if (r2inv < cutsq)
        {
            r2inv = 1.0f/r2inv;
            FPTYPE r6inv = r2inv * r2inv * r2inv;
            FPTYPE forceC = r2inv*r6inv*(lj1*r6inv - lj2);

            f.x += delx * forceC;
            f.y += dely * forceC;
            f.z += delz * forceC;
        }
        j++;
    }
    // store the results
    force[idx] = f;
}
