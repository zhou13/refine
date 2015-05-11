#define __ICC
#include <fftw3.h>
#undef __ICC

#include <moderngpu.cuh>
#include "multidim_array.h"
#include "complex.h"
#include <cmath>
#include <cassert>
#include <iostream>

using namespace mgpu;
using std::vector;


#define FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(V) \
    for (long int k = 0, kp = 0; k<ZSIZE(V); k++, kp = (k < XSIZE(V)) ? k : k - ZSIZE(V)) \
        for (long int i = 0, ip = 0 ; i<YSIZE(V); i++, ip = (i < XSIZE(V)) ? i : i - YSIZE(V)) \
            for (long int j = 0, jp = 0; j<XSIZE(V); j++, jp = j)

#define FFTW_ELEM(V, kp, ip, jp) \
    DIRECT_A3D_ELEM((V),((kp < 0) ? (kp + ZSIZE(V)) : (kp)), ((ip < 0) ? (ip + YSIZE(V)) : (ip)), (jp))

template<class T>
void windowFourierTransform(const MultidimArray<T > &in,
                            MultidimArray<T > &out,
                            long int newdim)
{
    // Check size of the input array
    if (YSIZE(in) > 1 && YSIZE(in)/2 + 1 != XSIZE(in))
        REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");
    long int newhdim = newdim/2 + 1;

    // If same size, just return input
    if (newhdim == XSIZE(in))
    {
        out = in;
        return;
    }

    // Otherwise apply a windowing operation
    // Initialise output array
    switch (in.getDim())
    {
    case 1:
        out.initZeros(newhdim);
        break;
    case 2:
        out.initZeros(newdim, newhdim);
        break;
    case 3:
        out.initZeros(newdim, newdim, newhdim);
        break;
    default:
        REPORT_ERROR("windowFourierTransform ERROR: dimension should be 1, 2 or 3!");
    }
    if (newhdim > XSIZE(in))
    {
        long int max_r2 = (XSIZE(in) -1) * (XSIZE(in) - 1);
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(in)
        {
            // Make sure windowed FT has nothing in the corners, otherwise we end up with an asymmetric FT!
            if (kp*kp + ip*ip + jp*jp <= max_r2)
                FFTW_ELEM(out, kp, ip, jp) = FFTW_ELEM(in, kp, ip, jp);
        }
    }
    else
    {
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(out)
        {
            FFTW_ELEM(out, kp, ip, jp) = FFTW_ELEM(in, kp, ip, jp);
        }
    }
}

inline float relativeDiff(double ground, float my)
{
    return abs((ground - my) / ground);
}

inline void checkDiffArray(
    const vector<double> &ground,
    const vector<float> &my,
    float diff = 0.05)
{
    assert(ground.size() == my.size());
    for (size_t i = 0; i < my.size(); ++i)
        assert(relativeDiff(ground[i], my[i]) < diff);
}
