#define __ICC
#include <fftw3.h>
#undef __ICC
#include <fftw.h>

#include <moderngpu.cuh>
#include "multidim_array.h"
#include <inttypes.h>
#include "complex.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <cstdint>

using namespace mgpu;
using std::vector;


inline int align_address(int address, int alignment)
{
    if (address % alignment == 0)
        return address;
    address &= -alignment;
    address += alignment;
    return address;
}

inline float relative_diff(double ground, float my)
{
    return abs((ground - my) / ground);
}

inline bool check_relative_diff(double ground, double my, double delta, double error, const char * msg, bool on_exit = true)
{
    if (relative_diff(abs(ground) + delta, abs(my) + delta) > error) {
        if (on_exit) {
            fprintf(stderr, "    check_relative_diff error at %s\n", msg);
            fprintf(stderr, "        ground: %.8f; my: %.8f\n", ground, my);
            assert(false);
        }
        return false;
    }
    return true;
}

inline float check_relative_diff(MultidimArray<double> &ground, MultidimArray<double> &my, double delta, double error, const char * msg, bool on_print = true)
{
    if (ground.nzyxdim != my.nzyxdim) {
        fprintf(stderr, "        nzyxdim %d vs %d is not same: %s\n",(int)ground.nzyxdim, (int)my.nzyxdim, msg);
        exit(1);
    }

    int nr_error = 0;
    int nr_total = 0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ground) {
        nr_error += !check_relative_diff(DIRECT_MULTIDIM_ELEM(ground, n),
                                         DIRECT_MULTIDIM_ELEM(my, n),
                                         delta, error, msg, false);
        nr_total ++;
    }

    if (on_print && nr_error) {
        fprintf(stderr, "    %s has %.3f%% error rate\n", msg, 100.f * nr_error / nr_total);
    }
    return float(nr_error) / nr_total;
}

inline float check_relative_diff(MultidimArray<Complex> &ground, MultidimArray<Complex> &my, double delta, double error, const char * msg, bool on_print = true)
{
    if (ground.nzyxdim != my.nzyxdim) {
        fprintf(stderr, "nzyxdim %d vs %d is not same: %s\n",(int)ground.nzyxdim, (int)my.nzyxdim, msg);
        exit(1);
    }

    int nr_error = 0;
    int nr_total = 0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ground) {
        nr_error += !check_relative_diff(DIRECT_MULTIDIM_ELEM(ground, n).real,
                                         DIRECT_MULTIDIM_ELEM(my, n).real,
                                         delta, error, (std::string(msg) + "'s real").c_str(),
                                         false);
        nr_error += !check_relative_diff(DIRECT_MULTIDIM_ELEM(ground, n).imag,
                                         DIRECT_MULTIDIM_ELEM(my, n).imag,
                                         delta, error, (std::string(msg) + "'s imag").c_str(),
                                         false);
        nr_total += 2;
    }
    if (on_print && nr_error) {
        fprintf(stderr, "    %s has %.3f%% error rate\n", msg, 100.f * nr_error / nr_total);
    }
    return float(nr_error) / nr_total;
}


inline void check_diff_array(
    const vector<double> &ground,
    const vector<float> &my,
    float diff = 0.05)
{
    assert(ground.size() == my.size());
    for (size_t i = 0; i < my.size(); ++i) {
        assert(!isnan(my[i]));
        assert(!isnan(ground[i]));
        if (relative_diff(ground[i], my[i]) >= diff) {
            fprintf(stderr, "    Incorrect result at (%d), ground is %.3f, mine is %.3f\n", (int)i, ground[i], my[i]);
            assert(false);
        }
    }
}

inline __host__ __device__ uint32_t flipFloat(float fl)
{
    union {
        float fl;
        uint32_t  u;
    } un;
    un.fl = fl;
    return un.u ^ ((int(un.u) >> 31) | 0x80000000);
}

inline __host__ __device__ float reverseFlipFloat(uint32_t u)
{
    union {
        float f;
        uint32_t u;
    } un;
    un.u = u ^ ((int(~u) >> 31) | 0x80000000);
    return un.f;
}

#define DIE(args...) { fprintf(stderr, args); exit(1); }
#define DIE_IF(cond, args...) { if (cond) { fprintf(stderr, args); exit(1); } }
#define CUDA_ASSERT(X) \
    if ( !(X) ) { \
        printf( "Thread %d:%d failed assert at %s:%d!\n", \
                blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); \
        return; \
    }
