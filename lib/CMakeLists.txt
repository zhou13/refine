cmake_minimum_required(VERSION 2.8)

include_directories(Healpix_2.15a)
add_library(
    healpix STATIC
    Healpix_2.15a/cxxutils.cc
    Healpix_2.15a/healpix_base.cc
)

include_directories(moderngpu/include)
cuda_add_library(
    moderngpu STATIC
    moderngpu/src/mgpucontext.cu
    moderngpu/src/mgpuutil.cpp
)
