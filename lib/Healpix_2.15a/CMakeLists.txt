cmake_minimum_required(VERSION 2.8)

project(healpix)

add_library(
    healpix STATIC
    cxxutils.cc
    healpix_base.cc
)
