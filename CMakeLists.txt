cmake_minimum_required(VERSION 2.8)
project(refine CXX C)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RELEASE")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

if(NOT MSVC)
    set(GCC_PARAMETERS "-Wall -std=c++1y")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_PARAMETERS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GCC_PARAMETERS} -g3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GCC_PARAMETERS} -O2")
    set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE} -g3")

    if(CMAKE_BUILD_TYPE STREQUAL "PROFILE")
        find_package(GProfiler REQUIRED)
        add_definitions(-DWITH_GPERFTOOLS)
        set(PROFILER ${GPROFILER_LIBRARIES})
        include_directories(${GPROFILER_INCLUDE_DIR})
    endif(CMAKE_BUILD_TYPE STREQUAL "PROFILE")

    set(CMAKE_REQUIRED_FLAGS "-Werror -fsanitize=address")
    CHECK_CXX_COMPILER_FLAG(-fsanitize=address COMPILER_SUPPORT_FSANITIZE_ADDRESS)
    unset(CMAKE_REQUIRED_FLAGS)
    if(COMPILER_SUPPORT_FSANITIZE_ADDRESS)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    endif()
endif(NOT MSVC)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
find_package(FFTW3 COMPONENTS double threads REQUIRED)
include_directories(${FFTW3_INCLUDE_DIR})
add_subdirectory(lib/Healpix_2.15a)
include_directories(lib/Healpix_2.15a)

include_directories(src)

set(
    refine_SRC

    src/args.cpp
    src/assembly.cpp
    src/backprojector.cpp
    src/complex.cpp
    src/ctf.cpp
    src/error.cpp
    src/euler.cpp
    src/exp_model.cpp
    src/fftw.cpp
    src/filename.cpp
    src/funcs.cpp
    src/healpix_sampling.cpp
    src/image.cpp
    src/mask.cpp
    src/matrix1d.cpp
    src/matrix2d.cpp
    src/memory.cpp
    src/metadata_container.cpp
    src/metadata_label.cpp
    src/metadata_table.cpp
    src/ml_model.cpp
    src/ml_optimiser.cpp
    src/ml_optimiser_mpi.cpp
    src/mpi_node.cpp
    src/multidim_array.cpp
    src/numerical_recipes.cpp
    src/parallel.cpp
    src/projector.cpp
    src/strings.cpp
    src/symmetries.cpp
    src/tabfuncs.cpp
    src/timer.cpp
    src/transformations.cpp

    src/apps/refine.cpp
)

add_executable(
    refine
    ${refine_SRC}
)

target_link_libraries(
    refine
    healpix
    ${PROFILER}
    ${FFTW3_LIBRARY}
    ${FFTW3_THREADS_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    ${MPI_CXX_LIBRARIES}
)
