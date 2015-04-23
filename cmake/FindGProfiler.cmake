# Tries to find profiler from Gperftools.
#
# Usage of this module as follows:
#
#     find_package(GProfiler)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  GProfiler_ROOT_DIR  Set this variable to the root installation of
#                       Gperftools if the module has problems finding
#                       the proper installation path.
#
# Variables defined by this module:
#
#  GPROFILER_FOUND              System has Gperftools libs/headers
#  GPROFILER_LIBRARIES          The Gperftools libraries (tcmalloc & profiler)
#  GPROFILER_INCLUDE_DIR        The location of Gperftools headers

find_library(GPROFILER_LIBRARIES
  NAMES profiler
  HINTS ${GProfiler_ROOT_DIR}/lib)

find_path(GPROFILER_INCLUDE_DIR
  NAMES gperftools/heap-profiler.h
  HINTS ${GProfiler_ROOT_DIR}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  GProfiler
  DEFAULT_MSG
  GPROFILER_LIBRARIES
  GPROFILER_INCLUDE_DIR)

mark_as_advanced(
  GProfiler_ROOT_DIR
  GPROFILER_LIBRARIES
  GPROFILER_INCLUDE_DIR)
