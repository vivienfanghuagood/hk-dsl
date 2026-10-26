cmake_minimum_required(VERSION 3.20.0)
project(hk-mlir LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(HK_MLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(HK_MLIR_BINARY_DIR ${PROJECT_BINARY_DIR})

# Reproducible binaries. Assertion messages carry __FILE__, and debug info the
# paths of every source and header, so without this the bindings name the
# checkout, the build directory, the MLIR install and nanobind wherever they
# happen to be: two builds of one commit in different directories differed in
# those bytes and, through them, in every library's GNU build-id (0.7.0's rc and
# tag wheels). -ffile-prefix-map (= -fmacro-prefix-map + -fdebug-prefix-map)
# rewrites each to a fixed name under /hk/. GCC uses the last map that matches,
# so the maps go shortest first and a build directory inside the checkout gets
# its own name (clang 20 takes the longest match, the same answer). Paths of
# the prebuilt MLIR's own build, inside its static libraries, are fixed by that
# install (docs/building.md, "Reproducible wheels").
option(HK_REPRODUCIBLE_PATHS "Map build-machine paths out of __FILE__ and debug info" ON)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-ffile-prefix-map=/a=/b" HK_HAVE_FILE_PREFIX_MAP)
function(hk_map_prefixes)
  if(NOT HK_REPRODUCIBLE_PATHS OR NOT HK_HAVE_FILE_PREFIX_MAP)
    return()
  endif()
  # pairs of <dir> <name>; each dir both as given and with symlinks resolved
  set(maps)
  while(ARGN)
    list(POP_FRONT ARGN dir name)
    if(dir STREQUAL "")
      continue()
    endif()
    file(REAL_PATH "${dir}" real)
    foreach(d IN ITEMS "${dir}" "${real}")
      string(LENGTH "${d}" len)
      list(APPEND maps "${len}|${d}=${name}")
    endforeach()
  endwhile()
  list(REMOVE_DUPLICATES maps)
  list(SORT maps COMPARE NATURAL)
  foreach(m IN LISTS maps)
    string(REGEX REPLACE "^[0-9]+\\|" "" m "${m}")
    add_compile_options("-ffile-prefix-map=${m}")
    message(STATUS "hk: -ffile-prefix-map=${m}")
  endforeach()
endfunction()
get_filename_component(HK_REPO_DIR "${PROJECT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(HK_MLIR_PREFIX "${MLIR_CMAKE_DIR}/../../.." ABSOLUTE)
get_filename_component(HK_LLVM_PREFIX "${LLVM_CMAKE_DIR}/../../.." ABSOLUTE)
set(HK_PREFIX_MAPS "${HK_REPO_DIR}" /hk/src "${PROJECT_BINARY_DIR}" /hk/build
                   "${HK_MLIR_PREFIX}" /hk/mlir)
if(NOT HK_LLVM_PREFIX STREQUAL HK_MLIR_PREFIX)   # an MLIR build tree, not an install
  list(APPEND HK_PREFIX_MAPS "${HK_LLVM_PREFIX}" /hk/llvm)
endif()
hk_map_prefixes(${HK_PREFIX_MAPS})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${HK_MLIR_SOURCE_DIR}/include)
include_directories(${HK_MLIR_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling the hk._mlir Python package")
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  # nanobind's sources are compiled into the bindings too (NB_DIR is its root)
  hk_map_prefixes("${NB_DIR}" /hk/nanobind)
  set(MLIR_PYTHON_PACKAGE_PREFIX "hk._mlir" CACHE STRING "" FORCE)
  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "hk" CACHE STRING "" FORCE)
  set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "hk/_mlir" CACHE STRING "" FORCE)
  add_subdirectory(python)
endif()

add_subdirectory(test)
