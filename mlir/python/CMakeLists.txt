include(AddMLIRPython)

# hk's libraries carry their own LLVM and MLIR. In a process that loaded
# another LLVM first -- vLLM's _C links ROCm's libLLVM.so -- their calls into
# LLVM resolved to that copy and the first MLIR context segfaulted; bound
# locally, each library calls its own.
add_link_options("LINKER:-Bsymbolic")

# Every MLIR module lives under hk._mlir (the API is embedded relocatably).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(HKPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HKPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hk_mlir"
  TD_FILE dialects/HKOps.td
  SOURCES
    dialects/hk.py
  DIALECT_NAME hk
  GEN_ENUM_BINDINGS)

declare_mlir_python_extension(HKPythonSources.Extension
  MODULE_NAME _hkDialects
  ADD_TO_PARENT HKPythonSources
  SOURCES
    HKExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIMath
    MLIRCAPISCF
    MLIRCAPIFunc
    MLIRCAPITransforms
    HKCAPI
)

set(_hk_declared_sources
  HKPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.func
)

add_mlir_python_common_capi_library(HKPythonCAPI
  INSTALL_COMPONENT HKPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES ${_hk_declared_sources}
)

add_mlir_python_modules(HKPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_hk_declared_sources}
  COMMON_CAPI_LINK_LIBS
    HKPythonCAPI
)

# The code-object launcher (hk.runtime.codeobj): plain CPython API, no MLIR and
# no HIP -- it calls hipModuleLaunchKernel through the address the Python side
# hands it -- built here so that it ships in the wheel beside the bindings.
Python3_add_library(_hkLaunch MODULE WITH_SOABI HKLaunch.c)
set_target_properties(_hkLaunch PROPERTIES
  C_STANDARD 11
  LIBRARY_OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs")
install(TARGETS _hkLaunch
  LIBRARY DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  COMPONENT HKPythonModules)

# Every library here is linked with its install RUNPATH ($ORIGIN) from the
# start. MLIR's functions set BUILD_WITH_INSTALL_RPATH OFF, and CMake then adds
# the build tree's absolute _mlir_libs path for the three that link
# libHKPythonCAPI / libnanobind-hk; `cmake --install` overwrites it in place,
# so .dynstr keeps the old string's length (the layout follows how long the
# build directory's path is) and the build-id, hashed at link time, the old
# path. In the build tree they all sit in one _mlir_libs, so $ORIGIN is enough
# there too.
get_property(_hk_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _hk_targets)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "SHARED_LIBRARY" OR _type STREQUAL "MODULE_LIBRARY")
    set_target_properties(${_t} PROPERTIES BUILD_WITH_INSTALL_RPATH ON)
  endif()
endforeach()
