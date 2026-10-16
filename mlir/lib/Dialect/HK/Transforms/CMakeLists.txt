add_mlir_dialect_library(MLIRHKTransforms
  Passes.cpp
  Fragments.cpp
  LdsPipeline.cpp

  DEPENDS
  MLIRHKPassesIncGen

  LINK_LIBS PUBLIC
  MLIRHK
  HKSupport
  HKTarget
  MLIRPass
)
