#include "hk/Dialect/HK/IR/HKDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hk;

#define GET_TYPEDEF_CLASSES
#include "hk/Dialect/HK/IR/HKOpsTypes.cpp.inc"

void HKDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hk/Dialect/HK/IR/HKOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Parsing helpers
//===----------------------------------------------------------------------===//

static ParseResult parseRowCol(AsmParser &p, bool &col) {
  StringRef kw;
  SMLoc at = p.getCurrentLocation();
  if (p.parseKeyword(&kw))
    return failure();
  if (kw == "row")
    col = false;
  else if (kw == "col")
    col = true;
  else
    return p.emitError(at, "expected 'row' or 'col', got '") << kw << "'";
  return success();
}

/// A tile's layout: `row`, `col`, or `strip<RUN>`.
static ParseResult parseTileLayout(AsmParser &p, bool &col, unsigned &run) {
  StringRef kw;
  SMLoc at = p.getCurrentLocation();
  run = 0;
  col = false;
  if (p.parseKeyword(&kw))
    return failure();
  if (kw == "row")
    return success();
  if (kw == "col") {
    col = true;
    return success();
  }
  if (kw == "strip")
    return failure(p.parseLess() || p.parseInteger(run) || p.parseGreater());
  return p.emitError(at, "expected 'row', 'col' or 'strip<RUN>', got '")
         << kw << "'";
}

static void printTileLayout(AsmPrinter &p, bool col, unsigned run) {
  if (run)
    p << "strip<" << run << ">";
  else
    p << (col ? "col" : "row");
}

static ParseResult parseExtents(AsmParser &p, unsigned &rows, unsigned &cols) {
  SmallVector<int64_t, 2> dims;
  SMLoc at = p.getCurrentLocation();
  if (p.parseDimensionList(dims, /*allowDynamic=*/false,
                           /*withTrailingX=*/false))
    return failure();
  if (dims.size() != 2)
    return p.emitError(at, "expected ROWSxCOLS");
  rows = dims[0];
  cols = dims[1];
  return success();
}

static LogicalResult checkExtent(function_ref<InFlightDiagnostic()> emitError,
                                 StringRef what, StringRef n, unsigned v) {
  if (v == 0 || v % 16)
    return emitError() << what << " " << n << "=" << v
                       << " must be a positive multiple of 16";
  return success();
}

// i1 (`b1`, a mask) is known only as a strip's element: hk.map's comparisons
static bool knownDtype(Type t) { return !t.isInteger(1) && !dtypeName(t).empty(); }

//===----------------------------------------------------------------------===//
// rt
//===----------------------------------------------------------------------===//

/// `, frag RxC`, if present.
static ParseResult parseOptionalFrag(AsmParser &p, unsigned &rows,
                                     unsigned &cols) {
  rows = cols = 0;
  if (failed(p.parseOptionalKeyword("frag")))
    return success();
  return parseExtents(p, rows, cols);
}

Type RegTileType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols, fr = 0, fc = 0, run, stride = 0;
  bool col;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols) || p.parseComma() ||
      parseTileLayout(p, col, run))
    return {};
  if (succeeded(p.parseOptionalComma()) && parseOptionalFrag(p, fr, fc))
    return {};
  if (fr && succeeded(p.parseOptionalKeyword("stride")) &&
      p.parseInteger(stride))
    return {};
  if (p.parseGreater())
    return {};
  return RegTileType::getChecked([&] { return p.emitError(at); },
                                 p.getContext(), et, rows, cols, col, fr, fc,
                                 run, stride);
}

void RegTileType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getRows() << "x" << getCols() << ", ";
  printTileLayout(p, getColLayout(), getRun());
  if (hasFrag())
    p << ", frag " << getFragRows() << "x" << getFragCols();
  if (getFragStride())
    p << " stride " << getFragStride();
  p << ">";
}

LogicalResult RegTileType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type et, unsigned rows, unsigned cols,
                                  bool col, unsigned fr, unsigned fc,
                                  unsigned run, unsigned stride) {
  if (stride) {
    // include/cdna4's rt_shape list: rt_32x32_8, rt_32x16_4, rt_16x32_4 are
    // the shapes with a lane run other than their own.
    bool known = (fr == 32 && fc == 32 && stride == 8) ||
                 (fr == 32 && fc == 16 && stride == 4) ||
                 (fr == 16 && fc == 32 && stride == 4);
    if (!known)
      return emitError() << "register tile fragment " << fr << "x" << fc
                         << " stride " << stride
                         << ": the strided fragments are 32x32 stride 8, "
                            "32x16 stride 4 and 16x32 stride 4 (a shape's "
                            "own stride is written without one)";
  }
  if (run) {
    if (!et.isInteger(1) && (!knownDtype(et) || dtypeBits(et) < 16))
      return emitError() << "there is no " << dtypeName(et)
                         << " strip: rs holds 16- and 32-bit elements, and b1 masks";
    if (col || fr)
      return emitError() << "a strip has no col layout and no fragment";
    if (run > 16 || (run & (run - 1)))
      return emitError() << "strip run " << run
                         << " must be a power of two of at most 16";
    if (rows == 0)
      return emitError() << "a strip has at least one row";
    if (cols == 0 || cols % (32 * run))
      return emitError() << "strip cols=" << cols
                         << " must be a positive multiple of 32*run = "
                         << 32 * run << ": a whole number of chunks";
    return success();
  }
  if ((fr == 0) != (fc == 0) || (fr && (fr % 16 || fc % 16)))
    return emitError() << "register tile fragment " << fr << "x" << fc
                       << ": both extents, multiples of 16 (or unset)";
  if (fr && (rows % fr || cols % fc))
    return emitError() << "register tile " << rows << "x" << cols
                       << " is not a whole number of " << fr << "x" << fc
                       << " fragments";
  if (!knownDtype(et))
    return emitError() << "register tile of " << et
                       << ": not an element type include/rdna3 has";
  if (dtypeBits(et) < 16 && !isFp8(et) && !et.isInteger(8))
    return emitError() << "there is no " << dtypeName(et)
                       << " register tile: rt_base exists only for 16- and "
                          "32-bit elements (fp8 on gfx1201, i8 operands on gfx1100 and gfx1201)";
  if (failed(checkExtent(emitError, "register tile", "rows", rows)) ||
      failed(checkExtent(emitError, "register tile", "cols", cols)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// st
//===----------------------------------------------------------------------===//

Type SharedTileType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols, count = 1;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols))
    return {};
  if (succeeded(p.parseOptionalComma()) && p.parseInteger(count))
    return {};
  if (p.parseGreater())
    return {};
  return SharedTileType::getChecked([&] { return p.emitError(at); },
                                    p.getContext(), et, rows, cols, count);
}

void SharedTileType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getRows() << "x" << getCols();
  if (getCount() != 1)
    p << ", " << getCount();
  p << ">";
}

LogicalResult
SharedTileType::verify(function_ref<InFlightDiagnostic()> emitError, Type et,
                       unsigned rows, unsigned cols, unsigned count) {
  if (!knownDtype(et))
    return emitError() << "shared tile of " << et;
  if (failed(checkExtent(emitError, "shared tile", "rows", rows)) ||
      failed(checkExtent(emitError, "shared tile", "cols", cols)))
    return failure();
  if (count == 0)
    return emitError() << "shared tile count must be positive";
  return success();
}

uint64_t SharedTileType::getNumBytes() const {
  return uint64_t(getRows()) * getCols() * (dtypeBits(getElementType()) / 8) *
         getCount();
}

//===----------------------------------------------------------------------===//
// rv
//===----------------------------------------------------------------------===//

Type RegVecType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols, fr = 0, fc = 0, run;
  bool tileCol, kindCol, uniform = false;
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols) || p.parseComma() ||
      parseTileLayout(p, tileCol, run) || p.parseComma() ||
      parseRowCol(p, kindCol))
    return {};
  while (succeeded(p.parseOptionalComma())) {
    if (succeeded(p.parseOptionalKeyword("uniform"))) {
      uniform = true;
      continue;
    }
    if (p.parseKeyword("frag") || parseExtents(p, fr, fc))
      return {};
  }
  if (p.parseGreater())
    return {};
  return RegVecType::get(p.getContext(), et, rows, cols, tileCol, kindCol,
                         uniform, fr, fc, run);
}

void RegVecType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getTileRows() << "x" << getTileCols()
    << ", ";
  printTileLayout(p, getTileColLayout(), getTileRun());
  p << ", " << getKind();
  if (getTileFragRows())
    p << ", frag " << getTileFragRows() << "x" << getTileFragCols();
  if (getUniform())
    p << ", uniform";
  p << ">";
}

//===----------------------------------------------------------------------===//
// sv
//===----------------------------------------------------------------------===//

Type SharedVecType::parse(AsmParser &p) {
  Type et;
  unsigned length, count = 1;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      p.parseInteger(length))
    return {};
  if (succeeded(p.parseOptionalComma()) && p.parseInteger(count))
    return {};
  if (p.parseGreater())
    return {};
  return SharedVecType::getChecked([&] { return p.emitError(at); },
                                   p.getContext(), et, length, count);
}

void SharedVecType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getLength();
  if (getCount() != 1)
    p << ", " << getCount();
  p << ">";
}

LogicalResult
SharedVecType::verify(function_ref<InFlightDiagnostic()> emitError, Type et,
                      unsigned length, unsigned count) {
  if (failed(checkExtent(emitError, "shared vector", "length", length)))
    return failure();
  if (count == 0)
    return emitError() << "shared vector count must be positive";
  return success();
}

uint64_t SharedVecType::getNumBytes() const {
  return uint64_t(getLength()) * getCount() * (dtypeBits(getElementType()) / 8);
}

//===----------------------------------------------------------------------===//
// coord
//===----------------------------------------------------------------------===//

Type CoordType::parse(AsmParser &p) {
  StringRef kw;
  SMLoc at = p.getCurrentLocation();
  if (p.parseLess() || p.parseKeyword(&kw) || p.parseGreater())
    return {};
  if (kw != "tile" && kw != "element") {
    p.emitError(at, "coord unit must be 'tile' or 'element'");
    return {};
  }
  return CoordType::get(p.getContext(), kw == "element");
}

void CoordType::print(AsmPrinter &p) const {
  p << "<" << (getElement() ? "element" : "tile") << ">";
}

//===----------------------------------------------------------------------===//
// granules
//===----------------------------------------------------------------------===//

Type GranulesType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols) || p.parseGreater())
    return {};
  return GranulesType::getChecked([&] { return p.emitError(at); },
                                  p.getContext(), et, rows, cols);
}

void GranulesType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getRows() << "x" << getCols() << ">";
}

/// Transcribed from st<T,R,C>::swizzle_bytes, 2-byte case.
unsigned GranulesType::getSwizzleBytes() const {
  unsigned width = getCols() / 16;
  return width % 4 == 0 ? 128 : width % 2 == 0 ? 64 : 32;
}

LogicalResult MxScalesType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   unsigned rows, unsigned kblocks, unsigned fmt) {
  if (fmt > 2)
    return emitError() << "mx scales: fmt is 0 (packed), 1 (loose) or 2 (AITER raw)";
  if (fmt == 2 && rows % 32)
    return emitError() << "mx scales: AITER raw dwords cover 32-row blocks";
  if (rows == 0 || rows % 16)
    return emitError() << "mx scales: " << rows
                       << " rows is not a whole number of 16-row base tiles";
  if (kblocks == 0 || kblocks % 4)
    return emitError() << "mx scales: " << kblocks
                       << " blocks of 32 is not a whole number of 128-wide K steps";
  return success();
}

LogicalResult GranulesType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   Type et, unsigned rows, unsigned cols) {
  if (dtypeBits(et) != 16)
    return emitError() << "granules: the decomposition is derived and checked "
                          "for 2-byte tiles only";
  unsigned width = cols / 16;
  unsigned swz = width % 4 == 0 ? 128 : width % 2 == 0 ? 64 : 32;
  if (swz < 64)
    return emitError() << "granules: swizzle_bytes is " << swz
                       << "; needs >= 32 columns of a 2-byte tile";
  return success();
}

//===----------------------------------------------------------------------===//
// stage
//===----------------------------------------------------------------------===//

Type StageType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols, threads, depth = 1;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols) || p.parseComma() || p.parseInteger(threads))
    return {};
  if (succeeded(p.parseOptionalComma()) && p.parseInteger(depth))
    return {};
  if (p.parseGreater())
    return {};
  return StageType::getChecked([&] { return p.emitError(at); },
                               p.getContext(), et, rows, cols, threads, depth);
}

void StageType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getRows() << "x" << getCols() << ", "
    << getThreads();
  if (getDepth() != 1)
    p << ", " << getDepth();
  p << ">";
}

/// Transcribed from stage_calls in global_to_shared.cuh. Must agree exactly:
/// a buffer one float4 short is an out-of-bounds write to the stack.
unsigned StageType::getCalls() const {
  unsigned perFloat4 = 16 / (dtypeBits(getElementType()) / 8);
  unsigned n = getRows() * getCols();
  unsigned chunks = n / perFloat4;
  return (chunks + getThreads() - 1) / getThreads();
}

LogicalResult StageType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type, unsigned, unsigned, unsigned threads,
                                unsigned depth) {
  if (threads == 0 || threads % 32)
    return emitError() << "stage buffer threads=" << threads
                       << " must be a multiple of 32";
  if (depth == 0)
    return emitError() << "stage buffer depth must be positive";
  return success();
}
