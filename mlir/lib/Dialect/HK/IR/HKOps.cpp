#include "hk/Dialect/HK/IR/HKDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::hk;

#define GET_OP_CLASSES
#include "hk/Dialect/HK/IR/HKOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// An op that makes a tile has a result; one that writes a tile has `into`.
/// Never both, never neither.
static LogicalResult verifyDest(Operation *op, Value dst, Value result) {
  if (dst && result)
    return op->emitOpError("has both a destination and a result");
  if (!dst && !result)
    return op->emitOpError("needs a destination (`into %x`) or a result");
  return success();
}

static Type destType(Value dst, Value result) {
  return dst ? dst.getType() : result.getType();
}

static Type elementOf(Type t) {
  if (auto r = dyn_cast<RegTileType>(t))
    return r.getElementType();
  if (auto v = dyn_cast<RegVecType>(t))
    return v.getElementType();
  return {};
}

/// Two types that compile to the same C++: rv's identity is (element,
/// length, kind, layout), not the tile it carries.
static bool sameCpp(Type a, Type b) {
  auto va = dyn_cast<RegVecType>(a);
  auto vb = dyn_cast<RegVecType>(b);
  if (va && vb)
    return va.sameCpp(vb);
  return a == b;
}

static bool isStrip(Type t) {
  if (auto r = dyn_cast<RegTileType>(t))
    return r.isStrip();
  if (auto v = dyn_cast<RegVecType>(t))
    return v.isStrip();
  return false;
}

/// The ops whose C++ exists only for WMMA-layout tiles.
static LogicalResult noStrip(Operation *op, Type t) {
  if (isStrip(t))
    return op->emitOpError()
           << "takes WMMA-layout tiles, not the strip " << t
           << ": a strip has loads, stores, maps, row and column reductions, "
              "broadcasts and left/right fills, and nothing else";
  return success();
}

static LogicalResult checkMapDtype(Operation *op, Type t) {
  static const llvm::StringSet<> fp32Only = {"gelu", "dgelu", "sqrt", "rsqrt",
                                             "silu", "step"};
  StringRef name = op->getName().stripDialect();
  StringRef dn = dtypeName(elementOf(t));
  bool ok = fp32Only.contains(name)
                ? dn == "fp32"
                : (dn == "bf16" || dn == "fp16" || dn == "fp32");
  if (!ok)
    return op->emitOpError() << "has no " << dn
                             << " implementation in include/rdna3";
  return success();
}

//===----------------------------------------------------------------------===//
// Indices
//===----------------------------------------------------------------------===//

LogicalResult AnyNeOp::verify() {
  if (!getLhs().getType().sameCpp(getRhs().getType()))
    return emitOpError("compares vectors of different shapes");
  return success();
}

//===----------------------------------------------------------------------===//
// Global memory
//===----------------------------------------------------------------------===//

/// A strip moves in elements; its row vector is a one-row strip and moves
/// the same way, bounded or not. Its col vector loads one value per row and
/// is stored only with store_scalar.
static LogicalResult checkStripAccess(Operation *op, Type t, CoordType idx,
                                      bool bounded, bool store) {
  if (!isStrip(t))
    return success();
  if (!idx.getElement())
    return op->emitOpError("a strip indexes in elements, not tiles");
  auto v = dyn_cast<RegVecType>(t);
  if (v && v.getColKind() && (bounded || store))
    return op->emitOpError(store ? "stores a strip's col vector; use store_scalar"
                                 : "bounded loads a strip, or its row vector");
  return success();
}

/// The cache-policy field of a scoped access: 0 (none), 16 (SC1: device
/// scope) or 17 (SC0|SC1: system scope) -- the two hk.comm uses on gfx950 --
/// or 2 (NT: a "stream" tensor, data a launch reads once).
static LogicalResult verifyAux(Operation *op, int64_t aux) {
  if (aux != 0 && aux != 2 && aux != 16 && aux != 17)
    return op->emitOpError("aux ") << aux << " is 0, 2 (non-temporal), 16 (device scope) or 17 (system scope)";
  return success();
}

/// What a scoped access moves: a register tile (hk_load_tile / hk_store_tile carry the bits), or a
/// register vector of a 32-bit global (each element a relaxed atomic of the scope); not a strip.
static bool scopedAccessOk(Type value, Type globalElement, bool period) {
  if (period || isStrip(value))
    return false;
  if (isa<RegTileType>(value))
    return true;
  return isa<RegVecType>(value) && globalElement.getIntOrFloatBitWidth() == 32;
}

LogicalResult LoadOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  Type t = destType(getDst(), getResult());
  if (getBounded() && !isa<RegTileType>(t) && !isStrip(t))
    return emitOpError("bounded predicates tile rows; the destination is not a tile");
  if (auto p = getPeriod()) {
    auto s = dyn_cast<RegTileType>(t);
    if (!s || !s.isStrip())
      return emitOpError("period repeats a strip's columns; the destination is ")
             << t;
    if (getBounded())
      return emitOpError("period reads `period` columns from idx.c, which the "
                         "caller has bounded; drop `bounded`");
    if (*p <= 0 || *p % s.getRun())
      return emitOpError("period ")
             << *p << " is not a positive number of whole runs of " << s.getRun();
  }
  if (failed(verifyAux(*this, getAux())))
    return failure();
  if (getAux() && !scopedAccessOk(t, getSrc().getType().getElementType(), getPeriod().has_value()))
    return emitOpError("a scoped load (aux) reads a register tile, or a register vector of 32-bit "
                       "elements");
  return checkStripAccess(*this, t, getIdx().getType(), getBounded(), false);
}

LogicalResult StoreOp::verify() {
  Type t = getValue().getType();
  if (getBounded() && !isa<RegTileType>(t) && !isStrip(t))
    return emitOpError("bounded predicates tile rows; the value is not a tile");
  if (failed(verifyAux(*this, getAux())))
    return failure();
  if (getAux() && !scopedAccessOk(t, getDst().getType().getElementType(), false))
    return emitOpError("a scoped store (aux) writes a register tile, or a register vector of 32-bit "
                       "elements");
  return checkStripAccess(*this, t, getIdx().getType(), getBounded(), true);
}

LogicalResult LoadRowsOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  return noStrip(*this, destType(getDst(), getResult()));
}

LogicalResult StoreRowsOp::verify() {
  return noStrip(*this, getValue().getType());
}

LogicalResult StoreScalarOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getValue().getType().getUniform())
    return emitOpError("needs a uniform vector: every entry the same value");
  return success();
}

LogicalResult LoadScalarOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getSrc().getType().getElementType().isInteger(32))
    return emitOpError("reads an int32 global");
  return success();
}

LogicalResult WgArriveOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getCounter().getType().getElementType().isInteger(32))
    return emitOpError("counts in an int32 global");
  return success();
}

/// Gather and scatter: a lane's elements of a row-layout base tile are all in
/// its row, which is what lets each lane address its own row.
static LogicalResult checkIndexedRows(Operation *op, Type tile, GlobalType idx) {
  if (!idx.getElementType().isInteger(32))
    return op->emitOpError("takes its row indices from an int32 global");
  if (failed(noStrip(op, tile)))
    return failure();
  if (cast<RegTileType>(tile).getColLayout())
    return op->emitOpError("moves row-layout tiles");
  return success();
}

LogicalResult GatherRowsOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  return checkIndexedRows(*this, destType(getDst(), getResult()),
                          getIdx().getType());
}

LogicalResult ScatterRowsOp::verify() {
  if (getStaged() < 0)
    return emitOpError("staged is the staged id count (> 0), or 0");
  if (getAdd()) {
    Type d = getDst().getType().getElementType();
    if (!d.isF32() && !d.isBF16())
      return emitOpError("adds rows into an fp32 or bf16 global");
    if (failed(noStrip(*this, getValue().getType())))
      return failure();
  }
  return checkIndexedRows(*this, getValue().getType(), getIdx().getType());
}

LogicalResult AtomicAddOp::verify() {
  // A 16-bit operand tile is replicated across the wave halves; adding one
  // would count every element twice.
  if (!getValue().getType().getElementType().isF32() ||
      !getDst().getType().getElementType().isF32())
    return emitOpError("adds an fp32 tile into an fp32 global");
  return noStrip(*this, getValue().getType());
}

LogicalResult AtomicScalarOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getDst().getType().getElementType().isInteger(32))
    return emitOpError("works on an int32 global");
  static const char *kinds[] = {"add", "max", "min", "or", "and", "xor", "exch"};
  if (llvm::none_of(kinds, [&](const char *k) { return getKind() == k; }))
    return emitOpError("kind is add, max, min, or, and, xor or exch");
  return verifyAux(*this, getAux());
}

LogicalResult SignalOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getDst().getType().getElementType().isInteger(32))
    return emitOpError("signals an int32 element");
  if (!getAux())
    return emitOpError("a signal has a scope: aux 16 (device) or 17 (system)");
  return verifyAux(*this, getAux());
}

LogicalResult ArriveOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getDst().getType().getElementType().isInteger(32))
    return emitOpError("counts in an int32 element");
  if (!getAux())
    return emitOpError("an arrival has a scope: aux 16 (device) or 17 (system)");
  return verifyAux(*this, getAux());
}

LogicalResult WaitOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getSrc().getType().getElementType().isInteger(32))
    return emitOpError("polls an int32 element");
  if (getCmp() != "ge" && getCmp() != "eq")
    return emitOpError("cmp is \"ge\" or \"eq\"");
  if (!getAux())
    return emitOpError("a wait has a scope: aux 16 (device) or 17 (system)");
  if (getTimeout() < 0)
    return emitOpError("timeout is a tick count >= 0");
  if (bool(getErr()) != bool(getErrIdx()))
    return emitOpError("err and its element come together");
  if (getErr()) {
    if (!getErr().getType().getElementType().isInteger(32))
      return emitOpError("err is an int32 element");
    if (!getErrIdx().getType().getElement())
      return emitOpError("err takes an element coordinate");
  }
  return verifyAux(*this, getAux());
}

LogicalResult PostOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getDst().getType().getElementType().isInteger(32))
    return emitOpError("a mailbox is int32 elements");
  if (!getAux())
    return emitOpError("a post has a scope: aux 16 (device) or 17 (system)");
  return verifyAux(*this, getAux());
}

LogicalResult TakeOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getSrc().getType().getElementType().isInteger(32))
    return emitOpError("a mailbox is int32 elements");
  if (!getAux())
    return emitOpError("a take has a scope: aux 16 (device) or 17 (system)");
  if (getTimeout() < 0)
    return emitOpError("timeout is a tick count >= 0");
  if (bool(getErr()) != bool(getErrIdx()))
    return emitOpError("err and its element come together");
  if (getErr()) {
    if (!getErr().getType().getElementType().isInteger(32))
      return emitOpError("err is an int32 element");
    if (!getErrIdx().getType().getElement())
      return emitOpError("err takes an element coordinate");
  }
  return verifyAux(*this, getAux());
}

LogicalResult PhiloxOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  if (!cast<RegTileType>(destType(getDst(), getResult())).getElementType().isF32())
    return emitOpError("draws into an fp32 tile");
  return noStrip(*this, destType(getDst(), getResult()));
}

LogicalResult LoadW4Op::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto tile = cast<RegTileType>(destType(getDst(), getResult()));
  Type et = tile.getElementType();
  if (!(et.isBF16() || et.isF16()) || tile.getColLayout() || tile.isStrip())
    return emitOpError("dequantizes into a bf16 or f16 row-layout tile");
  if (!getQw().getType().getElementType().isInteger(32))
    return emitOpError("reads its codes from an int32 global, eight to a word");
  if (getS().getType().getElementType() != et ||
      getZs().getType().getElementType() != et)
    return emitOpError("reads scales and zero-times-scales of the tile's type");
  if (getGroup() <= 0 || getGroup() % 16)
    return emitOpError("group must be a positive multiple of 16, got ")
           << getGroup();
  return success();
}

LogicalResult LoadMxScalesOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto et = getSrc().getType().getElementType();
  if (et.isInteger(32)) {
    auto t = cast<MxScalesType>(destType(getDst(), getResult()));
    if (t.getRows() % 64)
      return emitOpError("packed (i32) scales come in 64-row groups; ")
             << t.getRows() << " rows";
    return success();
  }
  if (!et.isInteger(8))
    return emitOpError("reads E8M0 scales from a uint8 (i8) global, one byte a "
                       "row and 32 columns, or packed dwords (i32)");
  if (getLayout() < 0 || getLayout() > 2)
    return emitOpError("layout is 0 (OCP), 1 (AITER gate|up) or 2 (AITER); got ")
           << getLayout();
  if (getLayout()) {
    auto t = cast<MxScalesType>(destType(getDst(), getResult()));
    if (getGroupRows() <= 0 || getGroupRows() % 32)
      return emitOpError("AITER scales need group_rows, a positive multiple of 32");
    if (getLayout() == 1 && t.getRows() % 32)
      return emitOpError("a gate|up tile is whole (gate, up) block pairs: 32-row multiples");
  }
  return success();
}

LogicalResult MxQuantStoreOp::verify() {
  auto st = getSrc().getType();
  if (!st.getElementType().isBF16() && !st.getElementType().isF32())
    return emitOpError("quantizes a bf16 or fp32 LDS tile");
  if (st.getCols() % 32)
    return emitOpError("whole 32-column groups a row");
  if (!isa<Float8E4M3FNType>(getQ().getType().getElementType()))
    return emitOpError("stores FP8 e4m3 values");
  if (!getS().getType().getElementType().isInteger(8) &&
      !getS().getType().getElementType().isInteger(32))
    return emitOpError("stores E8M0 scale bytes into a uint8 (i8) global (AITER's 32-row "
                       "layout) or an int32 one (the packed layout)");
  return success();
}

LogicalResult LoadFp4FragsLdsOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto t = cast<RegTileType>(destType(getDst(), getResult()));
  if (!isa<Float8E4M3FNType>(t.getElementType()) || t.getColLayout() || t.isStrip() ||
      t.getRows() % 16 || t.getCols() % 128)
    return emitOpError("fills an fp8 e4m3 row tile of 16x128 fragments (the FP4 container)");
  auto st = getSrc().getType();
  if (!isa<Float8E4M3FNType>(st.getElementType()) || st.getCols() % 1024)
    return emitOpError("reads an fp8-typed LDS tile of whole 1 KB chunks a row");
  return success();
}

LogicalResult LoadFp4FragsOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto t = cast<RegTileType>(destType(getDst(), getResult()));
  if (!isa<Float8E4M3FNType>(t.getElementType()) || t.getColLayout() || t.isStrip())
    return emitOpError("fills an fp8 e4m3 row-layout tile (the FP4 operand's container)");
  if (t.getRows() % 16 || t.getCols() % 128)
    return emitOpError("whole 16x128 fragments");
  if (getGateUp() && t.getRows() % 32)
    return emitOpError("a gate|up tile is whole (gate, up) block pairs: 32-row multiples");
  if (getGroupRows() <= 0 || getGroupRows() % 32)
    return emitOpError("group_rows is a positive multiple of 32");
  if (!getSrc().getType().getElementType().isInteger(8))
    return emitOpError("reads packed FP4 bytes from a uint8 (i8) global");
  return success();
}

LogicalResult MxScalesCopyOp::verify() {
  if (getDst().getType() != getSrc().getType())
    return emitOpError("copies scales of one shape: ") << getSrc().getType() << " into " << getDst().getType();
  return success();
}

LogicalResult MmaABtMxOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto d = cast<RegTileType>(destType(getDst(), getResult()));
  RegTileType a = getA().getType(), b = getB().getType(), c = getC().getType();
  for (RegTileType t : {a, b})
    if (!isa<Float8E4M3FNType>(t.getElementType()) || t.getColLayout() ||
        t.isStrip())
      return emitOpError("A and B are fp8 e4m3 row-layout tiles, not ") << t;
  if ((getAFmt() != 0 && getAFmt() != 4) || (getBFmt() != 0 && getBFmt() != 4))
    return emitOpError("a_fmt and b_fmt are 0 (fp8 e4m3) or 4 (fp4 e2m1)");
  if (!c.getElementType().isF32() || !c.getColLayout() || c != d)
    return emitOpError("accumulates into an fp32 col-layout tile, the result's type");
  if (a.getCols() != b.getCols() || a.getCols() % 128)
    return emitOpError("A and B share K, a multiple of 128; got ")
           << a.getCols() << " and " << b.getCols();
  if (c.getRows() != a.getRows() || c.getCols() != b.getRows())
    return emitOpError("C is A's rows x B's rows");
  MxScalesType sa = getSa().getType(), sb = getSb().getType();
  // with offsets the scales may cover more: the operand's tiles and steps from the offsets on
  auto covers = [](MxScalesType s, RegTileType t, int64_t tile, int64_t step) {
    if (tile < 0 || step < 0)
      return false;
    return s.getRows() >= t.getRows() + 16 * tile && s.getKblocks() * 32 >= t.getCols() + 128 * step;
  };
  if (!covers(sa, a, getSaTile(), getSaStep()))
    return emitOpError("sa holds A's ") << a.getRows() << " rows and "
                                        << a.getCols() / 32 << " blocks (from tile "
                                        << getSaTile() << ", step " << getSaStep() << ")";
  if (!covers(sb, b, getSbTile(), getSbStep()))
    return emitOpError("sb holds B's ") << b.getRows() << " rows and "
                                        << b.getCols() / 32 << " blocks (from tile "
                                        << getSbTile() << ", step " << getSbStep() << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// LDS
//===----------------------------------------------------------------------===//

LogicalResult SharedAtOp::verify() {
  auto buf = getBuf().getType();
  if (buf.getCount() == 1)
    return emitOpError("indexes a single LDS tile, not a stack of them");
  if (getResult().getType() != buf.element())
    return emitOpError("result must be one buffer of the stack");
  return success();
}

LogicalResult SubtileOp::verify() {
  auto src = getSrc().getType();
  auto res = getResult().getType();
  if (src.getCount() != 1)
    return emitOpError("subtile of a stack; index it first");
  if (res.getElementType() != src.getElementType() || res.getCount() != 1)
    return emitOpError("a subtile keeps the element type");
  if (src.getRows() % res.getRows() || src.getCols() % res.getCols())
    return emitOpError("the window does not tile its parent");
  return success();
}

LogicalResult LoadSharedOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto t = cast<RegTileType>(destType(getDst(), getResult()));
  auto st = getSrc().getType();
  if (failed(noStrip(*this, t)))
    return failure();
  if (t.getRows() != st.getRows() || t.getCols() != st.getCols())
    return emitOpError("register and shared tiles must have the same extents");
  if (!getWait() && getLdsLoads() == 0)
    return emitOpError("wait = false on a pair that does not vectorise");
  return success();
}

LogicalResult StoreSharedOp::verify() {
  return noStrip(*this, getValue().getType());
}

LogicalResult GranulesOp::verify() {
  auto st = getSrc().getType();
  if (st.getCount() != 1)
    return emitOpError("granules of a stack; index it first");
  if (getResult().getType().getTile() != st)
    return emitOpError("granules must be built for the tile they address");
  return success();
}

LogicalResult LoadFragOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto t = cast<RegTileType>(destType(getDst(), getResult()));
  auto gr = getGranules().getType();
  if (t.getRows() != 16 || t.getCols() != 16 || t.getColLayout() ||
      t.getElementType() != gr.getElementType())
    return emitOpError("reads a 16x16 row tile of the shared tile's element type");
  if (getRow() < 0 || 16 * getRow() + 16 > gr.getRows() || getCol() < 0 ||
      16 * getCol() + 16 > gr.getCols())
    return emitOpError("fragment coordinate out of the tile");
  return success();
}

LogicalResult StoreFragOp::verify() {
  auto t = getValue().getType();
  auto gr = getGranules().getType();
  if (t.getRows() != 16 || t.getCols() != 16 || t.getColLayout() ||
      t.getElementType() != gr.getElementType())
    return emitOpError("writes a 16x16 row tile of the shared tile's element type");
  if (bool(getRowIndex()) == getRow().has_value())
    return emitOpError("takes a constant row or a runtime row, exactly one");
  return success();
}

LogicalResult LdsWaitForOp::verify() {
  if (getTiles().empty())
    return emitOpError("needs the tiles it retires");
  return success();
}

/// The async copy is gfx950's: 64-lane waves, 16 bytes a lane an issue.
static constexpr int64_t kDmaPerWarp = 16 * 64;

static int64_t groupLoadBytes(GroupLoadOp op) {
  auto t = op.getDst().getType();
  return int64_t(t.getRows()) * t.getCols() *
         (dtypeBits(t.getElementType()) / 8);
}

int64_t GroupLoadOp::dmaCalls() {
  // Every warp issues bytes / (16 * threads) copies and, when that is not
  // whole, the first warps one more (hk/cdna.cuh): the fewest any warp
  // issues is what a wait behind the copy may assume of every warp.
  int64_t bytes = groupLoadBytes(*this);
  if (bytes < kDmaPerWarp || bytes % kDmaPerWarp)
    return 0;
  return bytes / (16 * getThreads());
}

LogicalResult GroupLoadOp::verify() {
  if (getDst().getType().getCount() != 1)
    return emitOpError("copies into a stack of LDS tiles; index it");
  int64_t bytes = groupLoadBytes(*this);
  if (getAsync() && (bytes < kDmaPerWarp || bytes % kDmaPerWarp))
    return emitOpError("async: the tile is not a whole number of 16-byte "
                       "copies per warp");
  if (getRows() && !getAsync())
    return emitOpError("rows: a row limit is the async copy's, whose "
                       "descriptor it narrows");
  if (getNontemporal() && !getAsync())
    return emitOpError("nontemporal: the cache policy is the async copy's");
  if (bool(getGatherIds()) != bool(getGatherAt()))
    return emitOpError("gather: the index list and the position in it, both");
  if (getGatherIds() && !getAsync())
    return emitOpError("gather: an async copy's rows by index");
  if (getGatherIds() && !getGatherIds().getType().getElementType().isInteger(32))
    return emitOpError("gather: the index list is int32");
  return success();
}

LogicalResult GroupWaitForOp::verify() {
  if (getTiles().empty())
    return emitOpError("needs the LDS tiles it retires");
  return success();
}

/// A pingpong pair: the begin, and one end later in the same block on the
/// same condition, with no other pingpong between them.
static LogicalResult verifyPingPong(Operation *op, Value cond, bool begin) {
  Operation *other = nullptr;
  for (Operation *o = begin ? op->getNextNode() : op->getPrevNode(); o;
       o = begin ? o->getNextNode() : o->getPrevNode()) {
    if (isa<PingPongBeginOp, PingPongEndOp>(o)) {
      other = o;
      break;
    }
  }
  bool paired = other && (begin ? isa<PingPongEndOp>(other)
                                : isa<PingPongBeginOp>(other));
  if (!paired)
    return op->emitOpError(begin ? "has no pingpong_end after it in its block"
                                 : "has no pingpong_begin before it in its block");
  if (other->getOperand(0) != cond)
    return op->emitOpError("pairs with a pingpong on another condition");
  return success();
}

LogicalResult PingPongBeginOp::verify() {
  return verifyPingPong(*this, getCond(), true);
}

LogicalResult PingPongEndOp::verify() {
  return verifyPingPong(*this, getCond(), false);
}

//===----------------------------------------------------------------------===//
// Staging
//===----------------------------------------------------------------------===//

LogicalResult StageBufferOp::verify() {
  if (getResult().getType().getTile() != getDst().getType().element())
    return emitOpError("stages a tile of a different shape than its destination");
  return success();
}

LogicalResult StageLoadOp::verify() {
  if (getSlot() < 0 || getSlot() >= getBuf().getType().getDepth())
    return emitOpError("slot out of the buffer's depth");
  return success();
}

LogicalResult StageCommitOp::verify() {
  auto dst = getDst().getType();
  auto buf = getBuf().getType();
  if (dst.getCount() != 1)
    return emitOpError("commits into a stack; index it");
  if (buf.getTile() != dst)
    return emitOpError("buffer stages a different tile than the destination");
  if (getSlot() < 0 || getSlot() >= buf.getDepth())
    return emitOpError("slot out of the buffer's depth");
  return success();
}

/// The instruction classes sched_barrier and sched_group_barrier name.
static constexpr int64_t kSchedClasses = 0x7FF;

LogicalResult SchedBarrierOp::verify() {
  if (getMask() < 0 || getMask() > kSchedClasses)
    return emitOpError("mask names classes 0x1..0x400; got ") << getMask();
  return success();
}

LogicalResult SchedGroupBarrierOp::verify() {
  if (getMask() <= 0 || getMask() > kSchedClasses)
    return emitOpError("mask names at least one class of 0x1..0x400; got ")
           << getMask();
  if (getSize() <= 0)
    return emitOpError("a group has at least one instruction");
  if (getSyncid() < 0)
    return emitOpError("syncid is a non-negative group id");
  return success();
}

LogicalResult SetPrioOp::verify() {
  if (getLevel() < 0 || getLevel() > 3)
    return emitOpError("the priority field is 2 bits, 0..3");
  return success();
}

//===----------------------------------------------------------------------===//
// Creation and maps
//===----------------------------------------------------------------------===//

#define HK_CREATE_VERIFY(OP)                                                   \
  LogicalResult OP::verify() {                                                 \
    return verifyDest(*this, getDst(), getResult());                           \
  }
HK_CREATE_VERIFY(ZeroOp)
HK_CREATE_VERIFY(FullOp)
HK_CREATE_VERIFY(NegInftyOp)
#undef HK_CREATE_VERIFY

template <typename OpT>
static LogicalResult verifyBinary(OpT op) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  Type lhs = op.getLhs().getType();
  if (failed(checkMapDtype(op, lhs)))
    return failure();
  Value rhs = op.getRhs();
  if (bool(rhs) == op.getScalar().has_value())
    return op.emitOpError("takes a right operand or a literal scalar, exactly one");
  if (rhs && !rhs.getType().isInteger(32) && !sameCpp(rhs.getType(), lhs))
    return op.emitOpError("operand types differ; elementwise ops do not broadcast");
  if (!sameCpp(destType(op.getDst(), op.getResult()), lhs))
    return op.emitOpError("result type differs from the operands'");
  return success();
}

#define HK_BINARY_VERIFY(OP)                                                   \
  LogicalResult OP::verify() { return verifyBinary(*this); }
HK_BINARY_VERIFY(AddOp)
HK_BINARY_VERIFY(SubOp)
HK_BINARY_VERIFY(MulOp)
HK_BINARY_VERIFY(DivOp)
HK_BINARY_VERIFY(MaxOp)
HK_BINARY_VERIFY(MinOp)
#undef HK_BINARY_VERIFY

template <typename OpT>
static LogicalResult verifyUnary(OpT op) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  if (failed(checkMapDtype(op, op.getSrc().getType())))
    return failure();
  if (!sameCpp(destType(op.getDst(), op.getResult()), op.getSrc().getType()))
    return op.emitOpError("result type differs from the operand's");
  return success();
}

#define HK_UNARY_VERIFY(OP)                                                    \
  LogicalResult OP::verify() { return verifyUnary(*this); }
HK_UNARY_VERIFY(ExpOp)
HK_UNARY_VERIFY(Exp2Op)
HK_UNARY_VERIFY(LogOp)
HK_UNARY_VERIFY(Log2Op)
HK_UNARY_VERIFY(AbsOp)
HK_UNARY_VERIFY(ReluOp)
HK_UNARY_VERIFY(CopyOp)
HK_UNARY_VERIFY(GeluOp)
HK_UNARY_VERIFY(DgeluOp)
HK_UNARY_VERIFY(SqrtOp)
HK_UNARY_VERIFY(RsqrtOp)
HK_UNARY_VERIFY(SiluOp)
HK_UNARY_VERIFY(StepOp)
#undef HK_UNARY_VERIFY

static LogicalResult verifyElems(Operation *op, Type valT, Type idxT) {
  auto v = dyn_cast<RegTileType>(valT);
  auto i = dyn_cast<RegTileType>(idxT);
  if (!v || !v.isStrip() || !i || !i.isStrip())
    return op->emitOpError("moves strips' elements: values and indices are strips");
  if (!i.getElementType().isInteger(32))
    return op->emitOpError("indices are an i32 strip, not ") << idxT;
  if (i.getRows() != v.getRows() || i.getCols() != v.getCols() || i.getRun() != v.getRun())
    return op->emitOpError("indices ") << idxT << " and values " << valT
           << " have one shape and run";
  if (!v.getElementType().isF32() && !v.getElementType().isInteger(32))
    return op->emitOpError("values are an fp32 or i32 strip, not ") << valT;
  return success();
}

LogicalResult Fp4OperandOp::verify() {
  auto w = dyn_cast<RegTileType>(getWords().getType());
  auto r = dyn_cast<RegTileType>(getResult().getType());
  if (!w || !w.isStrip() || !w.getElementType().isInteger(32) || w.getRows() != 1 || w.getCols() != 256 ||
      w.getRun() != 4)
    return emitOpError("takes an i32 strip of 1 x 256, run 4 (four dwords a lane)");
  if (!r || r.isStrip() || r.getRows() != 16 || r.getCols() != 128 || r.getLayout() != "row")
    return emitOpError("gives a 16x128 row tile (the FP4 container)");
  return success();
}

LogicalResult Fp8OperandOp::verify() {
  auto w = dyn_cast<RegTileType>(getWords().getType());
  auto r = dyn_cast<RegTileType>(getResult().getType());
  if (!w || !w.isStrip() || !w.getElementType().isInteger(32) || w.getRows() != 1 || w.getCols() != 512 ||
      w.getRun() != 4)
    return emitOpError("takes an i32 strip of 1 x 512, run 4 (eight dwords a lane, two runs)");
  if (!r || r.isStrip() || r.getRows() != 16 || r.getCols() != 128 || r.getLayout() != "row")
    return emitOpError("gives a 16x128 row tile (an fp8 fragment)");
  return success();
}

LogicalResult MxScalesLooseOp::verify() {
  auto b = dyn_cast<RegTileType>(getBytes().getType());
  auto r = cast<MxScalesType>(getResult().getType());
  if (!b || !b.isStrip() || !b.getElementType().isInteger(32) || b.getRows() != 1 || b.getCols() != 64 ||
      b.getRun() != 1)
    return emitOpError("takes an i32 strip of 1 x 64, run 1 (one byte a lane)");
  if (r.getRows() != 16 || r.getKblocks() != 4 || r.getFmt() != 1)
    return emitOpError("gives loose (fmt 1) scales of 16 rows and 4 blocks");
  return success();
}

LogicalResult FirstLaneOp::verify() {
  auto s = dyn_cast<RegTileType>(getSrc().getType());
  if (!s || !s.isStrip() || !s.getElementType().isInteger(32))
    return emitOpError("reads an i32 strip");
  return success();
}

LogicalResult LaneGatherOp::verify() {
  auto s = dyn_cast<RegTileType>(getSrc().getType());
  auto l = dyn_cast<RegTileType>(getLanes().getType());
  if (!s || !s.isStrip() || !l || !l.isStrip() || !l.getElementType().isInteger(32))
    return emitOpError("gathers a strip's elements by an i32 strip of lanes");
  if (l.getRows() != s.getRows() || l.getCols() != s.getCols() || l.getRun() != s.getRun())
    return emitOpError("lanes have the operand's shape and run");
  if (getResult().getType() != getSrc().getType())
    return emitOpError("gives a strip of its operand's type");
  if (s.getElementType().getIntOrFloatBitWidth() != 32)
    return emitOpError("gathers 32-bit elements");
  return success();
}

/// Elements a lane holds of a register tile or strip (wave64).
static int64_t perLane(RegTileType t) { return t.getRows() * t.getCols() / 64; }

static LogicalResult laneStripResult(Operation *op, RegTileType s, Type res, bool i32) {
  auto r = dyn_cast<RegTileType>(res);
  if (!r || !r.isStrip() || r.getRows() != 1)
    return op->emitOpError("gives a 1-row strip");
  if (perLane(r) != perLane(s))
    return op->emitOpError("gives a strip of the operand's ") << perLane(s) << " elements a lane, not "
                                                              << perLane(r);
  if (i32 ? !r.getElementType().isInteger(32) : r.getElementType() != s.getElementType())
    return op->emitOpError(i32 ? "gives an i32 strip" : "gives a strip of the operand's element type");
  return success();
}

LogicalResult LaneStripOp::verify() {
  auto s = dyn_cast<RegTileType>(getSrc().getType());
  if (!s || s.getElementType().getIntOrFloatBitWidth() != 32)
    return emitOpError("takes a register tile of 32-bit elements");
  return laneStripResult(*this, s, getResult().getType(), false);
}

LogicalResult LaneCoordOp::verify() {
  if (getAxis() != 0 && getAxis() != 1)
    return emitOpError("axis is 0 (rows) or 1 (columns)");
  if (!getSrc()) {
    auto t = dyn_cast<RegTileType>(getResult().getType());
    if (!t || !t.isStrip() || !t.getElementType().isInteger(32))
      return emitOpError("without a tile makes an i32 strip (its own coordinates), not ")
             << getResult().getType();
    return success();
  }
  auto s = dyn_cast<RegTileType>(getSrc().getType());
  if (!s)
    return emitOpError("takes a register tile");
  return laneStripResult(*this, s, getResult().getType(), true);
}

LogicalResult RunXorOp::verify() {
  auto s = dyn_cast<RegTileType>(getSrc().getType());
  if (!s || !s.isStrip())
    return emitOpError("permutes a strip's runs");
  if (getResult().getType() != getSrc().getType())
    return emitOpError("gives a strip of its operand's type");
  int64_t m = getMask();
  if (m <= 0 || m >= s.getRun())
    return emitOpError("mask is in 1 .. run - 1");
  return success();
}

LogicalResult MfmaStripOp::verify() {
  if (getKind() != "f32_16x16x4")
    return emitOpError("kind is f32_16x16x4");
  auto a = dyn_cast<RegTileType>(getA().getType());
  auto b = dyn_cast<RegTileType>(getB().getType());
  auto c = dyn_cast<RegTileType>(getAcc().getType());
  auto one = [](RegTileType t, int cols, int run) {
    return t && t.isStrip() && t.getElementType().isF32() && t.getRows() == 1 &&
           t.getCols() == cols && t.getRun() == run;
  };
  if (!one(a, 64, 1) || !one(b, 64, 1))
    return emitOpError("A and B are fp32 strips of 1 x 64, run 1: a value a lane");
  if (!one(c, 256, 4))
    return emitOpError("acc is an fp32 strip of 1 x 256, run 4: four values a lane");
  if (getResult().getType() != getAcc().getType())
    return emitOpError("gives acc's type");
  return success();
}

static LogicalResult verifyMailElems(Operation *op, Type mbox, Type valT, Type idxT, int64_t aux) {
  if (!cast<GlobalType>(mbox).getElementType().isInteger(32))
    return op->emitOpError("a mailbox is int32 pairs");
  if (failed(verifyElems(op, valT, idxT)))
    return failure();
  if (aux != 16 && aux != 17)
    return op->emitOpError("a mailbox has a scope: aux 16 (device) or 17 (system)");
  return success();
}

LogicalResult PostElemsOp::verify() {
  return verifyMailElems(*this, getDst().getType(), getVal().getType(), getIdx().getType(), getAux());
}

LogicalResult TakeElemsOp::verify() {
  if (getTimeout() < 0)
    return emitOpError("timeout is a tick count >= 0");
  return verifyMailElems(*this, getSrc().getType(), getResult().getType(), getIdx().getType(), getAux());
}

static LogicalResult verifyLdsElems(Operation *op, Type st, Type valT, Type idxT) {
  auto s = cast<SharedTileType>(st);
  if (!s.getElementType().isF32() || s.getCount() != 1)
    return op->emitOpError("addresses one fp32 LDS tile (not a stack), not ") << st;
  return verifyElems(op, valT, idxT);
}

LogicalResult LdsLoadElemsOp::verify() {
  return verifyLdsElems(*this, getSrc().getType(), getResult().getType(), getIdx().getType());
}

LogicalResult LdsStoreElemsOp::verify() {
  // stores also take a bf16 tile (the value rounded): an MMA operand built element by element
  auto s = cast<SharedTileType>(getDst().getType());
  if (s.getElementType().isBF16() && s.getCount() == 1)
    return verifyElems(*this, getVal().getType(), getIdx().getType());
  return verifyLdsElems(*this, getDst().getType(), getVal().getType(), getIdx().getType());
}

LogicalResult LdsAtomicAddElemsOp::verify() {
  auto v = dyn_cast<RegTileType>(getVal().getType());
  if (!v || !v.getElementType().isInteger(32))
    return emitOpError("adds i32 elements, not ") << getVal().getType();
  return verifyLdsElems(*this, getDst().getType(), getVal().getType(), getIdx().getType());
}

LogicalResult LoadElemsOp::verify() {
  return verifyElems(*this, getResult().getType(), getIdx().getType());
}

LogicalResult StoreElemsOp::verify() {
  return verifyElems(*this, getVal().getType(), getIdx().getType());
}

LogicalResult GroupReduceOp::verify() {
  auto t = dyn_cast<RegTileType>(getSrc().getType());
  if (!t || !t.isStrip())
    return emitOpError("reduces a strip's lane groups, not ") << getSrc().getType();
  if (getResult().getType() != getSrc().getType())
    return emitOpError("gives a strip of its operand's type");
  if (getKind() != "sum" && getKind() != "max" && getKind() != "min")
    return emitOpError("kind is sum, max or min");
  auto [lanes, offs] = getOrder().split(':');
  if (lanes != "seq" && lanes != "tree")
    return emitOpError("order's lane part is seq or tree");
  SmallVector<StringRef> parts;
  if (!offs.empty())
    offs.split(parts, ',');
  unsigned seen = 0;
  for (StringRef p : parts) {
    unsigned v = 0;
    if (p.getAsInteger(10, v) || v == 0 || v > 32 || (v & (v - 1)) || (seen & v))
      return emitOpError("order's offsets are powers of two of at most 32, each once");
    seen |= v;
  }
  return success();
}

namespace {
// a map's element types: f32, i32 and (masks) i1
bool mapElem(Type t) { return t.isF32() || t.isInteger(32) || t.isInteger(1); }
}  // namespace

LogicalResult MapOp::verify() {
  auto res = getResult().getType();
  auto rt = dyn_cast<RegTileType>(res);
  if (!rt || !rt.isStrip())
    return emitOpError("makes a strip, not ") << res;
  if (!mapElem(rt.getElementType()))
    return emitOpError("makes an fp32, i32 or i1 strip, not ") << res;
  bool hasBody = getBody().has_value(), hasRegion = !getRegion().empty();
  if (hasBody == hasRegion)
    return emitOpError("is a region of scalar ops or a C++ `body`, exactly one of them");
  if (hasBody) {
    StringRef e = *getBody();
    if (e.trim().empty() || e.contains(';') || e.contains('{') || e.contains('}'))
      return emitOpError("body is one C++ expression: no ';' and no braces");
    if (rt.getElementType().isInteger(1))
      return emitOpError("with a C++ body makes an fp32 or i32 strip");
  }
  for (auto [i, a] : llvm::enumerate(getArgs())) {
    auto t = dyn_cast<RegTileType>(a.getType());
    if (!t)
      continue;  // an i32 / f32 scalar
    if (!t.isStrip() || t.getRows() != rt.getRows() || t.getCols() != rt.getCols() ||
        t.getRun() != rt.getRun())
      return emitOpError("argument x") << i << " is " << t
             << "; every strip argument has the result's shape and run, " << res;
    if (!mapElem(t.getElementType()) || (hasBody && t.getElementType().isInteger(1)))
      return emitOpError("argument x") << i << " is " << t
             << ": strip arguments are fp32 or i32" << (hasBody ? "" : " or i1")
             << " (convert a 16-bit strip with .to)";
  }
  return success();
}

LogicalResult MapOp::verifyRegions() {
  if (getRegion().empty())
    return success();
  if (!getRegion().hasOneBlock())
    return emitOpError("region is one block");
  Block &b = getRegion().front();
  if (b.getNumArguments() != getArgs().size())
    return emitOpError("region takes one argument an operand: ")
           << getArgs().size() << ", not " << b.getNumArguments();
  for (auto [i, a] : llvm::enumerate(getArgs())) {
    Type want = a.getType();
    if (auto t = dyn_cast<RegTileType>(want))
      want = t.getElementType();
    if (b.getArgument(i).getType() != want)
      return emitOpError("region argument ") << i << " is " << b.getArgument(i).getType()
             << ", operand x" << i << "'s element type is " << want;
  }
  auto y = dyn_cast<YieldOp>(b.getTerminator());
  Type elem = cast<RegTileType>(getResult().getType()).getElementType();
  if (!y || y.getValue().getType() != elem)
    return emitOpError("region yields the result's element type, ") << elem;
  for (Operation &op : b.without_terminator()) {
    StringRef d = op.getDialect() ? op.getDialect()->getNamespace() : StringRef();
    if (d != "arith" && d != "math" && !isa<RcpOp>(op))
      return op.emitOpError("inside an hk.map: only arith, math and hk.rcp ops on scalars");
    if (op.getNumRegions())
      return op.emitOpError("inside an hk.map: no nested regions");
    SmallVector<Type> types(op.getOperandTypes());
    llvm::append_range(types, op.getResultTypes());
    for (Type t : types)
      if (!mapElem(t) && !t.isBF16())
        return op.emitOpError("inside an hk.map: values are f32, i32, i1 or bf16 scalars, not ")
               << t;
  }
  return success();
}

LogicalResult SwapHalvesOp::verify() {
  if (failed(verifyUnary(*this)))
    return failure();
  auto t = dyn_cast<RegTileType>(getSrc().getType());
  if (!t || !t.isStrip())
    return emitOpError("exchanges the halves of a strip's row segments; it takes a "
                       "strip, not ")
           << getSrc().getType();
  if (getSpan() < 0)
    return emitOpError("a span of ") << getSpan() << " columns";
  int64_t span = getSpan() ? getSpan() : t.getCols(), run = t.getRun(), chunk = 64 * run;
  if (span <= 0 || t.getCols() % span)
    return emitOpError("a span of ")
           << span << " does not divide the strip's " << t.getCols() << " columns";
  int64_t lanes = span / (2 * run);
  bool inChunk = span % (2 * run) == 0 && (lanes == 8 || lanes == 16 || lanes == 32);
  bool chunks = span % (2 * chunk) == 0;
  if (!inChunk && !chunks)
    return emitOpError("a span of ")
           << span << " at a run of " << run
           << " puts its halves neither 8, 16 or 32 lanes nor whole chunks apart";
  return success();
}

//===----------------------------------------------------------------------===//
// Reductions and broadcasts
//===----------------------------------------------------------------------===//

/// A strided operand fragment's rows and columns are the reduction index
/// permuted: a vector off it would name the wrong rows.
static LogicalResult noStrided(Operation *op, RegTileType t) {
  if (t.getFragStride())
    return op->emitOpError()
           << t << " is laid out as an mma operand with its reduction index "
              "permuted; reduce or broadcast over the tile it came from";
  return success();
}

template <typename OpT>
static LogicalResult verifyReduce(OpT op, bool colKind) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  auto src = op.getSrc().getType();
  if (failed(noStrided(op, src)))
    return failure();
  auto want = RegVecType::get(op.getContext(), src.getElementType(),
                              src.getRows(), src.getCols(), src.getColLayout(),
                              colKind, false, src.getFragRows(),
                              src.getFragCols(), src.getRun());
  auto got = cast<RegVecType>(destType(op.getDst(), op.getResult()));
  if (!got.sameCpp(want))
    return op.emitOpError() << "reducing " << src << " gives " << want
                            << ", not " << got;
  if (op.getAccumulate() && !op.getDst())
    return op.emitOpError("accumulate needs a destination to fold into");
  if (auto order = op.getOrder()) {
    if (!colKind)
      return op.emitOpError("an order is for row reductions");
    if (!src.isStrip())
      return op.emitOpError("an order is for strips: a WMMA tile's reduction "
                            "order is its layout's");
    auto [lanes, offs] = order->split(':');
    if (lanes != "seq" && lanes != "tree")
      return op.emitOpError("order \"") << *order << "\": the lane part is seq or tree";
    SmallVector<StringRef> parts;
    offs.split(parts, ',');
    unsigned seen = 0;
    for (StringRef p : parts) {
      unsigned v = 0;
      if (p.getAsInteger(10, v) || v == 0 || v > 32 || (v & (v - 1)) || (seen & v))
        return op.emitOpError("order \"") << *order
               << "\": the offsets are 1 2 4 8 16 32, each once";
      seen |= v;
    }
    if (seen != 63)
      return op.emitOpError("order \"") << *order
             << "\": every offset 1 2 4 8 16 32 once (a wave of 64 lanes)";
  }
  if (Value times = op.getTimes()) {
    if (op->getName().stripDialect() != "row_sum")
      return op.emitOpError("times is a row_sum's: the sum of src * times");
    if (!op.getOrder() || !op.getOrder()->starts_with("seq:"))
      return op.emitOpError("times needs a \"seq\" order: its products are fused into the "
                            "running sum left to right");
    if (times.getType() != op.getSrc().getType())
      return op.emitOpError("times is ") << times.getType() << ", not the source's "
                                          << op.getSrc().getType();
  }
  return success();
}

#define HK_REDUCE_VERIFY(OP, COL)                                              \
  LogicalResult OP::verify() { return verifyReduce(*this, COL); }
HK_REDUCE_VERIFY(RowMaxOp, true)
HK_REDUCE_VERIFY(RowMinOp, true)
HK_REDUCE_VERIFY(RowSumOp, true)
HK_REDUCE_VERIFY(RowProdOp, true)
HK_REDUCE_VERIFY(ColMaxOp, false)
HK_REDUCE_VERIFY(ColMinOp, false)
HK_REDUCE_VERIFY(ColSumOp, false)
HK_REDUCE_VERIFY(ColProdOp, false)
#undef HK_REDUCE_VERIFY

template <typename OpT>
static LogicalResult verifyBroadcast(OpT op, bool rowOp) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  auto src = op.getSrc().getType();
  auto vec = op.getVec().getType();
  if (failed(noStrided(op, src)))
    return failure();
  // A row op takes one value per row: the column vector row_sum gives.
  if (vec.getColKind() != rowOp)
    return op.emitOpError() << "takes a " << (rowOp ? "col" : "row")
                            << " vector";
  if (vec.getTileRows() != src.getRows() || vec.getTileCols() != src.getCols() ||
      vec.getTileRun() != src.getRun())
    return op.emitOpError("the vector came off a tile of another shape");
  if (vec.getTile().fragR() != src.fragR() || vec.getTile().fragC() != src.fragC())
    return op.emitOpError("the vector came off a tile of another fragment");
  if (destType(op.getDst(), op.getResult()) != src)
    return op.emitOpError("result type differs from the tile's");
  return success();
}

#define HK_BROADCAST_VERIFY(OP, ROW)                                           \
  LogicalResult OP::verify() { return verifyBroadcast(*this, ROW); }
HK_BROADCAST_VERIFY(AddRowOp, true)
HK_BROADCAST_VERIFY(SubRowOp, true)
HK_BROADCAST_VERIFY(MulRowOp, true)
HK_BROADCAST_VERIFY(DivRowOp, true)
HK_BROADCAST_VERIFY(AddColOp, false)
HK_BROADCAST_VERIFY(SubColOp, false)
HK_BROADCAST_VERIFY(MulColOp, false)
HK_BROADCAST_VERIFY(DivColOp, false)
#undef HK_BROADCAST_VERIFY

template <typename OpT>
static LogicalResult verifyBroadcastFill(OpT op, bool row) {
  auto vec = op.getVec().getType();
  auto res = op.getResult().getType();
  if (failed(noStrip(op, res)) || failed(noStrip(op, vec)))
    return failure();
  if (vec.getColKind() != row)
    return op.emitOpError() << "takes a " << (row ? "col" : "row") << " vector";
  if (vec.getTileRows() != res.getRows() || vec.getTileCols() != res.getCols())
    return op.emitOpError("the vector came off a tile of another shape");
  return success();
}

LogicalResult BroadcastRowOp::verify() { return verifyBroadcastFill(*this, true); }
LogicalResult BroadcastColOp::verify() { return verifyBroadcastFill(*this, false); }

LogicalResult RetypeVecOp::verify() {
  auto a = getSrc().getType();
  auto b = getResult().getType();
  if (a.getElementType() != b.getElementType() ||
      a.getLength() != b.getLength() ||
      a.getTileColLayout() != b.getTileColLayout() ||
      a.getColKind() != b.getColKind() || !a.sameCpp(b.plain()))
    return emitOpError("an rv's C++ type is (element, length, layout); these differ");
  return success();
}

//===----------------------------------------------------------------------===//
// Masks
//===----------------------------------------------------------------------===//

template <typename OpT>
static LogicalResult verifyFill(OpT op, bool columns) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  if (destType(op.getDst(), op.getResult()) != op.getSrc().getType())
    return op.emitOpError("a fill keeps shape, layout and element type");
  if (failed(noStrided(op, op.getSrc().getType())))
    return failure();
  // A strip fills by column only: its rows are not lanes.
  if (!columns)
    return noStrip(op, op.getSrc().getType());
  return success();
}

#define HK_FILL_VERIFY(OP, COLUMNS)                                            \
  LogicalResult OP::verify() { return verifyFill(*this, COLUMNS); }
HK_FILL_VERIFY(LeftFillOp, true)
HK_FILL_VERIFY(RightFillOp, true)
HK_FILL_VERIFY(UpperFillOp, false)
HK_FILL_VERIFY(LowerFillOp, false)
HK_FILL_VERIFY(TriuOp, false)
HK_FILL_VERIFY(TrilOp, false)
#undef HK_FILL_VERIFY


//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto src = dyn_cast<RegTileType>(getSrc().getType());
  auto dst = dyn_cast<RegTileType>(destType(getDst(), getResult()));
  if (!src || !dst)
    return emitOpError("casts register tiles");
  if (src.getRows() != dst.getRows() || src.getCols() != dst.getCols() ||
      src.getColLayout() != dst.getColLayout() || src.getRun() != dst.getRun())
    return emitOpError("a cast keeps shape and layout");
  if (src.fragR() != dst.fragR() || src.fragC() != dst.fragC() ||
      src.fragS() != dst.fragS())
    return emitOpError() << "a cast keeps the fragment: " << src << " into "
                         << dst << " is a relayout across lanes, not a cast";
  return success();
}

LogicalResult BitcastOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto src = dyn_cast<RegTileType>(getSrc().getType());
  auto dst = dyn_cast<RegTileType>(destType(getDst(), getResult()));
  if (!src || !dst)
    return emitOpError("reinterprets register tiles");
  if (src.getRows() != dst.getRows() || src.getCols() != dst.getCols() ||
      src.getColLayout() != dst.getColLayout() || src.getRun() != dst.getRun() ||
      src.fragR() != dst.fragR() || src.fragC() != dst.fragC() || src.fragS() != dst.fragS())
    return emitOpError("a bitcast keeps shape, layout and fragment");
  if (src.getElementType().getIntOrFloatBitWidth() != dst.getElementType().getIntOrFloatBitWidth())
    return emitOpError() << "a bitcast keeps the element width: " << src.getElementType()
                         << " into " << dst.getElementType() << " is a conversion (.to)";
  return success();
}

/// The accumulator fragments whose lanes hold exactly what an operand
/// fragment of the same tile holds, with the reduction index running down
/// the rows in another order: a 32x32 accumulator's lane holds rows
/// 4(l/32) + 8k + s, which is a 16x32 stride-4 operand two fragments a
/// 32-row band; a 16x16 one's, rows 4(l/16) + s, which stacked in pairs is
/// a 32x16 stride-4 operand. Returns the operand fragment, or nothing.
static std::optional<std::pair<unsigned, unsigned>>
operandOfAccumulator(RegTileType src) {
  if (src.getFragStride())
    return std::nullopt;
  if (src.fragR() == 32 && src.fragC() == 32)
    return std::make_pair(16u, 32u);
  if (src.fragR() == 16 && src.fragC() == 16)
    return std::make_pair(32u, 16u);
  return std::nullopt;
}

LogicalResult AsOperandOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto src = getSrc().getType();
  auto dst = cast<RegTileType>(destType(getDst(), getResult()));
  if (failed(noStrip(*this, src)))
    return failure();
  // fp32 -> 16-bit converts each element on the way (rounding to nearest
  // even); a 16-bit source is read as it is.
  bool fromF32 = src.getElementType().isF32();
  if (!src.getColLayout() ||
      (!fromF32 && dtypeBits(src.getElementType()) != 16))
    return emitOpError() << "reads a col-layout accumulator tile, fp32 or "
                            "16-bit (an MFMA result, or its cast), not "
                         << src;
  if (dtypeBits(dst.getElementType()) != 16 ||
      (!fromF32 && dst.getElementType() != src.getElementType()))
    return emitOpError() << "makes a bf16 or fp16 operand of " << src
                         << ", not " << dst;
  auto op = operandOfAccumulator(src);
  if (!op)
    return emitOpError() << src
                         << " is not in an accumulator fragment (16x16 or "
                            "32x32)";
  auto want = RegTileType::get(getContext(), dst.getElementType(),
                               src.getRows(), src.getCols(), true, op->first,
                               op->second, 0, 4);
  if (src.getRows() % op->first || dst != want)
    return emitOpError() << src << " read as an operand is " << want
                         << (src.getRows() % op->first
                                 ? ", which needs rows in multiples of " +
                                       std::to_string(op->first)
                                 : std::string());
  return success();
}

LogicalResult RegSubtileOp::verify() {
  auto src = getSrc().getType();
  auto dst = getResult().getType();
  if (failed(noStrip(*this, src)))
    return failure();
  if (dst.getElementType() != src.getElementType() ||
      dst.getCols() != src.getCols() ||
      dst.getColLayout() != src.getColLayout() ||
      dst.getFragRows() != src.getFragRows() ||
      dst.getFragCols() != src.getFragCols() ||
      dst.getFragStride() != src.getFragStride())
    return emitOpError() << "a subtile keeps the element type, columns, "
                            "layout and fragment of " << src;
  if (!dst.getRows() || src.getRows() % dst.getRows() ||
      dst.getRows() % dst.fragR())
    return emitOpError() << dst.getRows() << " rows: a whole number of "
                         << dst.fragR() << "-row fragments that tiles the "
                         << src.getRows() << " rows of " << src;
  if (getIndex() < 0 ||
      uint64_t(getIndex()) >= uint64_t(src.getRows() / dst.getRows()))
    return emitOpError() << "subtile " << getIndex() << " of "
                         << src.getRows() / dst.getRows();
  return success();
}

LogicalResult AnyAboveOp::verify() {
  auto a = getLhs().getType(), b = getRhs().getType();
  if (!a.sameCpp(b.plain()) || !a.getElementType().isF32())
    return emitOpError("compares two fp32 vectors of one type");
  return success();
}

LogicalResult TransposeOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto src = dyn_cast<RegTileType>(getSrc().getType());
  if (src && failed(noStrip(*this, src)))
    return failure();
  if (!src || destType(getDst(), getResult()) != src.transposed())
    return emitOpError("transposes a tile into the mirrored shape and opposite layout");
  return success();
}

LogicalResult TransposeSepOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto src = dyn_cast<RegTileType>(getSrc().getType());
  if (!src)
    return emitOpError("transposes register tiles");
  if (failed(noStrip(*this, src)))
    return failure();
  auto want = RegTileType::get(getContext(), src.getElementType(),
                               src.getCols(), src.getRows(),
                               src.getColLayout(), src.getFragCols(),
                               src.getFragRows());
  if (destType(getDst(), getResult()) != want)
    return emitOpError("transpose_sep mirrors the shape and keeps the layout");
  if (getDst() && getDst() == getSrc())
    return emitOpError("needs separate registers for source and destination");
  return success();
}

//===----------------------------------------------------------------------===//
// WMMA
//===----------------------------------------------------------------------===//

namespace {
struct MmaSpec {
  bool aCol, bCol;
  // Which extent of A gives D.rows, of B gives D.cols, and the reduction.
  bool dRowsFromACols, dColsFromBRows, aRedIsRows, bRedIsCols;
};
} // namespace

template <typename OpT>
static LogicalResult verifyMma(OpT op, MmaSpec s) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  auto a = op.getA().getType(), b = op.getB().getType(),
       c = op.getC().getType();
  for (Type t : {Type(a), Type(b), Type(c)})
    if (failed(noStrip(op, t)))
      return failure();
  bool fp8 = isFp8(a.getElementType()) && isFp8(b.getElementType());
  bool i8 = a.getElementType().isInteger(8) && b.getElementType().isInteger(8);
  if (a.getElementType() != b.getElementType() && !fp8)
    return op.emitOpError("takes one operand type, not a mixed pair");
  if (!fp8 && !i8 && dtypeBits(a.getElementType()) != 16)
    return op.emitOpError("takes bf16 or fp16 operands (fp8 on gfx1201, i8 on gfx1100 and gfx1201)");
  if (i8) {
    if (!c.getElementType().isInteger(32) || !c.getColLayout())
      return op.emitOpError("i8 x i8 accumulates into a col-layout i32 tile");
    if (s.aCol || s.bCol)
      return op.emitOpError("i8 operands are row-layout: C += A @ B.T only");
  } else if (!c.getElementType().isF32() || !c.getColLayout())
    return op.emitOpError("accumulates into a col-layout fp32 tile");
  if (a.getColLayout() != s.aCol || b.getColLayout() != s.bCol)
    return op.emitOpError("operand layouts do not match this variant");
  unsigned redA = s.aRedIsRows ? a.getRows() : a.getCols();
  unsigned redB = s.bRedIsCols ? b.getCols() : b.getRows();
  if (redA != redB)
    return op.emitOpError("reduction extents disagree");
  unsigned rows = s.dRowsFromACols ? a.getCols() : a.getRows();
  unsigned cols = s.dColsFromBRows ? b.getRows() : b.getCols();
  // The accumulator is 16x16 fragments (unset), or 32x32 for gfx950's
  // 32x32x16 MFMA, whose operands are then 32x16 (row) / 16x32 (col).
  bool big = c.getFragRows() == 32 && c.getFragCols() == 32 &&
             !c.getFragStride();
  auto want = RegTileType::get(op.getContext(), c.getElementType(), rows, cols,
                               true, big ? 32 : 0, big ? 32 : 0);
  if (c != want || destType(op.getDst(), op.getResult()) != want)
    return op.emitOpError() << "A " << a << " times B " << b << " is " << want;
  if (big)
    for (RegTileType t : {a, b})
      if (t.hasFrag() &&
          (t.getColLayout() ? t.fragR() != 16 || t.fragC() != 32
                            : t.fragR() != 32 || t.fragC() != 16))
        return op.emitOpError()
               << "a 32x32 accumulator takes 32x16 (row) and 16x32 (col) "
                  "operand fragments, not "
               << t;
  if (a.hasFrag() && b.hasFrag() && a.fragS() != b.fragS())
    return op.emitOpError() << "A's lane run is " << a.fragS() << " and B's "
                            << b.fragS()
                            << ": the MFMA pairs A and B element by element "
                               "along the reduction, so both are laid out "
                               "with one stride";
  return success();
}

LogicalResult MmaABOp::verify() {
  return verifyMma(*this, {false, true, false, false, false, false});
}
LogicalResult MmaABtOp::verify() {
  return verifyMma(*this, {false, false, false, true, false, true});
}
LogicalResult MmaAtBOp::verify() {
  return verifyMma(*this, {true, true, true, false, true, false});
}
LogicalResult MmaAtBtOp::verify() {
  return verifyMma(*this, {true, false, true, true, true, true});
}
