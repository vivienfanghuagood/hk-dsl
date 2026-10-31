// hk-map-fuse: merge a chain of region-form hk.map ops into one.
//
// The front end builds one map an operator (`(c >> 4) & 63` is two), so that
// the operators compose; this pass makes the chain one map, as the C++
// expression a kernel author would have written: a map whose operand is the
// result of another region-form map, read by it alone, takes that map's
// operands instead and computes its value inline. A value read twice -- by two
// maps, or by a store_elems index and a map -- is computed once and read, as
// the emitter does with a value used twice inside a region: nothing is
// computed twice. A string-body map (hk.map_cpp) is never fused, either way.
//
// The merged map sits where the consumer was. A producer is inlined only
// when nothing but maps and scalar arithmetic lies between it and its user:
// fusion never moves arithmetic across a load, a store, a lane exchange or a
// barrier, where it would change how the kernel is scheduled (a map written
// before a lane_gather stays before it). A producer whose user is in another
// block (inside a loop) is left alone, so arithmetic outside a loop never
// moves into its body.

#include "hk/Dialect/HK/Transforms/Passes.h"
#include "hk/Target/Target.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hk {
#define GEN_PASS_DEF_HKMAPFUSE
#include "hk/Dialect/HK/Transforms/Passes.h.inc"
} // namespace mlir::hk

using namespace mlir;
using namespace mlir::hk;

namespace {

bool isRegionMap(Operation *op) {
  auto m = dyn_cast_or_null<MapOp>(op);
  return m && !m.getBody() && !m.getRegion().empty();
}

/// Scalar arithmetic: an arith op, or an op on scalars and coordinates alone with no memory
/// effect (hk.coord, hk.extent, hk.block_idx): what a producer may be moved past.
bool scalarArith(Operation *o) {
  if (isa<arith::ArithDialect>(o->getDialect()))
    return true;
  auto scalar = [](Type t) { return t.isIntOrIndexOrFloat() || isa<CoordType, GlobalType>(t); };
  return isMemoryEffectFree(o) && o->getNumRegions() == 0 &&
         llvm::all_of(o->getOperandTypes(), scalar) && llvm::all_of(o->getResultTypes(), scalar);
}

/// A region-form map read once, by a region-form map later in its block with nothing but maps
/// and scalar arithmetic between them: what is inlined.
bool inlinable(Operation *op) {
  if (!isRegionMap(op) || !op->hasOneUse())
    return false;
  Operation *u = *op->getUsers().begin();
  if (!isRegionMap(u) || u->getBlock() != op->getBlock())
    return false;
  for (Operation *o = op->getNextNode(); o && o != u; o = o->getNextNode())
    if (!isa<MapOp>(o) && !scalarArith(o))
      return false;
  return true;
}

/// Inline operand `i` of `m`, produced by the region-form map `p`, into m.
void inlineProducer(MapOp m, unsigned i, MapOp p) {
  Block &body = m.getRegion().front();
  Block &pbody = p.getRegion().front();
  OpBuilder b(m.getContext());
  IRMapping map;
  // p's operands: reuse m's argument for an operand m already has, else add one
  for (auto [k, v] : llvm::enumerate(p.getArgs())) {
    BlockArgument arg;
    for (auto [j, w] : llvm::enumerate(m.getArgs()))
      if (w == v) {
        arg = body.getArgument(j);
        break;
      }
    if (!arg) {
      m.getArgsMutable().append(v);
      arg = body.addArgument(pbody.getArgument(k).getType(), p.getLoc());
    }
    map.map(pbody.getArgument(k), arg);
  }
  b.setInsertionPointToStart(&body);
  for (Operation &op : pbody.without_terminator())
    b.clone(op, map);
  Value y = map.lookup(cast<YieldOp>(pbody.getTerminator()).getValue());
  body.getArgument(i).replaceAllUsesWith(y);
  body.eraseArgument(i);
  m.getArgsMutable().erase(i);
}

/// Inline every region-form producer in m's block into m, until none is left; the producers
/// go into `fused`.
void fuseInto(MapOp m, llvm::SetVector<Operation *> &fused) {
  for (bool changed = true; changed;) {
    changed = false;
    for (auto [i, v] : llvm::enumerate(m.getArgs())) {
      Operation *def = v.getDefiningOp();
      if (!inlinable(def))
        continue;
      inlineProducer(m, i, cast<MapOp>(def));
      fused.insert(def);
      changed = true;
      break;
    }
  }
}

struct MapFusePass : public mlir::hk::impl::HKMapFuseBase<MapFusePass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    if (!isKernel(f))
      return;
    SmallVector<MapOp> maps;
    f.walk([&](MapOp m) {
      if (isRegionMap(m))
        maps.push_back(m);
    });
    // in program order: a map has taken in its own producers before it is inlined into the
    // next, so a chain is one pass
    llvm::SetVector<Operation *> fused;
    for (MapOp m : maps)
      fuseInto(m, fused);
    // the producers, now unused: last first, so a chain goes in one pass
    for (Operation *p : llvm::reverse(fused))
      p->erase();
    // RDNA's strips hold no bool (include/rdna3 / rdna4's base_types::T1): a mask may live only
    // inside a map, and one that leaves it here is refused, not a C++ static_assert
    auto arch = f->getAttrOfType<StringAttr>(attrs::kArch);
    const TargetInfo *tgt = getTarget(arch ? arch.getValue() : "gfx1100");
    if (tgt && tgt->wave == 32)
      f.walk([&](MapOp m) {
        auto t = cast<RegTileType>(m.getResult().getType());
        if (t.getElementType().isInteger(1)) {
          m.emitError() << "a mask strip leaves its hk.map on " << tgt->arch
                        << ": on RDNA a mask (a comparison) can only live inside a map -- "
                           "include/rdna3 and include/rdna4 have no bool strips. Use it where "
                           "it is made (hk.where(a < b, x, y)), or compute it inside one "
                           "hk.map(fn)" << atLoc(m);
          signalPassFailure();
        }
      });
  }
};

} // namespace
