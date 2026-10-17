// hk-fragments: name, in the type, the matrix-core fragment of every register
// tile that is not laid out in 16x16 ones.
//
// gfx950's bf16/fp16 MFMA is 16x16x32. Its A and B are 16x32 (row layout) and
// 32x16 (col) fragments, and include/cdna4 lays a register tile out fragment by
// fragment: rt<bf16, R, C, row, rt_16x32> is a different lane layout from the
// 16x16 one, a different count of ds_reads to load it, and a different C++
// type. Which tiles are operands is only known once the kernel is traced, so
// this pass assigns it: every tile type an mma reads as A or B, and every type
// tied to one by an op that keeps the fragment -- a cast, a transpose, a
// broadcast of a vector taken off it. The accumulators stay 16x16.
//
// A type tied to two fragments is an error naming both ops. The case that
// matters is attention's: S, an fp32 accumulator (16x16), cast into P, the
// bf16 A operand of P@V (16x32). On gfx950 that is a relayout across lanes,
// not a cast; before this pass it was a static_assert deep in include/cdna4.
// What attention does instead is compute Sᵀ and read the accumulator as an
// operand with its reduction index permuted (hk.as_operand), which is a
// fragment with a lane run of 4 rather than 8 ("stride 4"); an MMA's other
// operand then takes the same stride, since the two are paired element by
// element along the reduction.
//
// An accumulator declared in 32x32 fragments (gfx950's 32x32x16 MFMA) gives
// its operands 32x16 (row) and 16x32 (col) fragments.
//
// Targets whose fragment is 16x16 (RDNA, gfx942) are left untouched.

#include "hk/Dialect/HK/Transforms/Passes.h"
#include "hk/Target/Target.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::hk {
#define GEN_PASS_DEF_HKFRAGMENTS
#include "hk/Dialect/HK/Transforms/Passes.h.inc"
} // namespace mlir::hk

using namespace mlir;
using namespace mlir::hk;

namespace {

/// A fragment shape and its lane run along the reduction; s is 0 where the
/// run is the shape's own, as in the type.
struct Frag {
  unsigned r = 0, c = 0, s = 0;
  bool operator==(const Frag &o) const {
    return r == o.r && c == o.c && s == o.s;
  }
  bool operator!=(const Frag &o) const { return !(*this == o); }
};

std::string str(Frag f) {
  return std::to_string(f.r) + "x" + std::to_string(f.c) +
         (f.s ? " stride " + std::to_string(f.s) : std::string());
}

struct Assigner {
  /// The fragment each (unannotated) tile type takes, and the op that said so.
  llvm::MapVector<Type, std::pair<Frag, Operation *>> want;
  bool failed = false, changed = false;

  static Frag declared(RegTileType t) {
    return {t.fragR(), t.fragC(), t.getFragStride()};
  }

  void assign(RegTileType t, Frag f, Operation *why) {
    if (failed)
      return;
    if (t.getRows() % f.r || t.getCols() % f.c) {
      why->emitError() << "this op needs " << t << " in " << str(f)
                       << " fragments -- as an mma operand, or tied to a tile "
                          "that is one, or that names them -- and "
                       << t.getRows() << "x" << t.getCols()
                       << " is not a whole number of them. On this target an "
                          "mma operand is a 16x32 (row) or 32x16 (col) "
                          "fragment: make the tile a multiple of that, e.g. "
                          "32 wide along the reduction."
                       << atLoc(why);
      failed = true;
      return;
    }
    if (t.hasFrag()) {
      if (declared(t) != f) {
        why->emitError() << "this op needs " << t << " in " << str(f)
                         << " fragments, and the type says " << str(declared(t))
                         << atLoc(why);
        failed = true;
      }
      return;
    }
    auto it = want.find(t);
    if (it == want.end()) {
      want.insert({t, {f, why}});
      changed = true;
      return;
    }
    if (it->second.first == f)
      return;
    InFlightDiagnostic d = why->emitError();
    d << t << " would be laid out in " << str(f) << " fragments here, and in "
      << str(it->second.first)
      << " fragments for another (both below). One register tile has one lane "
         "layout: on this target an mma's operands are "
      << "16x32 (row) / 32x16 (col) fragments and its accumulator is "
         "16x16, so the same tile cannot be both, and a cast or transpose "
         "between them is a relayout across lanes, not a copy. Load the "
         "operand from memory (LDS or global) in its own type instead."
      << atLoc(why) << atLoc(it->second.second);
    failed = true;
  }

  /// What `t` is laid out in: its declaration, what this pass has assigned
  /// so far, or nothing yet.
  std::optional<Frag> current(RegTileType t) const {
    if (t.hasFrag())
      return declared(t);
    auto it = want.find(t);
    if (it == want.end())
      return std::nullopt;
    return it->second.first;
  }

  /// `a` and `b` share one fragment (`swap`: b's is a's transposed).
  void tie(RegTileType a, RegTileType b, Operation *why, bool swap = false) {
    auto flip = [&](Frag f) { return swap ? Frag{f.c, f.r, f.s} : f; };
    if (auto fa = current(a))
      assign(b, flip(*fa), why);
    if (auto fb = current(b))
      assign(a, flip(*fb), why);
  }
};

struct FragmentsPass : public mlir::hk::impl::HKFragmentsBase<FragmentsPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    if (!isKernel(f))
      return;
    auto arch = f->getAttrOfType<StringAttr>(attrs::kArch);
    const TargetInfo *tgt = getTarget(arch ? arch.getValue() : "gfx1100");
    if (!tgt || tgt->mma.empty() || tgt->mma.front().second.k == 16) {
      // These targets' tiles have no fragment shape to name (include/rdna3,
      // include/cdna3): a fragment in a type there has no C++ spelling.
      f.walk([&](Operation *op) {
        for (Value v : op->getResults())
          if (auto t = dyn_cast<RegTileType>(v.getType()); t && t.hasFrag()) {
            op->emitError() << t << ": " << (tgt ? tgt->arch : "this target")
                            << "'s register tiles have no fragment shape; "
                               "leave frag unset" << atLoc(op);
            return signalPassFailure();
          }
      });
      return;
    }
    unsigned k = tgt->mma.front().second.k;

    Assigner as;
    auto tile = [](Value v) { return dyn_cast<RegTileType>(v.getType()); };
    auto dest = [](auto o) -> Value { return o.getDst() ? o.getDst() : o.getResult(); };
    // Seeds: the operands and accumulators of every mma. An accumulator in
    // 32x32 fragments makes the operands 32x16 / 16x32; an operand already
    // laid out with a stride (an accumulator read by hk.as_operand) gives
    // the other operand that stride. In the fixed-point loop, so a stride
    // that reaches an operand through a cast or transpose still arrives.
    auto seed = [&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<MmaABOp, MmaABtOp, MmaAtBOp, MmaAtBtOp>([&](auto o) {
            RegTileType c = tile(o.getC());
            auto fc = as.current(c);
            bool big = fc && fc->r == 32 && fc->c == 32;
            unsigned s = 0;
            for (Value v : {o.getA(), o.getB()})
              if (auto fv = as.current(tile(v)); fv && fv->s)
                s = fv->s;
            for (Value v : {o.getA(), o.getB()}) {
              RegTileType t = tile(v);
              if (dtypeBits(t.getElementType()) != 16)
                continue;
              Frag want = t.getColLayout() ? (big ? Frag{16, 32} : Frag{k, 16})
                                           : (big ? Frag{32, 16} : Frag{16, k});
              if (s && s != RegTileType::naturalStride(want.r, want.c))
                want.s = s;
              as.assign(t, want, op);
            }
            as.assign(c, big ? Frag{32, 32} : Frag{16, 16}, op);
          });
    };
    f.walk([&](Operation *op) { seed(op); });
    // Ties, to a fixed point: each round can only add assignments.
    do {
      as.changed = false;
      f.walk([&](Operation *op) { seed(op); });
      f.walk([&](Operation *op) {
        llvm::TypeSwitch<Operation *>(op)
            .Case([&](CastOp o) {
              as.tie(tile(o.getSrc()), tile(dest(o)), op);
            })
            .Case([&](RegSubtileOp o) {
              as.tie(tile(o.getSrc()), tile(o.getResult()), op);
            })
            .Case<TransposeOp, TransposeSepOp>([&](auto o) {
              as.tie(tile(o.getSrc()), tile(dest(o)), op, /*swap=*/true);
            })
            .Case<AddRowOp, SubRowOp, MulRowOp, DivRowOp, AddColOp, SubColOp,
                  MulColOp, DivColOp>([&](auto o) {
              as.tie(o.getVec().getType().getTile(), o.getSrc().getType(), op);
            })
            .Case<BroadcastRowOp, BroadcastColOp>([&](auto o) {
              as.tie(o.getVec().getType().getTile(), o.getResult().getType(),
                     op);
            })
            .Case([&](RetypeVecOp o) {
              as.tie(o.getSrc().getType().getTile(),
                     o.getResult().getType().getTile(), op);
            });
      });
    } while (as.changed && !as.failed);
    if (as.failed)
      return signalPassFailure();

    // Rewrite. 16x16 is what an unannotated type already means.
    DenseMap<Type, Type> to;
    for (auto &[t, w] : as.want) {
      if (w.first == Frag{16, 16})
        continue;
      to[t] = cast<RegTileType>(t).withFrag(w.first.r, w.first.c, w.first.s);
    }
    if (to.empty())
      return;
    auto retype = [&](Value v) {
      Type t = v.getType();
      if (auto it = to.find(t); it != to.end()) {
        v.setType(it->second);
      } else if (auto rv = dyn_cast<RegVecType>(t)) {
        if (auto jt = to.find(rv.getTile()); jt != to.end())
          v.setType(rv.withTile(cast<RegTileType>(jt->second)));
      }
    };
    f.getBody().walk([&](Block *b) {
      for (BlockArgument a : b->getArguments())
        retype(a);
    });
    f.getBody().walk([&](Operation *op) {
      for (Value r : op->getResults())
        retype(r);
    });
  }
};

} // namespace
