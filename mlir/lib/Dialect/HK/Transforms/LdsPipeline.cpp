// hk-lds-pipeline: derive every `s_waitcnt lgkmcnt`, and prove every
// asynchronous tile is bound before it is read.
//
// `load_shared(wait=false)` issues its ds_reads through an inline-asm output,
// so LLVM believes the destination is ready the instant the asm ends, and
// nothing stops it hoisting a consuming WMMA above the wait. Binding the
// tile's registers after the wait fixes that; this pass makes sure every
// wait names the tiles it retires, and computes the N nobody should compute
// by hand: LDS returns in order, so retiring a tile means waiting until only
// what was issued after it is outstanding.
//
// Loops are walked twice, the second time from the state the first left,
// and must agree -- a body whose derived counts differ has no steady state.
//
// The same walk derives every `s_waitcnt vmcnt` for gfx950's asynchronous
// group loads (buffer_load ... lds, counted on vmcnt and returned in flight)
// and checks the two cross-warp hazards a copy into a shared buffer has:
// reading it before every warp's share has landed, and copying into it
// before every warp's reads of it have. Both come down to how many barriers
// separate the two events in program order -- one, or two inside an
// hk.pingpong, where half the workgroup runs a barrier behind the other.
// That is what lets a read of a named buffer stay in flight across a
// barrier: every later write into the buffer is checked against it instead.
//
// A port of python/hk/ir/passes/lds_pipeline.py; the messages are its.

#include "hk/Dialect/HK/Transforms/Passes.h"
#include "hk/Target/Target.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"

#include <optional>
#include <set>

namespace mlir::hk {
#define GEN_PASS_DEF_HKLDSPIPELINE
#include "hk/Dialect/HK/Transforms/Passes.h.inc"
} // namespace mlir::hk

using namespace mlir;
using namespace mlir::hk;

namespace {

using Scope = SmallVector<Operation *, 4>;

/// Which LDS buffer a shared-tile value is: its allocation and, for one
/// buffer of a stack, the constant index. A runtime index is -1: the
/// cross-warp checks below cannot tell which buffer that is, and leave it to
/// the rules that do not need to know.
///
/// A ring index -- `(base + rel) mod count`, with `base` a runtime value such
/// as a loop-carried rotation -- names a buffer relative to `base`: two of
/// them with the same base are the same buffer iff their offsets agree, which
/// is what lets a loop body of ONE stage index its ring at run time (the
/// buffer roles rotate each iteration) and still be checked. At the loop's
/// back edge the offsets shift by the rotation's step (see walk()).
struct Root {
  Operation *alloc = nullptr;
  int64_t index = -1;
  Value ring = {};   ///< the base of a ring index, else null
  int64_t rel = 0;   ///< the offset from it, in [0, count)
  bool known() const { return alloc && (index >= 0 || ring); }
  bool operator==(const Root &o) const {
    return alloc == o.alloc && index == o.index && ring == o.ring && rel == o.rel;
  }
  /// Whether two buffers may overlap: the same allocation, and the same
  /// buffer of it or one nobody can name.
  bool overlaps(const Root &o) const {
    if (!alloc || alloc != o.alloc)
      return false;
    if (ring && o.ring && ring == o.ring)
      return rel == o.rel;
    if (ring || o.ring)
      return true;
    return index < 0 || o.index < 0 || index == o.index;
  }
};

std::optional<int64_t> constOf(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
      return i.getInt();
  return std::nullopt;
}

/// `(base + rel) mod count` for an index `v` of a `count`-buffer stack: the
/// forms hk's `x % n` and `x + c` trace to. Null base if it is not one.
std::pair<Value, int64_t> ringOf(Value v, int64_t count) {
  Operation *def = v.getDefiningOp();
  if (!def || !isa<arith::RemSIOp, arith::RemUIOp>(def))
    return {};
  if (constOf(def->getOperand(1)) != count)
    return {};
  Value x = def->getOperand(0);
  int64_t rel = 0;
  // Peel constant additions and subtractions: rot + stages - 1 traces to
  // (rot + 2) - 1.
  while (true) {
    if (auto add = x.getDefiningOp<arith::AddIOp>()) {
      if (auto c = constOf(add.getRhs())) {
        rel += *c;
        x = add.getLhs();
        continue;
      }
      if (auto c = constOf(add.getLhs())) {
        rel += *c;
        x = add.getRhs();
        continue;
      }
    }
    if (auto sub = x.getDefiningOp<arith::SubIOp>())
      if (auto c = constOf(sub.getRhs())) {
        rel -= *c;
        x = sub.getLhs();
        continue;
      }
    break;
  }
  return {x, ((rel % count) + count) % count};
}

Root rootOf(Value v) {
  int64_t index = 0;
  while (Operation *def = v.getDefiningOp()) {
    if (auto s = dyn_cast<SubtileOp>(def)) {
      v = s.getSrc();
    } else if (auto g = dyn_cast<GranulesOp>(def)) {
      v = g.getSrc();
    } else if (auto a = dyn_cast<SharedAtOp>(def)) {
      index = -1;
      if (auto c = constOf(a.getIndex()))
        index = *c;
      v = a.getBuf();
      if (index < 0)
        if (auto alloc = v.getDefiningOp<AllocSharedOp>()) {
          auto [base, rel] = ringOf(a.getIndex(), alloc.getResult().getType().getCount());
          if (base)
            return {alloc, -1, base, rel};
        }
    } else if (isa<AllocSharedOp>(def)) {
      return {def, index};
    } else {
      break;
    }
  }
  return {};
}

/// When something happened, for the cross-warp checks: how many barriers
/// this warp had passed, and the hk.pingpong it was inside, if any.
struct Stamp {
  int64_t barriers = 0;
  Operation *region = nullptr;
};

struct Entry {
  const void *id;  ///< the tile for a read, the op for a write
  int64_t count;
  Value value;
  bool write;
  std::optional<Scope> scope;  ///< nullopt once a region it was issued in closed
  Root src = {};               ///< the LDS buffer a read reads
  bool crossed = false;        ///< a read still in flight at a barrier
};

/// One async group load, in flight on vmcnt.
struct Dma {
  Root root;
  int64_t count;
  Operation *op;
};

struct State {
  SmallVector<Entry> queue;
  llvm::MapVector<const void *, Value> bound;
  llvm::MapVector<const void *, Operation *> unbound;
  /// The latest ordinary LDS store nothing has retired yet. Its ds_writes are
  /// the compiler's, so they are not counted -- the compiler waits for them
  /// before this warp reads, and an estimate in a counted wait could only
  /// under-wait. But nothing waits for them before a barrier, and past one
  /// another warp reads the old value.
  Operation *plainStore = nullptr;

  /// Barriers passed (hk.pingpong's own are not counted: they are what makes
  /// the two halves' counts differ by one), and the pingpong we are inside.
  Stamp now;
  /// Async group loads this warp has issued and not retired, in issue order.
  SmallVector<Dma> dma;
  /// When this warp's share of the latest copy into a buffer landed.
  SmallVector<std::pair<Root, Stamp>> landed;
  /// When this warp's latest read of a buffer landed; `crossed` reads only
  /// for the second list, which every LDS write is checked against.
  SmallVector<std::pair<Root, Stamp>> readsDone, crossedDone;
  /// The op that last wrote a buffer through a group load, for messages.
  SmallVector<std::pair<Root, Operation *>> copiedBy;

  int indexOf(Value v) const {
    for (auto [i, e] : llvm::enumerate(queue))
      if (e.id == v.getAsOpaquePointer() && !e.write)
        return i;
    return -1;
  }
  static void stamp(SmallVector<std::pair<Root, Stamp>> &list, Root r,
                    Stamp s) {
    if (!r.known())
      return;
    for (auto &[k, v] : list)
      if (k == r) {
        v = s;
        return;
      }
    list.push_back({r, s});
  }
  /// A read's data has landed: when, for the copy that will overwrite it.
  void readLanded(const Entry &e) {
    if (e.write)
      return;
    stamp(readsDone, e.src, now);
    if (e.crossed)
      stamp(crossedDone, e.src, now);
  }
  void retireAllUnbound(Operation *by) {
    for (auto &e : queue) {
      if (e.write)
        continue;
      unbound[e.id] = by;
      readLanded(e);
    }
    queue.clear();
  }
  /// Rewrite every buffer name this state holds (a ring's back edge).
  template <typename F> void remapRoots(F f) {
    for (auto &e : queue)
      e.src = f(e.src);
    for (auto &d : dma)
      d.root = f(d.root);
    for (auto *list : {&landed, &readsDone, &crossedDone})
      for (auto &kv : *list)
        kv.first = f(kv.first);
    for (auto &kv : copiedBy)
      kv.first = f(kv.first);
  }
  /// vmcnt(0): every async copy this warp issued has landed.
  void retireAllDma() {
    for (auto &d : dma)
      stamp(landed, d.root, now);
    dma.clear();
  }
};

/// Barriers that must separate two events in different warps, the earlier
/// stamped `then`: one, so the later warp is past a barrier the earlier one
/// reached after its event; two when both are inside one hk.pingpong, where
/// the same barrier is one program point later in half the workgroup.
int64_t barriersNeeded(const Stamp &then, const Stamp &now) {
  return 1 + (then.region && then.region == now.region ? 1 : 0);
}

std::string nameOf(Value v) {
  if (auto r = dyn_cast<OpResult>(v))
    return valueNameOf(r.getOwner()->getLoc()).str();
  return valueNameOf(cast<BlockArgument>(v).getLoc()).str();
}

StringRef opcodeOf(Operation *op) {
  if (isa<HKDialect>(op->getDialect()))
    return op->getName().stripDialect();
  return op->getName().getStringRef();
}

/// Whether every wave that reaches `scope` issued an op issued in `issued`.
bool live(const std::optional<Scope> &issued, const Scope &scope) {
  if (!issued || issued->size() > scope.size())
    return false;
  for (auto [i, op] : llvm::enumerate(*issued))
    if (scope[i] != op)
      return false;
  return true;
}

/// ds_reads one load_shared issues on a wave64 target, where the count is
/// include/cdna{3,4}'s and depends on the fragment (hk-fragments has set it):
///   gfx950, per fragment: 16x16 row, one ds_read_b64; 16x32 row, one
///   ds_read_b128; 16x16 col, one ds_read_b64_tr_b16; 32x16 col, two;
///   32x16 row (from st_32x32), one ds_read_b128; 16x32 stride-4 col (from
///   st_8x32), two ds_read_b64_tr_b16; 32x16 stride-4 col (from st_32x16),
///   one ds_read_b64_tr_b16 a run, two (include/cdna4
///   tile/shared_to_register.cuh);
///   gfx942: one ds_read_b64 per 16x16 of a row-layout tile; a col-layout
///   tile is read with ordinary loads the compiler waits for itself.
/// 0: not inline asm, so nothing to count. -1: a pair hk has no count for.
/// The count has to be exact or low, never high: a load counted high makes
/// every wait issued while it is in flight too shallow.
int64_t cdnaLdsReads(const TargetInfo &t, RegTileType rt, SharedTileType st) {
  if (rt.getElementType() != st.getElementType())
    return 0;
  if (dtypeBits(rt.getElementType()) == 8) {
    // include/cdna4's fp8 16x128 row fragment: two runs of 16 bytes a lane,
    // one ds_read_b128 each (shared_to_register.cuh, stride 16).
    if (t.arch == "gfx950" && !rt.getColLayout() && rt.fragR() == 16 &&
        rt.fragC() == 128 && !rt.getFragStride())
      return 2 * int64_t(rt.getRows() / 16) * (rt.getCols() / 128);
    return 0;
  }
  if (dtypeBits(rt.getElementType()) != 16)
    return 0;
  int64_t frags = int64_t(rt.getRows() / rt.fragR()) * (rt.getCols() / rt.fragC());
  bool col = rt.getColLayout();
  unsigned fr = rt.fragR(), fc = rt.fragC();
  if (t.arch == "gfx942")
    return col ? 0 : frags;
  if (rt.getFragStride()) {
    if (col && fr == 16 && fc == 32 && rt.getFragStride() == 4)
      return 2 * frags;
    if (col && fr == 32 && fc == 16 && rt.getFragStride() == 4)
      return 2 * frags;
    return -1;
  }
  if (fr == 16 && fc == 16)
    return frags;
  if (!col && fr == 16 && fc == 32)
    return frags;
  if (col && fr == 32 && fc == 16)
    return 2 * frags;
  if (!col && fr == 32 && fc == 16)
    return frags;
  return -1;
}

struct Walker {
  bool failed = false;
  /// Set on wave64 targets: the per-load ds_read count is derived here.
  const TargetInfo *cdna = nullptr;

  bool fail(Operation *op, const Twine &msg) {
    op->emitError(msg);
    failed = true;
    return false;
  }

  /// The register tile a value is, through any hk.rt_subtile (an alias).
  static Value tileRoot(Value v) {
    while (auto s = v.getDefiningOp<RegSubtileOp>())
      v = s.getSrc();
    return v;
  }

  /// Nothing may read a tile that is in flight or retired-but-unbound; a
  /// subtile is read as the tile it is part of.
  bool checkReads(Operation *op, State &st) {
    for (Value operand : op->getOperands()) {
      if (!isa<RegTileType>(operand.getType()))
        continue;
      Value v = tileRoot(operand);
      if (st.indexOf(v) >= 0)
        return fail(op, Twine(opcodeOf(op)) + " reads " + nameOf(v) +
                            ", whose LDS reads are still in flight. Name it "
                            "in an lds_wait_for before using it. This is an "
                            "error rather than a race you might get away "
                            "with: the reads have not landed, and the "
                            "consumer will not wait for them on its own." +
                            atLoc(op));
      auto w = st.unbound.find(v.getAsOpaquePointer());
      if (w != st.unbound.end())
        return fail(op, Twine(opcodeOf(op)) + " reads " + nameOf(v) +
                            ". Its LDS reads were retired by an lds_wait_for "
                            "that did not name it, so the data has landed "
                            "but the register dependence has not been put "
                            "back -- this consumer is free to schedule above "
                            "the wait, and will, intermittently. Add " +
                            nameOf(v) + " to that lds_wait_for." +
                            atLoc(w->second) + atLoc(op));
    }
    return true;
  }

  bool waitFor(LdsWaitForOp op, State &st, const Scope &scope) {
    SmallVector<int> positions;
    for (Value v : op.getTiles()) {
      int i = st.indexOf(v);
      if (i < 0) {
        const void *id = v.getAsOpaquePointer();
        // Already retired by an earlier wait, bound or not: naming it here
        // is correct and costs nothing, and contributes no position.
        if (st.bound.count(id) || st.unbound.count(id))
          continue;
        return fail(op, "lds_wait_for names " + nameOf(v) +
                            ", which has no LDS reads outstanding and has "
                            "never had any. Either the load is missing, or "
                            "this names the wrong tile." +
                            atLoc(op));
      }
      positions.push_back(i);
    }
    int cut = positions.empty() ? -1
                                : *std::max_element(positions.begin(),
                                                    positions.end());
    int64_t n = 0;
    bool allLive = true;
    for (size_t j = cut + 1; j < st.queue.size(); ++j) {
      n += st.queue[j].count;
      allLive &= live(st.queue[j].scope, scope);
    }
    if (n && !allLive)
      return fail(op, "this lds_wait_for needs lgkmcnt(" + Twine(n) +
                          "), but some of the LDS ops it is counting were "
                          "issued inside an hk.if_. A wave that took the "
                          "other side of the branch has fewer in flight and "
                          "this wait lets it run on past reads that have not "
                          "landed. Retire the conditional ops before the "
                          "branch closes, or use barrier(drain=True), which "
                          "is lgkmcnt(0) and does not count." +
                          atLoc(op));
    llvm::DenseSet<const void *> named;
    for (Value v : op.getTiles())
      named.insert(v.getAsOpaquePointer());
    for (int j = 0; j <= cut; ++j) {
      auto &e = st.queue[j];
      if (e.write)
        continue;
      st.readLanded(e);
      if (!named.count(e.id))
        st.unbound[e.id] = op;
    }
    st.queue.erase(st.queue.begin(), st.queue.begin() + (cut + 1));
    for (Value v : op.getTiles()) {
      st.unbound.erase(v.getAsOpaquePointer());
      st.bound[v.getAsOpaquePointer()] = v;
    }
    if (auto prev = op.getN(); prev && *prev != uint64_t(n))
      return fail(op, "this lds_wait_for needs lgkmcnt(" + Twine(*prev) +
                          ") on one iteration and lgkmcnt(" + Twine(n) +
                          ") on the next: the loop has no steady state, so "
                          "the queue depth depends on how many iterations "
                          "have run. Peel the prologue out of the loop, or "
                          "issue the same reads every iteration." +
                          atLoc(op));
    op.setNAttr(Builder(op.getContext()).getI64IntegerAttr(n));
    return true;
  }

  /// `counted`: whether it advances the stamp. hk.pingpong's barriers are
  /// barriers for half the workgroup, and are what the stamp's region is for.
  bool barrier(Operation *op, bool drain, State &st, bool counted = true) {
    if (drain) {
      st.plainStore = nullptr;
    } else if (st.plainStore) {
      return fail(op, "barrier after " + Twine(opcodeOf(st.plainStore)) +
                          " with nothing waiting for its ds_writes. They are "
                          "ordinary stores: the compiler retires them before "
                          "this warp's own reads and not before a barrier, and "
                          "s_barrier orders execution, not memory -- past it "
                          "another warp can read what was there before. Write "
                          "barrier(drain=True), which spends an lgkmcnt(0) "
                          "here." +
                          atLoc(st.plainStore) + atLoc(op));
    }
    if (drain) {
      st.retireAllUnbound(op);
    } else if (!st.queue.empty()) {
      // A read of a buffer the pass can name may stay in flight: every
      // write into that buffer is checked against it (writeAfterRead). A
      // write in flight, or a read of a buffer picked at run time, may not.
      bool refused = llvm::any_of(st.queue, [](const Entry &e) {
        return e.write || !e.src.known();
      });
      if (refused) {
        std::set<std::string> kinds;
        for (auto &e : st.queue)
          kinds.insert(e.write ? "write" : "read");
        std::string what;
        for (auto &k : kinds) {
          if (!what.empty())
            what += " and ";
          what += "ds_" + k + "s";
        }
        return fail(op, "barrier with " + what +
                            " still in flight. s_barrier orders execution, not "
                            "memory: past it the other warps touch this LDS "
                            "the other way round, and nothing waits for these. "
                            "Either retire them first -- an lds_wait_for that "
                            "names the tiles, which is free if the math needed "
                            "them anyway -- or write barrier(drain=True) to "
                            "spend an lgkmcnt(0) here." +
                            atLoc(op));
      }
      for (auto &e : st.queue)
        e.crossed = true;
    }
    if (counted)
      ++st.now.barriers;
    return true;
  }

  static std::string bufName(Root r) {
    auto a = cast<AllocSharedOp>(r.alloc);
    std::string n = valueNameOf(a->getLoc()).str();
    if (a.getResult().getType().getCount() != 1)
      n += "[" + (r.ring ? "ring+" + std::to_string(r.rel)
                         : r.index < 0 ? std::string("?") : std::to_string(r.index)) +
           "]";
    return n;
  }

  static std::string barriersText(int64_t n) {
    return n == 1 ? "a barrier" : std::to_string(n) + " barriers";
  }

  static std::string pingpongText(const Stamp &then, const Stamp &now) {
    return barriersNeeded(then, now) > 1
               ? " Inside hk.pingpong half the workgroup is a barrier "
                 "behind, so it takes two."
               : "";
  }

  /// Every warp writes part of a group load's tile, so before one is issued
  /// every warp's reads of that buffer must have landed, a barrier (two in a
  /// pingpong) ago. Any other write is checked only against reads that were
  /// in flight at a barrier, which the barrier rule lets through for a
  /// buffer the pass can name and nowhere else.
  bool writeAfterRead(Operation *op, Root dst, bool workgroup, State &st) {
    if (!dst.alloc)
      return true;
    for (auto &e : st.queue) {
      if (e.write || !e.src.overlaps(dst) || (!workgroup && !e.crossed))
        continue;
      return fail(op, Twine(opcodeOf(op)) + " writes " + bufName(dst) +
                          " while this warp's reads of it into " +
                          nameOf(e.value) +
                          " are still in flight, and the other warps' reads "
                          "of it are no further along. Retire them with an "
                          "lds_wait_for, then put " +
                          barriersText(barriersNeeded(st.now, st.now)) +
                          " between that and this write." +
                          atLoc(op));
    }
    for (auto &[r, s] : workgroup ? st.readsDone : st.crossedDone) {
      if (!r.overlaps(dst))
        continue;
      int64_t have = st.now.barriers - s.barriers, need = barriersNeeded(s, st.now);
      if (have < need)
        return fail(op, Twine(opcodeOf(op)) + " overwrites " + bufName(dst) +
                            " " + Twine(have) + " barrier" +
                            (have == 1 ? "" : "s") +
                            " after this warp's reads of it landed. Another "
                            "warp may still be reading it: a warp is only "
                            "known to be past its reads once it has reached " +
                            barriersText(need) + " after them." +
                            pingpongText(s, st.now) + atLoc(op));
    }
    return true;
  }

  /// A read of a buffer a group load wrote: this warp's share must have
  /// landed (group_wait_for), and every other warp's, which a barrier (two
  /// in a pingpong) after their waits makes sure of.
  bool readAfterCopy(Operation *op, Root src, State &st) {
    if (!src.alloc)
      return true;
    for (auto &d : st.dma)
      if (d.root.overlaps(src))
        return fail(op, Twine(opcodeOf(op)) + " reads " + bufName(src) +
                            " while this warp's copy into it is in flight. "
                            "Name it in a group_wait_for, then a barrier, "
                            "before reading it." +
                            atLoc(d.op) + atLoc(op));
    for (auto &[r, s] : st.landed) {
      if (!r.overlaps(src))
        continue;
      int64_t have = st.now.barriers - s.barriers, need = barriersNeeded(s, st.now);
      if (have < need)
        return fail(op, Twine(opcodeOf(op)) + " reads " + bufName(src) +
                            " " + Twine(have) + " barrier" +
                            (have == 1 ? "" : "s") +
                            " after this warp's share of the copy into it "
                            "landed. The other warps' shares are only known "
                            "to have landed " +
                            barriersText(need) + " after their waits." +
                            pingpongText(s, st.now) + atLoc(op));
    }
    return true;
  }

  bool groupLoad(GroupLoadOp op, State &st) {
    Root dst = rootOf(op.getDst());
    if (op.getAsync()) {
      if (!cdna || cdna->arch != "gfx950")
        return fail(op, "an asynchronous group load on " +
                            (cdna ? cdna->arch : std::string("this target")) +
                            ": only gfx950 has the 16-byte buffer_load ... lds "
                            "it is. Use wait=True." +
                            atLoc(op));
      if (!dst.known())
        return fail(op, "an asynchronous group load into a buffer picked by "
                        "a runtime index. hk tracks which buffer a copy is in "
                        "flight to, to check every read and every later copy "
                        "against it; index the stack with a Python int." +
                            atLoc(op));
      for (auto &d : st.dma)
        if (d.root == dst)
          return fail(op, "a second copy into " + bufName(dst) +
                              " while the first is still in flight. Retire "
                              "the first with a group_wait_for." +
                              atLoc(d.op) + atLoc(op));
      if (!writeAfterRead(op, dst, true, st))
        return false;
      st.dma.push_back({dst, op.dmaCalls(), op});
      return true;
    }
    if (!writeAfterRead(op, dst, true, st))
      return false;
    // Issued and waited by the group: it ends in s_waitcnt 0.
    st.retireAllUnbound(op);
    st.retireAllDma();
    st.plainStore = nullptr;
    State::stamp(st.landed, dst, st.now);
    return true;
  }

  bool groupWaitFor(GroupWaitForOp op, State &st) {
    int cut = -1;
    for (Value v : op.getTiles()) {
      Root r = rootOf(v);
      int last = -1;
      for (auto [i, d] : llvm::enumerate(st.dma))
        if (d.root.overlaps(r))
          last = int(i);
      if (last < 0) {
        bool landed = llvm::any_of(st.landed, [&](auto &kv) {
          return kv.first.overlaps(r);
        });
        if (landed)
          continue;
        return fail(op, "group_wait_for names " +
                            (r.alloc ? bufName(r) : nameOf(v)) +
                            ", which has no copy in flight and has never had "
                            "one. Either the group load is missing, or this "
                            "names the wrong buffer." +
                            atLoc(op));
      }
      cut = std::max(cut, last);
    }
    // Only copies into LDS issued after the cut may stay in flight: loads
    // into registers (E8M0 scales, rootless entries) retire out of order with
    // them on gfx950 -- a vmcnt that counted scales issued after a copy as
    // the ones left in flight let the copy's own last loads through instead,
    // and a K-loop read them before they landed (results varying run to run).
    int64_t n = 0;
    for (size_t j = cut + 1; j < st.dma.size(); ++j)
      if (st.dma[j].root.alloc)
        n += st.dma[j].count;
    for (int j = 0; j <= cut; ++j)
      State::stamp(st.landed, st.dma[j].root, st.now);
    st.dma.erase(st.dma.begin(), st.dma.begin() + (cut + 1));
    if (auto prev = op.getN(); prev && *prev != uint64_t(n))
      return fail(op, "this group_wait_for needs vmcnt(" + Twine(*prev) +
                          ") on one iteration and vmcnt(" + Twine(n) +
                          ") on the next: the loop has no steady state. Issue "
                          "the same copies every iteration, or peel the "
                          "prologue out of the loop." +
                          atLoc(op));
    op.setNAttr(Builder(op.getContext()).getI64IntegerAttr(n));
    return true;
  }

  /// hk.arch.vm_wait(keep): the oldest copies land until `keep` are left.
  void vmWait(VmWaitOp op, State &st) {
    int64_t total = 0;
    for (auto &d : st.dma)
      total += d.count;
    size_t k = 0;
    while (k < st.dma.size() && total > int64_t(op.getKeep())) {
      total -= st.dma[k].count;
      State::stamp(st.landed, st.dma[k].root, st.now);
      ++k;
    }
    st.dma.erase(st.dma.begin(), st.dma.begin() + k);
  }

  bool load(Operation *op, Value dst, bool wait, int64_t count, State &st,
            const Scope &scope, Root src = {}) {
    if (dst.getDefiningOp<RegSubtileOp>())
      return fail(op, Twine(opcodeOf(op)) + " reads LDS into " + nameOf(dst) +
                          ", a subtile of another register tile. The wait "
                          "and the bind name tiles, not windows of them: "
                          "load into the whole tile." +
                          atLoc(op));
    if (st.indexOf(dst) >= 0)
      return fail(op, Twine(opcodeOf(op)) + " writes " + nameOf(dst) +
                          " while its previous LDS reads are still in "
                          "flight. The earlier reads would land on top of "
                          "the newer ones -- retire them with an "
                          "lds_wait_for first." +
                          atLoc(op));
    if (!readAfterCopy(op, src, st))
      return false;
    const void *id = dst.getAsOpaquePointer();
    if (wait) {
      // load<true> ends in lgkmcnt(0): it retires everything, binds nothing.
      st.retireAllUnbound(op);
      State::stamp(st.readsDone, src, st.now);
      st.plainStore = nullptr;
      st.unbound.erase(id);
    } else {
      st.unbound.erase(id);
      st.bound.erase(id);
      Entry e{id, count, dst, false, scope};
      e.src = src;
      st.queue.push_back(e);
    }
    return true;
  }

  State walk(Block &block, State st, const Scope &scope) {
    for (Operation &opRef : block) {
      if (failed)
        return st;
      Operation *op = &opRef;
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        // Once: a branch runs at most once per encounter. Ops issued inside
        // carry it in their scope, and once it closes a counted wait that
        // would count them is the error.
        Scope inner = scope;
        inner.push_back(op);
        st = walk(ifOp.getThenRegion().front(), std::move(st), inner);
        if (!ifOp.getElseRegion().empty())
          st = walk(ifOp.getElseRegion().front(), std::move(st), inner);
        for (auto &e : st.queue)
          if (e.scope && llvm::is_contained(*e.scope, op))
            e.scope = std::nullopt;
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        // Rings rotated by a loop-carried value: base B starts at a constant
        // r and is yielded as (B + k) mod n. Buffers named by constants on
        // the way in become ring names relative to B; at the back edge an
        // offset c becomes c - k (B moved on by k); on the way out they are
        // named relative to the loop's result.
        struct Rot { Value arg, result; int64_t init, step, count; };
        SmallVector<Rot> rots;
        auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
          auto init = constOf(forOp.getInitArgs()[i]);
          Value y = yield.getOperand(i);
          Operation *rem = y.getDefiningOp();
          if (!init || !rem || !isa<arith::RemSIOp, arith::RemUIOp>(rem))
            continue;
          auto n = constOf(rem->getOperand(1));
          if (!n || *n <= 0)
            continue;
          auto [base, k] = ringOf(y, *n);
          if (base != arg)
            continue;
          rots.push_back({arg, forOp.getResult(i), *init, k, *n});
        }
        // Which allocations each rotation indexes (its count must match).
        SmallVector<std::pair<Operation *, size_t>> ringed;
        forOp.getBody()->walk([&](SharedAtOp a) {
          Root r = rootOf(a.getResult());
          for (auto [j, rot] : llvm::enumerate(rots))
            if (r.ring == rot.arg)
              ringed.push_back({r.alloc, j});
        });
        auto entering = [&](Root r) {
          for (auto &[alloc, j] : ringed)
            if (r.alloc == alloc && r.index >= 0)
              return Root{alloc, -1, rots[j].arg,
                          ((r.index - rots[j].init) % rots[j].count + rots[j].count) % rots[j].count};
          return r;
        };
        auto backEdge = [&](Value to) {
          return [&, to](Root r) {
            for (auto &rot : rots)
              if (r.ring == rot.arg) {
                r.rel = ((r.rel - rot.step) % rot.count + rot.count) % rot.count;
                if (to)
                  r.ring = rot.result;
              }
            return r;
          };
        };
        if (!rots.empty())
          st.remapRoots(entering);
        State first = walk(*forOp.getBody(), st, scope);
        if (!rots.empty())
          first.remapRoots(backEdge(Value()));
        st = walk(*forOp.getBody(), std::move(first), scope);
        if (!rots.empty())
          st.remapRoots([&](Root r) {
            for (auto &rot : rots)
              if (r.ring == rot.arg) {
                r.rel = ((r.rel - rot.step) % rot.count + rot.count) % rot.count;
                r.ring = rot.result;
              }
            return r;
          });
        continue;
      }
      if (auto wl = dyn_cast<scf::WhileOp>(op)) {
        // The test runs once more than the body: twice round, then the exit.
        Block &test = wl.getBefore().front(), &body = wl.getAfter().front();
        for (int round = 0; round < 2; ++round)
          st = walk(body, walk(test, std::move(st), scope), scope);
        st = walk(test, std::move(st), scope);
        continue;
      }
      if (auto w = dyn_cast<LdsWaitForOp>(op)) {
        waitFor(w, st, scope);
        continue;
      }
      if (auto b = dyn_cast<BarrierOp>(op)) {
        barrier(op, b.getDrain(), st);
        continue;
      }
      if (isa<PingPongBeginOp>(op)) {
        barrier(op, false, st, /*counted=*/false);
        st.now.region = op;
        continue;
      }
      if (isa<PingPongEndOp>(op)) {
        barrier(op, false, st, /*counted=*/false);
        st.now.region = nullptr;
        continue;
      }
      if (auto m = dyn_cast<LoadMxScalesOp>(op)) {
        // Byte loads of E8M0 scales: on vmcnt with the copies, one per 16-row
        // base tile and 128-wide K step. They land in registers the compiler
        // waits for itself; here they only count, so that a wait for a copy
        // issued before them keeps them in flight instead of retiring them.
        auto t = m.getResult() ? m.getResult().getType() : m.getDst().getType();
        auto mt = cast<MxScalesType>(t);
        // Packed (i32) scales: one dword load per 64-row group and K step. AITER's shuffled
        // layouts (layout 1, 2): one dword per 32-row block and pair of K steps.
        int64_t n;
        if (m.getLayout())
          n = int64_t(mt.getRows() / 32) * ((mt.getKblocks() / 4 + 1) / 2);
        else {
          int64_t per = m.getSrc().getType().getElementType().isInteger(32) ? 64 : 16;
          n = int64_t(mt.getRows() / per) * (mt.getKblocks() / 4);
        }
        st.dma.push_back({Root{}, n, op});
        continue;
      }
      if (auto g = dyn_cast<GroupLoadOp>(op)) {
        groupLoad(g, st);
        continue;
      }
      if (auto w = dyn_cast<GroupWaitForOp>(op)) {
        groupWaitFor(w, st);
        continue;
      }
      if (auto v = dyn_cast<VmWaitOp>(op)) {
        vmWait(v, st);
        continue;
      }
      if (auto l = dyn_cast<LoadSharedOp>(op)) {
        Value dst = l.getDst() ? l.getDst() : l.getResult();
        int64_t n = l.getLdsLoads();
        if (cdna) {
          n = cdnaLdsReads(*cdna, cast<RegTileType>(dst.getType()),
                           l.getSrc().getType());
          if (!l.getWait() && n <= 0) {
            fail(op, "load(wait=False) of " + nameOf(dst) + " on " +
                         cdna->arch +
                         (n == 0 ? Twine(": this tile is read with ordinary "
                                         "loads the compiler waits for, so "
                                         "there is nothing to keep in flight")
                                 : Twine(": hk has no ds_read count for this "
                                         "fragment, and a wait derived from "
                                         "a guess could under-wait")) +
                         ". Use wait=True." + atLoc(op));
            continue;
          }
          l.setLdsLoads(n < 0 ? 0 : n);
        }
        load(op, dst, l.getWait(), n, st, scope, rootOf(l.getSrc()));
        continue;
      }
      if (auto l = dyn_cast<LoadFragOp>(op)) {
        Value dst = l.getDst() ? l.getDst() : l.getResult();
        load(op, dst, l.getWait(), l.getLdsLoads(), st, scope,
             rootOf(l.getGranules()));
        continue;
      }
      if (auto s = dyn_cast<StoreFragOp>(op)) {
        if (!writeAfterRead(op, rootOf(s.getGranules()), false, st))
          continue;
        // Two raw ds_write_b128 with no wait of their own: they only raise
        // the immediate of every wait issued while they are in flight.
        st.queue.push_back({op, 2, s.getGranules(), true, scope});
        continue;
      }
      if (auto c = dyn_cast<StageCommitOp>(op)) {
        if (!writeAfterRead(op, rootOf(c.getDst()), false, st))
          continue;
        if (c.getWait()) {
          st.retireAllUnbound(op);
          st.plainStore = nullptr;
        } else {
          st.queue.push_back(
              {op, c.getBuf().getType().getCalls(), c.getDst(), true, scope});
        }
        continue;
      }
      if (isa<StoreSharedVecOp>(op)) {
        checkReads(op, st);
        st.plainStore = op;
        continue;
      }
      if (auto s = dyn_cast<StoreSharedOp>(op)) {
        checkReads(op, st);
        if (!writeAfterRead(op, rootOf(s.getDst()), false, st))
          continue;
        // The vectorised form -- a row-layout 16-bit tile into an LDS tile of
        // its own type -- writes through inline asm and ends in lgkmcnt(0);
        // every other form is ordinary stores.
        auto src = cast<RegTileType>(s.getValue().getType());
        auto dst = cast<SharedTileType>(s.getDst().getType());
        bool vectorised = src.getLayout() == "row" &&
                          dtypeBits(src.getElementType()) == 16 &&
                          src.getElementType() == dst.getElementType();
        if (vectorised) {
          st.retireAllUnbound(op);
          st.plainStore = nullptr;
        } else {
          st.plainStore = op;
        }
        continue;
      }
      checkReads(op, st);
    }
    return st;
  }
};

struct LdsPipelinePass : public mlir::hk::impl::HKLdsPipelineBase<LdsPipelinePass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    if (!isKernel(f))
      return;
    // A stale count from an earlier run would read as a missing steady state.
    f.walk([](LdsWaitForOp op) { op.removeNAttr(); });
    f.walk([](GroupWaitForOp op) { op.removeNAttr(); });
    Walker w;
    auto arch = f->getAttrOfType<StringAttr>(attrs::kArch);
    if (const TargetInfo *t = getTarget(arch ? arch.getValue() : "gfx1100");
        t && t->wave == 64)
      w.cdna = t;
    State fin = w.walk(f.getBody().front(), State(), Scope());
    if (w.failed)
      return signalPassFailure();
    std::string names;
    for (auto &e : fin.queue) {
      if (e.write)
        continue;
      if (!names.empty())
        names += ", ";
      names += nameOf(e.value);
    }
    if (!names.empty()) {
      f.emitError() << "the kernel ends with LDS reads still in flight for "
                    << names
                    << ". Either they are dead -- drop the load -- or a wait "
                       "is missing.";
      return signalPassFailure();
    }
    // Scale loads (no buffer) land in registers the compiler waits for.
    llvm::erase_if(fin.dma, [](const Dma &d) { return !d.root.alloc; });
    if (!fin.dma.empty()) {
      f.emitError() << "the kernel ends with a copy into "
                    << Walker::bufName(fin.dma.front().root)
                    << " still in flight. Either nothing reads it -- drop the "
                       "group load -- or a group_wait_for is missing."
                    << atLoc(fin.dma.front().op);
      return signalPassFailure();
    }
  }
};

} // namespace
