// hk -> HipKittens C++.
//
// The output is meant to be read: when a kernel spills or mis-schedules, the
// generated .cpp is what goes to the resource check and the disassembler. So
// real names, one type alias per tile type, and the same structure a
// hand-written kernel in kernels/rdna3 has.
//
// Tiles are buffers in the dialect, so every op is already in the
// destination-passing form include/rdna3 exposes: an op with a result
// declares it, an op with `into %dst` writes a local that already exists.

#include "hk/Target/Cpp/HKToCpp.h"

#include "hk/Support/PyRepr.h"
#include "hk/Target/Target.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <vector>

using namespace mlir;
using namespace mlir::hk;

namespace {

std::string join(ArrayRef<std::string> xs, StringRef sep = ", ") {
  std::string out;
  for (auto [i, x] : llvm::enumerate(xs)) {
    if (i)
      out += sep;
    out += x;
  }
  return out;
}

/// A float literal for C++: Python's repr, except where that is not a token.
std::string floatLit(double v) {
  if (std::isnan(v))
    return "NAN";
  if (std::isinf(v))
    return v > 0 ? "INFINITY" : "-INFINITY";
  return pyFloat(v);
}

std::string fragName(unsigned r, unsigned c) {
  return std::to_string(r) + "x" + std::to_string(c);
}

/// A tile's fragment as include/cdna4 names it: rt_16x32, or rt_16x32_4
/// where the lane run is not the shape's own.
std::string fragName(RegTileType r) {
  std::string s = fragName(r.getFragRows(), r.getFragCols());
  if (r.getFragStride())
    s += "_" + std::to_string(r.getFragStride());
  return s;
}

std::string rawCpp(Type t) {
  if (auto r = dyn_cast<RegTileType>(t); r && r.isStrip())
    return "kittens::rs<" + dtypeCpp(r.getElementType()).str() + ", " +
           std::to_string(r.getRows()) + ", " + std::to_string(r.getCols()) +
           ", " + std::to_string(r.getRun()) + ">";
  if (auto r = dyn_cast<RegTileType>(t))
    return "kittens::rt<" + dtypeCpp(r.getElementType()).str() + ", " +
           std::to_string(r.getRows()) + ", " + std::to_string(r.getCols()) +
           ", kittens::ducks::rt_layout::" + r.getLayout().str() +
           (r.hasFrag() ? ", kittens::ducks::rt_shape::rt_" + fragName(r)
                        : std::string()) +
           ">";
  if (auto s = dyn_cast<SharedTileType>(t))
    return "kittens::st<" + dtypeCpp(s.getElementType()).str() + ", " +
           std::to_string(s.getRows()) + ", " + std::to_string(s.getCols()) +
           ">";
  if (auto v = dyn_cast<RegVecType>(t))
    return "typename " + rawCpp(v.getTile()) + "::" + v.getKind().str() +
           "_vec";
  if (auto s = dyn_cast<SharedVecType>(t))
    return "kittens::sv<" + dtypeCpp(s.getElementType()).str() + ", " +
           std::to_string(s.getLength()) + ">";
  if (auto g = dyn_cast<GlobalType>(t))
    return "kittens::gl<" + dtypeCpp(g.getElementType()).str() +
           ", -1, -1, -1, -1>";
  if (auto g = dyn_cast<GranulesType>(t))
    return "kittens::lds_granules<" + rawCpp(g.getTile()) + ">";
  if (isa<StageType>(t))
    return "float4";
  if (auto m = dyn_cast<MxScalesType>(t))
    return "kittens::hk_mxs<" + std::to_string(m.getRows()) + ", " +
           std::to_string(m.getKblocks()) +
           (m.getFmt() ? ", " + std::to_string(m.getFmt()) : std::string()) + ">";
  if (t.isIntOrFloat())
    return dtypeCpp(t).str();
  return "/*?*/";
}

Type plainOf(Type t) {
  if (auto v = dyn_cast<RegVecType>(t))
    return v.plain();
  return t;
}

/// hk.atomic_scalar's kind as kittens::atomic_scalar's op code.
static std::string atomicKindCode(StringRef kind) {
  static const char *kinds[] = {"add", "max", "min", "or", "and", "xor", "exch"};
  for (int i = 0; i < 7; ++i)
    if (kind == kinds[i])
      return std::to_string(i);
  return "0";
}

/// What makes two types one alias: the C++ they spell. An rv's identity is
/// (element, length, kind, layout), not the tile it carries; a strip's
/// col_vec is rs_cv<T, rows> and its row_vec rs<T, 1, cols, run>.
std::string typeKey(Type t) {
  t = plainOf(t);
  if (auto v = dyn_cast<RegVecType>(t))
    return "rv|" + dtypeName(v.getElementType()).str() + "|" +
           std::to_string(v.getLength()) + "|" + v.getKind().str() + "|" +
           (v.isStrip() ? (v.getColKind() ? std::string("strip")
                                          : "strip" + std::to_string(v.getTileRun()))
                        : std::string(v.getTileColLayout() ? "col" : "row")) +
           (v.getTileFragRows()
                ? "|" + fragName(v.getTileFragRows(), v.getTileFragCols())
                : std::string());
  std::string s;
  llvm::raw_string_ostream os(s);
  os << t;
  return s;
}

std::string aliasName(Type t) {
  if (auto r = dyn_cast<RegTileType>(t); r && r.isStrip())
    return "rs_" + dtypeName(r.getElementType()).str() + "_" +
           std::to_string(r.getRows()) + "x" + std::to_string(r.getCols()) +
           "_r" + std::to_string(r.getRun());
  if (auto v = dyn_cast<RegVecType>(t); v && v.isStrip())
    return "rv_" + dtypeName(v.getElementType()).str() + "_" +
           std::to_string(v.getTileRows()) + "x" +
           std::to_string(v.getTileCols()) + "_r" +
           std::to_string(v.getTileRun()) + "_" + v.getKind().str();
  if (auto r = dyn_cast<RegTileType>(t))
    return "rt_" + dtypeName(r.getElementType()).str() + "_" +
           std::to_string(r.getRows()) + "x" + std::to_string(r.getCols()) +
           "_" + r.getLayout().str() +
           (r.hasFrag() ? "_f" + fragName(r) : std::string());
  if (auto v = dyn_cast<RegVecType>(t))
    return "rv_" + dtypeName(v.getElementType()).str() + "_" +
           std::to_string(v.getTileRows()) + "x" +
           std::to_string(v.getTileCols()) + "_" +
           (v.getTileColLayout() ? "col" : "row") +
           (v.getTileFragRows()
                ? "_f" + fragName(v.getTileFragRows(), v.getTileFragCols())
                : std::string()) +
           "_" + v.getKind().str();
  auto s = cast<SharedTileType>(t);
  return "st_" + dtypeName(s.getElementType()).str() + "_" +
         std::to_string(s.getRows()) + "x" + std::to_string(s.getCols());
}

std::string pyValue(Attribute a) {
  if (auto b = dyn_cast<BoolAttr>(a))
    return b.getValue() ? "True" : "False";
  if (auto i = dyn_cast<IntegerAttr>(a))
    return std::to_string(i.getInt());
  if (auto f = dyn_cast<FloatAttr>(a))
    return pyFloat(f.getValueAsDouble());
  if (auto s = dyn_cast<StringAttr>(a))
    return s.getValue().str();
  return "?";
}

class Emitter {
public:
  Emitter(func::FuncOp kernel) : kernel(kernel) {}

  LogicalResult emit(ArrayRef<std::string> includes, raw_ostream &os) {
    auto arch = kernel->getAttrOfType<StringAttr>(attrs::kArch);
    tgt = getTarget(arch ? arch.getValue() : "gfx1100");
    if (!tgt)
      return kernel.emitError("unknown arch");
    if (failed(findGrid()))
      return failure();
    findSharedShapes();
    assignNames();
    header(includes);
    if (failed(globalsStruct()))
      return failure();
    if (failed(kernelBody()))
      return failure();
    launch();
    for (auto &l : lines)
      os << l << "\n";
    return failure(bad);
  }

  void applyNames() {
    assignNames();
    auto under = [&](const std::string &n, Location loc) -> Location {
      if (auto nl = dyn_cast<NameLoc>(loc))
        loc = nl.getChildLoc();
      return NameLoc::get(StringAttr::get(kernel.getContext(), n), loc);
    };
    for (auto &[v, n] : names) {
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        ba.setLoc(under(n, ba.getLoc()));
        continue;
      }
      // One location per op: a multi-result op keeps MLIR's numbering.
      Operation *op = v.getDefiningOp();
      if (op->getNumResults() == 1)
        op->setLoc(under(n, op->getLoc()));
    }
  }

private:
  func::FuncOp kernel, grid;
  const TargetInfo *tgt = nullptr;
  std::vector<std::string> lines;
  int indent = 1;
  int opaque = 0;
  bool bad = false;
  DenseMap<Value, std::string> names;
  llvm::MapVector<std::string, std::pair<Type, std::string>,
                  std::map<std::string, unsigned>>
      aliases;
  std::map<std::string, std::string> sharedShape;
  llvm::MapVector<Value, std::string> allocShape;
  std::set<std::string> conflicted;

  // -- helpers ----------------------------------------------------------

  void w(const std::string &line = "") {
    if (line.empty())
      lines.emplace_back();
    else
      lines.push_back(std::string(4 * indent, ' ') + line);
  }

  StringRef kernelName() { return kernel.getSymName(); }
  int64_t intAttr(StringRef n, int64_t d = 0) {
    if (auto a = kernel->getAttrOfType<IntegerAttr>(n))
      return a.getInt();
    return d;
  }
  int64_t threads() { return intAttr(attrs::kWarps, 1) * tgt->wave; }

  /// The st_shape an LDS tile read into a register tile of fragment `dt`
  /// is swizzled as. Mostly the fragment's own (a 16x32 row operand from
  /// st_16x32); a 32x16 row operand from st_32x32's swizzle, one
  /// ds_read_b128 a fragment, and a 16x32 stride-4 col operand from
  /// st_8x32, two ds_read_b64_tr_b16 -- the pair HipKittens' CDNA4
  /// attention reads K and V with.
  static std::string sharedShapeFor(RegTileType dt) {
    unsigned r = dt.getFragRows(), c = dt.getFragCols(), s = dt.fragS();
    if (!dt.getColLayout() && r == 32 && c == 16 && s == 8)
      return "st_32x32";
    if (dt.getColLayout() && r == 16 && c == 32 && s == 4)
      return "st_8x32";
    return "st_" + fragName(r, c);
  }

  /// The allocation an LDS tile value is a buffer or window of.
  static Value sharedRoot(Value src) {
    while (Operation *def = src.getDefiningOp()) {
      if (auto s = dyn_cast<SubtileOp>(def))
        src = s.getSrc();
      else if (auto a = dyn_cast<SharedAtOp>(def))
        src = a.getBuf();
      else
        break;
    }
    return src;
  }

  /// gfx950's shared tile names a shape, and it has to match the register
  /// tile read out of it: the allocation a fragment tile is loaded from
  /// (through any subtile) takes that fragment's shape (sharedShapeFor).
  /// hk-fragments has put the fragment in the register tile's type. Two
  /// reads that want two shapes of one allocation are refused: the swizzle
  /// is the allocation's, so one of them would read the wrong elements.
  /// Allocations of one type that want different shapes (attention's K and
  /// V) are spelled apart: their type is `conflicted`, and the allocation
  /// and its buffers are declared with their own shape.
  /// Whether include/cdna4 reads fragment `dt` out of an allocation laid
  /// out as `shape`: its own, or one of the layouts its reads also take
  /// (tile/shared_to_register.cuh's generic paths).
  static bool readsFrom(RegTileType dt, StringRef shape) {
    std::string own = sharedShapeFor(dt);
    if (shape == own)
      return true;
    unsigned r = dt.getFragRows(), c = dt.getFragCols(), s = dt.fragS();
    if (!dt.getColLayout() && r == 16 && c == 32 && s == 8)
      return shape == "st_16x64";
    if (dt.getColLayout() && r == 32 && c == 16 && s == 4)
      return shape == "st_32x32";
    return false;
  }

  void findSharedShapes() {
    if (!sharedTilesHaveShape())
      return;
    // An allocation that names its layout keeps it; its reads are held to
    // fragments that layout serves.
    kernel.walk([&](AllocSharedOp a) {
      if (auto s = a.getShapeAttr())
        allocShape.insert({a.getResult(), s.getValue().str()});
    });
    // include/cdna4's st_subtile points at the layout's subtile holding its
    // first element and keeps no offset inside it: a window narrower than
    // the layout's subtile (32 columns of st_16x64's 64) reads that
    // subtile's first columns, whatever its index.
    kernel.walk([&](SubtileOp o) {
      auto a = sharedRoot(o.getSrc()).getDefiningOp<AllocSharedOp>();
      if (!a || !a.getShapeAttr())
        return;
      StringRef shape = a.getShapeAttr().getValue();
      unsigned r = shape == "st_32x32" ? 32 : 16, c = shape == "st_32x32" ? 32 : 64;
      auto t = o.getResult().getType();
      if (t.getRows() % r || t.getCols() % c) {
        o.emitError() << "a " << t.getRows() << "x" << t.getCols()
                      << " window of an LDS tile laid out as " << shape
                      << " is not whole " << r << "x" << c
                      << " subtiles of it, which is all include/cdna4's "
                         "subtile addresses; read whole subtiles";
        bad = true;
      }
    });
    // An allocation only load_fp4_frags_lds reads holds raw bytes in 1 KB chunks: what an fp8
    // 16x128 operand read would pick (an unnamed byte tile has no include/cdna4 layout at all).
    kernel.walk([&](LoadFp4FragsLdsOp l) {
      allocShape.insert({sharedRoot(l.getSrc()), "st_16x128"});
    });
    std::map<Operation *, Operation *> by;
    kernel.walk([&](LoadSharedOp l) {
      auto dt = cast<RegTileType>(l.getDst() ? l.getDst().getType()
                                             : l.getResult().getType());
      if (!dt.hasFrag())
        return;
      Value root = sharedRoot(l.getSrc());
      if (auto a = root.getDefiningOp<AllocSharedOp>(); a && a.getShapeAttr()) {
        if (!readsFrom(dt, a.getShapeAttr().getValue())) {
          l.emitError() << "this read's fragment is not one an LDS tile laid "
                           "out as "
                        << a.getShapeAttr().getValue()
                        << " serves; it wants " << sharedShapeFor(dt);
          bad = true;
        }
        return;
      }
      std::string shape = sharedShapeFor(dt);
      auto [it, fresh] = allocShape.insert({root, shape});
      if (!fresh && it->second != shape) {
        InFlightDiagnostic d = l.emitError();
        d << "this read wants its LDS tile swizzled as " << shape
          << ", and another read of the same allocation wants " << it->second
          << ". The swizzle is the allocation's, so read both with one "
             "fragment, or stage them in two allocations";
        d.attachNote(by[root.getDefiningOp()]->getLoc()) << "the other read";
        bad = true;
      }
      by.insert({root.getDefiningOp(), l});
    });
    for (auto &[root, shape] : allocShape) {
      std::string key = rawCpp(root.getType());
      auto [it, fresh] = sharedShape.insert({key, shape});
      if (!fresh && it->second != shape)
        conflicted.insert(key);
    }
  }

  /// An allocation's C++ type, with its own shape where its type is
  /// conflicted.
  std::string allocCpp(Value root) {
    auto t = cast<SharedTileType>(root.getType());
    std::string s = rawCpp(t);
    if (!conflicted.count(s))
      return tileCpp(t);
    auto it = allocShape.find(root);
    std::string shape = it != allocShape.end() ? it->second : "st_16x16";
    s.insert(s.size() - 1, ", kittens::ducks::st_shape::" + shape);
    return s;
  }

  /// gfx950's shared tile takes a shape parameter and has no default.
  bool sharedTilesHaveShape() {
    return !tgt->mma.empty() && tgt->mma.front().second.k != 16;
  }

  /// rawCpp, with the shared tile shape findSharedShapes assigned.
  std::string tileCpp(Type t) {
    std::string s = rawCpp(t);
    if (isa<SharedTileType>(t) && sharedTilesHaveShape()) {
      auto it = sharedShape.find(s);
      std::string shape = it != sharedShape.end() ? it->second : "st_16x16";
      s.insert(s.size() - 1, ", kittens::ducks::st_shape::" + shape);
    }
    return s;
  }
  int64_t ldsBytes() { return intAttr(attrs::kLdsBytes); }

  StringRef argName(func::FuncOp f, unsigned i) {
    if (auto n = f.getArgAttrOfType<StringAttr>(i, attrs::kArgName))
      return n.getValue();
    return "";
  }

  /// A strided_gl's fields as the flat kernel takes them, each across every
  /// tensor before the next: (parameter suffix, accessor). The pointers
  /// first, then what addresses a row, so the first loads' operands are the
  /// arguments LLVM preloads into SGPRs at dispatch.
  static constexpr std::pair<const char *, const char *> kFlatFields[] = {
      {"", "raw_ptr"},          {"_rs", "row_stride()"}, {"_r", "rows()"},
      {"_c", "cols()"},         {"_b", "batch()"},       {"_d", "depth()"}};

  /// CDNA with hk.noalias: the kernel takes the globals' fields, not the
  /// struct. A struct argument is passed by reference into the kernarg
  /// segment, which LLVM neither preloads nor lets the pointers in it be
  /// __restrict__ (docs/history/43).
  bool flatArgs() {
    if (tgt->wave != 64 || !kernel->hasAttr(attrs::kNoAlias))
      return false;
    for (unsigned i = 0; i < kernel.getNumArguments(); ++i)
      if (!isa<GlobalType>(kernel.getArgument(i).getType()))
        return false;
    return true;
  }

  std::string flatParam(unsigned i, unsigned field) {
    return "hk_" + argName(kernel, i).str() + kFlatFields[field].first;
  }

  void fail(Operation *op, const Twine &msg) {
    op->emitError(msg);
    bad = true;
  }

  LogicalResult findGrid() {
    auto mod = kernel->getParentOfType<ModuleOp>();
    if (mod)
      for (auto f : mod.getOps<func::FuncOp>())
        if (auto of = f->getAttrOfType<FlatSymbolRefAttr>(attrs::kGridOf))
          if (of.getValue() == kernelName())
            grid = f;
    if (!grid)
      return kernel.emitError("no grid function (hk.grid_of) for this kernel");
    return success();
  }

  /// Deterministic names, in program order: a base name the first time,
  /// base1, base2, ... after. The frontend's naming, so the textual IR and
  /// the C++ call a value the same thing.
  void assignNames() {
    llvm::StringMap<int> counts;
    // A frontend name that is a C++ keyword (the `&` of two flags is named
    // "and", an alternative token) gets a trailing underscore.
    static const llvm::StringSet<> keywords = {
        "and", "and_eq", "bitand", "bitor", "compl", "not", "not_eq", "or",
        "or_eq", "xor", "xor_eq", "int", "float", "double", "char", "bool",
        "long", "short", "signed", "unsigned", "void", "auto", "const", "if",
        "else", "for", "while", "do", "return", "switch", "case", "default",
        "break", "continue", "goto", "new", "delete", "this", "true", "false",
        "template", "typename", "class", "struct", "union", "enum", "static",
        "inline", "operator", "sizeof", "typedef", "namespace", "using",
        "register", "volatile", "extern", "asm", "try", "catch", "throw",
        "public", "private", "protected", "friend", "virtual", "mutable",
        "explicit", "export", "nullptr", "constexpr", "decltype", "noexcept",
        "concept", "requires", "alignas", "alignof", "static_assert",
        "thread_local", "co_await", "co_return", "co_yield", "consteval",
        "constinit", "char8_t", "char16_t", "char32_t", "wchar_t"};
    auto bind = [&](Value v, StringRef base) {
      std::string b = keywords.contains(base) ? (base + "_").str() : base.str();
      int n = counts[b]++;
      names[v] = n == 0 ? b : b + std::to_string(n);
    };
    kernel.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == kernel.getOperation())
        return;
      if (auto f = dyn_cast<scf::ForOp>(op)) {
        StringRef iv = "arg";
        if (auto a = f->getAttrOfType<StringAttr>(attrs::kIvName))
          iv = a.getValue();
        else if (auto n = valueNameOf(f.getInductionVar().getLoc());
                 !n.empty())
          iv = n;
        bind(f.getInductionVar(), iv);
        auto carried = f->getAttrOfType<ArrayAttr>("hk.carried_names");
        for (auto [i, a] : llvm::enumerate(f.getRegionIterArgs())) {
          StringRef n = valueNameOf(a.getLoc());
          if (carried && i < carried.size())
            n = cast<StringAttr>(carried[i]).getValue();
          bind(a, n.empty() ? "arg" : n);
        }
        return;
      }
      if (auto wl = dyn_cast<scf::WhileOp>(op)) {
        auto carried = wl->getAttrOfType<ArrayAttr>("hk.carried_names");
        for (auto [i, a] : llvm::enumerate(wl.getBeforeArguments())) {
          StringRef n = valueNameOf(a.getLoc());
          if (carried && i < carried.size())
            n = cast<StringAttr>(carried[i]).getValue();
          bind(a, n.empty() ? "arg" : n);
        }
        return;
      }
      if (auto i = dyn_cast<scf::IfOp>(op)) {
        auto rn = i->getAttrOfType<ArrayAttr>("hk.result_names");
        for (auto [k, r] : llvm::enumerate(i.getResults())) {
          StringRef n = rn && k < rn.size()
                            ? cast<StringAttr>(rn[k]).getValue()
                            : StringRef("v");
          bind(r, n);
        }
        return;
      }
      for (Value r : op->getResults()) {
        StringRef n = valueNameOf(op->getLoc());
        if (!n.empty())
          bind(r, n);
      }
    });
    // A loop's results are its iter_args' variables.
    kernel.walk([&](scf::ForOp f) {
      for (auto [r, a] : llvm::zip(f.getResults(), f.getRegionIterArgs()))
        names[r] = names[a];
    });
    // A while's condition forwards its arguments unchanged (emitWhile
    // refuses anything else), so the body's arguments and the loop's results
    // are the same locals as the test's.
    kernel.walk([&](scf::WhileOp wl) {
      for (auto [a, b] :
           llvm::zip(wl.getAfterArguments(), wl.getBeforeArguments()))
        names[a] = names[b];
      for (auto [r, b] : llvm::zip(wl.getResults(), wl.getBeforeArguments()))
        names[r] = names[b];
    });
  }

  std::string name(Value v) {
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (ba.getOwner() == &kernel.getBody().front())
        return "g." + argName(kernel, ba.getArgNumber()).str();
    }
    auto it = names.find(v);
    if (it != names.end())
      return it->second;
    return inlineExpr(v);
  }

  /// A value nobody named: a literal, or a width conversion of one that is.
  std::string inlineExpr(Value v) {
    Operation *def = v.getDefiningOp();
    if (!def) {
      bad = true;
      return "/*unnamed block argument*/";
    }
    if (auto c = dyn_cast<arith::ConstantOp>(def)) {
      if (auto i = dyn_cast<IntegerAttr>(c.getValue())) {
        if (i.getType().isInteger(1))
          return i.getValue().getBoolValue() ? "1" : "0";
        return std::to_string(i.getInt());
      }
      if (auto f = dyn_cast<FloatAttr>(c.getValue()))
        return floatLit(f.getValueAsDouble());
    }
    if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp,
            arith::IndexCastOp>(def))
      return name(def->getOperand(0));
    if (auto e = dyn_cast<ExtentOp>(def))
      return name(e.getTensor()) + "." +
             stringifyExtent(e.getAxis()).str() + "()";
    if (auto s = scalarBinary(def))
      return "(" + name(def->getOperand(0)) + " " + *s + " " +
             name(def->getOperand(1)) + ")";
    fail(def, "value has no name and is not a literal");
    return "/*?*/";
  }

  /// The C++ operator of a host-expression arith op, for inline use.
  std::optional<std::string> scalarBinary(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<std::string>>(op)
        .Case([](arith::AddIOp) { return "+"; })
        .Case([](arith::SubIOp) { return "-"; })
        .Case([](arith::MulIOp) { return "*"; })
        .Case([](arith::DivSIOp) { return "/"; })
        .Case([](arith::RemSIOp) { return "%"; })
        .Case([](arith::AndIOp) { return "&"; })
        .Case([](arith::OrIOp) { return "|"; })
        .Case([](arith::XOrIOp) { return "^"; })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::string cpp(Type t) {
    auto it = aliases.find(typeKey(t));
    if (it != aliases.end())
      return it->second.second;
    return tileCpp(t);
  }

  // -- top level -------------------------------------------------------

  void header(ArrayRef<std::string> includes) {
    indent = 0;
    w("// Generated by hk (HipKittens Python IR). Do not edit by hand;");
    w("// edit the kernel that produced it and regenerate.");
    w("//");
    w("// kernel : " + kernelName().str());
    w("// arch   : " + tgtArch() + "  warps=" +
      std::to_string(intAttr(attrs::kWarps, 1)) +
      " threads=" + std::to_string(threads()));
    if (auto consts = kernel->getAttrOfType<DictionaryAttr>(attrs::kConsts);
        consts && !consts.empty()) {
      std::vector<std::string> kv;
      for (NamedAttribute na : consts)
        kv.push_back(na.getName().str() + "=" + pyValue(na.getValue()));
      w("// const  : " + join(kv));
    }
    w("// LDS    : " + std::to_string(ldsBytes()) + " B");
    if (auto warns = kernel->getAttrOfType<ArrayAttr>(attrs::kWarnings))
      for (Attribute a : warns) {
        StringRef s = cast<StringAttr>(a).getValue();
        SmallVector<StringRef> parts;
        s.split(parts, '\n');
        for (StringRef p : parts)
          w("// WARN   : " + p.trim().str());
      }
    w();
    for (auto &inc : includes)
      w("#include " + inc);
    w();
    w("using namespace kittens;");
    w();
  }

  std::string tgtArch() {
    auto arch = kernel->getAttrOfType<StringAttr>(attrs::kArch);
    return arch ? arch.getValue().str() : "gfx1100";
  }

  LogicalResult globalsStruct() {
    w("struct globals {");
    indent = 1;
    for (unsigned i = 0; i < kernel.getNumArguments(); ++i) {
      Type t = kernel.getArgument(i).getType();
      // include/cdna{3,4}'s gl has no row stride; hk/cdna.cuh's strided_gl
      // is that gl with the one the launchers pass.
      std::string ty = rawCpp(t);
      if (auto g = dyn_cast<GlobalType>(t); g && tgt->wave == 64)
        ty = "kittens::strided_gl<" + dtypeCpp(g.getElementType()).str() + ">";
      w(ty + " " + argName(kernel, i).str() + ";");
    }
    w();
    w("dim3 grid() const {");
    indent = 2;
    w("const globals &g = *this;");
    auto ret = cast<func::ReturnOp>(grid.getBody().front().getTerminator());
    if (ret.getNumOperands() != 3)
      return grid.emitError("a grid is three dimensions");
    std::vector<std::string> dims;
    for (Value v : ret.getOperands())
      dims.push_back(gridExpr(v));
    w("return dim3(" + join(dims) + ");");
    indent = 1;
    w("}");
    w("dim3 block() const { return dim3(" + std::to_string(threads()) + "); }");
    if (ldsBytes())
      w("size_t dynamic_shared_memory() const { return " +
        std::to_string(ldsBytes()) + "; }");
    indent = 0;
    w("};");
    w();
    return success();
  }

  std::string gridExpr(Value v) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return "g." + argName(grid, ba.getArgNumber()).str();
    Operation *def = v.getDefiningOp();
    if (auto c = dyn_cast<arith::ConstantOp>(def))
      return std::to_string(cast<IntegerAttr>(c.getValue()).getInt());
    if (auto e = dyn_cast<ExtentOp>(def))
      return gridExpr(e.getTensor()) + "." +
             stringifyExtent(e.getAxis()).str() + "()";
    if (auto s = scalarBinary(def))
      return "(" + gridExpr(def->getOperand(0)) + " " + *s + " " +
             gridExpr(def->getOperand(1)) + ")";
    fail(def, "a grid dimension combines extents, constants and + - * / %");
    return "0";
  }

  void collectAliases() {
    auto note = [&](Type t, auto &self) -> void {
      t = plainOf(t);
      if (auto v = dyn_cast<RegVecType>(t))
        self(v.getTile(), self);
      if (!isa<RegTileType, SharedTileType, RegVecType>(t))
        return;
      std::string key = typeKey(t);
      if (!aliases.count(key))
        aliases[key] = {t, aliasName(t)};
    };
    kernel.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == kernel.getOperation())
        return;
      for (Value v : op->getResults())
        note(v.getType(), note);
      for (Value v : op->getOperands())
        note(v.getType(), note);
      for (Region &r : op->getRegions())
        for (Block &b : r)
          for (BlockArgument a : b.getArguments())
            note(a.getType(), note);
    });
    if (aliases.empty())
      return;
    for (auto &kv : aliases) {
      Type t = kv.second.first;
      if (auto v = dyn_cast<RegVecType>(t))
        w("using " + kv.second.second + " = " +
          aliases[typeKey(v.getTile())].second + "::" + v.getKind().str() +
          "_vec;");
      else
        w("using " + kv.second.second + " = " + tileCpp(t) + ";");
    }
    w();
  }

  LogicalResult kernelBody() {
    collectAliases();
    if (flatArgs()) {
      unsigned n = kernel.getNumArguments();
      std::vector<std::string> params, fields, gls;
      for (unsigned f = 0; f < std::size(kFlatFields); ++f)
        for (unsigned i = 0; i < n; ++i) {
          std::string ty = "int ";
          if (!f)
            ty = dtypeCpp(cast<GlobalType>(kernel.getArgument(i).getType())
                              .getElementType())
                     .str() +
                 " *";
          params.push_back(ty + (f ? "" : "__restrict__ ") + flatParam(i, f));
          fields.push_back(ty + flatParam(i, f) + ";");
        }
      for (unsigned i = 0; i < n; ++i)
        gls.push_back("decltype(globals::" + argName(kernel, i).str() + ")(" +
                      flatParam(i, 0) + ", " + flatParam(i, 4) + ", " +
                      flatParam(i, 5) + ", " + flatParam(i, 2) + ", " +
                      flatParam(i, 3) + ", " + flatParam(i, 1) + ")");
      // The parameters as one struct, for a launcher that packs them by
      // hand (scaffold.device's layout table).
      w("struct hk_kernarg {");
      indent = 1;
      for (auto &f : fields)
        w(f);
      indent = 0;
      w("};");
      w();
      w("namespace hk_flat {");
      w("__global__ __launch_bounds__(" + std::to_string(threads()) + ", 1)");
      w("void " + kernelName().str() + "_kernel(" + join(params) + ") {");
      indent = 1;
      w("const globals g{" + join(gls) + "};");
    } else {
      w("__global__ __launch_bounds__(" + std::to_string(threads()) + ", 1)");
      w("void " + kernelName().str() + "_kernel(const globals g) {");
      indent = 1;
    }
    // RDNA: the arguments' scalar-cache lines all at once, rather than a cold
    // miss wherever LLVM sinks a late field's load (include/rdna3, util.cuh).
    if (tgt->wave == 32)
      w("kittens::touch_kernargs<sizeof(globals)>();");
    if (ldsBytes()) {
      w("extern __shared__ alignment_dummy __shm[];");
      w("shared_allocator al((int*)&__shm[0]);");
      w();
    }
    emitBlock(kernel.getBody().front());
    indent = 0;
    w("}");
    if (flatArgs())
      w("} // namespace hk_flat");
    w();
    return success();
  }

  void launch() {
    std::string n = kernelName().str();
    std::string args = "g";
    if (flatArgs()) {
      n = "hk_flat::" + n;
      std::vector<std::string> a;
      for (auto &[suffix, field] : kFlatFields)
        for (unsigned i = 0; i < kernel.getNumArguments(); ++i)
          a.push_back("g." + argName(kernel, i).str() + "." + field);
      args = join(a);
    }
    w("static void launch_on(const globals &g, hipStream_t stream) {");
    indent = 1;
    if (ldsBytes()) {
      std::string lds = std::to_string(ldsBytes());
      w("hipFuncSetAttribute((void *)&" + n +
        "_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, " + lds + ");");
      w(n + "_kernel<<<g.grid(), g.block(), " + lds + ", stream>>>(" + args + ");");
    } else {
      w(n + "_kernel<<<g.grid(), g.block(), 0, stream>>>(" + args + ");");
    }
    indent = 0;
    w("}");
    w();
    w("static void launch(const globals &g) { launch_on(g, 0); }");
    w();
  }

  // -- ops ----------------------------------------------------------------

  void emitBlock(Block &b) {
    for (Operation &op : b)
      emitOp(&op);
  }

  bool named(Value v) { return names.count(v); }

  /// The op's destination: `dst` when it writes one, else its result,
  /// declared here. A fresh declaration of a loop-carried accumulator would
  /// shadow it and keep only the last iteration.
  Value dest(Value dst, Value result) {
    if (dst)
      return dst;
    w(cpp(result.getType()) + " " + name(result) + ";");
    return result;
  }

  void call(StringRef fn, Value dst, ArrayRef<std::string> args) {
    std::vector<std::string> all{name(dst)};
    all.insert(all.end(), args.begin(), args.end());
    w("kittens::" + fn.str() + "(" + join(all) + ");");
  }

  std::string coordArg(Value idx, const std::string &tileCpp) {
    auto ct = cast<CoordType>(idx.getType());
    if (!ct.getElement())
      return "kittens::coord<" + tileCpp + ">(" + name(idx) + ")";
    return name(idx);
  }

  std::string svEntry(Value sv, Value index) {
    if (cast<SharedVecType>(sv.getType()).getCount() == 1)
      return name(sv);
    return name(sv) + "[" + name(index) + "]";
  }

  std::string elemCpp(Type t) {
    if (auto r = dyn_cast<RegTileType>(t))
      return dtypeCpp(r.getElementType()).str();
    if (auto v = dyn_cast<RegVecType>(t))
      return dtypeCpp(v.getElementType()).str();
    return "float";
  }

  void scalarStmt(Operation *op, const std::string &expr) {
    w("const int " + name(op->getResult(0)) + " = " + expr + ";");
  }

  void emitArith(Operation *op) {
    if (op->getNumResults() == 1 && !named(op->getResult(0)))
      return; // written inline where it is used
    auto a = [&](unsigned i) { return name(op->getOperand(i)); };
    llvm::TypeSwitch<Operation *>(op)
        .Case([&](arith::ConstantOp c) {
          std::string v;
          if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
            v = std::to_string(i.getInt());
          else if (auto f = dyn_cast<FloatAttr>(c.getValue()))
            v = pyFloat(f.getValueAsDouble());
          w("const int " + name(c.getResult()) + " = " + v + ";");
        })
        .Case([&](arith::AddIOp) { scalarStmt(op, a(0) + " + " + a(1)); })
        .Case([&](arith::SubIOp) { scalarStmt(op, a(0) + " - " + a(1)); })
        .Case([&](arith::MulIOp) { scalarStmt(op, a(0) + " * " + a(1)); })
        .Case([&](arith::DivSIOp) { scalarStmt(op, a(0) + " / " + a(1)); })
        .Case([&](arith::RemSIOp) { scalarStmt(op, a(0) + " % " + a(1)); })
        .Case([&](arith::AndIOp) { scalarStmt(op, a(0) + " & " + a(1)); })
        .Case([&](arith::OrIOp) { scalarStmt(op, a(0) + " | " + a(1)); })
        .Case([&](arith::XOrIOp) { scalarStmt(op, a(0) + " ^ " + a(1)); })
        .Case([&](arith::MinSIOp) {
          scalarStmt(op, "(" + a(0) + " < " + a(1) + ") ? " + a(0) + " : " +
                             a(1));
        })
        .Case([&](arith::MaxSIOp) {
          scalarStmt(op, "(" + a(0) + " > " + a(1) + ") ? " + a(0) + " : " +
                             a(1));
        })
        .Case([&](arith::CeilDivSIOp) {
          scalarStmt(op, "(" + a(0) + " + " + a(1) + " - 1) / " + a(1));
        })
        .Case([&](arith::CmpIOp c) {
          static const char *sym[] = {"==", "!=", "<", "<=", ">", ">=",
                                      "<",  "<=", ">", ">="};
          scalarStmt(op, a(0) + " " + sym[int(c.getPredicate())] + " " + a(1));
        })
        .Case<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp,
              arith::IndexCastOp>(
            [&](Operation *) { scalarStmt(op, a(0)); })
        .Default([&](Operation *) {
          fail(op, "no C++ for this arith op: " +
                       op->getName().getStringRef());
        });
  }

  /// The C++ type of a mutable local: a scalar is an int, whatever its width.
  std::string varCpp(Type t) {
    if (t.isInteger())
      return "int";
    return cpp(t);
  }

  /// `var = value;` for each yielded value that is not already the variable.
  void assignYield(Block &b, ValueRange vars) {
    auto y = dyn_cast<scf::YieldOp>(b.getTerminator());
    if (!y)
      return;
    for (auto [var, val] : llvm::zip(vars, y.getOperands()))
      if (name(var) != name(val))
        w(name(var) + " = " + name(val) + ";");
  }

  void emitFor(scf::ForOp f) {
    // A loop-carried value is a mutable local declared in front of the loop;
    // the body's block argument and the loop's result are the same local.
    for (auto [arg, init] : llvm::zip(f.getRegionIterArgs(), f.getInitArgs()))
      w(varCpp(arg.getType()) + " " + name(arg) + " = " + name(init) + ";");
    std::string n = name(f.getInductionVar());
    int64_t unroll = 1;
    if (auto u = f->getAttrOfType<IntegerAttr>(attrs::kUnroll))
      unroll = u.getInt();
    if (unroll > 1)
      w("#pragma unroll " + std::to_string(unroll));
    auto isConst = [](Value v, int64_t want) {
      if (auto c = v.getDefiningOp<arith::ConstantOp>())
        if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
          return i.getInt() == want;
      return false;
    };
    std::string lb = name(f.getLowerBound()), ub = name(f.getUpperBound()),
                step = name(f.getStep());
    if (isConst(f.getStep(), 1))
      w("for (int " + n + " = " + lb + "; " + n + " < " + ub + "; ++" + n +
        ") {");
    else
      w("for (int " + n + " = " + lb + "; " + n + " < " + ub + "; " + n +
        " += " + step + ") {");
    indent++;
    emitBlock(*f.getBody());
    assignYield(*f.getBody(), f.getRegionIterArgs());
    indent--;
    w("}");
  }

  void emitWhile(scf::WhileOp wl) {
    Block &before = wl.getBefore().front(), &after = wl.getAfter().front();
    auto cond = cast<scf::ConditionOp>(before.getTerminator());
    for (auto [fwd, arg] : llvm::zip(cond.getArgs(), before.getArguments()))
      if (fwd != arg)
        return fail(wl, "a while loop's condition has to forward the loop's "
                        "carried values unchanged");
    for (auto [arg, init] : llvm::zip(before.getArguments(), wl.getInits()))
      w(varCpp(arg.getType()) + " " + name(arg) + " = " + name(init) + ";");
    w("while (true) {");
    indent++;
    emitBlock(before);
    w("if (!(" + name(cond.getCondition()) + ")) break;");
    emitBlock(after);
    assignYield(after, before.getArguments());
    indent--;
    w("}");
  }

  void emitIf(scf::IfOp i) {
    // A branch result is a local declared in front of the branch and
    // assigned at the end of each arm.
    for (Value r : i.getResults())
      w(varCpp(r.getType()) + " " + name(r) + ";");
    std::string c = name(i.getCondition());
    if (i->hasAttr(attrs::kOpaque)) {
      // The condition is `const int` everywhere else, and an "+v" operand has
      // to be writable: hand the optimizer a copy it cannot see through.
      ++opaque;
      std::string live = c + "_live" + std::to_string(opaque);
      w("int " + live + " = " + c + ";");
      w("asm volatile(\"\" : \"+v\"(" + live + "));");
      c = live;
    }
    w("if (" + c + ") {");
    indent++;
    emitBlock(i.getThenRegion().front());
    assignYield(i.getThenRegion().front(), i.getResults());
    indent--;
    if (!i.getElseRegion().empty() &&
        (!i.getElseRegion().front().without_terminator().empty() ||
         i.getNumResults())) {
      w("} else {");
      indent++;
      emitBlock(i.getElseRegion().front());
      assignYield(i.getElseRegion().front(), i.getResults());
      indent--;
    }
    w("}");
  }

  template <typename OpT>
  std::vector<std::string> names_(OpT range) {
    std::vector<std::string> out;
    for (Value v : range)
      out.push_back(name(v));
    return out;
  }

  void emitOp(Operation *op) {
    if (isa<arith::ArithDialect>(op->getDialect()))
      return emitArith(op);
    llvm::TypeSwitch<Operation *>(op)
        .Case([&](scf::ForOp f) { emitFor(f); })
        .Case([&](scf::IfOp i) { emitIf(i); })
        .Case([&](scf::WhileOp wl) { emitWhile(wl); })
        .Case<scf::YieldOp, scf::ConditionOp, func::ReturnOp>(
            [](Operation *) {})
        // indices
        .Case([&](BlockIdxOp o) {
          w("const int " + name(o.getResult()) + " = blockIdx." +
            stringifyAxis(o.getAxis()).str() + ";");
        })
        .Case([&](WarpIdOp o) {
          // Uniform: the IR hoists this into the prologue, where every lane
          // is active, so the readfirstlane is always well defined.
          w("const int " + name(o.getResult()) +
            " = kittens::warpid_uniform();");
        })
        .Case([&](XcdIdOp o) {
          w("const int " + name(o.getResult()) + " = kittens::xcd_id();");
        })
        .Case([&](LaneIdOp o) {
          w("const int " + name(o.getResult()) + " = kittens::laneid();");
        })
        .Case([&](CoordOp o) {
          std::string parts = join(names_(o->getOperands()));
          if (!o.getResult().getType().getElement())
            w("const int4 " + name(o.getResult()) + " = make_int4(" + parts +
              ");");
          else
            w("const kittens::coord<> " + name(o.getResult()) + "(" + parts +
              ");");
        })
        .Case([&](ExtentOp o) {
          if (!named(o.getResult()))
            return;
          w("const int " + name(o.getResult()) + " = " + name(o.getTensor()) +
            "." + stringifyExtent(o.getAxis()).str() + "();");
        })
        .Case([&](AnyNeOp o) {
          std::string a = name(o.getLhs()), b = name(o.getRhs()),
                      r = name(o.getResult());
          w("bool " + r + "_v = false;");
          w("#pragma unroll");
          w("for (int o = 0; o < decltype(" + a + ")::outer_dim; o++)");
          w("    #pragma unroll");
          w("    for (int i = 0; i < decltype(" + a + ")::inner_dim; i++)");
          w("        " + r + "_v |= (" + a + ".data[o][i] != " + b +
            ".data[o][i]);");
          w("const int " + r + " = __any(" + r + "_v);");
        })
        .Case([&](AnyAboveOp o) {
          std::string a = name(o.getLhs()), b = name(o.getRhs()),
                      r = name(o.getResult());
          std::string t =
              "(float)" + floatLit(o.getThreshold().convertToDouble());
          w("bool " + r + "_v = false;");
          w("#pragma unroll");
          w("for (int o = 0; o < decltype(" + a + ")::outer_dim; o++)");
          w("    #pragma unroll");
          w("    for (int i = 0; i < decltype(" + a + ")::inner_dim; i++)");
          w("        " + r + "_v |= (" + a + ".data[o][i] - " + b +
            ".data[o][i] > " + t + ");");
          w("const int " + r + " = __any(" + r + "_v);");
        })
        // global memory
        .Case([&](LoadOp o) {
          Value d = dest(o.getDst(), o.getResult());
          std::string fn = o.getBounded() ? "load_bounded" : "load";
          if (auto p = o.getPeriod())
            fn = "load_period<" + std::to_string(*p) + ">";
          if (o.getAux())
            fn = "load_scoped<" + std::to_string(o.getAux()) + ">";
          w("kittens::" + fn + "(" + name(d) + ", " + name(o.getSrc()) + ", " +
            coordArg(o.getIdx(), cpp(d.getType())) + ");");
        })
        .Case([&](StoreOp o) {
          std::string fn = o.getBounded() ? "store_bounded" : "store";
          if (o.getAux())
            fn = "store_scoped<" + std::to_string(o.getAux()) + ">";
          w("kittens::" + fn + "(" + name(o.getDst()) + ", " +
            name(o.getValue()) + ", " +
            coordArg(o.getIdx(), cpp(o.getValue().getType())) + ");");
        })
        .Case([&](LoadRowsOp o) {
          Value d = dest(o.getDst(), o.getResult());
          w("kittens::load_bounded(" + name(d) + ", " + name(o.getSrc()) +
            ", " + coordArg(o.getIdx(), cpp(d.getType())) + ", " +
            name(o.getRows()) + ");");
        })
        .Case([&](StoreRowsOp o) {
          w("kittens::store_bounded(" + name(o.getDst()) + ", " +
            name(o.getValue()) + ", " +
            coordArg(o.getIdx(), cpp(o.getValue().getType())) + ", " +
            name(o.getRows()) + ");");
        })
        .Case([&](StoreScalarOp o) {
          w("kittens::store_scalar(" + name(o.getDst()) + ", " +
            name(o.getValue()) + ", " + name(o.getIdx()) + ");");
        })
        .Case([&](LoadScalarOp o) {
          // include/rdna3 has one load_scalar; the scalar-cache read is
          // hk's shim's, on wave64 targets.
          std::string fn = o.getInvariant() && tgt->wave == 64
                               ? "load_scalar_invariant"
                               : "load_scalar";
          w("const int " + name(o.getResult()) + " = kittens::" + fn + "(" +
            name(o.getSrc()) + ", " + name(o.getIdx()) + ");");
        })
        .Case([&](WgArriveOp o) {
          w("const int " + name(o.getResult()) + " = kittens::wg_arrive" +
            (o.getXcd() ? "_xcd(" : "(") +
            name(o.getCounter()) + ", " + name(o.getIdx()) + ", " +
            name(o.getExpected()) + ");");
        })
        .Case([&](GatherRowsOp o) {
          Value d = dest(o.getDst(), o.getResult());
          w("kittens::load_gather(" + name(d) + ", " + name(o.getSrc()) +
            ", " + name(o.getIdx()) + ", " + name(o.getAt()) + ", " +
            name(o.getCount()) + ", " + name(o.getCol()) + ", " +
            name(o.getDiv()) + ");");
        })
        .Case([&](LoadMxScalesOp o) {
          Value d = dest(o.getDst(), o.getResult());
          if (o.getLayout())
            w("kittens::load_mx_scales_aiter<" + std::to_string(o.getLayout()) + ", " +
              std::to_string(o.getGroupRows()) + ">(" + name(d) + ", " + name(o.getSrc()) +
              ", " + name(o.getRow0()) + ", " + name(o.getKb0()) + ");");
          else
            call("load_mx_scales", d,
                 {name(o.getSrc()), name(o.getRow0()), name(o.getKb0())});
        })
        .Case([&](MxQuantStoreOp o) {
          w("kittens::mx_quant_store(" + name(o.getQ()) + ", " + name(o.getS()) + ", " +
            name(o.getSrc()) + ", " + name(o.getRow0()) + ", " + name(o.getCol0()) + ");");
        })
        .Case([&](LoadFp4FragsLdsOp o) {
          Value d = dest(o.getDst(), o.getResult());
          w("kittens::load_fp4_frags_lds(" + name(d) + ", " + name(o.getSrc()) + ", " +
            name(o.getRow0()) + ", " + name(o.getChunk0()) + ");");
        })
        .Case([&](LoadFp4FragsOp o) {
          Value d = dest(o.getDst(), o.getResult());
          w("kittens::load_fp4_frags<" + std::to_string(o.getGateUp()) + ", " +
            std::to_string(o.getGroupRows()) + ">(" + name(d) + ", " + name(o.getSrc()) +
            ", " + name(o.getRow0()) + ", " + name(o.getKc0()) + ");");
        })
        .Case([&](MxScalesCopyOp o) {
          w(name(o.getDst()) + " = " + name(o.getSrc()) + ";");
        })
        .Case([&](MmaABtMxOp o) {
          Value d = dest(o.getDst(), o.getResult());
          std::string offs;
          if (o.getSaTile() || o.getSaStep() || o.getSbTile() || o.getSbStep())
            offs = ", " + std::to_string(o.getSaTile()) + ", " + std::to_string(o.getSaStep()) +
                   ", " + std::to_string(o.getSbTile()) + ", " + std::to_string(o.getSbStep());
          w("kittens::mma_ABt_mx<" + std::to_string(o.getAFmt()) + ", " +
            std::to_string(o.getBFmt()) + offs + ">(" + name(d) + ", " +
            name(o.getA()) + ", " + name(o.getB()) + ", " + name(o.getC()) + ", " +
            name(o.getSa()) + ", " + name(o.getSb()) + ");");
        })
        .Case([&](LoadW4Op o) {
          Value d = dest(o.getDst(), o.getResult());
          w("kittens::load_w4<" + std::to_string(o.getGroup()) + ">(" + name(d) +
            ", " + name(o.getQw()) + ", " + name(o.getS()) + ", " +
            name(o.getZs()) + ", " + name(o.getN0()) + ", " + name(o.getK0()) +
            ");");
        })
        .Case([&](GatherPrepOp o) {
          w("kittens::gather_prep<" + std::to_string(o.getRows()) + ">(" + name(o.getIds()) +
            ", " + name(o.getAt()) + ");");
        })
        .Case([&](ScatterRowsOp o) {
          w("kittens::store_scatter" +
            (o.getSortTopk() || o.getAdd()
                 ? "<" + std::to_string(o.getSortTopk()) + (o.getAdd() ? ", 1>" : ">")
                 : std::string()) +
            "(" + name(o.getDst()) + ", " +
            name(o.getValue()) + ", " + name(o.getIdx()) + ", " +
            name(o.getAt()) + ", " + name(o.getCount()) + ", " +
            name(o.getCol()) + ");");
        })
        .Case([&](ScatterAddLdsOp o) {
          w("kittens::scatter_add_lds<" + std::to_string(o.getSortTopk()) + ">(" +
            name(o.getDst()) + ", " + name(o.getSrc()) + ", " + name(o.getIdx()) + ", " +
            name(o.getAt()) + ", " + name(o.getCount()) + ", " + name(o.getCol()) + ");");
        })
        .Case([&](AtomicAddOp o) {
          w("kittens::atomic_add(" + name(o.getDst()) + ", " +
            name(o.getValue()) + ", " +
            coordArg(o.getIdx(), cpp(o.getValue().getType())) + ");");
        })
        .Case([&](SignalOp o) {
          w("kittens::signal<" + std::to_string(o.getAux()) + ">(" + name(o.getDst()) + ", " +
            name(o.getIdx()) + ", " + name(o.getValue()) + ");");
        })
        .Case([&](ArriveOp o) {
          w("const int " + name(o.getResult()) + " = kittens::" +
            std::string(o.getNodrain() ? "claim" : "arrive") + "<" +
            std::to_string(o.getAux()) + ">(" + name(o.getDst()) + ", " + name(o.getIdx()) +
            ", " + name(o.getValue()) + ");");
        })
        .Case([&](WaitOp o) {
          std::string err = "nullptr";
          if (o.getErr())
            err = "(int *)&" + name(o.getErr()) + "[(" + name(o.getErrIdx()) +
                  ").template unit_coord<-1, 3>()]";
          w("const int " + name(o.getResult()) + " = kittens::wait<" +
            std::to_string(o.getAux()) + ", " + std::string(o.getCmp() == "eq" ? "1" : "0") +
            ">(" + name(o.getSrc()) + ", " + name(o.getIdx()) + ", " + name(o.getValue()) +
            ", " + std::to_string(o.getTimeout()) + "LL, " + err + ");");
        })
        .Case([&](PostOp o) {
          w("kittens::post<" + std::to_string(o.getAux()) + ">(" + name(o.getDst()) + ", " +
            name(o.getIdx()) + ", " + name(o.getValue()) + ", " + name(o.getTag()) + ");");
        })
        .Case([&](TakeOp o) {
          std::string err = "nullptr";
          if (o.getErr())
            err = "(int *)&" + name(o.getErr()) + "[(" + name(o.getErrIdx()) +
                  ").template unit_coord<-1, 3>()]";
          w("const int " + name(o.getResult()) + " = kittens::take<" +
            std::to_string(o.getAux()) + ">(" + name(o.getSrc()) + ", " + name(o.getIdx()) + ", " +
            name(o.getTag()) + ", " + std::to_string(o.getTimeout()) + "LL, " + err + ");");
        })
        .Case([&](AtomicScalarOp o) {
          if (o.getAux()) {
            w("const int " + name(o.getResult()) + " = kittens::atomic_scalar_scoped<" +
              std::to_string(o.getAux()) + ">(" + name(o.getDst()) + ", " + name(o.getIdx()) +
              ", " + name(o.getValue()) + ", " +
              atomicKindCode(o.getKind()) + ");");
            return;
          }
          w("const int " + name(o.getResult()) + " = kittens::atomic_scalar(" +
            name(o.getDst()) + ", " + name(o.getIdx()) + ", " +
            name(o.getValue()) + ", " +
            atomicKindCode(o.getKind()) + ");");
        })
        .Case([&](PhiloxOp o) {
          Value d = dest(o.getDst(), o.getResult());
          w("kittens::philox_uniform(" + name(d) + ", (uint32_t)" +
            name(o.getSeed()) + ", 0u, " + name(o.getOffset()) + ", " +
            name(o.getRowStride()) + ");");
        })
        // LDS
        .Case([&](AllocSharedOp o) {
          auto t = o.getResult().getType();
          std::string el = allocCpp(o.getResult());
          if (t.getCount() == 1)
            w(el + " &" + name(o.getResult()) + " = al.allocate<" + el +
              ">();");
          else
            w(el + " (&" + name(o.getResult()) + ")[" +
              std::to_string(t.getCount()) + "] = al.allocate<" + el + ", " +
              std::to_string(t.getCount()) + ">();");
        })
        .Case([&](AllocSharedVecOp o) {
          auto t = o.getResult().getType();
          std::string el = rawCpp(t);
          if (t.getCount() == 1)
            w(el + " &" + name(o.getResult()) + " = al.allocate<" + el +
              ">();");
          else
            w(el + " (&" + name(o.getResult()) + ")[" +
              std::to_string(t.getCount()) + "] = al.allocate<" + el + ", " +
              std::to_string(t.getCount()) + ">();");
        })
        .Case([&](SharedAtOp o) {
          // A buffer of an allocation spelled with its own shape has that
          // allocation's element type, not the alias.
          std::string t = conflicted.count(rawCpp(o.getResult().getType()))
                              ? "auto"
                              : cpp(o.getResult().getType());
          w(t + " &" + name(o.getResult()) + " = " + name(o.getBuf()) + "[" +
            name(o.getIndex()) + "];");
        })
        .Case([&](SubtileOp o) {
          auto t = o.getResult().getType();
          w("auto " + name(o.getResult()) + " = kittens::subtile_inplace<" +
            std::to_string(t.getRows()) + ", " + std::to_string(t.getCols()) +
            ">(" + name(o.getSrc()) + ", {" + name(o.getRow()) + ", " +
            name(o.getCol()) + "});");
        })
        .Case([&](LoadSharedOp o) {
          Value d = dest(o.getDst(), o.getResult());
          if (tgt->wave == 64) {
            // include/cdna{3,4}'s load(rt, st) issues its ds_reads through
            // inline-asm outputs and never waits, so `wait` is ours to keep:
            // the wait, then the bind that orders the tile's readers after it.
            w("kittens::load(" + name(d) + ", " + name(o.getSrc()) + ");");
            if (o.getWait())
              w("kittens::lds_wait_for<0>(" + name(d) + ");");
            return;
          }
          std::string t = o.getWait() ? "" : "<false>";
          w("kittens::load" + t + "(" + name(d) + ", " + name(o.getSrc()) +
            ");");
        })
        .Case([&](StoreSharedOp o) {
          w("kittens::store(" + name(o.getDst()) + ", " + name(o.getValue()) +
            ");");
          // hk-lds-pipeline takes the vectorised store (a row-layout 16-bit
          // tile into LDS of its type) to end in lgkmcnt(0), as
          // include/rdna3's does. include/cdna4's issues inline-asm ds_writes
          // and returns, so a barrier after it would let another warp read
          // the LDS before they land.
          auto v = cast<RegTileType>(o.getValue().getType());
          if (tgt->wave == 64 && !v.getColLayout() &&
              dtypeBits(v.getElementType()) == 16 &&
              v.getElementType() == o.getDst().getType().getElementType())
            w("kittens::lds_wait<0>();");
        })
        .Case([&](GranulesOp o) {
          std::string r = name(o.getResult());
          w(cpp(o.getResult().getType()) + " " + r + ";");
          w(r + ".init(" + name(o.getSrc()) + ");");
        })
        .Case([&](LoadFragOp o) {
          Value d = dest(o.getDst(), o.getResult());
          w("kittens::lds_read_frag<" + std::to_string(16 * o.getRow()) + ", " +
            std::to_string(16 * o.getCol()) + ">(" + name(d) + ", " +
            name(o.getGranules()) + ");");
        })
        .Case([&](StoreFragOp o) {
          auto gt = o.getGranules().getType();
          std::string off, tmpl;
          if (!o.getRow()) {
            off = "(uint32_t)(" + std::to_string(16 * gt.getSwizzleBytes()) +
                  " * " + name(o.getRowIndex()) + ")";
            tmpl = "<0, " + std::to_string(16 * o.getCol()) + ">";
          } else {
            off = "0u";
            tmpl = "<" + std::to_string(16 * *o.getRow()) + ", " +
                   std::to_string(16 * o.getCol()) + ">";
          }
          w("kittens::lds_write_frag" + tmpl + "(" + name(o.getGranules()) +
            ", " + off + ", " + name(o.getValue()) + ");");
        })
        .Case([&](LdsWaitForOp o) {
          if (!o.getN()) {
            fail(o, "lds_wait_for has no count: run hk-lds-pipeline first");
            return;
          }
          // A wave cannot have more LDS ops outstanding than the counter
          // holds (issue stalls at the cap), so a deeper wait is the cap.
          int64_t n = std::min<int64_t>(*o.getN(), tgt->lgkmcntMax);
          w("kittens::lds_wait_for<" + std::to_string(n) + ">(" +
            join(names_(o.getTiles())) + ");");
        })
        .Case([&](StoreSharedVecOp o) {
          w("kittens::store(" + svEntry(o.getDst(), o.getIndex()) + ", " +
            name(o.getValue()) + ");");
        })
        .Case([&](LoadSharedVecOp o) {
          std::string r = name(o.getResult());
          w(cpp(o.getResult().getType()) + " " + r + ";");
          w("kittens::load(" + r + ", " + svEntry(o.getSrc(), o.getIndex()) +
            ");");
        })
        // staging
        .Case([&](StageBufferOp o) {
          auto t = o.getResult().getType();
          std::string dim =
              t.getDepth() > 1 ? "[" + std::to_string(t.getDepth()) + "]" : "";
          w("float4 " + name(o.getResult()) + dim + "[" +
            std::to_string(t.getCalls()) + "];");
        })
        .Case([&](StageLoadOp o) {
          auto [g, expr, tmpl] = stageArgs(o.getBuf(), o.getSlot());
          auto t = o.getBuf().getType();
          w(g + "load_global_to_register_buffer(" + expr + ", " +
            std::to_string(t.getCalls()) + ", " + name(o.getSrc()) + ", " +
            coordArg(o.getIdx(), tileCpp(t.getTile())) + ", " + tmpl + ");");
        })
        .Case([&](StageCommitOp o) {
          auto [g, expr, tmpl] = stageArgs(o.getBuf(), o.getSlot());
          (void)tmpl;
          w(g + "store_register_buffer_to_shared<" +
            (o.getWait() ? "true" : "false") + ">(" + name(o.getDst()) + ", " +
            expr + ", " + std::to_string(o.getBuf().getType().getCalls()) +
            ");");
        })
        .Case([&](VmWaitOp o) {
          w("kittens::vm_wait<" + std::to_string(o.getKeep()) + ">();");
        })
        .Case([&](GroupLoadOp o) {
          if (o.getAsync() && o.getGatherIds()) {
            w("kittens::group_load_async_gather<" + std::to_string(o.getThreads() / tgt->wave) +
              ", " + std::to_string(o.getGatherStaged()) +
              ">(" + name(o.getDst()) + ", " + name(o.getSrc()) + ", " +
              coordArg(o.getIdx(), tileCpp(o.getDst().getType())) + ", " +
              name(o.getGatherIds()) + ", " + name(o.getGatherAt()) + ");");
            return;
          }
          if (o.getAsync()) {
            // hk/cdna.cuh: gfx950's DMA, left in flight on vmcnt until the
            // hk.group_wait_for hk-lds-pipeline counted.
            w("kittens::group_load_async<" +
              std::to_string(o.getThreads() / tgt->wave) +
              (o.getNontemporal() ? ", true" : "") + ">(" +
              name(o.getDst()) + ", " + name(o.getSrc()) + ", " +
              coordArg(o.getIdx(), tileCpp(o.getDst().getType())) +
              (o.getRows() ? ", " + name(o.getRows()) : std::string()) + ");");
            return;
          }
          std::string nw = std::to_string(o.getThreads() / tgt->wave);
          // CDNA: hk/cdna.cuh's copy, its descriptor ending at the tensor's
          // last row; include/cdna4's spans the tile's whole rows, so a last,
          // partial tile read past the tensor (mxfp8.quant's fault, #47).
          // include/rdna3's group load is bounded already.
          if (tgt->wave == 64)
            w("kittens::group_load_sync<" + nw + ">(" + name(o.getDst()) + ", " +
              name(o.getSrc()) + ", " +
              coordArg(o.getIdx(), tileCpp(o.getDst().getType())) + ");");
          else
            w("kittens::group<" + nw + ">::load(" + name(o.getDst()) + ", " +
              name(o.getSrc()) + ", " +
              coordArg(o.getIdx(), tileCpp(o.getDst().getType())) + ");");
          // include/cdna4's group load is buffer_load ... lds, a DMA that
          // counts on vmcnt and returns in flight; gfx90a+ has a back-off
          // barrier, so the compiler puts no wait before the s_barrier that
          // follows, and the LDS reads after it are inline asm it cannot see.
          // Measured on an MI355X without this: a different wrong answer per
          // run. The op promises "issued and waited", so wait here.
          // hk-lds-pipeline also takes it to retire every LDS op, as on
          // RDNA3, where the group load ends in s_waitcnt 0.
          if (tgt->wave == 64) {
            w("kittens::vm_wait<0>();");
            w("kittens::lds_wait<0>();");
          }
        })
        .Case([&](GroupStoreOp o) {
          // CDNA: hk/cdna.cuh's store, rows past the tensor's last dropped;
          // include/cdna4's wrote the whole tile, past the tensor.
          std::string nw = std::to_string(o.getThreads() / tgt->wave);
          std::string fn = tgt->wave == 64
                               ? "kittens::group_store_bounded<" + nw + ">"
                               : "kittens::group<" + nw + ">::store";
          w(fn + "(" + name(o.getDst()) + ", " + name(o.getSrc()) + ", " +
            coordArg(o.getIdx(), tileCpp(o.getSrc().getType())) + ");");
        })
        .Case([&](GroupWaitForOp o) {
          if (!o.getN()) {
            fail(o, "group_wait_for has no count: run hk-lds-pipeline first");
            return;
          }
          // vmcnt is 6 bits on gfx9, and issue stalls at the cap.
          w("kittens::vm_wait<" +
            std::to_string(std::min<int64_t>(*o.getN(), 63)) + ">();");
        })
        // synchronization
        .Case([&](PingPongBeginOp o) {
          w("if (" + name(o.getCond()) + ") __builtin_amdgcn_s_barrier();");
        })
        .Case([&](PingPongEndOp o) {
          w("if (!(" + name(o.getCond()) + ")) __builtin_amdgcn_s_barrier();");
        })
        .Case([&](BarrierOp o) {
          // The drain goes before the barrier: s_barrier has no memory
          // semantics of its own. The builtin, not __syncthreads(), which on
          // gfx11 adds an L0 flush this op does not promise.
          if (o.getDrain())
            w("kittens::lds_wait<0>();");
          w("__builtin_amdgcn_s_barrier();");
        })
        .Case([&](SchedBarrierOp o) {
          w("__builtin_amdgcn_sched_barrier(" + std::to_string(o.getMask()) +
            ");");
        })
        .Case([&](SchedGroupBarrierOp o) {
          w("__builtin_amdgcn_sched_group_barrier(" +
            std::to_string(o.getMask()) + ", " + std::to_string(o.getSize()) +
            ", " + std::to_string(o.getSyncid()) + ");");
        })
        .Case([&](PinOp o) {
          for (Value v : o.getValues())
            w("kittens::lds_bind(" + name(v) + ");");
        })
        .Case([&](SetPrioOp o) {
          w("__builtin_amdgcn_s_setprio(" + std::to_string(o.getLevel()) +
            ");");
        })
        // creation
        .Case([&](ZeroOp o) {
          call("zero", dest(o.getDst(), o.getResult()), {});
        })
        .Case([&](NegInftyOp o) {
          call("neg_infty", dest(o.getDst(), o.getResult()), {});
        })
        .Case([&](FullOp o) {
          Value d = dest(o.getDst(), o.getResult());
          std::string n = name(d);
          double v = o.getValue().convertToDouble();
          if (v == 0)
            w("kittens::zero(" + n + ");");
          else if (v == 1)
            w("kittens::one(" + n + ");");
          else if (std::isinf(v) && v < 0)
            w("kittens::neg_infty(" + n + ");");
          else if (std::isinf(v))
            w("kittens::pos_infty(" + n + ");");
          else {
            // No fill-with-a-constant in maps.cuh; zero then add folds to the
            // same VALU sequence.
            w("kittens::zero(" + n + ");");
            w("kittens::add(" + n + ", " + n + ", (" + elemCpp(d.getType()) +
              ")" + floatLit(v) + ");");
          }
        })
        // maps
        .Case<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp>(
            [&](auto o) { emitBinary(o); })
        .Case<ExpOp, Exp2Op, LogOp, Log2Op, AbsOp, ReluOp, CopyOp, GeluOp,
              DgeluOp, SqrtOp, RsqrtOp, SiluOp, StepOp, SwapHalvesOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          call(o->getName().stripDialect(), d, {name(o.getSrc())});
        })
        .Case([&](RotateHalfOp o) {
          Value d = dest(o.getDst(), o.getResult());
          call("rotate_half<" + std::to_string(o.getSpan()) + ">", d,
               {name(o.getSrc())});
        })
        .Case<MmaABOp, MmaABtOp, MmaAtBOp, MmaAtBtOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          call(o->getName().stripDialect(), d,
               {name(o.getA()), name(o.getB()), name(o.getC())});
        })
        .Case([&](CastOp o) {
          Value d = dest(o.getDst(), o.getResult());
          call("copy", d, {name(o.getSrc())});
        })
        .Case([&](LaneGatherOp o) {
          Value d = dest(Value(), o.getResult());
          w("kittens::lane_gather(" + name(d) + ", " + name(o.getSrc()) + ", " + name(o.getLanes()) + ");");
        })
        .Case([&](MfmaStripOp o) {
          Value d = dest(Value(), o.getResult());
          w("kittens::mfma_strip_f32_16x16x4(" + name(d) + ", " + name(o.getA()) + ", " +
            name(o.getB()) + ", " + name(o.getAcc()) + ");");
        })
        .Case([&](PostElemsOp o) {
          w("kittens::post_elems<" + std::to_string(o.getAux()) + ">(" + name(o.getDst()) + ", " +
            name(o.getVal()) + ", " + name(o.getIdx()) + ", " + name(o.getTag()) + ");");
        })
        .Case([&](TakeElemsOp o) {
          Value d = dest(Value(), o.getResult());
          w("kittens::take_elems<" + std::to_string(o.getAux()) + ">(" + name(d) + ", " +
            name(o.getSrc()) + ", " + name(o.getIdx()) + ", " + name(o.getTag()) + ", " +
            std::to_string(o.getTimeout()) + "LL);");
        })
        .Case([&](LdsLoadElemsOp o) {
          Value d = dest(Value(), o.getResult());
          w("kittens::lds_load_elems(" + name(d) + ", " + name(o.getSrc()) + ", " + name(o.getIdx()) + ");");
        })
        .Case([&](LdsStoreElemsOp o) {
          w("kittens::lds_store_elems(" + name(o.getDst()) + ", " + name(o.getVal()) + ", " + name(o.getIdx()) + ");");
        })
        .Case([&](IotaOp o) {
          Value d = dest(Value(), o.getResult());
          w("kittens::iota(" + name(d) + ");");
        })
        .Case([&](LoadElemsOp o) {
          Value d = dest(Value(), o.getResult());
          w("kittens::load_elems<" + std::to_string(o.getAux()) + ">(" + name(d) + ", " +
            name(o.getSrc()) + ", " + name(o.getIdx()) + ");");
        })
        .Case([&](StoreElemsOp o) {
          w("kittens::store_elems<" + std::to_string(o.getAux()) + ">(" + name(o.getDst()) + ", " +
            name(o.getVal()) + ", " + name(o.getIdx()) + ");");
        })
        .Case([&](GroupReduceOp o) {
          Value d = dest(Value(), o.getResult());
          auto [lanes, offs] = o.getOrder().split(':');
          std::string tmpl = std::string(o.getKind() == "sum" ? "kittens::base_ops::sum"
                                         : o.getKind() == "max" ? "kittens::base_ops::max"
                                                                : "kittens::base_ops::min") +
                             ", " + (lanes == "seq" ? "1" : "0");
          SmallVector<StringRef> parts;
          if (!offs.empty())
            offs.split(parts, ',');
          for (StringRef p : parts)
            tmpl += ", " + p.str();
          w("kittens::group_reduce_ordered<" + tmpl + ">(" + name(d) + ", " +
            name(o.getSrc()) + ");");
        })
        .Case([&](MapCppOp o) {
          // element (r, c) = (T)(expr) with x0.. the arguments' elements (or scalars)
          Value d = dest(Value(), o.getResult());
          std::vector<std::string> params, args{name(d)};
          for (auto [i, a] : llvm::enumerate(o.getArgs())) {
            params.push_back("const auto x" + std::to_string(i));
            args.push_back(name(a));
          }
          // contraction off: the expression is compiled as written -- a product and a sum stay two
          // roundings unless it says fmaf (HIP's __fmul_rn / __fadd_rn are plain operators clang may
          // otherwise fuse, and whether it does changed with the code around it)
          w("kittens::map_cpp(" + args[0] + ", [&](" + join(params) +
            ") { _Pragma(\"clang fp contract(off)\") return (" + o.getExpr().str() + "); }" +
            (args.size() > 1 ? ", " + join(std::vector<std::string>(args.begin() + 1, args.end()))
                             : std::string()) +
            ");");
        })
        .Case([&](BitcastOp o) {
          // the same registers read as another type of the same width: no instruction
          Value d = dest(o.getDst(), o.getResult());
          w("static_assert(sizeof(" + name(d) + ") == sizeof(" + name(o.getSrc()) + "));");
          w("__builtin_memcpy(&" + name(d) + ", &" + name(o.getSrc()) + ", sizeof(" + name(d) + "));");
        })
        // reductions and broadcasts
        .Case<RowMaxOp, RowMinOp, RowSumOp, RowProdOp, ColMaxOp, ColMinOp,
              ColSumOp, ColProdOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          std::vector<std::string> args{name(o.getSrc())};
          if (o.getAccumulate())
            args.push_back(name(d));
          if (auto order = o.getOrder()) {
            // row_sum_ordered<SEQ, offsets...>: verified "seq|tree:o1,o2,..."
            auto [lanes, offs] = order->split(':');
            std::string tmpl = lanes == "seq" ? "1" : "0";
            SmallVector<StringRef> parts;
            offs.split(parts, ',');
            for (StringRef p : parts)
              tmpl += ", " + p.str();
            std::vector<std::string> all{name(d)};
            all.insert(all.end(), args.begin(), args.end());
            w("kittens::" + o->getName().stripDialect().str() + "_ordered<" + tmpl + ">(" +
              join(all) + ");");
            return;
          }
          call(o->getName().stripDialect(), d, args);
        })
        .Case<AddRowOp, SubRowOp, MulRowOp, DivRowOp, AddColOp, SubColOp,
              MulColOp, DivColOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          call(o->getName().stripDialect(), d,
               {name(o.getSrc()), name(o.getVec())});
        })
        .Case<BroadcastRowOp, BroadcastColOp>([&](auto o) {
          Value d = dest(Value(), o.getResult());
          call(o->getName().stripDialect(), d, {name(o.getVec())});
        })
        .Case([&](RetypeVecOp o) {
          std::string a = cpp(o.getResult().getType()),
                      b = cpp(o.getSrc().getType());
          w("static_assert(std::is_same_v<" + a + ", " + b +
            ">, \"hk: as_vec between distinct rv types\");");
          w(a + " &" + name(o.getResult()) + " = " + name(o.getSrc()) + ";");
        })
        // masks
        .Case<LeftFillOp, RightFillOp, UpperFillOp, LowerFillOp, TriuOp,
              TrilOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          call(o->getName().stripDialect(), d,
               {name(o.getSrc()), name(o.getIdx()),
                "(" + elemCpp(d.getType()) + ")" +
                    floatLit(o.getValue().convertToDouble())});
        })
        .Case<MakeCausalOp, MakeCausalTOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          // The fill value is the unpacked element type; for bf16 the tile's
          // dtype is the packed pair, so no cast at all.
          call(o->getName().stripDialect(), d,
               {name(o.getSrc()), floatLit(o.getValue().convertToDouble())});
        })
        .Case([&](TransposeOp o) {
          Value d = dest(o.getDst(), o.getResult());
          call("transpose", d, {name(o.getSrc())});
        })
        .Case([&](TransposeSepOp o) {
          Value d = dest(o.getDst(), o.getResult());
          call("transpose_sep", d, {name(o.getSrc())});
        })
        .Case([&](RegSubtileOp o) {
          w("auto &" + name(o.getResult()) + " = kittens::subtile_inplace<" +
            std::to_string(o.getResult().getType().getRows()) + ">(" +
            name(o.getSrc()) + ", " + std::to_string(o.getIndex()) + ");");
        })
        .Case([&](AsOperandOp o) {
          // hk/cdna.cuh: register renaming only; the verifier has held the
          // two fragments to the same lanes.
          Value d = dest(o.getDst(), o.getResult());
          call("as_operand", d, {name(o.getSrc())});
        })
        .Default([&](Operation *o) {
          fail(o, "no C++ emitter for " + o->getName().getStringRef());
        });
  }

  template <typename OpT>
  void emitBinary(OpT o) {
    Value d = dest(o.getDst(), o.getResult());
    std::vector<std::string> args{name(o.getLhs())};
    if (Value rhs = o.getRhs()) {
      if (rhs.getType().isInteger(32))
        // A device scalar: cast to the element type so the map's templated
        // right operand resolves as it does for a literal.
        args.push_back("(" + elemCpp(d.getType()) + ")" + name(rhs));
      else
        args.push_back(name(rhs));
    }
    if (auto s = o.getScalar())
      args.push_back("(" + elemCpp(d.getType()) + ")" +
                     floatLit(s->convertToDouble()));
    call(o->getName().stripDialect(), d, args);
  }

  std::tuple<std::string, std::string, std::string> stageArgs(Value buf,
                                                              int64_t slot) {
    auto t = cast<StageType>(buf.getType());
    std::string g =
        "kittens::group<" + std::to_string(t.getThreads() / tgt->wave) + ">::";
    std::string expr =
        name(buf) + (t.getDepth() > 1 ? "[" + std::to_string(slot) + "]" : "");
    auto sb = buf.getDefiningOp<StageBufferOp>();
    std::string tmpl = name(sb.getDst());
    if (sb.getDst().getType().getCount() != 1)
      tmpl += "[0]";
    return {g, expr, tmpl};
  }
};

} // namespace

LogicalResult mlir::hk::translateKernelToCpp(func::FuncOp kernel,
                                             ArrayRef<std::string> includes,
                                             raw_ostream &os) {
  return Emitter(kernel).emit(includes, os);
}

void mlir::hk::nameValuesAsCpp(func::FuncOp kernel) {
  Emitter(kernel).applyNames();
}

void mlir::hk::registerToCppTranslation() {
  TranslateFromMLIRRegistration reg(
      "hk-to-cpp", "translate hk kernels to HipKittens C++",
      [](Operation *op, raw_ostream &os) -> LogicalResult {
        auto mod = dyn_cast<ModuleOp>(op);
        if (!mod)
          return op->emitError("expected a module");
        for (auto f : mod.getOps<func::FuncOp>())
          if (f->hasAttr(attrs::kKernel))
            if (failed(translateKernelToCpp(f, {"\"kittens.cuh\""}, os)))
              return failure();
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<HKDialect, arith::ArithDialect, scf::SCFDialect,
                        func::FuncDialect>();
      });
}
