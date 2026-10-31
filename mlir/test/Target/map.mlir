// RUN: hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s

// hk.map: a region prints as one C++ expression (a value used twice becomes a
// local), a body as written; both with contraction off.

// CHECK-LABEL: map_kernel
// hk-map-fuse made the chain idx -> sel -> bits -> f one map; f, which the body map
// also reads, is computed once and m reads it
// CHECK: kittens::map_cpp(f, [&](const auto x0, const auto x1) { _Pragma("clang fp contract(off)") const int v0 = (((x0 * 64) + x1) & 63); const int v1 = ((v0 < x0) ? (v0 >> 2) : (0 - (v0 % 4))); const float v2 = (float)(__builtin_popcount(v1) + __builtin_ctzg((unsigned)v1, 32)); return (__builtin_fmaf(v2, v2, __builtin_amdgcn_rcpf((float)__float2bfloat16(v2)))); }, w, col);
// CHECK: kittens::map_cpp(m, [&](const auto x0, const auto x1) { _Pragma("clang fp contract(off)") return ((!(x0 >= (float)0.5))); }, f, w);
// CHECK: kittens::map_cpp(raw, [&](const auto x0) { _Pragma("clang fp contract(off)") return (fabsf(x0)); }, f);

module {
  func.func @map(%o: !hk.gl<f32> {hk.name = "o"})
      attributes {hk.kernel, hk.arch = "gfx950", hk.warps = 1 : i64, hk.consts = {}} {
    %w = hk.warp_id loc("w")
    %col = hk.lane_coord {axis = 1 : i64} -> !hk.rt<i32, 1x64, strip<1>> loc("col")
    %idx = hk.map(%col, %w) : (!hk.rt<i32, 1x64, strip<1>>, i32) -> !hk.rt<i32, 1x64, strip<1>> {
    ^bb0(%x: i32, %y: i32):
      %c64 = arith.constant 64 : i32
      %p = arith.muli %y, %c64 : i32
      %s = arith.addi %p, %x : i32
      hk.yield %s : i32
    } loc("idx")
    %sel = hk.map(%idx, %w) : (!hk.rt<i32, 1x64, strip<1>>, i32) -> !hk.rt<i32, 1x64, strip<1>> {
    ^bb0(%x: i32, %y: i32):
      %c63 = arith.constant 63 : i32
      %c2 = arith.constant 2 : i32
      %c4 = arith.constant 4 : i32
      %a = arith.andi %x, %c63 : i32
      %lt = arith.cmpi slt, %a, %y : i32
      %sh = arith.shrsi %a, %c2 : i32
      %r = arith.remsi %a, %c4 : i32
      %c0 = arith.constant 0 : i32
      %n = arith.subi %c0, %r : i32
      %v = arith.select %lt, %sh, %n : i32
      hk.yield %v : i32
    } loc("sel")
    %bits = hk.map(%sel) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
    ^bb0(%x: i32):
      %p = math.ctpop %x : i32
      %z = math.cttz %x : i32
      %s = arith.addi %p, %z : i32
      hk.yield %s : i32
    } loc("bits")
    %f = hk.map(%bits) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<f32, 1x64, strip<1>> {
    ^bb0(%x: i32):
      %v = arith.sitofp %x : i32 to f32
      %b = arith.truncf %v : f32 to bf16
      %e = arith.extf %b : bf16 to f32
      %r = hk.rcp %e : f32
      %m = math.fma %v, %v, %r : f32
      hk.yield %m : f32
    } loc("f")
    %m = hk.map(%f, %w) : (!hk.rt<f32, 1x64, strip<1>>, i32) -> !hk.rt<i1, 1x64, strip<1>> {
    ^bb0(%x: f32, %y: i32):
      %h = arith.constant 0.5 : f32
      %ge = arith.cmpf oge, %x, %h : f32
      %t = arith.constant true
      %n = arith.xori %ge, %t : i1
      hk.yield %n : i1
    } loc("m")
    %raw = hk.map(%f) {body = "fabsf(x0)"} : (!hk.rt<f32, 1x64, strip<1>>) -> !hk.rt<f32, 1x64, strip<1>> loc("raw")
    return
  }
  func.func @map__grid(%o: !hk.gl<f32> {hk.name = "o"}) -> (i32, i32, i32)
      attributes {hk.grid_of = @map} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
}
