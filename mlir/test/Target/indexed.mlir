// RUN: hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s

// Rows by index, atomics and random draws: each op becomes the library call
// of the same name, with its runtime operands in order.

// CHECK-LABEL: indexed_kernel
// CHECK: kittens::load_gather(ld, g.x, g.ids, at, n, c);
// CHECK: kittens::store_scatter(g.o, ld, g.ids, at, n, c);
// CHECK: kittens::philox_uniform(rnd, (uint32_t)at, 0u, c, n);
// CHECK: kittens::atomic_add(g.acc, rnd, {{.*}});
// CHECK: const int old = kittens::atomic_scalar(g.cnt, li, n, 1);

module {
  func.func @indexed(%x: !hk.gl<bf16> {hk.name = "x"}, %o: !hk.gl<bf16> {hk.name = "o"},
                     %ids: !hk.gl<i32> {hk.name = "ids"},
                     %acc: !hk.gl<f32> {hk.name = "acc"},
                     %cnt: !hk.gl<i32> {hk.name = "cnt"})
      attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 1 : i64,
                  hk.consts = {}} {
    %by = hk.block_idx y loc("bid_y")
    %c = arith.constant 0 : i32 loc("c")
    %at = arith.constant 16 : i32 loc("at")
    %n = arith.constant 64 : i32 loc("n")
    %li = hk.coord %c, %c, %c, %c : !hk.coord<element> loc("li")
    %ld = hk.gather_rows %x[%ids[%at, %n], %c] : !hk.gl<bf16>, !hk.gl<i32> -> !hk.rt<bf16, 16x64, row> loc("ld")
    hk.scatter_rows %ld, %o[%ids[%at, %n], %c] : !hk.rt<bf16, 16x64, row>, !hk.gl<bf16>, !hk.gl<i32>
    %rnd = hk.philox_uniform %at, %c, %n -> !hk.rt<f32, 16x64, row> loc("rnd")
    %idx = hk.coord %c, %c, %by, %c : !hk.coord<tile> loc("idx")
    hk.atomic_add %rnd, %acc[%idx] : !hk.rt<f32, 16x64, row>, !hk.gl<f32>, !hk.coord<tile>
    %old = hk.atomic_scalar "max" %cnt[%li], %n : !hk.gl<i32>, !hk.coord<element> loc("old")
    return
  }
  func.func @indexed__grid(%x: !hk.gl<bf16> {hk.name = "x"}, %o: !hk.gl<bf16> {hk.name = "o"},
                           %ids: !hk.gl<i32> {hk.name = "ids"},
                           %acc: !hk.gl<f32> {hk.name = "acc"},
                           %cnt: !hk.gl<i32> {hk.name = "cnt"}) -> (i32, i32, i32)
      attributes {hk.grid_of = @indexed} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
}
