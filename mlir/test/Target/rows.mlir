// RUN: hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s

// A tile whose rows end before the tensor's: `rows` is a runtime count, and
// both ops become the bounded load and store with it as the row limit.

// CHECK-LABEL: rows_kernel
// CHECK: kittens::load_bounded(ld, g.a, {{.*}}, n);
// CHECK: kittens::store_bounded(g.o, ld, {{.*}}, n);

module {
  func.func @rows(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"},
                  %len: !hk.gl<i32> {hk.name = "len"})
      attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 1 : i64,
                  hk.consts = {}} {
    %by = hk.block_idx y loc("bid_y")
    %c0 = arith.constant 0 : i32 loc("c")
    %li = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element> loc("li")
    %n = hk.load_scalar %len[%li] : !hk.gl<i32>, !hk.coord<element> loc("n")
    %idx = hk.coord %c0, %c0, %by, %c0 : !hk.coord<tile> loc("idx")
    %ld = hk.load_rows %a[%idx] rows %n : !hk.gl<bf16>, !hk.coord<tile> -> !hk.rt<bf16, 16x16, row> loc("ld")
    hk.store_rows %ld, %o[%idx] rows %n : !hk.rt<bf16, 16x16, row>, !hk.gl<bf16>, !hk.coord<tile>
    return
  }
  func.func @rows__grid(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"},
                        %len: !hk.gl<i32> {hk.name = "len"}) -> (i32, i32, i32)
      attributes {hk.grid_of = @rows} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
}
