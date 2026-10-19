// RUN: hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s

// A scalar the kernel only reads. `invariant` on a wave64 target is the
// scalar-cache read, an s_load counted on lgkmcnt; include/rdna3 has one
// load_scalar, which gfx1100 keeps.

// CHECK-LABEL: scalar950_kernel
// CHECK: const int n = kittens::load_scalar_invariant(g.len, li);
// CHECK: const int m = kittens::load_scalar(g.len, li);
// CHECK-LABEL: scalar1100_kernel
// CHECK: const int n = kittens::load_scalar(g.len, li);

module {
  func.func @scalar950(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"},
                       %len: !hk.gl<i32> {hk.name = "len"})
      attributes {hk.kernel, hk.arch = "gfx950", hk.warps = 1 : i64,
                  hk.consts = {}} {
    %c0 = arith.constant 0 : i32 loc("c")
    %li = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element> loc("li")
    %n = hk.load_scalar %len[%li] {invariant} : !hk.gl<i32>, !hk.coord<element> loc("n")
    %m = hk.load_scalar %len[%li] : !hk.gl<i32>, !hk.coord<element> loc("m")
    %r = arith.addi %n, %m : i32 loc("r")
    %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<tile> loc("idx")
    %ld = hk.load_rows %a[%idx] rows %r : !hk.gl<bf16>, !hk.coord<tile> -> !hk.rt<bf16, 16x16, row> loc("ld")
    hk.store_rows %ld, %o[%idx] rows %r : !hk.rt<bf16, 16x16, row>, !hk.gl<bf16>, !hk.coord<tile>
    return
  }
  func.func @scalar950__grid(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"},
                             %len: !hk.gl<i32> {hk.name = "len"}) -> (i32, i32, i32)
      attributes {hk.grid_of = @scalar950} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
  func.func @scalar1100(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"},
                        %len: !hk.gl<i32> {hk.name = "len"})
      attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 1 : i64,
                  hk.consts = {}} {
    %c0 = arith.constant 0 : i32 loc("c")
    %li = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element> loc("li")
    %n = hk.load_scalar %len[%li] {invariant} : !hk.gl<i32>, !hk.coord<element> loc("n")
    %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<tile> loc("idx")
    %ld = hk.load_rows %a[%idx] rows %n : !hk.gl<bf16>, !hk.coord<tile> -> !hk.rt<bf16, 16x16, row> loc("ld")
    hk.store_rows %ld, %o[%idx] rows %n : !hk.rt<bf16, 16x16, row>, !hk.gl<bf16>, !hk.coord<tile>
    return
  }
  func.func @scalar1100__grid(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"},
                              %len: !hk.gl<i32> {hk.name = "len"}) -> (i32, i32, i32)
      attributes {hk.grid_of = @scalar1100} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
}
