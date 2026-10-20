// RUN: hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s

// hk.noalias on CDNA: the kernel takes the globals' fields as scalar
// arguments -- the pointers __restrict__ and first, then each field across the
// tensors -- in namespace hk_flat, rebuilds `g` from them, and the launch
// passes them. Without the attribute, or on RDNA, the kernel takes `globals`.

module {
  func.func @copy950(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"},
                     %s: !hk.gl<f32> {hk.name = "s"})
      attributes {hk.kernel, hk.arch = "gfx950", hk.warps = 1 : i64, hk.noalias,
                  hk.consts = {}} {
    %c0 = arith.constant 0 : i32 loc("c")
    %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<tile> loc("idx")
    %ld = hk.load %a[%idx] : !hk.gl<bf16>, !hk.coord<tile> -> !hk.rt<bf16, 16x16, row> loc("ld")
    hk.store %ld, %o[%idx] : !hk.rt<bf16, 16x16, row>, !hk.gl<bf16>, !hk.coord<tile>
    return
  }
  func.func @copy950__grid(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"},
                           %s: !hk.gl<f32> {hk.name = "s"})
      -> (i32, i32, i32) attributes {hk.grid_of = @copy950} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
  func.func @plain950(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"})
      attributes {hk.kernel, hk.arch = "gfx950", hk.warps = 1 : i64, hk.consts = {}} {
    %c0 = arith.constant 0 : i32 loc("c")
    %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<tile> loc("idx")
    %ld = hk.load %a[%idx] : !hk.gl<bf16>, !hk.coord<tile> -> !hk.rt<bf16, 16x16, row> loc("ld")
    hk.store %ld, %o[%idx] : !hk.rt<bf16, 16x16, row>, !hk.gl<bf16>, !hk.coord<tile>
    return
  }
  func.func @plain950__grid(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"})
      -> (i32, i32, i32) attributes {hk.grid_of = @plain950} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
  func.func @copy1100(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"})
      attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 1 : i64, hk.noalias,
                  hk.consts = {}} {
    %c0 = arith.constant 0 : i32 loc("c")
    %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<tile> loc("idx")
    %ld = hk.load %a[%idx] : !hk.gl<bf16>, !hk.coord<tile> -> !hk.rt<bf16, 16x16, row> loc("ld")
    hk.store %ld, %o[%idx] : !hk.rt<bf16, 16x16, row>, !hk.gl<bf16>, !hk.coord<tile>
    return
  }
  func.func @copy1100__grid(%a: !hk.gl<bf16> {hk.name = "a"}, %o: !hk.gl<bf16> {hk.name = "o"})
      -> (i32, i32, i32) attributes {hk.grid_of = @copy1100} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
}

// CHECK: // kernel : copy950
// CHECK: struct hk_kernarg {
// CHECK-NEXT:     bf16 *hk_a;
// CHECK-NEXT:     bf16 *hk_o;
// CHECK-NEXT:     float *hk_s;
// CHECK-NEXT:     int hk_a_rs;
// CHECK-NEXT:     int hk_o_rs;
// CHECK-NEXT:     int hk_s_rs;
// CHECK-NEXT:     int hk_a_r;
// CHECK:          int hk_s_d;
// CHECK-NEXT: };
// CHECK: namespace hk_flat {
// CHECK-NEXT: __global__ __launch_bounds__(64, 1)
// CHECK-NEXT: void copy950_kernel(bf16 *__restrict__ hk_a, bf16 *__restrict__ hk_o, float *__restrict__ hk_s, int hk_a_rs, int hk_o_rs, int hk_s_rs, int hk_a_r, int hk_o_r, int hk_s_r, int hk_a_c, int hk_o_c, int hk_s_c, int hk_a_b, int hk_o_b, int hk_s_b, int hk_a_d, int hk_o_d, int hk_s_d) {
// CHECK-NEXT:     const globals g{decltype(globals::a)(hk_a, hk_a_b, hk_a_d, hk_a_r, hk_a_c, hk_a_rs), decltype(globals::o)(hk_o, hk_o_b, hk_o_d, hk_o_r, hk_o_c, hk_o_rs), decltype(globals::s)(hk_s, hk_s_b, hk_s_d, hk_s_r, hk_s_c, hk_s_rs)};
// CHECK: } // namespace hk_flat
// CHECK: hk_flat::copy950_kernel<<<g.grid(), g.block(), 0, stream>>>(g.a.raw_ptr, g.o.raw_ptr, g.s.raw_ptr, g.a.row_stride(), g.o.row_stride(), g.s.row_stride(), g.a.rows(), g.o.rows(), g.s.rows(), g.a.cols(), g.o.cols(), g.s.cols(), g.a.batch(), g.o.batch(), g.s.batch(), g.a.depth(), g.o.depth(), g.s.depth());

// CHECK: // kernel : plain950
// CHECK-NOT: hk_kernarg
// CHECK: void plain950_kernel(const globals g) {
// CHECK: plain950_kernel<<<g.grid(), g.block(), 0, stream>>>(g);

// CHECK: // kernel : copy1100
// CHECK-NOT: hk_kernarg
// CHECK: void copy1100_kernel(const globals g) {
// CHECK: copy1100_kernel<<<g.grid(), g.block(), 0, stream>>>(g);
