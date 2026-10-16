// RUN: hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s

// gfx950's asynchronous group load, the vmcnt hk-lds-pipeline derives for it,
// and the pingpong pair: two copies in flight, the first retired with the
// second's two instructions still outstanding, one half of the workgroup a
// barrier behind the other between pingpong_begin and pingpong_end.

module {
  func.func @dma(%a: !hk.gl<bf16> {hk.name = "a"})
      attributes {hk.kernel, hk.arch = "gfx950", hk.warps = 8 : i64, hk.consts = {}} {
    %c0 = arith.constant 0 : i32 loc("c")
    %c1 = arith.constant 1 : i32 loc("c")
    %c4 = arith.constant 4 : i32 loc("c")
    %w = hk.warp_id loc("warp_id")
    %half = arith.divsi %w, %c4 : i32 loc("half")
    %behind = arith.cmpi eq, %half, %c1 : i32 loc("behind")
    %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element> loc("idx")
    %As = hk.alloc_shared : !hk.st<bf16, 128x64, 2> loc("As")
    %a0 = hk.shared_at %As[%c0] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64> loc("a0")
    %a1 = hk.shared_at %As[%c1] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64> loc("a1")
    // CHECK: kittens::group_load_async<8>(a0, g.a, idx);
    hk.group_load %a[%idx] into %a0 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
    // CHECK: kittens::group_load_async<8>(a1, g.a, idx);
    hk.group_load %a[%idx] into %a1 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
    // CHECK: if (behind) __builtin_amdgcn_s_barrier();
    hk.pingpong_begin %behind
    // CHECK: kittens::vm_wait<2>();
    hk.group_wait_for %a0 : !hk.st<bf16, 128x64>
    // CHECK: kittens::vm_wait<0>();
    hk.group_wait_for %a1 : !hk.st<bf16, 128x64>
    hk.barrier
    // CHECK: if (!(behind)) __builtin_amdgcn_s_barrier();
    hk.pingpong_end %behind
    return
  }
  func.func @dma__grid(%a: !hk.gl<bf16> {hk.name = "a"}) -> (i32, i32, i32)
      attributes {hk.grid_of = @dma} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
}
