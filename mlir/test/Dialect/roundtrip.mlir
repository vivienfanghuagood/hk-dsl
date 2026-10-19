// RUN: hk-opt %s | hk-opt | FileCheck %s

// Every type, and one op of each shape: a fresh tile (a result) and a write
// into one that exists (`into`).

// CHECK-LABEL: func.func @types
func.func @types(%g: !hk.gl<bf16>) attributes {hk.kernel} {
  // CHECK: hk.block_idx y
  %by = hk.block_idx y
  %w = hk.warp_id
  %c0 = arith.constant 0 : i32
  // CHECK: hk.coord {{.*}} : !hk.coord<element>
  %idx = hk.coord %c0, %c0, %by, %w : !hk.coord<element>
  // CHECK: hk.load {{.*}} -> !hk.rt<f32, 16x64, row>
  %t = hk.load %g[%idx] : !hk.gl<bf16>, !hk.coord<element> -> !hk.rt<f32, 16x64, row>
  // CHECK: hk.row_sum {{.*}} -> !hk.rv<f32, 16x64, row, col>
  %v = hk.row_sum %t : !hk.rt<f32, 16x64, row> -> !hk.rv<f32, 16x64, row, col>
  // CHECK: hk.row_max {{.*}} into {{.*}} {accumulate}
  hk.row_max %t into %v {accumulate} : !hk.rt<f32, 16x64, row> into !hk.rv<f32, 16x64, row, col>
  // CHECK: hk.mul_row {{.*}} into
  hk.mul_row %t, %v into %t : !hk.rt<f32, 16x64, row>, !hk.rv<f32, 16x64, row, col> into !hk.rt<f32, 16x64, row>
  // CHECK: hk.add {{.*}} {scalar = 1.000000e+00 : f64}
  hk.add %t into %t {scalar = 1.0 : f64} : !hk.rt<f32, 16x64, row> into !hk.rt<f32, 16x64, row>
  // CHECK: hk.mul {{.*}}, %{{.*}} : !hk.rt<f32, 16x64, row>, i32
  %s = hk.mul %t, %by : !hk.rt<f32, 16x64, row>, i32 -> !hk.rt<f32, 16x64, row>
  %b = hk.cast %s : !hk.rt<f32, 16x64, row> -> !hk.rt<bf16, 16x64, row>
  hk.store %b, %g[%idx] : !hk.rt<bf16, 16x64, row>, !hk.gl<bf16>, !hk.coord<element>
  // CHECK: hk.alloc_shared : !hk.st<bf16, 128x64, 2>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64, 2>
  %a0 = hk.shared_at %As[%c0] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64>
  // CHECK: hk.stage_buffer {{.*}} -> !hk.stage<bf16, 128x64, 256>
  %buf = hk.stage_buffer %As : !hk.st<bf16, 128x64, 2> -> !hk.stage<bf16, 128x64, 256>
  %sv = hk.alloc_shared_vec : !hk.sv<f32, 16, 4>
  // CHECK: hk.granules {{.*}} -> !hk.granules<bf16, 128x64>
  %gr = hk.granules %a0 : !hk.st<bf16, 128x64> -> !hk.granules<bf16, 128x64>
  %u = hk.retype_vec %v : !hk.rv<f32, 16x64, row, col> -> !hk.rv<f32, 16x16, row, col, uniform>
  // CHECK: hk.barrier {drain}
  hk.barrier {drain}
  hk.vm_wait 0
  hk.setprio 1
  // CHECK: hk.group_load {{.*}} {async, threads = 512 : i64}
  hk.group_load %g[%idx] into %a0 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  // CHECK: hk.group_load {{.*}} rows %{{.*}} into
  hk.group_load %g[%idx] rows %by into %a0 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  // CHECK: hk.group_wait_for {{.*}} : !hk.st<bf16, 128x64>
  hk.group_wait_for %a0 : !hk.st<bf16, 128x64>
  %wid = hk.warp_id
  %c4 = arith.constant 4 : i32
  %half = arith.divsi %wid, %c4 : i32
  %behind = arith.cmpi eq, %half, %c0 : i32
  // CHECK: hk.pingpong_begin
  hk.pingpong_begin %behind
  // CHECK: hk.pingpong_end
  hk.pingpong_end %behind
  return
}

// gfx950 attention's forms: a 32x32 accumulator, a half of it, the operand
// read off its registers (a 16x32 fragment with a lane run of 4), the vote on
// the running max, and the pin that holds values to their phase.

// CHECK-LABEL: func.func @cdna_attention
func.func @cdna_attention(%s: !hk.rt<f32, 64x32, col, frag 32x32>,
                          %m: !hk.rv<f32, 64x32, col, row, frag 32x32>) {
  // CHECK: hk.rt_subtile %{{.*}}[1] : !hk.rt<f32, 64x32, col, frag 32x32> -> !hk.rt<f32, 32x32, col, frag 32x32>
  %h = hk.rt_subtile %s[1] : !hk.rt<f32, 64x32, col, frag 32x32> -> !hk.rt<f32, 32x32, col, frag 32x32>
  hk.exp2 %h into %h : !hk.rt<f32, 32x32, col, frag 32x32> into !hk.rt<f32, 32x32, col, frag 32x32>
  // CHECK: hk.zero -> !hk.rt<bf16, 64x32, col, frag 16x32 stride 4>
  %p = hk.zero -> !hk.rt<bf16, 64x32, col, frag 16x32 stride 4>
  // CHECK: hk.as_operand %{{.*}} into %{{.*}} : !hk.rt<f32, 64x32, col, frag 32x32> into !hk.rt<bf16, 64x32, col, frag 16x32 stride 4>
  hk.as_operand %s into %p : !hk.rt<f32, 64x32, col, frag 32x32> into !hk.rt<bf16, 64x32, col, frag 16x32 stride 4>
  // CHECK: hk.any_above %{{.*}}, %{{.*}}, 8.000000e+00
  %g = hk.any_above %m, %m, 8.0 : !hk.rv<f32, 64x32, col, row, frag 32x32>, !hk.rv<f32, 64x32, col, row, frag 32x32>
  // CHECK: hk.pin %{{.*}}, %{{.*}} : !hk.rt<bf16, 64x32, col, frag 16x32 stride 4>, !hk.rv<f32, 64x32, col, row, frag 32x32>
  hk.pin %p, %m : !hk.rt<bf16, 64x32, col, frag 16x32 stride 4>, !hk.rv<f32, 64x32, col, row, frag 32x32>
  return
}

// A strip's halves exchanged (RoPE's rotate_half), fresh and in place.
// CHECK-LABEL: func.func @strip_swap_halves
func.func @strip_swap_halves(%s: !hk.rt<f32, 1x128, strip<2>>) {
  // CHECK: hk.swap_halves %{{.*}} : !hk.rt<f32, 1x128, strip<2>> -> !hk.rt<f32, 1x128, strip<2>>
  %h = hk.swap_halves %s : !hk.rt<f32, 1x128, strip<2>> -> !hk.rt<f32, 1x128, strip<2>>
  // CHECK: hk.swap_halves %{{.*}} into %{{.*}}
  hk.swap_halves %h into %s : !hk.rt<f32, 1x128, strip<2>> into !hk.rt<f32, 1x128, strip<2>>
  return
}
