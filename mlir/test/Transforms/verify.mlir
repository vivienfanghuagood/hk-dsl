// RUN: hk-opt %s -split-input-file --hk-verify -verify-diagnostics

// A workgroup operation every warp has to reach the same number of times
// cannot sit in a loop whose test may differ between warps.

func.func @barrier_in_while(%x: !hk.gl<bf16> {hk.name = "x"})
    attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 2 : i64, hk.consts = {}} {
  %c16 = arith.constant 16 : i32
  %c0 = arith.constant 0 : i32
  %k = scf.while (%i = %c0) : (i32) -> i32 {
    %cols = hk.extent %x cols : !hk.gl<bf16>
    %lt = arith.cmpi slt, %i, %cols : i32
    scf.condition(%lt) %i : i32
  } do {
  ^bb0(%i: i32):
    // expected-error@+1 {{is inside a runtime while loop}}
    hk.barrier
    %n = arith.addi %i, %c16 : i32
    scf.yield %n : i32
  }
  return
}

// -----

func.func @barrier_in_if(%x: !hk.gl<bf16> {hk.name = "x"})
    attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 2 : i64, hk.consts = {}} {
  %w = hk.warp_id
  %c0 = arith.constant 0 : i32
  %is0 = arith.cmpi eq, %w, %c0 : i32
  scf.if %is0 {
    // expected-error@+1 {{is inside an hk.if_}}
    hk.barrier
  }
  return
}

// -----

// hk.pingpong splits the barriers of everything between its ends, so both
// ends are in the kernel's own block, on a condition from warp_id.

func.func @pingpong_in_loop(%x: !hk.gl<bf16> {hk.name = "x"}, %p: i1)
    attributes {hk.kernel, hk.arch = "gfx950", hk.warps = 8 : i64, hk.consts = {}} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %c4 step %c1 : i32 {
    // expected-error@+1 {{hk.pingpong is inside a loop or a branch}}
    hk.pingpong_begin %p
    hk.pingpong_end %p
  }
  return
}

// -----

func.func @pingpong_by_lane(%x: !hk.gl<bf16> {hk.name = "x"})
    attributes {hk.kernel, hk.arch = "gfx950", hk.warps = 8 : i64, hk.consts = {}} {
  %l = hk.lane_id
  %c32 = arith.constant 32 : i32
  %p = arith.cmpi slt, %l, %c32 : i32
  // expected-error@+1 {{hk.pingpong's condition depends on lane_id()}}
  hk.pingpong_begin %p
  hk.pingpong_end %p
  return
}
