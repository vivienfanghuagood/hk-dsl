// RUN: hk-opt %s -split-input-file --pass-pipeline='builtin.module(func.func(hk-lds-pipeline))' -verify-diagnostics | FileCheck %s

// gfx950's asynchronous group load: buffer_load ... lds, counted on vmcnt. A
// 128x64 bf16 tile over 512 threads is two 16-byte copies a thread, so
// retiring the first of two copies leaves the second's two in flight.

// CHECK-LABEL: func.func @counts
func.func @counts(%g: !hk.gl<bf16>) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64, 2>
  %a0 = hk.shared_at %As[%c0] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64>
  %a1 = hk.shared_at %As[%c1] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %a0 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %a1 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  // CHECK: hk.group_wait_for {{.*}} {n = 2 : i64}
  hk.group_wait_for %a0 : !hk.st<bf16, 128x64>
  hk.barrier
  %r = hk.load_shared %a0 {lds_loads = 0 : i64, wait = true} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
  // CHECK: hk.group_wait_for {{.*}} {n = 0 : i64}
  hk.group_wait_for %a1 : !hk.st<bf16, 128x64>
  return
}

// -----

func.func @read_in_flight(%g: !hk.gl<bf16>) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %As {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.barrier
  // expected-error@+1 {{reads  while this warp's copy into it is in flight}}
  %r = hk.load_shared %As {lds_loads = 0 : i64, wait = true} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
  hk.group_wait_for %As : !hk.st<bf16, 128x64>
  return
}

// -----

// This warp's share has landed; the other warps' have not, until a barrier.

func.func @read_before_barrier(%g: !hk.gl<bf16>) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %As {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.group_wait_for %As : !hk.st<bf16, 128x64>
  // expected-error@+1 {{0 barriers after this warp's share of the copy into it landed}}
  %r = hk.load_shared %As {lds_loads = 0 : i64, wait = true} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
  return
}

// -----

// Inside a pingpong half the workgroup is one barrier behind: one barrier
// after the wait is not enough, two is.

func.func @pingpong_needs_two(%g: !hk.gl<bf16>, %p: i1) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %As {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.pingpong_begin %p
  hk.group_wait_for %As : !hk.st<bf16, 128x64>
  hk.barrier
  // expected-error@+1 {{1 barrier after this warp's share of the copy into it landed. The other warps' shares are only known to have landed 2 barriers after their waits. Inside hk.pingpong}}
  %r = hk.load_shared %As {lds_loads = 0 : i64, wait = true} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
  hk.pingpong_end %p
  return
}

// -----

// CHECK-LABEL: func.func @pingpong_two
func.func @pingpong_two(%g: !hk.gl<bf16>, %p: i1) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %As {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.pingpong_begin %p
  hk.group_wait_for %As : !hk.st<bf16, 128x64>
  hk.barrier
  hk.barrier
  %r = hk.load_shared %As {lds_loads = 0 : i64, wait = true} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
  hk.pingpong_end %p
  return
}

// -----

// A read of a named buffer may stay in flight across a barrier; a copy into
// that buffer while it is in flight may not.

func.func @copy_over_reads(%g: !hk.gl<bf16>) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %As {threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.barrier
  %r = hk.load_shared %As {lds_loads = 32 : i64, wait = false} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
  hk.barrier
  // expected-error@+1 {{group_load writes  while this warp's reads of it into  are still in flight}}
  hk.group_load %g[%idx] into %As {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.lds_wait_for %r : !hk.rt<bf16, 128x64, row>
  return
}

// -----

func.func @copy_too_soon(%g: !hk.gl<bf16>, %p: i1) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %As {threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.barrier
  hk.pingpong_begin %p
  %r = hk.load_shared %As {lds_loads = 32 : i64, wait = false} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
  hk.lds_wait_for %r : !hk.rt<bf16, 128x64, row>
  hk.barrier
  // expected-error@+1 {{group_load overwrites  1 barrier after this warp's reads of it landed}}
  hk.group_load %g[%idx] into %As {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  hk.group_wait_for %As : !hk.st<bf16, 128x64>
  hk.pingpong_end %p
  return
}

// -----

func.func @runtime_index(%g: !hk.gl<bf16>, %i: i32) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64, 2>
  %a = hk.shared_at %As[%i] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64>
  // expected-error@+1 {{an asynchronous group load into a buffer picked by a runtime index}}
  hk.group_load %g[%idx] into %a {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  return
}

// -----

func.func @not_gfx950(%g: !hk.gl<bf16>) attributes {hk.kernel, hk.arch = "gfx942"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64>
  // expected-error@+1 {{an asynchronous group load on gfx942}}
  hk.group_load %g[%idx] into %As {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  return
}

// -----

// expected-error@+1 {{the kernel ends with a copy into  still in flight}}
func.func @ends_in_flight(%g: !hk.gl<bf16>) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %As {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  return
}

// -----

// A loop that issues the next copy each iteration and retires the one before
// has a steady state: one copy (two instructions) left in flight.

// CHECK-LABEL: func.func @steady
func.func @steady(%g: !hk.gl<bf16>, %n: i32) attributes {hk.kernel, hk.arch = "gfx950"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %idx = hk.coord %c0, %c0, %c0, %c0 : !hk.coord<element>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64, 2>
  %a0 = hk.shared_at %As[%c0] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64>
  %a1 = hk.shared_at %As[%c1] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64>
  hk.group_load %g[%idx] into %a0 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  scf.for %t = %c0 to %n step %c2 : i32 {
    hk.group_load %g[%idx] into %a1 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
    // CHECK: hk.group_wait_for {{.*}} {n = 2 : i64}
    hk.group_wait_for %a0 : !hk.st<bf16, 128x64>
    hk.barrier
    %r = hk.load_shared %a0 {lds_loads = 0 : i64, wait = true} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
    hk.barrier
    hk.group_load %g[%idx] into %a0 {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
    // CHECK: hk.group_wait_for {{.*}} {n = 2 : i64}
    hk.group_wait_for %a1 : !hk.st<bf16, 128x64>
    hk.barrier
    %q = hk.load_shared %a1 {lds_loads = 0 : i64, wait = true} : !hk.st<bf16, 128x64> -> !hk.rt<bf16, 128x64, row>
    hk.barrier
  }
  hk.group_wait_for %a0 : !hk.st<bf16, 128x64>
  return
}
