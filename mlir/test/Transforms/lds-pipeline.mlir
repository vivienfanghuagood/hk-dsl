// RUN: hk-opt %s -split-input-file --pass-pipeline='builtin.module(func.func(hk-lds-pipeline))' -verify-diagnostics | FileCheck %s

// Two async reads in flight; retiring the first leaves the second's two
// ds_read_b128 outstanding, so the wait is lgkmcnt(2). Retiring the second
// then waits for nothing behind it.

// CHECK-LABEL: func.func @counts
func.func @counts() attributes {hk.kernel} {
  %s = hk.alloc_shared : !hk.st<bf16, 16x16>
  %a = hk.load_shared %s {lds_loads = 2 : i64, wait = false} : !hk.st<bf16, 16x16> -> !hk.rt<bf16, 16x16, row>
  %b = hk.load_shared %s {lds_loads = 2 : i64, wait = false} : !hk.st<bf16, 16x16> -> !hk.rt<bf16, 16x16, row>
  // CHECK: hk.lds_wait_for {{.*}} {n = 2 : i64}
  hk.lds_wait_for %a {} : !hk.rt<bf16, 16x16, row>
  // CHECK: hk.lds_wait_for {{.*}} {n = 0 : i64}
  hk.lds_wait_for %b {} : !hk.rt<bf16, 16x16, row>
  return
}

// -----

func.func @unbound_read() attributes {hk.kernel} {
  %s = hk.alloc_shared : !hk.st<bf16, 16x16>
  %a = hk.load_shared %s {lds_loads = 2 : i64, wait = false} : !hk.st<bf16, 16x16> -> !hk.rt<bf16, 16x16, row>
  // expected-error@+1 {{exp reads , whose LDS reads are still in flight}}
  %e = hk.exp %a : !hk.rt<bf16, 16x16, row> -> !hk.rt<bf16, 16x16, row>
  hk.lds_wait_for %a : !hk.rt<bf16, 16x16, row>
  return
}

// -----

// A vector store is ordinary ds_writes: nothing retires them before an
// s_barrier, and a warp past it reads what was there before. Found in the
// cross-warp combine of the shipped norms, where it corrupted rows whenever
// enough workgroups shared a WGP.

func.func @plain_store_then_barrier(%v: !hk.rv<f32, 16x64, row, col>) attributes {hk.kernel} {
  %i = arith.constant 0 : i32
  %s = hk.alloc_shared_vec : !hk.sv<f32, 16, 4>
  hk.store_shared_vec %v, %s[%i] : !hk.rv<f32, 16x64, row, col>, !hk.sv<f32, 16, 4>
  // expected-error@+1 {{barrier after store_shared_vec with nothing waiting for its ds_writes}}
  hk.barrier
  return
}

// -----

// CHECK-LABEL: func.func @plain_store_then_drain
func.func @plain_store_then_drain(%v: !hk.rv<f32, 16x64, row, col>) attributes {hk.kernel} {
  %i = arith.constant 0 : i32
  %s = hk.alloc_shared_vec : !hk.sv<f32, 16, 4>
  hk.store_shared_vec %v, %s[%i] : !hk.rv<f32, 16x64, row, col>, !hk.sv<f32, 16, 4>
  hk.barrier {drain}
  hk.barrier
  return
}

// -----

// An fp32 tile cannot take the vectorised path, so it is ordinary stores too.

func.func @f32_tile_store_then_barrier(%t: !hk.rt<f32, 16x16, row>) attributes {hk.kernel} {
  %s = hk.alloc_shared : !hk.st<f32, 16x16>
  hk.store_shared %t, %s : !hk.rt<f32, 16x16, row>, !hk.st<f32, 16x16>
  // expected-error@+1 {{barrier after store_shared}}
  hk.barrier
  return
}

// -----

// A row-layout bf16 tile into a bf16 LDS tile is the vectorised path, which
// ends in lgkmcnt(0) of its own.

// CHECK-LABEL: func.func @bf16_tile_store_then_barrier
func.func @bf16_tile_store_then_barrier(%t: !hk.rt<bf16, 16x16, row>) attributes {hk.kernel} {
  %s = hk.alloc_shared : !hk.st<bf16, 16x16>
  hk.store_shared %t, %s : !hk.rt<bf16, 16x16, row>, !hk.st<bf16, 16x16>
  hk.barrier
  return
}
