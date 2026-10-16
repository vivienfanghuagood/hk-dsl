// RUN: hk-opt %s -split-input-file -verify-diagnostics

func.func @both(%t: !hk.rt<f32, 16x64, row>) {
  // expected-error@+1 {{has both a destination and a result}}
  %r = hk.exp %t into %t : !hk.rt<f32, 16x64, row> into !hk.rt<f32, 16x64, row> -> !hk.rt<f32, 16x64, row>
  return
}

// -----

func.func @gelu_bf16(%t: !hk.rt<bf16, 16x64, row>) {
  // expected-error@+1 {{has no bf16 implementation in include/rdna3}}
  %r = hk.gelu %t : !hk.rt<bf16, 16x64, row> -> !hk.rt<bf16, 16x64, row>
  return
}

// -----

func.func @mma_layout(%a: !hk.rt<bf16, 16x16, col>, %b: !hk.rt<bf16, 16x16, row>, %c: !hk.rt<f32, 16x16, col>) {
  // expected-error@+1 {{operand layouts do not match this variant}}
  hk.mma_ABt %a, %b, %c into %c : !hk.rt<bf16, 16x16, col>, !hk.rt<bf16, 16x16, row>, !hk.rt<f32, 16x16, col> into !hk.rt<f32, 16x16, col>
  return
}

// -----

func.func @reduce_kind(%t: !hk.rt<f32, 16x64, row>) {
  // expected-error@+1 {{reducing}}
  %v = hk.row_sum %t : !hk.rt<f32, 16x64, row> -> !hk.rv<f32, 16x64, row, row>
  return
}

// -----

// expected-error@+1 {{register tile rows=8 must be a positive multiple of 16}}
func.func @bad_tile(%t: !hk.rt<f32, 8x64, row>) {
  return
}

// -----

func.func @async_too_small(%g: !hk.gl<bf16>, %i: !hk.coord<element>) {
  %s = hk.alloc_shared : !hk.st<bf16, 32x64>
  // expected-error@+1 {{async: the tile is not a whole number of 16-byte copies per thread}}
  hk.group_load %g[%i] into %s {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 32x64>
  return
}

// -----

func.func @group_load_into_stack(%g: !hk.gl<bf16>, %i: !hk.coord<element>) {
  %s = hk.alloc_shared : !hk.st<bf16, 128x64, 2>
  // expected-error@+1 {{copies into a stack of LDS tiles; index it}}
  hk.group_load %g[%i] into %s {threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64, 2>
  return
}

// -----

func.func @pingpong_unpaired(%p: i1) {
  // expected-error@+1 {{has no pingpong_end after it in its block}}
  hk.pingpong_begin %p
  return
}

// -----

func.func @pingpong_end_alone(%p: i1) {
  // expected-error@+1 {{has no pingpong_begin before it in its block}}
  hk.pingpong_end %p
  return
}

// -----

func.func @pingpong_two_conditions(%p: i1, %q: i1) {
  // expected-error@+1 {{pairs with a pingpong on another condition}}
  hk.pingpong_begin %p
  hk.pingpong_end %q
  return
}
