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
  %s = hk.alloc_shared : !hk.st<bf16, 16x16>
  // expected-error@+1 {{async: the tile is not a whole number of 16-byte copies per warp}}
  hk.group_load %g[%i] into %s {async, threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 16x16>
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

// -----

func.func @rows_on_a_waited_copy(%g: !hk.gl<bf16>, %i: !hk.coord<element>, %n: i32) {
  %s = hk.alloc_shared : !hk.st<bf16, 128x64>
  // expected-error@+1 {{rows: a row limit is the async copy's}}
  hk.group_load %g[%i] rows %n into %s {threads = 512 : i64} : !hk.gl<bf16>, !hk.coord<element> into !hk.st<bf16, 128x64>
  return
}

// -----

// expected-error@+1 {{the strided fragments are 32x32 stride 8, 32x16 stride 4 and 16x32 stride 4}}
func.func @unknown_stride(%t: !hk.rt<bf16, 64x32, col, frag 16x32 stride 2>) {
  return
}

// -----

func.func @operand_of_a_row_tile(%s: !hk.rt<f32, 64x32, row, frag 32x32>,
                                 %p: !hk.rt<bf16, 64x32, col, frag 16x32 stride 4>) {
  // expected-error@+1 {{reads a col-layout accumulator tile}}
  hk.as_operand %s into %p : !hk.rt<f32, 64x32, row, frag 32x32> into !hk.rt<bf16, 64x32, col, frag 16x32 stride 4>
  return
}

// -----

func.func @operand_in_the_wrong_fragment(%s: !hk.rt<f32, 64x32, col, frag 32x32>,
                                         %p: !hk.rt<bf16, 64x32, col, frag 16x32>) {
  // expected-error@+1 {{read as an operand is '!hk.rt<bf16, 64x32, col, frag 16x32 stride 4>'}}
  hk.as_operand %s into %p : !hk.rt<f32, 64x32, col, frag 32x32> into !hk.rt<bf16, 64x32, col, frag 16x32>
  return
}

// -----

func.func @subtile_past_the_tile(%s: !hk.rt<f32, 64x32, col, frag 32x32>) {
  // expected-error@+1 {{subtile 2 of 2}}
  %h = hk.rt_subtile %s[2] : !hk.rt<f32, 64x32, col, frag 32x32> -> !hk.rt<f32, 32x32, col, frag 32x32>
  return
}

// -----

func.func @subtile_that_does_not_tile(%s: !hk.rt<f32, 96x32, col, frag 32x32>) {
  // expected-error@+1 {{64 rows: a whole number of 32-row fragments that tiles the 96 rows}}
  %h = hk.rt_subtile %s[1] : !hk.rt<f32, 96x32, col, frag 32x32> -> !hk.rt<f32, 64x32, col, frag 32x32>
  return
}

// -----

func.func @vote_on_bf16(%m: !hk.rv<bf16, 64x32, col, row, frag 32x32>) {
  // expected-error@+1 {{compares two fp32 vectors of one type}}
  %g = hk.any_above %m, %m, 8.0 : !hk.rv<bf16, 64x32, col, row, frag 32x32>, !hk.rv<bf16, 64x32, col, row, frag 32x32>
  return
}

// -----

func.func @swap_halves_of_a_tile(%t: !hk.rt<f32, 16x64, row>) {
  // expected-error@+1 {{exchanges the halves of a strip's rows; it takes a strip}}
  %s = hk.swap_halves %t : !hk.rt<f32, 16x64, row> -> !hk.rt<f32, 16x64, row>
  return
}

// -----

func.func @rotate_half_four_lanes_apart(%s: !hk.rt<f32, 1x512, strip<8>>) {
  // expected-error@+1 {{puts its halves neither 8, 16 or 32 lanes nor whole chunks apart}}
  %h = hk.rotate_half %s {span = 64 : i64} : !hk.rt<f32, 1x512, strip<8>> -> !hk.rt<f32, 1x512, strip<8>>
  return
}

// -----

func.func @rotate_half_of_a_tile(%t: !hk.rt<f32, 16x64, row>) {
  // expected-error@+1 {{rotates the halves of a strip's segments; it takes a strip}}
  %h = hk.rotate_half %t {span = 64 : i64} : !hk.rt<f32, 16x64, row> -> !hk.rt<f32, 16x64, row>
  return
}

// -----

func.func @period_bounded(%g: !hk.gl<f32>, %idx: !hk.coord<element>) {
  // expected-error@+1 {{drop `bounded`}}
  %c = hk.load %g[%idx] {bounded, period = 64 : i64} : !hk.gl<f32>, !hk.coord<element> -> !hk.rt<f32, 1x512, strip<8>>
  return
}

// -----

func.func @period_of_part_of_a_run(%g: !hk.gl<f32>, %idx: !hk.coord<element>) {
  // expected-error@+1 {{is not a positive number of whole runs of 8}}
  %c = hk.load %g[%idx] {period = 4 : i64} : !hk.gl<f32>, !hk.coord<element> -> !hk.rt<f32, 1x512, strip<8>>
  return
}
