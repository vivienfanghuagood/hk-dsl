# Tiles, strips and LDS: the data a kernel holds

An hk-dsl kernel holds its data in four kinds of values. Which one a kernel
uses decides how its lanes touch memory, so it is the first choice to make:
**register strips** for kernels that stream rows (elementwise, norms,
softmax, quantization), **register tiles staged through LDS** for what feeds
the matrix core (GEMM, attention). The templates make that choice for you;
the [language](language.md#types) has every detail.

| value | built with | lives in | for |
|---|---|---|---|
| register tile | `hk.RT(dtype, rows, cols, layout)` | each warp's registers | matrix-core operands and accumulators |
| register strip | `hk.RS(dtype, rows, cols, run=8)` | each warp's registers | streaming rows: one op an element, reductions along a row |
| LDS tile | `hk.ST`, from `hk.lds.alloc` | the workgroup's LDS | staging operands that several warps read |
| vector | `hk.col_vec(t)`, `hk.row_vec(t)`, `hk.SV` | registers, or LDS | what a reduction gives, and what scales a row or a column |

## Register tiles: the matrix core's layout

A register tile is laid out as the matrix core reads it: one warp holds the
tile, and each lane holds pieces of several rows. On RDNA a 16-bit row tile
holds each row twice and an fp32 tile gives a lane alternate columns, which is
what WMMA wants and what a streaming kernel does not. On gfx950 an operand's
fragment is 16x32 for bf16 rather than 16x16; the `hk-fragments` pass names it
in the tile's type, so the same kernel source builds for both
([IR types](ir.md#types)).

The layout of an operand matters for speed: on RDNA3 only row-layout 16-bit
operands are read from LDS 16 bytes at a time, and a column-layout one takes
eight times the instructions. The passes warn about it while tracing
([the gates](correctness-gates.md)).

## Register strips: a row, run by run

A strip gives each lane contiguous runs of a row instead: lane `l` owns, of
every row, the runs of `run` columns starting at `j*wave*run + l*run`. A
chunk is the wave's lanes times the run -- 32 x 8 on gfx1100 and gfx1201,
64 x 8 on gfx950 -- and a strip's columns are whole chunks. A bf16 run of 8
is one 16-byte access, and nothing is held twice.

- It loads and stores by element (`x.at[r, c]`), converting the element type
  on the way; `bounded=True` reads zero past the tensor's last row and column
  and writes nothing there.
- It maps, reduces along a row (the answer lands in every lane) and along
  columns, and broadcasts its vectors.
- Its elements are bf16, fp16, fp32 or i32. It is never a matrix-core operand
  or an LDS tile.
- It exists on gfx1100, gfx1201 and gfx950; gfx942 refuses it while tracing
  ([supported hardware](supported-hardware.md#what-the-language-has-on-each)).

Elementwise and row-wise kernels belong on strips: the same math on 16x64
register tiles leaves bandwidth unused. `hk.templates.elementwise` and
`rowwise` build strip kernels from the math of one element or one row
([templates](templates/elementwise-rowwise.md)).

## LDS: staging for the matrix core

Operands that several warps read -- a GEMM's panels, attention's K and V --
are copied into LDS tiles (`hk.lds.alloc`) and read from there into register
tiles. The passes do the bookkeeping C++ leaves to the author:

- **`hk-lds-alloc`** places every allocation at an offset, against the
  target's LDS limit; the two arms of a runtime `if` share their space, and an
  allocation inside a runtime loop is refused (allocate it before the loop).
- **`hk-lds-pipeline`** derives every LDS wait, and proves that every tile
  loaded asynchronously is waited for before it is read; on gfx950 it also
  derives the global-to-LDS copies' waits and checks, by counting barriers,
  that no warp reads a buffer before every warp's share of the copy landed.

On gfx950 an LDS tile's C++ layout is chosen by how it is read, per
allocation ([translation](ir.md#translation)).

## Choosing

- Elementwise, a norm, a softmax, a quantizer: a template, or a strip kernel
  of your own.
- A GEMM with something done to its output: `hk.templates.gemm` with an
  epilogue ([GEMM template](templates/gemm.md)).
- Attention, a GEMM schedule of your own: register tiles through LDS; the
  examples and `hk.ops`' kernels are the models ([examples](../examples/README.md),
  [API](api/hk.ops.md)).
