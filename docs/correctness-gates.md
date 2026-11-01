# The correctness gates: what hk-dsl refuses before a kernel runs

At the tile level, the mistakes that cost the most produce no diagnostic: the
kernel compiles, runs, and is wrong some of the time. hk-dsl's answer is a set
of gates, each either making a mistake impossible to write or refusing the
kernel before it runs -- at trace time, or at `k.build()`, which needs hipcc
but no GPU. This page is what each gate catches and where it lives; the
[runtime](runtime.md#the-resource-gates) and [IR](ir.md#passes) pages have the
details.

## The traps

1. **A wait carries no register dependence.** `s_waitcnt` does not tie the
   registers it waits for, so the compiler can hoist a matrix-core instruction
   above the wait that was supposed to guard its operand. The results are
   randomly wrong, nothing spills, nothing is reported. **The translation**
   binds the fragment with a volatile asm after every wait, so a kernel cannot
   forget to ([translation](ir.md#translation)). On an MI355X, LDS GEMMs built
   without the bind returned NaN.
2. **A register spill is silent corruption, not slowness**, because hk places
   its waits by hand and scratch traffic reorders against them. **The resource
   gate**: `k.build()` reads the compiler's resource report
   (`-Rpass-analysis=kernel-resource-usage`) for every instantiation and
   refuses to return a kernel that spills to memory, raising
   `hk.ResourceError`; a build that emits no report is refused too
   ([resource gates](runtime.md#the-resource-gates)).
3. **An operand in the wrong layout takes eight times the instructions.** On
   RDNA3 only row-layout 16-bit operands are read from LDS 16 bytes at a time;
   a column-layout one degrades to scalar reads. **The passes** flag it while
   tracing, not at benchmark time ([passes](ir.md#passes)).
4. **On CDNA4, a copy from global memory into LDS is still in flight at the
   barrier after it.** It counts on `vmcnt`, the barrier gets no wait from
   LLVM, and the reads after it are inline asm LLVM's wait insertion cannot
   see. **The translation** follows every such copy with its wait, and
   **`hk-lds-pipeline`** derives every wait and proves, by counting barriers,
   that no warp reads a buffer before every warp's share of the copy landed.

## The gates, by stage

| gate | stage | refuses or fixes |
|---|---|---|
| `hk-verify` | trace | an illegal op, an unsupported dtype, a workgroup operation inside a branch |
| `hk-lds-alloc` | trace | LDS over the target's limit; an allocation inside a runtime loop |
| `hk-lds-pipeline` | trace | a tile read before its asynchronous load was waited for; a copy that races its readers (gfx950) |
| `hk-mma-fence` | trace (CDNA) | puts a scheduling barrier between a matrix-core chain and the first other op on its accumulator |
| `hk-check-budgets` | trace | checks the register estimate against the occupancy the kernel asked for; warns on slow layouts |
| the bind after every wait | translate | closes trap 1 by construction |
| the resource gate | build | a memory spill, in any instantiation; `max_vgprs` / `min_occupancy` the kernel asked for; a build with no resource report |
| the MFMA hazard gate | build (CDNA) | an instruction too few wait states after the MFMA whose result it reads or overwrites (a compiler bug before llvm/llvm-project#225989): padded and checked again (`HK_MFMA_GATE=fix`), or refused (`refuse`) |
| `hk.check` | launch | a wrong answer: the output against a torch reference computed in fp32, next to torch's own error |

The resource, wait and MFMA-hazard gates also run on any AMDGPU code object,
hk's or not: `python3 -m hk.runtime.cocheck`
([cocheck](runtime.md#any-code-object-cocheck)).

## Arch facts as data

The architecture facts the gates and passes need -- register granules, LDS
sizes, wait-counter widths, which layouts are fast -- are data, not comments:
`python/hk/target/` for Python and `mlir/lib/Target/Target.cpp` for the
passes, with a test that the two agree field by field. Two that matter: on
gfx1100 the register granule is 24, so occupancy 6 needs at most 240
registers, not 256; on gfx9 `lgkmcnt` has 4 bits, so a derived wait above 15
is emitted as the cap ([target facts](ir.md#target-facts),
[supported hardware](supported-hardware.md)).

An op the library does not implement for a dtype is caught the same way:
`gelu` on a bf16 tile raises while tracing rather than failing at the end of a
30-second compile.

## What the gates do not catch

A kernel that computes the wrong function, correctly. That is what
[`hk.check`](check.md) is for: run it after every edit, against a torch
reference, at the shapes you ship.
