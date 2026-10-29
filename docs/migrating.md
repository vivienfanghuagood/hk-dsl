# Migrating to the converged DSL surface

The DSL-surface migration ([design](design/dsl-surface.md#5-migration-path)) folds kernel-specific
names into general ones. Each old spelling keeps working as a shim that warns
(`DeprecationWarning`) and traces to exactly the IR of the new one. A kernel moved to the new
spelling is the same kernel -- **as long as its ops are emitted in the same order**; see
[the order rule](#the-order-rule-bind-inline-loads-first) below.

**When the old spellings go.** One policy for every name below: a deprecated name is removed in
the release after the last call to it from outside the repository is migrated, and at most two
minor releases after the release that deprecated it. Until then it works and warns.

| old (deprecated) | new | step |
|---|---|---|
| `t.make_causal(v)` (`transposed=True`) | `tril_(0, v)` (`triu_(0, v)`) on a copy of `t` | S1 |
| `hk.mma_ABt_mx(a, b, acc, sa, sb, a_fmt=, b_fmt=, sa_at=, sb_at=, into=)` | `hk.mma("mma_ABt", a.scaled(sa, fmt=, at=), b.scaled(sb, fmt=, at=), acc, into=)` | S2 |
| `hk.copy_mx_scales(dst, src)` | `src.copy(into=dst)` | S2 |
| `hk.load_fp4_frags_lds(ty, lds, r, c)` | `hk.load_fp4_frags(ty, lds, r, c)` | S2 |
| `x.gather(ty, ids, at, div=k)` | `x.gather(ty, hk.RowIndex(ids, at, div=k))` | S3 |
| `out.scatter(t, ids, at, sort_topk=k)` | `out.scatter(t, hk.RowIndex(ids, at, sort_topk=k))` | S3 |
| `g.load(dst, src, wait=False, gather=(ids, at)[, rows=n])` | `g.load(dst, src, wait=False, gather=hk.RowIndex(ids, at)[, rows=n])` | S3 |
| `g.gather_prep(ids, at, rows, raw=r)` (`g` a group) | `ix = g.stage_rows(hk.RowIndex(ids, at), rows, keep_choice=r)` | S3 |
| `staged=N` (on `g.load` / `out.scatter`) | pass the index `g.stage_rows` returned (`ix.staged(N)` names it ahead of a loop body) | S3 |
| `hk.fp8_operand(w)` | `w.as_fragment(hk.RT(hk.fp8e4m3, 16, 128))` | S5 |
| `hk.fp4_operand(w)` | `w.as_fragment(hk.RT(hk.fp8e4m3, 16, 128), fmt="fp4")` | S5 |
| `hk.mx_scales_loose(b)` | `b.as_scales()` | S5 |

`x.gather(ty, ids, at)` / `out.scatter(t, ids, at)` with plain ids are not deprecated.

Some steps add a general form and keep the old name as permanent sugar, with no warning:
`Tile.rotate_half(span)` is `swap_halves(span, negate_first=True)` (S4), and `hk.iota` stays.

## The rewriter

```bash
python3 tools/surface/migrate.py FILE.py...            # rewrite in place
python3 tools/surface/migrate.py --check FILE.py...    # report only; exit 1 if anything would change
```

It handles S2's three names (`mma_ABt_mx`, `copy_mx_scales`, `load_fp4_frags_lds`) and nothing
else: migrate the S1, S3 and S5 spellings by hand from the table. It does not re-wrap long lines,
and it prints every call it leaves for a person (`left as it is: ...`).

## The order rule: bind inline loads first

A kernel is its trace, and the trace is the order the DSL emits ops in -- Python's evaluation
order. `hk.mma_ABt_mx(a, b, acc, sa, sb)` evaluates `a, b, acc, sa, sb`; the new spelling
`hk.mma("mma_ABt", a.scaled(sa), b.scaled(sb), acc)` evaluates `a, sa, b, sb, acc`. When the
arguments are names, nothing moves. When two or more of them emit ops -- an inline load, a
`hk.load_mx_scales(...)`, a `hk.zeros(...)` -- writing the new spelling in place issues them in
another order. That is another kernel: different generated source, and on gfx950 a different load
issue order, which is performance (the `vmcnt` waits retire in issue order).

```python
# old: the loads issue weights, x, the weights' scales, x's scales
hk.mma_ABt_mx(w.at[r, k].load(T_W), x.at[0, k].load(T_W), acc,
              hk.load_mx_scales(ws, 16, 4, r, k // 32), hk.load_mx_scales(xs, 16, 4, 0, k // 32), into=acc)

# WRONG: weights, the weights' scales, x, x's scales
hk.mma("mma_ABt", w.at[r, k].load(T_W).scaled(hk.load_mx_scales(ws, 16, 4, r, k // 32)),
       x.at[0, k].load(T_W).scaled(hk.load_mx_scales(xs, 16, 4, 0, k // 32)), acc, into=acc)

# right: bind them to locals in the old order, then the new spelling
w_ = w.at[r, k].load(T_W)
x_ = x.at[0, k].load(T_W)
ws_ = hk.load_mx_scales(ws, 16, 4, r, k // 32)
xs_ = hk.load_mx_scales(xs, 16, 4, 0, k // 32)
hk.mma("mma_ABt", w_.scaled(ws_), x_.scaled(xs_), acc, into=acc)
```

`hk.copy_mx_scales(dst, src)` has the same trap (`dst` first; `src.copy(into=dst)` evaluates
`src` first).

What the rewriter does about it:

- It binds the arguments that may emit ops to locals, in the old order, whenever the new spelling
  would evaluate them in another relative order -- and only then.
- A name, a constant, an attribute of one and a subscript of a name by names, constants and their
  arithmetic (`ws[k]`, `S[u % 2]`) are taken not to emit: the subscript is read as indexing a
  Python list. Indexing a DSL value does emit (an LDS stack `S[i]`, a global); for a file that
  passes such an argument, run with `--strict`, which treats everything but a name or a constant
  as emitting.
- A call nested in a larger expression (`o.store(hk.mma_ABt_mx(...))`) cannot be reordered
  without moving what is evaluated before it; it is left as it is and reported. Bind its
  arguments by hand.
- The shims need nothing: an old-spelling call still evaluates its arguments in the old order.

## Checking a migrated kernel

Trace it before and after and compare: `kernel.source()` (the generated C++) must be
byte-identical, and so must `kernel.trace().dump()`. For the shipped kernels,
`tools/surface/snapshot.py sources` hashes both at every route-table configuration
([testing](testing.md)). `tests/hk/ir/test_surface_migrate.py` holds the rewriter to this,
including a kernel with inline loads in every argument.
