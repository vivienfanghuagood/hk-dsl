"""The V0 / V1 evidence for a change to the DSL surface (docs/design/dsl-surface.md §7).

Run in two checkouts -- the merge base and the change -- each with its own hk._mlir, for the
same HK_ARCH, and diff:

    python3 tools/surface/snapshot.py sources > base.jsonl     # V0: no hipcc, no GPU
    python3 tools/surface/snapshot.py builds  > base.jsonl     # V1: hipcc, no GPU
    python3 tools/surface/snapshot.py diff base.jsonl change.jsonl

`sources` traces every kernel of the catalog below and hashes what hk generates for it: every
scaffold's C++ and the IR dump. Byte-identical sources compiled against identical headers are
identical code objects (V0). `builds` also compiles each one and records, per function of its
code object, the resources (VGPR, AGPR, SGPR, LDS, scratch, spills, occupancy) and a hash of the
disassembled instructions (V1: the ISA is identical). `--only REGEX` picks kernels by name; `-j`
compiles in parallel.

The catalog: every example (examples/_catalog), the row-kernel tables of hk.ops, and the
kernels the shipped routes build for the target -- on gfx950 K1 (moe_a8w4: the vLLM route's
stage 1 with the MX epilogue and stage 2 resident, at DeepSeek-V4.1's TP1 and TP2 shapes), K2
(mxfp8: the four-warp schedules of the route's ranges, the eight-warp kernel, the tail, quant),
gemm_cdna's candidates, attention (dense, causal, paged prefill, its split and merge), the paged
decode (plain, split, fused at 1-512 splits, two pages a step, merges) and int4; on gfx1100 /
gfx1201 attention and K3, the paged decode, the MoE GEMM, int4, the GEMMs and the decode GEMVs.
A kernel a target refuses is a line too (its error), so a change in what is refused shows.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _catalog(arch: str):
    """(name, factory) for every kernel to snapshot; a factory returns an hk Kernel."""
    sys.path.insert(0, str(REPO / "examples"))
    import _catalog as ex  # noqa: PLC0415

    jobs = [(f"example.{n}", (lambda k=k: k)) for n, k in ex.kernels()]
    for mod in ("norm", "fused", "quant", "elementwise", "attn", "sdpa", "paged"):
        m = importlib.import_module(f"hk.ops.{mod}")
        for table in ("KERNELS", "STRIP_KERNELS", "ROTARY_KERNELS", "ROTARY16_KERNELS"):
            for n, k in sorted(getattr(m, table, {}).items(), key=lambda kv: str(kv[0])):
                jobs.append((f"{mod}.{table}.{n}", (lambda k=k: k)))
    from hk.ops import decode_gemm, gemm  # noqa: PLC0415

    jobs.append(("gemm.default", lambda: gemm.TUNER._kernel_for(gemm.DEFAULT)))
    jobs.append(("gemm.thin", lambda: gemm.TUNER._kernel_for(gemm.THIN)))
    for t, s in sorted(decode_gemm.SCHEDULES.items()):
        jobs.append((f"decode_gemm.t{t}", lambda t=t, s=s: decode_gemm.kernel_for(tokens=t, **s)))
    from hk.ops import w4a16  # noqa: PLC0415

    for t, s in sorted(w4a16.SCHEDULES.items()):
        for g in (32, 64, 128):
            jobs.append((f"w4a16.t{t}.g{g}", lambda t=t, s=s, g=g: w4a16.kernel_for(t, group=g, **s)))
    if arch == "gfx950":
        jobs += _gfx950()
    else:
        jobs += _rdna()
    return jobs


def _gfx950():
    from hk.ops import attn_cdna, gemm_cdna, gemm_cdna_dp, moe_a8w4, mxfp8, paged_cdna  # noqa: PLC0415

    jobs = []
    for n, (kind, sched) in sorted(gemm_cdna.CANDIDATES.items()):
        if kind == "pp":
            jobs.append((f"gemm_cdna.{n}", lambda n=n, s=sched: gemm_cdna.gemm_kernel(
                "gemm_cdna_" + n, **{**gemm_cdna.DEFAULT, **s})))
        elif kind == "dp":
            jobs.append((f"gemm_cdna.{n}", lambda n=n, s=sched: gemm_cdna_dp.gemm_kernel(
                "gemm_cdna_" + n, **{**gemm_cdna_dp.DEFAULT, **s})))
    # K1, the vLLM route (hk.integration.vllm_moe_a8w4.moe): DeepSeek-V4.1 at TP2 (inter 1152,
    # stage 2 resident) and TP1 (inter 2304), hidden 5120, top-6, swiglu limit 10, MT 4
    for inter, res in ((1152, 5), (2304, 0)):
        jobs.append((f"moe_a8w4.s1.i{inter}", lambda inter=inter: moe_a8w4._sorted(
            1, 2 * inter, 4, 3, 5120, 10.0, 6, 2, 4, mxq=True, xpipe=True)))
        if res:
            jobs.append((f"moe_a8w4.s2r.i{inter}", lambda inter=inter, res=res:
                         moe_a8w4._sorted_resident(5120, 4, 4, inter, 6, res)))
        else:
            jobs.append((f"moe_a8w4.s2.i{inter}", lambda inter=inter: moe_a8w4._sorted(
                2, 5120, 4, 4, inter, 10.0, 6, 2, 4)))
    # K1's other forms, kept and tested (hk.ops.moe_a8w4): each stage-1 / stage-2 kernel family
    # at DeepSeek-V4.1's TP2 shapes, and the top-k reduce
    jobs += [
        ("moe_a8w4.s1.v1", lambda: moe_a8w4.stage1_kernel("moe_s1v1", group_rows=2304)),
        ("moe_a8w4.s1.lds", lambda: moe_a8w4.stage1_lds_kernel("moe_s1lds", group_rows=2304)),
        ("moe_a8w4.s1.ring", lambda: moe_a8w4.stage1_ring_kernel("moe_s1ring", group_rows=2304)),
        ("moe_a8w4.s1.dma", lambda: moe_a8w4._stage1_dma(2304, 4, 4, 2, 10.0)),
        ("moe_a8w4.s1.direct", lambda: moe_a8w4.stage1_direct_kernel("moe_s1dir", group_rows=2304,
                                                                     k=5120)),
        ("moe_a8w4.s1.grid", lambda: moe_a8w4._sorted_grid(2304, 4, 4, 5120, 10.0, 2, 2)),
        ("moe_a8w4.s1.sorted", lambda: moe_a8w4._sorted(1, 2304, 4, 3, 5120, 10.0, 6)),
        ("moe_a8w4.s1.mxq", lambda: moe_a8w4._sorted(1, 2304, 4, 3, 5120, 10.0, 6, mxq=True)),
        ("moe_a8w4.s2.v1", lambda: moe_a8w4.stage2_kernel("moe_s2v1", group_rows=5120)),
        ("moe_a8w4.s2.persistent", lambda: moe_a8w4._sorted_persistent(5120, 4, 4, 1152, 6, 768)),
        ("moe_a8w4.reduce", lambda: moe_a8w4._reduce(128, 6)),
    ]
    jobs.append(("mxfp8.4p", lambda: mxfp8.mx4p_kernel()))
    # the DeepSeek-V4.1 index front (hk.integration.vllm_dsv41_index): softmax_scale * n_head^-0.5
    from hk.ops import dsv41_index  # noqa: PLC0415

    jobs.append(("dsv41_index", lambda: dsv41_index.index_kernel(128 ** -0.5 * 64 ** -0.5)))
    # K2, the vLLM route (hk.integration.vllm_mxfp8): the four-warp schedules of RANGES and
    # _SMALL_N, the tail's, the eight-warp kernel (gemm_8w) and the activation quant
    four = [dict(block_m=128, block_n=128, split_m=1, split_n=1, stages=3),
            dict(block_m=128, block_n=128, split_m=1, split_n=1, stages=2),
            dict(block_m=128, split_m=1, stages=3),
            dict(block_m=128, block_n=128, split_m=1, split_n=1, stages=3, rolled=True)]
    for s in four:
        name = "mxfp8_4w" + "".join(f"_{k}{v}" for k, v in sorted(s.items()))
        jobs.append((f"mxfp8.{name}", lambda name=name, s=s: mxfp8.mx4w_kernel(name, **s)))
    jobs.append(("mxfp8.8w", lambda: gemm_cdna.gemm_kernel(
        "mxfp8_pp_256", mx=True, k_step=128, block_n=256, arch="gfx950", mx_scales="direct")))
    jobs.append(("mxfp8.gemm_128", lambda: mxfp8.mxfp8_kernel()))
    for packed in (True, False):
        jobs.append((f"mxfp8.quant.p{int(packed)}", lambda p=packed: mxfp8.quant_kernel(p)))
    # attention and the paged prefill (hk.ops.sdpa / attn on gfx950 -> attn_cdna)
    for d in attn_cdna.HEAD_DIMS:
        for causal in (False, True):
            jobs.append((f"attn_cdna.d{d}.c{int(causal)}",
                         lambda d=d, c=causal: attn_cdna.kernel_for(d, c)))
        jobs.append((f"attn_cdna.paged.d{d}", lambda d=d: attn_cdna.paged_kernel_for(d)))
        jobs.append((f"attn_cdna.split.d{d}", lambda d=d: attn_cdna.split_kernels_for(d)[0]))
        jobs.append((f"attn_cdna.merge.d{d}", lambda d=d: attn_cdna.split_kernels_for(d)[1]))
        # the paged decode (hk.ops.paged on gfx950 -> paged_cdna)
        jobs.append((f"paged_cdna.plain.d{d}", lambda d=d: paged_cdna.kernel_for(d, 1, "plain")))
        for s in (2, 4, 8, 16, 32, 64, 128, 256, 512):
            jobs.append((f"paged_cdna.split.d{d}.s{s}",
                         lambda d=d, s=s: paged_cdna.kernel_for(d, s, "split")))
            for pp in (1, 2):
                jobs.append((f"paged_cdna.fused.d{d}.s{s}.pp{pp}", lambda d=d, s=s, pp=pp:
                             paged_cdna.kernel_for(d, s, "fused",
                                                   **({"ppstep": pp} if pp != 1 else {}))))
            jobs.append((f"paged_cdna.merge.d{d}.s{s}", lambda d=d, s=s: paged_cdna.merge_for(d, s)))
    return jobs


def _rdna():
    from hk.ops import attn_k3, moe, paged, paged_prefill  # noqa: PLC0415

    jobs = [("paged.decode", lambda: paged.paged_decode_kernel("rd_paged_decode")),
            ("paged.split", lambda: paged.paged_decode_split_kernel("rd_paged_split"))]
    for d in (64, 128):
        for s in (2, 8, 32):
            jobs.append((f"paged.fused.d{d}.s{s}", lambda d=d, s=s: paged.fused_kernel(d, s)))
        for g in (1, 4, 8):
            jobs.append((f"paged_prefill.d{d}.g{g}",
                         lambda d=d, g=g: paged_prefill.kernel_for(d, g)))
    for part in ("vt", "attn", "combine"):
        jobs.append((f"attn_k3.{part}", lambda part=part: attn_k3._kernels()[part]))
    for bm in (16, 32, 64):
        for topk in (1, 8):
            for routed in (False, True):
                s = moe.schedule_for(bm, 2048)
                if s:
                    jobs.append((f"moe.m{bm}.k{topk}.r{int(routed)}",
                                 lambda bm=bm, topk=topk, r=routed, s=s:
                                 moe.kernel_for(bm, topk, r, **s)))
    return jobs


def _trace_hash(kernel):
    from hk.codegen import scaffold  # noqa: PLC0415

    tr = kernel.trace()
    parts = [kernel.with_defines(scaffold.render(tr, sc)) for sc in ("pybind", "bare", "device")]
    try:
        parts.append(kernel.with_defines(scaffold.render(tr, "torch")))
    except ValueError:
        pass
    parts.append(tr.dump())
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def sources(only):
    from hk.target import current_arch  # noqa: PLC0415

    arch = current_arch()
    for name, make in _catalog(arch):
        if only and not re.search(only, name):
            continue
        try:
            row = {"kernel": name, "src": _trace_hash(make())}
        except Exception as e:  # noqa: BLE001 -- a refusal is a line to diff too
            row = {"kernel": name, "refused": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}
        print(json.dumps(row), flush=True)


def _isa(so_path):
    from hk.runtime import cocheck  # noqa: PLC0415

    d = tempfile.mkdtemp(prefix="hk-surface-isa-")
    try:
        out = {}
        for co in cocheck.unbundle(so_path, Path(d)):
            for fn, ins in cocheck.disassemble(co).items():
                txt = "\n".join(i.text for i in ins)
                out[fn] = [hashlib.sha256(txt.encode()).hexdigest()[:16], len(ins)]
        return out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _build_one(name, make):
    try:
        k = make()
        src = _trace_hash(k)
        b = k.build()
        res = {kk.name: {"vgpr": kk.vgpr, "agpr": kk.agpr, "sgpr": kk.sgpr, "lds": kk.lds,
                         "scratch": kk.scratch, "vgpr_spill": kk.vgpr_spill,
                         "sgpr_spill": kk.sgpr_spill, "occupancy": kk.occupancy}
               for kk in b.kernels}
        return {"kernel": name, "src": src, "resources": res, "isa": _isa(b.so_path)}
    except Exception as e:  # noqa: BLE001
        return {"kernel": name, "refused": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}


_TODO: list = []


def _init(only):
    sys.path.insert(0, str(REPO / "python"))
    from hk.target import current_arch  # noqa: PLC0415

    _TODO[:] = [(n, m) for n, m in _catalog(current_arch()) if not only or re.search(only, n)]


def _build_index(i):
    return _build_one(*_TODO[i])


def builds(only, jobs):
    """Each kernel traced and compiled in a worker process: tracing is not thread-safe (the
    builder's state is per process), and a forked worker can inherit a lock held by one of the
    parent's threads, so the workers are spawned and build the catalog themselves."""
    import multiprocessing as mp  # noqa: PLC0415

    _init(only)
    n = len(_TODO)
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"),
                                initializer=_init, initargs=(only,)) as pool:
        for row in pool.map(_build_index, range(n)):
            print(json.dumps(row), flush=True)


def diff(a, b):
    def load(p):
        return {r["kernel"]: r for r in map(json.loads, Path(p).read_text().splitlines()) if r}

    old, new = load(a), load(b)
    bad = 0
    for name in sorted(set(old) | set(new)):
        o, n = old.get(name), new.get(name)
        if o is None or n is None:
            print(f"{'only in ' + (a if n is None else b)}: {name}")
            bad += 1
            continue
        for key in ("refused", "src", "resources", "isa"):
            if o.get(key) != n.get(key):
                print(f"{name}: {key} differs")
                bad += 1
    same = len(set(old) & set(new))
    print(f"{same} kernels in both; {bad} differences")
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sources")
    s.add_argument("--only")
    b = sub.add_parser("builds")
    b.add_argument("--only")
    b.add_argument("-j", type=int, default=min(16, os.cpu_count() or 1))
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = p.parse_args(argv)
    sys.path.insert(0, str(REPO / "python"))
    if args.cmd == "sources":
        sources(args.only)
    elif args.cmd == "builds":
        builds(args.only, args.j)
    else:
        return diff(args.a, args.b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
