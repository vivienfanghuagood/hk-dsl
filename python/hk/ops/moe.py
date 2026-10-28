"""The GEMM of a mixture-of-experts layer, on vLLM's sorted token lists.

vLLM's fused MoE sorts the (token, choice) pairs of a batch by expert
(`moe_align_block_size`): `sorted_ids` lists them as `token * top_k + choice`,
each expert's run padded to a multiple of BLOCK_M with the sentinel
`tokens * top_k`, and `expert_ids[b]` names the expert of block b. Its GEMM is

    C[sorted_ids[i]] = A[sorted_ids[i] // top_k] @ W[expert_ids[i // BLOCK_M]].T
                       (* topk_weights[sorted_ids[i]] for the second GEMM)

and this is that GEMM, laid out as `decode_gemm` is, because at the batches
a server decodes the whole cost is streaming each chosen expert's weights:

  * **W goes straight to registers**, its rows on the WMMA's 16-row side;
    the accumulator is C^T, and its transpose is a row-layout tile whose rows
    are tokens -- which is what lets the store scatter them.
  * **One 16-row block of one expert's W per workgroup, K split across the
    warps**, interleaved, the partial sums meeting through LDS at the end.
  * **A's rows come by index**: `A.gather(ty, hk.RowIndex(ids, at, div=top_k))`. A sentinel
    lands on row `tokens`, past A's end, and reads zeros; the scatter drops it.

A block past `num_tokens_post_padded` runs its K loop zero times rather than
branching around the kernel: the combine has barriers, which every warp must
reach, and a zero-trip loop reads no weights.

With few tokens against many experts (tokens * top_k * 4 <= experts) vLLM
skips the sort: block b is pair b alone, of expert `topk_ids[b]`. `naive`
kernels take that: the index list is 0..pairs-1, and block b gathers from
position b with the list ending at b + 1, so its other rows read zeros and
store nothing.
"""

import functools
from typing import Any, Dict, Optional, Tuple

from .. import dsl as hk
from ..types import bf16, fp32, i32


def moe_gemm_kernel(name: str, *, block_m: int = 16, top_k: int = 1,
                    routed: bool = False, slices: int = 4, warps: int = 16,
                    naive: bool = False):
    """`block_m` token rows per block (vLLM's BLOCK_SIZE_M, a multiple of 16),
    `top_k` index values per A row, `routed` multiplies each row by its
    routing weight, `slices` 16-deep K slices per load, `warps` waves sharing
    each 16-row block of W, `naive` one pair per block (see the module
    docstring). K must be a multiple of 16 * slices * warps."""
    if block_m % 16:
        raise ValueError(f"block_m={block_m}: token rows come in 16-row tiles")
    kc = 16 * slices
    ty_w = hk.RT(bf16, 16, kc, "row")
    ty_x = hk.RT(bf16, block_m, kc, "row")
    ty_acc = hk.RT(fp32, 16, block_m, "col")     # C^T: W rows x tokens
    ty_out = hk.RT(fp32, block_m, 16, "row")     # its transpose: tokens x W rows
    st_part = hk.ST(fp32, block_m, 16)
    in_flight = max(2, 8 // (block_m // 16))

    def body(a: hk.Tensor[bf16], w: hk.Tensor[bf16], c: hk.Tensor[bf16],
             sorted_ids: hk.Tensor[i32], expert_ids: hk.Tensor[i32],
             ntp: hk.Tensor[i32], topk_w: hk.Tensor[fp32]):
        n0 = hk.arch.block_idx.x * 16
        blk = hk.arch.block_idx.y
        wid = hk.arch.warp_id()
        at = blk if naive else blk * block_m
        count = blk + 1 if naive else sorted_ids.cols
        e = expert_ids.at[0, blk].load_scalar()
        # 1 for a block of real rows, 0 past the padded end or for an expert
        # this rank does not hold (-1): its loop then reads nothing.
        live = hk.min(hk.max(ntp.at[0, 0].load_scalar() - blk * block_m, 0), 1) * \
            hk.min(hk.max(e + 1, 0), 1)
        acc = hk.zeros(ty_acc)
        for t in hk.range(live * (a.cols // (kc * warps))):
            k = (t * warps + wid) * kc
            w_t = w.at[e, n0, k].load(ty_w)
            x_t = a.gather(ty_x, hk.RowIndex(sorted_ids, at, count=count, div=top_k), col=k)
            acc += w_t @ x_t.T

        out = hk.transpose(acc)
        if warps > 1:
            parts = hk.lds.alloc(st_part, count=warps, name="parts")
            parts[wid].store(out)
            hk.arch.barrier(drain=True)
        if wid == 0:
            if warps > 1:
                for i in range(1, warps):
                    # A partial is 8 VGPRs per 16 tokens, and all fifteen
                    # read at once spill from 32 tokens up: batch them.
                    if i > 1 and (i - 1) % in_flight == 0:
                        hk.arch.sched_barrier()
                    out += parts[i].load(ty_out)
            if routed:
                # A row's weight is element 0 of its row of the (pairs, 1)
                # view; the rest of the gathered tile is its neighbours'.
                rw = topk_w.gather(ty_out, sorted_ids, at, count=count)
                rw.right_fill_(1, 0.0)
                out *= rw.sum(axis=1)[:, None]
            c.scatter(out, sorted_ids, at, count=count, col=n0)

    body.__name__ = name
    if naive:
        grid = lambda p: (p.w.rows // 16, p.sorted_ids.cols, 1)  # noqa: E731
    else:
        grid = lambda p: (p.w.rows // 16, hk.cdiv(p.sorted_ids.cols, block_m), 1)  # noqa: E731
    return hk.kernel(body, warps=warps, name=name, grid=grid)


_KERNELS: Dict[Tuple[Any, ...], Any] = {}
_OPS: Dict[Tuple[Any, ...], Any] = {}
_ARANGE: Dict[Tuple[Any, ...], Any] = {}


def kernel_for(block_m: int, top_k: int, routed: bool, slices: int = 4,
               warps: int = 16, naive: bool = False):
    key = (block_m, top_k, routed, slices, warps, naive)
    if key not in _KERNELS:
        _KERNELS[key] = moe_gemm_kernel(
            f"moe_gemm_m{block_m}_k{top_k}_r{int(routed)}_s{slices}_w{warps}"
            + ("_naive" if naive else ""),
            block_m=block_m, top_k=top_k, routed=routed, slices=slices,
            warps=warps, naive=naive)
    return _KERNELS[key]


def _op_for(*key):
    """The kernel as a torch op: launched on torch's current stream, so a
    CUDA graph being captured records it. Called without the dispatcher
    outside torch.compile (`torch_ext.launcher`)."""
    if key not in _OPS:
        from ..runtime.torch_ext import launcher  # noqa: PLC0415

        _OPS[key] = launcher(kernel_for(*key))
    return _OPS[key]


def _arange(n: int, device):
    """0..n-1 as int32, made once per length: the naive path's index list,
    stable across calls so a captured graph keeps pointing at it."""
    key = (n, str(device))
    if key not in _ARANGE:
        import torch  # noqa: PLC0415

        _ARANGE[key] = torch.arange(n, device=device, dtype=torch.int32)
    return _ARANGE[key]


@functools.lru_cache(maxsize=None)
def schedule_for(block_m: int, k: int) -> Optional[Dict[str, int]]:
    """slices and warps for a block of `block_m` tokens and depth `k`, or None
    when none fits. Two register limits, both measured:
      * the combine holds every other warp's partial at once (8 VGPRs per 16
        tokens each) whatever the sched_barriers say -- 77, 141 and 269 VGPRs
        at 4, 8 and 16 warps of 32 tokens -- so (warps - 1) partials stay
        under 128;
      * a round's token tile is 32 VGPRs per 16 x 64, held twice by the load
        pipeline, so slices x token tiles stays at four.
    """
    tiles = block_m // 16
    for slices, warps in ((4, 16), (2, 16), (4, 8), (2, 8), (1, 8), (2, 4),
                          (1, 4), (1, 2)):
        if (warps - 1) * 8 * tiles > 128 or slices * tiles > 4:
            continue
        if block_m * 16 * 4 * warps > 64 * 1024 - 1024:   # the combine's LDS
            continue
        if k % (16 * slices * warps) == 0:
            return dict(slices=slices, warps=warps)
    return None


def _placeholder(device):
    """The (1, 1) fp32 a kernel without routing weights is handed for them."""
    key = ("f32", str(device))
    if key not in _ARANGE:
        import torch  # noqa: PLC0415

        _ARANGE[key] = torch.zeros(1, 1, device=device, dtype=torch.float32)
    return _ARANGE[key]


def takes(a, w, block_m: int) -> bool:
    """Whether moe_gemm runs this GEMM: bf16, N a multiple of 16, and a
    schedule for this block size and depth."""
    import torch  # noqa: PLC0415

    return (a.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
            and w.dim() == 3 and w.shape[1] % 16 == 0 and w.is_contiguous()
            and block_m % 16 == 0 and schedule_for(block_m, w.shape[2]) is not None)


def moe_gemm(a, w, c, sorted_ids, expert_ids, num_tokens_post_padded, *,
             top_k: int, block_m: int, topk_weights=None) -> None:
    """vLLM's fused MoE GEMM, into `c` ((tokens * top_k, N), or the
    (tokens, top_k, N) it is a view of). `sorted_ids` None is vLLM's naive
    assignment, one pair per block, `expert_ids` then being the flattened
    top-k ids. `topk_weights` given multiplies each row by its routing
    weight. Raises ValueError for a GEMM `takes` refuses."""
    if not takes(a, w, block_m):
        raise ValueError(f"moe_gemm: does not take A {tuple(a.shape)} {a.dtype}, "
                         f"W {tuple(w.shape)} {w.dtype} at BLOCK_M={block_m}")
    _launch(a, w, c, sorted_ids, expert_ids, num_tokens_post_padded, top_k=top_k,
            block_m=block_m, topk_weights=topk_weights)


def _launch(a, w, c, sorted_ids, expert_ids, num_tokens_post_padded, *,
            top_k: int, block_m: int, topk_weights=None) -> None:
    """`moe_gemm` for a caller that has already asked `takes`."""
    n, k = w.shape[1], w.shape[2]
    s = schedule_for(block_m, k)
    m = a.shape[0]
    naive = sorted_ids is None
    if naive:
        ids = _arange(expert_ids.numel(), a.device)
    else:
        em = total = sorted_ids.numel()
        if m < block_m:
            # vLLM's own bound: with fewer tokens than a block, each of the
            # tokens * top_k pairs opens at most one block.
            em = min(em, m * top_k * block_m)
        ids = sorted_ids if em == total else sorted_ids[:em]
    routed = topk_weights is not None
    tw = topk_weights.reshape(-1, 1) if routed else _placeholder(a.device)
    if not a.is_contiguous():
        a = a.contiguous()
    _op_for(block_m, top_k, routed, s["slices"], s["warps"], naive)(
        a, w, c.view(-1, n), ids, expert_ids.view(-1),
        num_tokens_post_padded.view(1, 1), tw)


def moe_align(topk_ids, block_m: int, num_experts: int):
    """A torch moe_align_block_size, for tests and for running without vLLM:
    (sorted_ids, expert_ids, num_tokens_post_padded)."""
    import torch  # noqa: PLC0415

    flat = topk_ids.flatten().to(torch.int64)
    numel = flat.numel()
    dev = topk_ids.device
    sorted_ids, experts = [], []
    for e in range(num_experts):
        idx = (flat == e).nonzero().flatten()
        if idx.numel() == 0:
            continue
        pad = (-idx.numel()) % block_m
        sorted_ids.append(torch.cat([idx, torch.full((pad,), numel, device=dev)]))
        experts += [e] * ((idx.numel() + pad) // block_m)
    ids = torch.cat(sorted_ids).to(torch.int32)
    total = ids.numel()
    cap = numel + num_experts * (block_m - 1)
    out = torch.full((max(cap, total),), numel, device=dev, dtype=torch.int32)
    out[:total] = ids
    exp = torch.full((-(-out.numel() // block_m),), -1, device=dev, dtype=torch.int32)
    exp[:len(experts)] = torch.tensor(experts, device=dev, dtype=torch.int32)
    return out, exp, torch.tensor([total], device=dev, dtype=torch.int32)


__all__ = ["kernel_for", "moe_align", "moe_gemm", "moe_gemm_kernel", "schedule_for",
           "takes"]
