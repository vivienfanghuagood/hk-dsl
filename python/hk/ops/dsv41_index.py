"""DeepSeek-V4.1's index chain for a decode step, as one resident launch (gfx950).

An index layer of DeepSeek-V4.1 (an "index source": its compressor and the sparse-attention
indexer) runs, in vLLM's `--monokernel` path, 9-12 kernels between mono's two launches: the
compressor's projection, ring and RMSNorm, the compressed-cache insert, the indexer's K store,
the MXFP8 quant of qr and its wq_b GEMM, weights_proj, the Q RoPE / MXFP4 quant, then the
logits and the top-k. `index_kernel` is all of that in one launch of 256 resident workgroups of
8 waves, for up to `ROWS` decode rows.

Stage 1, every workgroup, the weight streams:
  G1  kv_score = normed @ wkv^T (ratio 2: 1024 rows, ratio 1: 512), fp32: 32 * ratio row tiles of
      16, K split 256 / tiles ways; a workgroup's 8 waves take its K slice in 32-column chunks
  G3  q = MXFP8(qr) @ wq_b^T (4096 rows, block-scaled e4m3): one 16-row tile a workgroup, the 10
      K steps of 128 over the waves. Each workgroup quantizes qr into its own scratch rows (vLLM's
      rule bit for bit), written and read on one CU, so no other workgroup is waited for
  G2  wproj = normed @ weights_proj^T (32 rows): workgroups 0..31
Partials go to tagged mailboxes (each value carries its tag: no flags). Stage 2, for row t < M:
  C   workgroup 248 + t: kv_score row (K parts summed in order) -> compress (ratio 2: the ring) ->
      RMSNorm -> the compressed-cache record; k = wk @ latent on the matrix cores -> K norm, RoPE,
      MXFP4 -> the indexer's K cache (AITER's preshuffled page)
  Q   workgroup 240 + t: q row + wproj -> RoPE, MXFP4 quant, head weights -> the logits' inputs
With `SEL`, stages 3-4 are the logits and the top-k as well (rows of at most `WIDTH_MAX`):
  3   rows past TOPK entries: every workgroup takes 16-token blocks of a row once its q and new K
      entry are stored, logit[t] = sum_h relu(q_h . k_t) w_h on scaled fp4 MFMAs straight off the
      preshuffled K pages, the keys (order-preserving uint32) by mailbox to the row's Q workgroup
  4   the row's Q workgroup: rows of at most TOPK entries take 0 .. len-1 (the stock decode
      kernel's answer), longer ones a radix select (`select_row`) in index order. `CAND`: the
      candidate source also writes its candidate blocks (every block while they fit).

Every cache byte, q / q_scale and the top-k sets are vLLM's (hk.integration.vllm_dsv41_index has
the evidence); the head weights are fp32-accumulated, vLLM's weights_proj rounds a few heads by
one bf16 ulp differently at M > 1.
"""

from __future__ import annotations

from .. import dsl as hk
from ..types import bf16, fp8e4m3, fp32, i8, i32

#: Decode rows a launch takes, at most.
ROWS = 8
#: DeepSeek-V4.1-Flash's hidden size, the indexer's q rows (heads x head dim), head dim, heads.
H, NQ, HD, NH = 5120, 4096, 128, 32
#: The indexer's q_lora rank: wq_b's K.
QR = 1280
#: Top-k of the indexer.
TOPK = 512
#: The widest row (the indexer's max_model_len) the in-launch selection takes; a longer model length
#: keeps vLLM's logits and top-k. The launch's WIDTH is the row width rounded up to `WIDTH_STEP`.
WIDTH_MAX = 12288
WIDTH_STEP = 4096
WAVES = 8
WS = WAVES - 1                  # the storing wave: every post and cache write, no poll after one
                                # (gfx950's vmcnt counts stores: a wave that stored waits out
                                # their acks at its next poll)
_T_X = hk.RT(bf16, 16, 32, "row")           # a 32-column chunk of W rows / token rows
_T_C = hk.RT(fp32, 16, 16, "col")
_ST_C = hk.ST(fp32, 16, 16)
_T_W8 = hk.RT(fp8e4m3, 16, 128, "row")

# mailbox regions (int32 pairs)
G1P0 = 0                                     # [4 or 8 parts][8 tokens][1024] fp32
WPP0 = G1P0 + 8 * ROWS * 1024                # [16 parts][8][32] fp32
QP0 = WPP0 + 16 * ROWS * 32                  # [8][4096] bf16, two a pair
LP0 = QP0 + ROWS * NQ // 2                   # [8][256] latents, bf16 pairs (unused)
KP0 = LP0 + ROWS * 256                       # [8][128] k_pre fp32 (unused)
QR0 = KP0 + ROWS * 128                       # [8] a row's q / q_scale / weights stored
KR0 = QR0 + ROWS                             # [8] a row's new K entry stored
LK0 = KR0 + ROWS                             # [8][WIDTH_MAX] a row's keys, to its Q workgroup
MB_PAIRS = LK0 + ROWS * WIDTH_MAX
#: The mailbox's int32 length (`mailbox`).
MB_WORDS = 2 * MB_PAIRS + 64
#: qr's MXFP8 scratch: 16 rows a workgroup (`qr_scratch`: scales [QS_ROWS, 40] start at 127).
QS_ROWS = 256 * 16

#: fp32 -> uint32, order kept: the logits as the selection's keys.
KEY = ("(int)((__builtin_bit_cast(unsigned, x0) & 0x80000000u) ? ~__builtin_bit_cast(unsigned, x0)"
       " : (__builtin_bit_cast(unsigned, x0) | 0x80000000u))")


def width(n: int) -> int:
    """The launch's WIDTH for rows of up to `n` entries."""
    return -(-n // WIDTH_STEP) * WIDTH_STEP


def mailbox(device):
    """The launch's tagged mailbox (one a device; every launch reuses it, tagged by its epoch)."""
    import torch  # noqa: PLC0415

    return torch.zeros(MB_WORDS, dtype=torch.int32, device=device)


def qr_scratch(device):
    """(values, scales): the per-workgroup rows qr is MXFP8-quantized into."""
    import torch  # noqa: PLC0415

    return (torch.zeros(QS_ROWS, QR, dtype=torch.float8_e4m3fn, device=device),
            torch.full((QS_ROWS, QR // 32), 127, dtype=torch.int8, device=device))


def alloc_select():
    """select_row's LDS: a 2048-bin histogram, the control words ([0] prefix, [1] above (f32), [2]
    closed, [3] cut, as bits) and the per-wave totals of a compaction round."""
    return (hk.lds.alloc(hk.ST(fp32, 128, 16), name="hist"), hk.lds.alloc(hk.ST(fp32, 16, 16), name="ctl"),
            hk.lds.alloc(hk.ST(fp32, 16, 16), name="wt"))


@hk.func
def select_row(lk, n, w, topk, r, WIDTH: int, bufs=None):
    """The top TOPK of keys lk[0 .. n) (a 16-column LDS tile of order-preserving uint32 keys, n > TOPK,
    every wave of the workgroup calling) into topk[r, :], in index order: the keys above the
    threshold key, then ties."""
    # radix select on the keys: 11 / 11 / 10 bits from the top, a 2048-bin LDS histogram of the keys
    # that match the prefix so far; wave 0 finds the bin where the count from the top reaches what
    # is still needed -- groups of 32 bins (a lane's two 16-bin runs), then the 32 bins of the group.
    # A bin holding exactly what is needed closes the selection at that level (later passes
    # skipped). Counts are fp32 (exact far past 8192; strip arithmetic and reductions are fp32's); a
    # value crosses between strip shapes as a scalar (first_lane, floats by their bits).
    # bufs: (hist, ctl, wt) allocated by the caller (e.g. at a launch's entry), else here
    hist, ctl, wt = bufs if bufs is not None else alloc_select()
    ES, FS = hk.RS(i32, 1, 64, 1), hk.RS(fp32, 1, 64, 1)
    R16, F16 = hk.RS(i32, 1, 1024, 16), hk.RS(fp32, 1, 1024, 16)
    Z4 = hk.RS(i32, 1, 256, 4)
    lid = hk.iota(ES)
    c16 = hk.iota(R16)
    z4 = hk.iota(Z4)

    def scan(x, lanes, span, up):         # inclusive scan over lanes 0 .. span-1: suffix (up) or prefix
        for d in (1, 2, 4, 8, 16, 32):
            if d < span:
                ok = f"x0 + {d} < {span}" if up else f"x0 >= {d}"
                src = f"x0 + {d}" if up else f"x0 - {d}"
                y = x.lane_gather(hk.map_cpp(f"{ok} ? {src} : x0", lanes, dtype=i32))
                x = hk.map_cpp(f"{ok} ? x1 + x2 : x1", lanes, x, y, dtype=fp32)
        return x

    def fbits(x):                         # an fp32 strip's value as an Int of its bits (lane 0)
        return hk.map_cpp("__builtin_bit_cast(int, x0)", x, dtype=i32).first_lane()

    ctl.store_elems(hk.map_cpp("0.0f + 0.0f * (float)x0", lid, dtype=fp32), hk.map_cpp("(x1 == 0 && x0 < 4) ? x0 : -1", lid, w))
    hk.arch.barrier(drain=True)
    for sh, nbits in ((21, 11), (10, 11), (0, 10)):
        if hk.uniform(ctl.load_elems(hk.map_cpp("2 + 0 * x0", lid), i32).first_lane() == 0):
            pfx = ctl.load_elems(hk.map_cpp("0 * x0", lid), i32).first_lane()          # the key bits fixed so far
            abv = fbits(ctl.load_elems(hk.map_cpp("1 + 0 * x0", lid)))                 # keys above them (f32 bits)
            hist.store_elems(hk.map_cpp("0 * x0", z4, dtype=i32), hk.map_cpp("x1 * 256 + x0", z4, w))
            hk.arch.barrier(drain=True)
            hi = sh + nbits
            same = "1" if hi >= 32 else f"((unsigned)x1 >> {hi}) == ((unsigned)x3 >> {hi})"
            for it in range(WIDTH // (256 * WAVES)):
                if hk.uniform(it * 256 * WAVES < n):
                    e = hk.map_cpp(f"({it} * {WAVES} + x1) * 256 + x0", z4, w)
                    key = lk.load_elems(hk.map_cpp("x0 < x1 ? x0 : -1", e, n), i32, runs=True)
                    hist.atomic_add_elems(hk.map_cpp("1 + 0 * x0", z4, dtype=i32), hk.map_cpp(
                        f"(x0 < x2 && {same}) ? (int)(((unsigned)x1 >> {sh}) & {(1 << nbits) - 1}u) : -1",
                        e, key, n, pfx, dtype=i32))
            hk.arch.barrier(drain=True)
            if w == 0:
                # groups: lane l's bins 32 l .. 32 l + 31 (two 16-bin runs), every element of the lane its total
                g = hk.zeros(F16)
                for half in (0, 16):
                    g = g + hk.map_cpp("(float)x0", hist.load_elems(hk.map_cpp(f"(x0 / 16) * 32 + {half} + x0 % 16", c16),
                                                                    i32, runs=True), dtype=fp32).group_sum(16, lanes="seq")
                lane16 = hk.map_cpp("x0 / 16", c16, dtype=i32)
                si = scan(g, lane16, 64, up=True)
                L = hk.zeros(F16)
                L += hk.map_cpp("(x1 - x2 < x4 - __builtin_bit_cast(float, x3) && x4 - __builtin_bit_cast(float, x3) <= x1)"
                                " ? (float)(x0 / 16) : -1.0f", c16, si, g, abv, TOPK, dtype=fp32).max(axis=1)
                Li = hk.map_cpp("(int)x0", L, dtype=i32)
                gL = hk.map_cpp("__builtin_bit_cast(int, x0)", g.lane_gather(Li), dtype=i32).first_lane()
                sL = hk.map_cpp("__builtin_bit_cast(int, x0)", si.lane_gather(Li), dtype=i32).first_lane()
                Ls = Li.first_lane()
                # the 32 bins of group L, a lane each; base = keys in the groups above L
                hb = hk.map_cpp("(float)x0", hist.load_elems(hk.map_cpp("x0 < 32 ? x1 * 32 + x0 : -1", lid, Ls), i32), dtype=fp32)
                bi = scan(hb, lid, 32, up=True)
                BASE = "(__builtin_bit_cast(float, x5) - __builtin_bit_cast(float, x6))"
                NEED = f"({TOPK}.0f - __builtin_bit_cast(float, x4))"
                Hm = hk.zeros(FS)
                Hm += hk.map_cpp(f"(x0 < 32 && {BASE} + x1 - x2 < {NEED} && {NEED} <= {BASE} + x1) ? (float)x0 : -1.0f",
                                 lid, bi, hb, Ls, abv, sL, gL, dtype=fp32).max(axis=1)
                Hi = hk.map_cpp("(int)x0", Hm, dtype=i32)
                bH, hH = bi.lane_gather(Hi), hb.lane_gather(Hi)
                last = 1 if sh == 0 else 0
                newp = hk.map_cpp(f"x1 | ((x2 * 32 + x3) << {sh})", lid, pfx, Ls, Hi, dtype=i32)
                nabv = hk.map_cpp(f"__builtin_bit_cast(float, x1) + {BASE.replace('x5', 'x2').replace('x6', 'x3')} + x4 - x5",
                                  lid, abv, sL, gL, bH, hH, dtype=fp32)
                # exact fit: the keys in bins >= H (groups above + this group's suffix) are what is needed
                closed = hk.map_cpp(f"({last} || __builtin_bit_cast(float, x3) - __builtin_bit_cast(float, x4) + x1"
                                    f" == {TOPK}.0f - __builtin_bit_cast(float, x2)) ? 1 : 0", lid, bH, abv, sL, gL, dtype=i32)
                ctl.store_elems(hk.map_cpp("__builtin_bit_cast(float, x0)", newp, dtype=fp32), hk.map_cpp("x0 == 0 ? 0 : -1", lid))
                ctl.store_elems(nabv, hk.map_cpp("x0 == 0 ? 1 : -1", lid))
                ctl.store_elems(hk.map_cpp("__builtin_bit_cast(float, x0)", closed, dtype=fp32), hk.map_cpp("x0 == 0 ? 2 : -1", lid))
                ctl.store_elems(hk.map_cpp(f"__builtin_bit_cast(float, {sh} + 0 * x0)", lid, dtype=fp32),
                                hk.map_cpp("x0 == 0 ? 3 : -1", lid))
            hk.arch.barrier(drain=True)
    # compaction in index order, 4 keys a lane a round (2048 a round): the keys above the cut first,
    # then the first TOPK - above ties (keys equal to the prefix above the cut shift)
    pfx = ctl.load_elems(hk.map_cpp("0 * x0", lid), i32).first_lane()
    abv = fbits(ctl.load_elems(hk.map_cpp("1 + 0 * x0", lid)))
    cut = ctl.load_elems(hk.map_cpp("3 + 0 * x0", lid), i32).first_lane()
    carry_g, carry_e = hk.zeros(FS), hk.zeros(FS)
    for it in range(WIDTH // (256 * WAVES)):
        if hk.uniform(it * 256 * WAVES < n):
            es, gts, eqs = [], [], []
            for j in range(4):
                e = hk.map_cpp(f"({it} * {WAVES} + x1) * 256 + 4 * x0 + {j}", lid, w)
                key = lk.load_elems(hk.map_cpp("x0 < x1 ? x0 : -1", e, n), i32)
                es.append(e)
                gts.append(hk.map_cpp("(x0 < x2 && ((unsigned)x1 >> x4) > ((unsigned)x3 >> x4)) ? 1.0f : 0.0f",
                                      e, key, n, pfx, cut, dtype=fp32))
                eqs.append(hk.map_cpp("(x0 < x2 && ((unsigned)x1 >> x4) == ((unsigned)x3 >> x4)) ? 1.0f : 0.0f",
                                      e, key, n, pfx, cut, dtype=fp32))
            G = gts[0] + gts[1] + gts[2] + gts[3]
            E = eqs[0] + eqs[1] + eqs[2] + eqs[3]
            ig, ie = scan(G, lid, 64, up=False), scan(E, lid, 64, up=False)
            wt.store_elems(ig, hk.map_cpp("x0 == 63 ? x1 : -1", lid, w))
            wt.store_elems(ie, hk.map_cpp("x0 == 63 ? 8 + x1 : -1", lid, w))
            hk.arch.barrier(drain=True)
            bg, be = carry_g + ig - G, carry_e + ie - E                 # before this lane's 4 keys
            for v in range(WAVES):
                tg = wt.load_elems(hk.map_cpp(f"{v} + 0 * x0", lid))
                te = wt.load_elems(hk.map_cpp(f"{8 + v} + 0 * x0", lid))
                bg = bg + hk.map_cpp(f"x1 > {v} ? x0 : 0.0f", tg, w)
                be = be + hk.map_cpp(f"x1 > {v} ? x0 : 0.0f", te, w)
                carry_g += tg
                carry_e += te
            hk.arch.barrier(drain=True)
            for j in range(4):
                pos = hk.map_cpp(f"x0 > 0.0f ? (int)x2 : ((x1 > 0.0f && __builtin_bit_cast(float, x4) + x3 < {TOPK}.0f)"
                                 f" ? (int)(__builtin_bit_cast(float, x4) + x3) : -1)", gts[j], eqs[j], bg, be, abv, dtype=i32)
                topk.store_elems(es[j], hk.map_cpp(f"x0 < 0 ? -1 : x1 * {TOPK} + x0", pos, r))
                bg = bg + gts[j]
                be = be + eqs[j]


def index_kernel(wscale: float):
    """The launch for one indexer softmax scale. `wscale`: the indexer's softmax scale x
    n_heads^-0.5, folded into its head weights.

    Arguments, in order (int32 views where the cache is bytes; `-1` slots write nothing):
      normed [M, H] bf16, wkv [512 RATIO, H] bf16, wp [NH, H] bf16, qr [M, QR] bf16,
      qs8 / qss: `qr_scratch`, wqb [NQ, QR] e4m3 and its per-row scales (expanded, int8),
      mb: `mailbox`, ep: the epoch whose tag this launch writes (it must move between launches
      that share `mb`), pos / rslot / cslot / kslot: positions and slot mappings (int64 as int32
      pairs), ring (fp32 rows), the compressed cache as e4m3 / bf16 / int8, the indexer K cache
      (int8), the two RoPE cos/sin caches, the compressor's norm weight (bf16 pairs as int32),
      wk [HD, 512] and k_norm [HD], the outputs q4 / q4s / wts, int32 views of q4 and the K
      cache, then (SEL) the decode block table, row lengths, the top-k, the candidates and
      block ends.
    Constants: RATIO (1 or 2), MC (rows, <= ROWS), HASKV (a compressor), CBLK / KBLK (records
    a compressed / K page), NPT / DPT / SL (AITER's MXFP4 cache layout), SEL / WIDTH / BTW
    (selection, `width(max_model_len)`, block-table width), CAND / NCAND (the candidate source)."""
    wscale = float(wscale)

    def body(normed: hk.Tensor[bf16], wkv: hk.Tensor[bf16], wp: hk.Tensor[bf16],
             qr: hk.Tensor[bf16], qs8: hk.Tensor[fp8e4m3], qss: hk.Tensor[i8], wqb: hk.Tensor[fp8e4m3],
             wqb_s: hk.Tensor[i8],
             mb: hk.Tensor[i32, "device"], ep: hk.Tensor[i32],
             pos: hk.Tensor[i32], rslot: hk.Tensor[i32], cslot: hk.Tensor[i32], kslot: hk.Tensor[i32],
             ring: hk.Tensor[fp32], comp8: hk.Tensor[fp8e4m3], comp16: hk.Tensor[bf16], compb: hk.Tensor[i8],
             kidx: hk.Tensor[i8, "device"], cs_a: hk.Tensor[fp32], cs_i: hk.Tensor[fp32], normw: hk.Tensor[i32],
             wk: hk.Tensor[bf16], knw: hk.Tensor[bf16],
             q4: hk.Tensor[i8, "device"], q4s: hk.Tensor[i8, "device"], wts: hk.Tensor[fp32, "device"],
             q4w: hk.Tensor[i32, "device"], kidx32: hk.Tensor[i32, "device"], bt: hk.Tensor[i32], lens: hk.Tensor[i32],
             topk: hk.Tensor[i32], cand: hk.Tensor[i32], bends: hk.Tensor[i32],
             *, RATIO: int = 2, MC: int = 8, HASKV: int = 1, CBLK: int = 64, KBLK: int = 64,
             NPT: int = 16, DPT: int = 16, SL: int = 4, SEL: int = 0, WIDTH: int = 4096,
             BTW: int = 64, CAND: int = 0, NCAND: int = 2048):
        b = hk.arch.block_idx.x
        w = hk.arch.warp_id()
        if SEL:
            # the selection's LDS first, at the lowest offsets (see the stage-4 note)
            lk = hk.lds.alloc(hk.ST(fp32, WIDTH // 16, 16), name="lk")
            selbufs = alloc_select()
        tag = ep.at[0].load_scalar(invariant=True) * 2 + 1
        N1 = 512 * RATIO
        TILES1 = N1 // 16
        SPLITS1 = 256 // TILES1
        KC1 = H // SPLITS1
        CH1 = KC1 // 32                                    # 32-column chunks of this workgroup's K slice
        red = hk.lds.alloc(_ST_C, count=WAVES, name="red")
        tot = hk.lds.alloc(_ST_C, name="tot")
        # stage 3's per-wave logit tiles, here rather than in its branch: LDS is placed in program
        # order, each tile aligned to its size, and here it fills the gap before stage C's tiles
        # (12288-entry rows fit gfx950's 160 KB with it here, 8192 without)
        lt_ = hk.lds.alloc(hk.ST(fp32, 16, 32), count=WAVES, name="lt_") if SEL else None

        def cross_waves(acc):
            # every wave's 16 x 16 partial into LDS, wave 0 sums them in wave order into `tot`
            red[w].store(acc)
            hk.arch.barrier(drain=True)
            if w == 0:
                s = hk.zeros(_T_C)
                for i in range(WAVES):
                    s += red[i].load(_T_C)
                tot.store(s)
            hk.arch.barrier(drain=True)

        # ---- stage-1 operands, issued at entry in the order they are used (one in-order vmcnt): G1's
        # W and token chunks, then G3's W tiles and qr. A wave's chunk past its slice loads a valid
        # address, no rows.
        # qr's rows first (wave w row w): the workgroup quantizes them itself, below
        QS, QT = hk.RS(fp32, 1, 512, 8), hk.RS(fp32, 1, 256, 4)
        qrow = hk.min(w, MC - 1)
        qv = [(qr.at[qrow, 0].load(QS), 0, QS, (1, 2)), (qr.at[qrow, 512].load(QS), 512, QS, (1, 2)),
              (qr.at[qrow, 1024].load(QT), 1024, QT, (1, 2, 4))]
        n1 = (b // SPLITS1) * 16
        part1 = b % SPLITS1
        g1w, g1x = [], []
        if HASKV:
            for i in range((CH1 + WAVES - 1) // WAVES):
                if CH1 % WAVES == 0 or i < CH1 // WAVES:
                    k = part1 * KC1 + (w + WAVES * i) * 32
                    rows = MC
                else:
                    k = part1 * KC1 + hk.min(w + WAVES * i, CH1 - 1) * 32
                    rows = MC * hk.max(0, hk.min(1, CH1 - (w + WAVES * i)))
                g1w.append(wkv.at[n1, k].load(_T_X))
                g1x.append(normed.at[0, k].load(_T_X, bounded=rows))

        # G3's weight tiles do not wait for the quant. Without a compressor (G3 -> Q is the critical
        # path) they go here with the other stage-1 operands; with one, after the quant's wait, which
        # G1's compute then does not share (stage C, behind G1, is the critical path there)
        def g3_weights():
            out = []
            for i in range(2):
                st = hk.min(w + WAVES * i, 9)
                out.append((wqb.at[b * 16, st * 128].load(_T_W8), hk.load_mx_scales(wqb_s, 16, 4, b * 16, st * 4, loose=True)))
            return out
        g3w = [] if HASKV else g3_weights()
        g2w, g2x = [], []
        g2on = hk.min(hk.max(32 - b, 0), 1)                          # workgroups 0 .. 31: G2's 2 tiles x 16 K parts
        for i in range(2):
            ch = hk.min(w + WAVES * i, 9)
            live2 = g2on * hk.max(0, hk.min(1, 10 - (w + WAVES * i)))
            g2w.append(wp.at[(b // 16) % 2 * 16, (b % 16) * 320 + ch * 32].load(_T_X, bounded=16 * live2))
            g2x.append(normed.at[0, (b % 16) * 320 + ch * 32].load(_T_X, bounded=MC * live2))
        # qr's MXFP8 quant (vLLM's rule: per 32 a block, sb = clamp(ceil(log2(amax / 448)) + 127, 0,
        # 254), values e4m3(x * 2^(127 - sb))) into this workgroup's own rows of a scratch, read back
        # below as G3's x operand: written and read by the one CU (its XCD's L2), so no other
        # workgroup is waited for and no quant kernel runs ahead
        qrow0 = b * 16
        for v, c0, ty, gx in qv:
            c = hk.iota(ty)
            amax = hk.map_cpp("__builtin_fabsf(x0)", v).group_max(32, lanes="seq", xor=gx)
            sb = hk.map_cpp("__builtin_fminf(__builtin_fmaxf(__builtin_ceilf(__builtin_log2f("
                            "__builtin_fmaxf(x0, 1.17549435082228751e-38f) / 448.0f)) + 127.0f, 0.0f), 254.0f)", amax)
            xs = hk.map_cpp("x0 * __builtin_exp2f(127.0f - x1)", v, sb)
            qs8.store_elems(xs, hk.map_cpp(f"x1 < {MC} ? (x2 + x1) * 1280 + {c0} + x0 : -1", c, w, qrow0), runs=True)
            qss.store_elems(hk.map_cpp("(int)x0", sb, dtype=i32),
                            hk.map_cpp(f"(x1 < {MC} && (x0 % 32) == 0) ? (x2 + x1) * 40 + ({c0} + x0) / 32 : -1", c, w, qrow0))
        hk.arch.vm_wait(0)                 # this wave's rows stored ...
        hk.arch.barrier()                  # ... and every wave's: a G3 tile reads all the rows
        if HASKV:
            g3w = g3_weights()
        g3 = []
        for i in range(len(g3w)):
            st = hk.min(w + WAVES * i, 9)
            g3.append(g3w[i] + (qs8.at[qrow0, st * 128].load(_T_W8, bounded=MC),
                                hk.load_mx_scales(qss, 16, 4, qrow0, st * 4, loose=True)))

        # the stage-2 roles' scalars (C: workgroup 248 + t, Q: 240 + t), behind the stage-1 operands;
        # s_loads (invariant), so no wait on them waits out the vector loads and stores issued before
        tc = hk.max(hk.min(b - 248, MC - 1), 0)
        tq = hk.max(hk.min(b - 240, MC - 1), 0)
        pC = pos.at[2 * tc].load_scalar(invariant=True)
        rsC = rslot.at[2 * tc].load_scalar(invariant=True)
        csC = cslot.at[2 * tc].load_scalar(invariant=True)
        ksC = kslot.at[2 * tc].load_scalar(invariant=True)
        pQ = pos.at[2 * tq].load_scalar(invariant=True)
        # ---- G1: kv_score partials (layers with a compressor)
        if HASKV:
            acc1 = hk.zeros(_T_C)
            for i in range(len(g1w)):
                acc1 += g1w[i] @ g1x[i].T
            cross_waves(acc1)
        if HASKV and w == WS:
            e = hk.iota(hk.RS(fp32, 1, 256, 4))
            v = tot.load_elems(hk.map_cpp("(x0 & 15) * 16 + (x0 >> 4)", e))
            mb.post_elems(v, hk.map_cpp(f"(x0 >> 4) < {MC} ? {G1P0} + (x1 * 8 + (x0 >> 4)) * 1024 + x2 + (x0 & 15) : -1",
                                        e, part1, n1), tag, runs=True)
        # ---- G3: indexer q rows b * 16 .. + 16, K = 1280 in 10 steps of 128 (waves w, w + 8)
        acc3 = hk.zeros(_T_C)
        for i in range(len(g3)):
            wt, ws, xt, xs = g3[i]
            if i == 0:
                hk.mma("mma_ABt", wt.scaled(ws), xt.scaled(xs), acc3, into=acc3)
            else:
                if w + WAVES * i < 10:
                    hk.mma("mma_ABt", wt.scaled(ws), xt.scaled(xs), acc3, into=acc3)
        cross_waves(acc3)
        if w == WS:
            e = hk.iota(hk.RS(fp32, 1, 256, 4))                     # token = e >> 4, row = e & 15
            v = tot.load_elems(hk.map_cpp("(x0 & 15) * 16 + (x0 >> 4)", e)).to(bf16).to(fp32)
            mb.post_elems(v, hk.map_cpp(f"(x0 >> 4) < {MC} ? {QP0} + ((x0 >> 4) * {NQ} + x1 * 16 + (x0 & 15)) / 2 : -1",
                                        e, b), tag, runs=True, bf16=True)

        # ---- G2: wproj partials, workgroups 0 .. 31: 2 tiles x 16 K parts of 320 (10 chunks)
        if hk.uniform(b < 32):
            n2 = (b // 16) * 16
            part2 = b % 16
            acc2 = hk.zeros(_T_C)
            for i in range(2):
                acc2 += g2w[i] @ g2x[i].T
            cross_waves(acc2)
            if w == WS:
                e = hk.iota(hk.RS(fp32, 1, 256, 4))
                v = tot.load_elems(hk.map_cpp("(x0 & 15) * 16 + (x0 >> 4)", e))
                mb.post_elems(v, hk.map_cpp(f"(x0 >> 4) < {MC} ? {WPP0} + (x1 * 8 + (x0 >> 4)) * 32 + x2 + (x0 & 15) : -1",
                                            e, part2, n2), tag, runs=True)

        # ---- stage 2, C: token t on workgroup 248 + t (layers with a compressor). Every wave holds the
        # whole latent as 256 (even, odd) dim pairs, pair P = 4 l + m (no cross-wave reduction); wave
        # WS writes the caches. k_pre = wk @ latent on the matrix cores: wave w rows 16 w .. + 16 over
        # 16 K steps (a 16 x 16 tile, the token in column 0); the 128 values meet in LDS.
        if HASKV and hk.uniform((b >= 248) & (b < 248 + MC)):
            t = b - 248
            PP = hk.RS(fp32, 1, 256, 4)
            P = hk.iota(PP)                                                         # pair P: dims 2 P, 2 P + 1
            p_, ks_, rs_, cs_ = pC, ksC, rsC, csC
            gp = (p_ // RATIO) * RATIO
            sl = hk.lds.alloc(hk.ST(bf16, 16, 512), name="sl")
            # operands that do not depend on this step's kv_score, issued before its poll
            w0 = hk.map_cpp(f"x1 == {WS} ? 1 : 0", P, w)                       # the storing wave writes
            nwp = normw.load_elems(hk.map_cpp("x0", P), runs=True)                 # bf16 pairs
            nwE = hk.map_cpp("__builtin_bit_cast(float, x0 << 16)", nwp, dtype=fp32)
            nwO = hk.map_cpp("__builtin_bit_cast(float, x0 & (int)0xFFFF0000)", nwp, dtype=fp32)
            jr = hk.map_cpp("x0 >= 224 ? x1 * 64 + (x0 - 224) : -1", P, gp)
            ccE = hk.map_cpp("x0 >= 224 ? x1 : 1.0f", P, cs_a.load_elems(jr, runs=True), dtype=fp32)
            ssE = hk.map_cpp("x0 >= 224 ? x1 : 0.0f", P, cs_a.load_elems(hk.map_cpp("x0 < 0 ? -1 : x0 + 32", jr), runs=True),
                             dtype=fp32)
            blk = hk.max(rs_, 0) // 8                                               # (ratio 2: the ring block)
            rowp = (p_ + 7) % 8                                                     # position p - 1's ring row
            # (ratio 2) position p - 1's ring row [kv 512 | score 512], once a workgroup: 8 bytes a lane,
            # into LDS after the poll
            RG = hk.RS(fp32, 1, 128, 2)
            rge = hk.map_cpp("x0 + x1 * 128", hk.iota(RG), w)
            rgv = ring.load_elems(hk.map_cpp(f"{int(RATIO == 2)} ? x1 * 1024 + x0 : -1", rge, rowp),
                                  runs=True, row=blk)
            # this step's kv_score row: the K parts of every element, in order
            el = hk.map_cpp("x0 + x1 * 512", hk.iota(hk.RS(fp32, 1, 512, 8)), w)     # (part, n) = e / N1, e % N1
            nall = SPLITS1 * N1                                                     # 4096 either ratio
            parts = mb.take_elems(hk.map_cpp(f"x0 < {nall} ? {G1P0} + ((x0 / {N1}) * 8 + x1) * 1024 + x0 % {N1} : -1",
                                             el, t), tag, runs=True, timeout_s=2.0)
            # (element-indexed LDS is 16 columns wide: one column of 16 x 16 subtiles is linear, where a
            # wider tile puts every 16th element 1 KB on -- the same banks, an 8-way conflict on each
            # strip read)
            sc = hk.lds.alloc(hk.ST(fp32, 256, 16), name="sc")                      # 4096 partials
            # retire the prefetches here, ahead of any store (a later wait for them would wait out the stores)
            hk.arch.pin(nwE, nwO, ccE, ssE)
            # wk's tiles behind the poll (ahead of it, their 128 KB held the poll back): they land under the sums
            wkt = [wk.at[w * 16, k * 32].load(_T_X) for k in range(16)]
            rg = hk.lds.alloc(hk.ST(fp32, 64, 16), name="rg")
            rg.store_elems(rgv, hk.map_cpp("(x0 & ~511) + (x0 & 1) * 256 + ((x0 & 511) >> 1)", rge))   # de-interleaved
            # de-interleaved: each 512-element segment as [even 256 | odd 256], so the sums below read P
            # contiguous
            sc.store_elems(parts, hk.map_cpp("(x0 & ~511) + (x0 & 1) * 256 + ((x0 & 511) >> 1)", el))
            hk.arch.barrier(drain=True)

            ks4 = []
            for off in ((0, 256, 512, 768) if RATIO == 2 else (0, 256)):  # kv even / odd, score even / odd
                ks4.append(sc.load_elems(hk.map_cpp(f"x0 + {off}", P), runs=True))
                for pp_ in range(1, SPLITS1):
                    ks4[-1] = ks4[-1] + sc.load_elems(hk.map_cpp(f"x0 + {off + pp_ * N1}", P), runs=True)
            kvE, kvO = ks4[0], ks4[1]
            if RATIO == 2:
                scE, scO = ks4[2], ks4[3]
                pkE = rg.load_elems(hk.map_cpp("x0", P), runs=True)
                pkO = rg.load_elems(hk.map_cpp("256 + x0", P), runs=True)
                psE = rg.load_elems(hk.map_cpp("512 + x0", P), runs=True)
                psO = rg.load_elems(hk.map_cpp("768 + x0", P), runs=True)
                POOL = ("(x0 * __expf(x1 - __builtin_fmaxf(x1, x3)) + x2 * __expf(x3 - __builtin_fmaxf(x1, x3)))"
                        " / (__expf(x1 - __builtin_fmaxf(x1, x3)) + __expf(x3 - __builtin_fmaxf(x1, x3)))")
                plE = hk.map_cpp(POOL, pkE, psE, kvE, scE)
                plO = hk.map_cpp(POOL, pkO, psO, kvO, scO)
                # the current row to the ring, after the previous row's read above
                hk.arch.pin(*wkt)                  # (retired before the ring stores: no wait for them waits out a store)
                if w == WS:
                    rrow = hk.map_cpp("(x1 < 0 || x2 == 0) ? -1 : (x1 % 8) * 1024 + 2 * x0", P, rs_, w0)
                    ring.store_elems(kvE, rrow, row=blk)
                    ring.store_elems(kvO, hk.map_cpp("x0 < 0 ? -1 : x0 + 1", rrow), row=blk)
                    ring.store_elems(scE, hk.map_cpp("x0 < 0 ? -1 : x0 + 512", rrow), row=blk)
                    ring.store_elems(scO, hk.map_cpp("x0 < 0 ? -1 : x0 + 513", rrow), row=blk)
                live = hk.map_cpp("(x1 >= 0 && ((x2 + 1) % 2) == 0) ? 1 : 0", P, rs_, p_)
            else:
                plE, plO = kvE, kvO
                live = hk.map_cpp("x1 >= 0 ? 1 : 0", P, cs_)
            # RMSNorm over the 512 dims, in this wave alone
            sq = (plE * plE + plO * plO).sum(axis=1, lanes="seq", xor=(32, 16, 8, 4, 2, 1))
            sqs = hk.zeros(PP)
            sqs += sq
            rr = hk.map_cpp("__builtin_amdgcn_rsqf(x0 * (float)(1.0 / 512.0) + 1e-20f)", sqs)
            lE = ((plE * rr) * nwE).to(bf16).to(fp32)
            lO = ((plO * rr) * nwO).to(bf16).to(fp32)
            if RATIO != 2:
                hk.arch.pin(*wkt)
            # the latent (bf16) as row 0 of the wk MMA's B operand (rows 1 .. 15 unused: their columns
            # are not read)
            sl.store_elems(lE, hk.map_cpp("x1 == 0 ? 2 * x0 : -1", P, w))
            sl.store_elems(lO, hk.map_cpp("x1 == 0 ? 2 * x0 + 1 : -1", P, w))
            # the compressed main-cache record: RoPE of pairs 224.. at the group's first position;
            # dims < 448 e4m3 in 7 groups of 64 (32 pairs = 8 lanes) with a UE8M0 scale each (capped
            # at 254), scale byte 7 zero
            rE = hk.map_cpp("x0 >= 224 ? (x1 * x3 - x2 * x4) : x1", P, lE, lO, ccE, ssE, dtype=fp32).to(bf16).to(fp32)
            rO = hk.map_cpp("x0 >= 224 ? (x2 * x3 + x1 * x4) : x2", P, lE, lO, ccE, ssE, dtype=fp32).to(bf16).to(fp32)
            amax = hk.map_cpp("__builtin_fmaxf(__builtin_fabsf(x0), __builtin_fabsf(x1))", lE, lO).group_max(
                32, lanes="seq", xor=(1, 2, 4))
            code = hk.map_cpp("((__builtin_bit_cast(int, __builtin_fmaxf(x0, 1e-4f) * (float)(1.0 / 448.0)) >> 23) & 0xFF)"
                              " + ((__builtin_bit_cast(int, __builtin_fmaxf(x0, 1e-4f) * (float)(1.0 / 448.0)) & 0x7FFFFF) != 0)",
                              amax, dtype=i32)
            code = hk.map_cpp("x0 > 254 ? 254 : x0", code, dtype=i32)
            QF = "__builtin_fmaxf(__builtin_fminf(x0 * __builtin_bit_cast(float, (254 - x1) << 23), 448.0f), -448.0f)"
            qE = hk.map_cpp(QF, lE, code, dtype=fp32)
            qO = hk.map_cpp(QF, lO, code, dtype=fp32)
            cb = hk.max(cs_, 0) // CBLK
            coff = hk.map_cpp(f"(x1 == 0 || x3 == 0 || x2 < 0) ? -1 : (x2 % {CBLK}) * 576", P, live, cs_, w0)
            if w == WS:
                comp8.store_elems(qE, hk.map_cpp("(x1 < 0 || x0 >= 224) ? -1 : x1 + 2 * x0", P, coff), row=cb)
                comp8.store_elems(qO, hk.map_cpp("(x1 < 0 || x0 >= 224) ? -1 : x1 + 2 * x0 + 1", P, coff), row=cb)
                comp16.store_elems(rE, hk.map_cpp("(x1 < 0 || x0 < 224) ? -1 : (x1 + 448) / 2 + 2 * (x0 - 224)", P, coff), row=cb)
                comp16.store_elems(rO, hk.map_cpp("(x1 < 0 || x0 < 224) ? -1 : (x1 + 448) / 2 + 2 * (x0 - 224) + 1", P, coff), row=cb)
                compb.store_elems(code, hk.map_cpp(f"(x1 < 0 || x0 >= 224 || (x0 & 31)) ? -1 : {CBLK} * 576 + (x2 % {CBLK}) * 8 + (x0 >> 5)",
                                                   P, coff, cs_), row=cb)
                compb.store_elems(hk.map_cpp("0 * x0", code, dtype=i32),
                                  hk.map_cpp(f"(x1 < 0 || x0 != 0) ? -1 : {CBLK} * 576 + (x2 % {CBLK}) * 8 + 7", P, coff, cs_),
                                  row=cb)
            hk.arch.barrier(drain=True)
            acc = hk.zeros(_T_C)
            for k in range(16):
                acc += wkt[k] @ sl.subtile(16, 32, 0, k).load(_T_X).T
            kq = hk.lds.alloc(hk.ST(fp32, 128, 16), name="kq")                        # [wk row][token column]
            kq.subtile(16, 16, w, 0).store(acc)
            hk.arch.barrier(drain=True)
            if w == 0:
                # the 128 k_pre values as (even, odd) pairs: lane l holds dims 2 l and 2 l + 1
                PL = hk.RS(fp32, 1, 64, 1)
                pl = hk.iota(PL)
                ke = kq.load_elems(hk.map_cpp("(2 * x0) * 16", pl)).to(bf16).to(fp32)
                ko = kq.load_elems(hk.map_cpp("(2 * x0 + 1) * 16", pl)).to(bf16).to(fp32)
                kvar = (ke * ke + ko * ko).sum(axis=1, lanes="seq", xor=(32, 16, 8, 4, 2, 1))
                kvs = hk.zeros(PL)
                kvs += kvar
                kr = hk.map_cpp("__builtin_amdgcn_rsqf(x0 * (float)(1.0 / 128.0) + 1e-20f)", kvs)
                ke = ((ke * kr) * knw.load_elems(hk.map_cpp("2 * x0", pl))).to(bf16).to(fp32)
                ko = ((ko * kr) * knw.load_elems(hk.map_cpp("2 * x0 + 1", pl))).to(bf16).to(fp32)
                kj = hk.map_cpp("x0 >= 32 ? x1 * 64 + (x0 - 32) : -1", pl, gp)
                kc = hk.map_cpp("x0 >= 32 ? x1 : 1.0f", pl, cs_i.load_elems(kj), dtype=fp32)
                ksn = hk.map_cpp("x0 >= 32 ? x1 : 0.0f", pl, cs_i.load_elems(hk.map_cpp("x0 < 0 ? -1 : x0 + 32", kj)), dtype=fp32)
                ke2 = hk.map_cpp("x0 >= 32 ? (x1 * x3 - x2 * x4) : x1", pl, ke, ko, kc, ksn, dtype=fp32).to(bf16).to(fp32)
                ko2 = hk.map_cpp("x0 >= 32 ? (x2 * x3 + x1 * x4) : x2", pl, ke, ko, kc, ksn, dtype=fp32).to(bf16).to(fp32)
                # MXFP4: 32 dims = 16 lanes a block; e8m0 = 2^ceil(log2(amax / 6))
                kam = hk.map_cpp("__builtin_fmaxf(__builtin_fabsf(x0), __builtin_fabsf(x1))", ke2, ko2).group_max(16, lanes="seq", xor=(1, 2, 4, 8))
                kbs = hk.map_cpp("(int)((__builtin_bit_cast(unsigned, __builtin_fmaxf(x0, 6.0f * 1.1754943508222875e-38f) * (float)(1.0 / 6.0)) + 0x7FFFFFu) >> 23)",
                                 kam, dtype=i32)
                kbyte = hk.map_cpp("(int)(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32(0u, x0 * __builtin_bit_cast(float, (254 - x2) << 23),"
                                   " x1 * __builtin_bit_cast(float, (254 - x2) << 23), 1.0f, 0) & 0xFF)", ke2, ko2, kbs, dtype=i32)
                kpage = hk.max(ks_, 0) // KBLK
                klive = hk.map_cpp("(x1 >= 0 && ((x2 + 1) % x3) == 0) ? 1 : 0", pl, ks_, p_, RATIO)
                # AITER's preshuffled page: a run of NPT tokens stores [64 / DPT chunks][NPT tokens][DPT
                # bytes], then its scales [4 % SL][NPT][4 / SL]
                kvo = hk.map_cpp(f"x1 == 0 ? -1 : (((x2 % {KBLK}) / {NPT}) * ({NPT} * 64) + (x0 / {DPT}) * ({NPT} * {DPT})"
                                 f" + ((x2 % {KBLK}) % {NPT}) * {DPT} + (x0 % {DPT}))", pl, klive, ks_)
                kidx.store_elems(kbyte, kvo, row=kpage)
                kso = hk.map_cpp(f"(x1 == 0 || (x0 & 15)) ? -1 : {KBLK} * 64 + ((x2 % {KBLK}) / {NPT}) * ({NPT} * 4)"
                                 f" + ((x0 >> 4) % {SL}) * ({NPT} * (4 / {SL})) + ((x2 % {KBLK}) % {NPT}) * (4 / {SL}) + (x0 >> 4) / {SL}",
                                 pl, klive, ks_)
                kidx.store_elems(kbs, kso, row=kpage)
                if SEL:
                    # the row's new K entry is stored (or there is none): the logits roles may read the cache
                    hk.arch.vm_wait(0)
                    mb.post_elems(hk.map_cpp("0 * x0", pl, dtype=i32), hk.map_cpp(f"x0 == 0 ? {KR0} + x1 : -1", pl, t), tag)

        # ---- stage 2, Q: token t on workgroup 240 + t
        if hk.uniform((b >= 240) & (b < 240 + MC)):
            t = b - 240
            p_ = pQ
            # a pair (dims 2 m, 2 m + 1 of a head) a value: wave w pairs w * 256 + 4 l + j
            QL = hk.RS(i32, 1, 256, 4)
            qp = hk.map_cpp("x0 + x1 * 256", hk.iota(QL), w)                      # head = qp >> 6, pair = qp & 63
            pr = mb.take_elems(hk.map_cpp(f"{QP0} + x1 * {NQ // 2} + x0", qp, t), tag, i32, runs=True, timeout_s=2.0)
            # wproj first: wave 0 polls before it stores anything
            sp = hk.lds.alloc(hk.ST(fp32, 32, 16), name="sp")      # (outside the wave-0 branch: see lds.alloc)
            if w == 0:
                pl = hk.iota(hk.RS(fp32, 1, 512, 8))                               # (part, head) = e / 32, e % 32
                ppv = mb.take_elems(hk.map_cpp(f"{WPP0} + ((x0 >> 5) * 8 + x1) * 32 + (x0 & 31)", pl, t), tag,
                                    runs=True, timeout_s=2.0)
                sp.store_elems(ppv, pl)
                hl = hk.iota(hk.RS(fp32, 1, 64, 1))
                hv = sp.load_elems(hk.map_cpp("x0 < 32 ? x0 : -1", hl))
                for pp_ in range(1, 16):
                    hv = hv + sp.load_elems(hk.map_cpp(f"x0 < 32 ? x0 + {32 * pp_} : -1", hl))
                wv = hk.map_cpp(f"x0 * (float){wscale!r}", hv.to(bf16).to(fp32), dtype=fp32)
                wts.store_elems(wv, hk.map_cpp("x0 < 32 ? x1 * 32 + x0 : -1", hl, t))
            qe = hk.map_cpp("__builtin_bit_cast(float, x0 << 16)", pr, dtype=fp32)
            qo = hk.map_cpp("__builtin_bit_cast(float, x0 & (int)0xFFFF0000)", pr, dtype=fp32)
            qm = hk.map_cpp("x0 & 63", qp)
            qj = hk.map_cpp("x0 >= 32 ? x1 * 64 + (x0 - 32) : -1", qm, p_)
            qc = hk.map_cpp("x0 >= 32 ? x1 : 1.0f", qm, cs_i.load_elems(qj), dtype=fp32)
            qs = hk.map_cpp("x0 >= 32 ? x1 : 0.0f", qm, cs_i.load_elems(hk.map_cpp("x0 < 0 ? -1 : x0 + 32", qj)), dtype=fp32)
            re = hk.map_cpp("x0 >= 32 ? (x1 * x3 - x2 * x4) : x1", qm, qe, qo, qc, qs, dtype=fp32)
            ro = hk.map_cpp("x0 >= 32 ? (x2 * x3 + x1 * x4) : x2", qm, qe, qo, qc, qs, dtype=fp32)
            re = hk.map_cpp("x0 >= 32 ? x1 : x2", qm, re.to(bf16).to(fp32), re, dtype=fp32)
            ro = hk.map_cpp("x0 >= 32 ? x1 : x2", qm, ro.to(bf16).to(fp32), ro, dtype=fp32)
            qam = hk.map_cpp("__builtin_fmaxf(__builtin_fabsf(x0), __builtin_fabsf(x1))", re, ro).group_max(16, lanes="seq", xor=(1, 2))
            qbs = hk.map_cpp("(int)((__builtin_bit_cast(unsigned, __builtin_fmaxf(x0, 6.0f * 1.1754943508222875e-38f) * (float)(1.0 / 6.0)) + 0x7FFFFFu) >> 23)",
                             qam, dtype=i32)
            qbyte = hk.map_cpp("(int)(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32(0u, x0 * __builtin_bit_cast(float, (254 - x2) << 23),"
                               " x1 * __builtin_bit_cast(float, (254 - x2) << 23), 1.0f, 0) & 0xFF)", re, ro, qbs, dtype=i32)
            q4.store_elems(qbyte, hk.map_cpp(f"x1 * {NQ // 2} + x0", qp, t), runs=True)
            q4s.store_elems(qbs, hk.map_cpp("(x0 & 15) ? -1 : x1 * 128 + (x0 >> 4)", qp, t))
            if SEL:
                hk.arch.vm_wait(0)                 # this wave's q / q_scale / weights stored ...
                hk.arch.barrier()                  # ... and every wave's: the row's q is ready
                if w == 0:
                    lq = hk.iota(hk.RS(i32, 1, 64, 1))
                    mb.post_elems(hk.map_cpp("0 * x0", lq, dtype=i32), hk.map_cpp(f"x0 == 0 ? {QR0} + x1 : -1", lq, t), tag)
        if SEL:
            # (the roles below reuse the earlier stages' LDS -- allocations in sibling branches share it
            # -- so every wave finishes the one before writing the next)
            hk.arch.barrier(drain=True)
            # ---- stage 3: the logits of a row past TOPK entries, on every workgroup -- row b // 32, its
            # 16-token blocks (b % 32) 8 + w + 256 k -- once the row's q (Q workgroup) and new K entry
            # (C workgroup) are stored; the 16 keys of a block (logits as ordered uint32, all 16
            # positions, the tail past the row too) by mailbox to the row's Q workgroup.
            # logit[t] = sum_h relu(q_h . k_t) w_h on two scaled fp4 MFMAs, K as A, q as B: [token][head]
            rr = b // 32
            cc = b % 32
            nr = lens.at[hk.min(rr, MC - 1)].load_scalar(invariant=True) * hk.min(hk.max(MC - rr, 0), 1)
            if hk.uniform(nr > TOPK):
                l1_ = hk.iota(hk.RS(i32, 1, 64, 1))
                hk.arch.pin(mb.take_elems(hk.map_cpp(f"x0 == 0 ? {QR0} + x1 : -1", l1_, rr), tag, i32, timeout_s=2.0))
                if HASKV:
                    hk.arch.pin(mb.take_elems(hk.map_cpp(f"x0 == 0 ? {KR0} + x1 : -1", l1_, rr), tag, i32, timeout_s=2.0))
                lw_ = hk.iota(hk.RS(i32, 1, 256, 4))
                qa_, qsc_, wv_ = [], [], []
                hl_ = hk.iota(hk.RS(i32, 1, 1024, 16))
                for hq_ in range(2):
                    qa_.append(hk.fp4_operand(q4w.load_elems(hk.map_cpp(
                        f"((x1 * {NH} + {16 * hq_} + (x0 >> 2) % 16) * 16) + 4 * ((x0 >> 2) / 16) + (x0 & 3)", lw_, rr), runs=True)))
                    qsc_.append(hk.mx_scales_loose(hk.map_cpp("x0 & 0xFF", q4s.load_elems(hk.map_cpp(
                        f"(x1 * {NH} + {16 * hq_} + x0 % 16) * 4 + x0 / 16", l1_, rr), i32), dtype=i32)))
                    wv_.append(wts.load_elems(hk.map_cpp(f"x1 * {NH} + {16 * hq_} + x0 % 16", hl_, rr), runs=True))
                nb_ = (nr + 15) // 16
                for j in range(WIDTH // 16 // 256):
                    blk = cc * 8 + w + 256 * j
                    if blk < nb_:                  # wave-uniform only
                        t0 = blk * 16
                        pg = bt.at[rr * BTW + t0 // KBLK].load_scalar(invariant=True)
                        kw = kidx32.load_elems(hk.map_cpp(
                            f"(((x1 % {KBLK}) + (x0 >> 2) % 16) / {NPT} * ({NPT} * 64) + ((x0 >> 2) / 16) * ({NPT} * {DPT})"
                            f" + (((x1 % {KBLK}) + (x0 >> 2) % 16) % {NPT}) * {DPT}) / 4 + (x0 & 3)", lw_, t0), runs=True, row=pg)
                        kb8 = kidx.load_elems(hk.map_cpp(
                            f"{KBLK} * 64 + (((x1 % {KBLK}) + x0 % 16) / {NPT}) * ({NPT} * 4) + ((x0 / 16) % {SL}) * ({NPT} * (4 / {SL}))"
                            f" + (((x1 % {KBLK}) + x0 % 16) % {NPT}) * (4 / {SL}) + (x0 / 16) / {SL}", l1_, t0), i32, row=pg)
                        kb = hk.fp4_operand(kw)
                        ks = hk.mx_scales_loose(hk.map_cpp("x0 & 0xFF", kb8, dtype=i32))
                        for im_ in range(2):
                            lt_[w].subtile(16, 16, 0, im_).store(hk.mma(
                                "mma_ABt", kb.scaled(ks, fmt="fp4"),
                                qa_[im_].scaled(qsc_[im_], fmt="fp4"), hk.zeros(_T_C)))
                        lg16 = hk.zeros(hk.RS(fp32, 1, 1024, 16))
                        for is_ in range(2):
                            sv = lt_[w].load_elems(hk.map_cpp(f"x0 / 16 < 16 ? (x0 / 16) * 32 + {16 * is_} + x0 % 16 : -1", hl_), runs=True)
                            lg16 = lg16 + hk.map_cpp("__builtin_fmaxf(x0, 0.0f) * x1", sv, wv_[is_]).group_sum(16, lanes="seq")
                        mb.post_elems(hk.map_cpp(KEY, lg16, dtype=i32),
                                      hk.map_cpp(f"(x0 % 16 == 0 && x0 / 16 < 16) ? {LK0} + x1 * {WIDTH} + x2 + x0 / 16 : -1", hl_, rr, t0), tag)
            hk.arch.barrier(drain=True)
            # ---- stage 4: the row's top TOPK on its Q workgroup -- up to TOPK entries 0 .. len-1 then -1
            # (the stock decode kernel's own answer, no logits), else the keys taken into LDS and selected
            if hk.uniform((b >= 240) & (b < 240 + MC)):
                tq_ = b - 240
                nt = lens.at[tq_].load_scalar(invariant=True)
                if CAND:
                    # the candidate source: its candidate blocks for the consumers -- every block of the
                    # row (the stock block top-k's answer while the blocks fit NCAND), 0 .. ends-1, then -1
                    be = bends.at[tq_].load_scalar(invariant=True)
                    cz = hk.iota(hk.RS(i32, 1, 256, 4))
                    for ic in range(NCAND // (256 * WAVES)):
                        ci = hk.map_cpp(f"({ic} * {WAVES} + x1) * 256 + x0", cz, w)
                        cand.store_elems(hk.map_cpp("x0 < x1 ? x0 : -1", ci, be, dtype=i32),
                                         hk.map_cpp(f"x1 * {NCAND} + x0", ci, tq_), runs=True)
                if hk.uniform(nt <= TOPK):
                    if w == 0:
                        sel = hk.iota(hk.RS(i32, 1, 512, 8))
                        topk.store_elems(hk.map_cpp("x0 < x1 ? x0 : -1", sel, nt, dtype=i32), hk.map_cpp(f"x1 * {TOPK} + x0", sel, tq_),
                                         runs=True)
                if hk.uniform(nt > TOPK):
                    z4 = hk.iota(hk.RS(i32, 1, 256, 4))
                    for it in range(WIDTH // 2048):
                        if hk.uniform(it * 2048 < nt):
                            e = hk.map_cpp(f"({it} * 8 + x1) * 256 + x0", z4, w)
                            kv = mb.take_elems(hk.map_cpp(f"x0 < ((x2 + 15) / 16) * 16 ? {LK0} + x1 * {WIDTH} + x0 : -1", e, tq_, nt),
                                               tag, i32, runs=True, timeout_s=2.0)
                            lk.store_elems(kv, hk.map_cpp("x0 < x1 ? x0 : -1", e, nt))
                    hk.arch.barrier(drain=True)
                    select_row(lk, nt, w, topk, tq_, WIDTH, bufs=selbufs)
    body.__name__ = "body"
    return hk.kernel(body, arch="gfx950", warps=WAVES, name="dsv41_index", grid=lambda p: (256, 1, 1), resident=True)
