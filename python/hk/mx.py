"""OCP MX operands on the host side: the E8M0 scale layouts and AITER weight layouts gfx950's
scaled-MFMA ops read (`hk.load_mx_scales`, `hk.mma_ABt_mx`, `hk.load_fp4_frags`,
`hk.mx_quant_store`), and the reference quantization they are checked against. Pure torch, used
by tests and by kernels that prepare their inputs.

    q, s = hk.mx.quant(x)                  # FP8 e4m3 rows, E8M0 a 32-column block (AITER's rule)
    sa = hk.mx.pack_scales(s)              # one coalesced dword a lane per 64 rows and K step
    sb = hk.mx.aiter_scales(s)             # AITER's 32-row two-step layout (shuffle_scale)
"""

from __future__ import annotations

#: The 16 FP4 (E2M1) values, by code.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def quant(x):
    """Rows of x to FP8 e4m3 with an E8M0 scale per 32 columns: 2^e, e = ceil(log2(amax / 448)),
    AITER's per_1x32 rule (the block max lands at <= 448, never clamped). OCP's
    floor(log2 amax) - 8 would clamp blocks whose amax mantissa is >= 1.75. Returns (q [M, K]
    float8_e4m3fn, s [M, K/32] uint8). `hk.mx_quant_store` matches it bit for bit."""
    import torch  # noqa: PLC0415

    m, k = x.shape
    v = x.float().view(m, k // 32, 32)
    amax = v.abs().amax(-1).clamp_min(2.0 ** -126)
    e = torch.floor(torch.log2(amax)) - 8
    e = e + (amax > 448.0 * torch.pow(2.0, e)).float()          # exact ceil(log2(amax / 448))
    q = (v / torch.pow(2.0, e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(m, k), (e + 127).clamp(0, 254).to(torch.uint8)


def dequant(q, s):
    """FP8 rows q [M, K] with E8M0 scales s [M, K/32] -> fp32."""
    import torch  # noqa: PLC0415

    return q.float() * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)


def fp4_values(w):
    """Packed FP4 bytes [..., K/2] (low nibble first) -> fp32 [..., K]."""
    import torch  # noqa: PLC0415

    t = torch.tensor(E2M1, device=w.device)
    b = w.long()
    return torch.stack([t[b & 15], t[b >> 4]], -1).reshape(*w.shape[:-1], -1)


def expand_block_scales(s, block_rows: int = 32):
    """32x32-block E8M0 scales [N/32, K/32] as one row each [N, K/32]."""
    return s.repeat_interleave(block_rows, dim=0).contiguous()


def pack_scales(s):
    """E8M0 scales [R, K/32] (uint8) in the packed layout `hk.load_mx_scales` reads from an i32
    tensor: [ceil(R/64), K/128 * 64] int32, row g, column st*64 + l the dword lane l takes for rows
    g*64.. and K step st -- byte i the scale of row g*64 + i*16 + l%16, block st*4 + l//16. Rows
    past R are unit scales (127). K a multiple of 128."""
    import torch  # noqa: PLC0415

    r, kb = s.shape
    if kb % 4:
        raise ValueError(f"pack_scales: K/32={kb} blocks, a multiple of 4 (K of 128)")
    g = -(-r // 64)
    if g * 64 != r:
        s = torch.cat([s, s.new_full((g * 64 - r, kb), 127)])
    t = s.view(torch.uint8).reshape(g, 4, 16, kb // 4, 4)      # [g, i, r16, st, kb4]
    t = t.permute(0, 3, 4, 2, 1).contiguous()                  # [g, st, kb4, r16, i]
    return t.view(g, kb // 4 * 64, 4).view(torch.int32).view(g, kb // 4 * 64)


def aiter_scales(s):
    """E8M0 scales [R, K/32] (uint8) in AITER's 32-row layout (shuffle_scale without gate|up;
    `hk.load_mx_scales(layout="aiter")`): for 32-row block n1 and 256-wide K pair k1, dword
    (n1 * K/256 + k1) * 64 + l holds, as byte kpack * 2 + h, the scale of row n1 * 32 + h * 16 +
    l % 16 and block k1 * 8 + kpack * 4 + l // 16. R a multiple of 32; K/32 is padded with unit
    scales to a multiple of 8."""
    import torch  # noqa: PLC0415

    r, kb = s.shape
    if r % 32:
        raise ValueError(f"aiter_scales: rows {r}, a multiple of 32")
    if kb % 8:
        s = torch.cat([s, s.new_full((r, 8 - kb % 8), 127)], 1)
        kb = s.shape[1]
    x = s.view(r // 32, 2, 16, kb // 8, 2, 4)                  # [n1, h, r16, k1, kpack, kb4]
    return x.permute(0, 3, 5, 2, 4, 1).contiguous().view(r, kb)


def aiter_gate_up_scales(s, e: int):
    """[E * 2I, K/32] E8M0 (per expert: gate rows, then up) -> AITER's shuffle_scale(is_guinterleave,
    gate_up) layout, what `hk.load_mx_scales(layout="aiter_gate_up")` reads."""
    rows, kb = s.shape
    x = s.view(e, 2, rows // e // 32, 16, kb // 8, 2, 4)
    return x.permute(0, 2, 4, 6, 3, 5, 1).contiguous().view(rows, kb)


def aiter_fp4_weights(w, gate_up: bool = False):
    """Packed FP4 weights [E, N, K/2] uint8 (with gate_up: per expert the gate rows, then the up
    rows) -> AITER's gfx950 MoE layout (shuffle_weight(is_guinterleave), 1 KB chunks of 16x128
    operands in lane order), as the [E * N, K/2] uint8 view `hk.load_fp4_frags` reads."""
    e, n, kp = w.shape
    if gate_up:
        x = w.view(e, 2, n // 32, 16, kp // 64, 4, 16)
        return x.permute(0, 2, 1, 4, 5, 3, 6).contiguous().view(e * n, kp)
    return w.view(e, n // 16, 16, kp // 64, 4, 16).permute(0, 1, 3, 4, 2, 5).contiguous().view(e * n, kp)
