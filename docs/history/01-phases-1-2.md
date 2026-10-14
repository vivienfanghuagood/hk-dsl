# Phases 1 and 2: Python to torch, end to end

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

**Phases 1 and 2 are done.** End to end: Python → IR → C++ → hipcc → `.so` →
torch. 291 shipped kernels — elementwise (5 binary × 3 dtypes plus the unary ops
the library specialises), RMSNorm / LayerNorm / softmax, RoPE, SiLU-mul, and
per-row int8 quantize/dequantize. Every one builds with `scratch=0 spill=0`
(a hard gate, not a check) and matches torch elementwise on a W7900D, including
shapes that do not divide the tile.

Phase 2's gate was "do not require a win". It won anyway: against
`torch.compile` on a W7900D, 13 of 15 cases are faster and the other two
(`quantize 16384x4096` and `layernorm 16384x5120`) land within 1% behind --
which is inside the same noise band [Phase 5](04-phase-5-autotune.md) later had to make explicit, and so
is reported as a tie rather than as a win for either side.

```
case                        hk      compile    eager
rmsnorm   4096x4096      0.060 ms   0.093     0.418
rmsnorm  16384x5120      0.486      0.775     2.823
layernorm 4096x4096      0.061      0.081     0.063
softmax   8192x16384     0.707      0.874     0.738
silu_mul  8192x11008     0.761      0.763     1.218
rope     128x1024x128    0.076      0.107     0.583
quantize 16384x5120      0.367      0.369     6.118
```

Getting there was mostly *not* kernel work. A 4096x4096 quantize is 60 µs of
GPU; the Python wrapper around it was 13.9 µs. Three rounds — memoising the
launcher on the kwargs as passed, caching the plan on the `torch.dtype` object
rather than `str(dtype)`, and a pybind converter that interns the attribute
names and reads a tensor in five Python calls (`pyutils/hk_bind.cuh`) — took the
worst wrapper to 9.9 µs and `rmsnorm` from 11.3 to 6.6. See
`tools/hk-bench/host_overhead.py`, which measures host time, queue depth and
device time separately so the three cannot be confused for each other.

---

[Development record](README.md) · [Phase 3](02-phase-3-gemm.md) →
