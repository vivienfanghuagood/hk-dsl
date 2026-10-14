# Strided globals, and the two patches they were for that lost

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

A global used to be a base pointer plus the strides implied by its shape, so
any non-contiguous tensor was refused, and an engine's views had to be
copied first. The case that motivated the change is vLLM's fused gate/up
output (M, 2I), whose two halves `x[:, :I]` and `x[:, I:]` are what SiLU-mul
reads.

A global now carries one runtime row stride. It defaults to the column count,
and `operator[]` and `stride<axis>()` use it, which every load and store goes
through. Both entry points -- the pybind launcher and the torch-op scaffold
-- accept a slice of the last axis: contiguous along it, rows evenly strided,
outer axes compact over the rows. Anything else is refused with a message
saying why, rather than read as if it were that. Contiguous tensors take the
same path and pay the same one call as before. Rebuilt against the previous
commit, 63 of 66 kernel families are identical; dequantize is +1 VGPR at the
same rounded count and occupancy.

What it was for did not survive measurement. The plan was SiLU-mul and a
weighted RMSNorm in vLLM, each opt-in and kept only if the end-to-end did not
lose. Before writing either, the kernels were timed against what the engine
runs instead. In the mode vLLM ships, that is not its HIP kernels: under
torch.compile both ops come from Inductor, fused with their neighbours.
Kernel time on Qwen3-8B shapes, one process:

```
                               Inductor    hk
SiLU-mul (4096, 2*12288)        409.5 us   429.7 us   (contiguous halves)
RMSNorm  (4096, 4096)            38.8 us    55.8 us   (hk unweighted)
RMSNorm  q/k heads (163840, 128) 42.9 us    70.4 us   (hk unweighted)
```

The library's RMSNorm loses by 44-64% before it even has a weight to
multiply. An opaque op in that position would also cost Inductor the fusion
of the residual add and the weight. Replacing it could only make the request
slower, so both integrations were dropped rather than built and then
measured losing. (At M=16-64 hk looks 2x faster in the same table; that is
the compiled function's Python overhead, which does not exist inside vLLM's
graph.) Not measured: vLLM's own eager kernels, the baseline only in
`enforce_eager` mode. The strided global stays: it is what the next patch
that receives a view instead of a tensor needs.

---

← [A reduction across workgroups](11-cross-workgroup-reduction.md) · [Development record](README.md) · [gfx1201](13-gfx1201.md) →
