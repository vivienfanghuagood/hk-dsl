# Your kernel in vLLM: the registration path, the wheels, and SGLang

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

The second goal of this project is that a kernel written in hk is easy to
put inside a serving engine and measure there. Before this chapter, the
only kernels that reached vLLM were hk's own. Each of them took a
hand-written patch module (`vllm_gemm`, `vllm_moe`, `vllm_int4`, the
backend), each with its own counters and its own trap around the compile
cache.

## One entry point

`hk.integration.register` is the path for a kernel someone else writes.

**`register.op(name, kernel=, reference=, out=, route=)`** makes an
`@hk.kernel` the opaque torch op `torch.ops.hk_user.<name>`:
- `out` allocates the outputs from the inputs, and on meta tensors it is the
  op's fake, which torch.compile traces;
- `route` decides per call whether the kernel or the reference runs. It runs
  inside the op, where a token count is a number, not a symbolic shape
  Dynamo would guard on and recompile for;
- the op launches on torch's stream, so CUDA graphs capture it, and counts
  both paths the way hk's own patches do;
- `op.check(x)` holds the kernel to the reference.

**`register.patch("vllm.<module>.<Class>.<method>", fn)`** rebinds a dotted
attribute. When the target is vLLM's, it also moves the engine to a compile
cache root keyed by what is patched, because [chapter 20](20-vllm-031.md)
showed that vLLM's cache does not know.

**`HK_VLLM_USER`** names the modules or `.py` files the plugin imports in
vLLM's own process, which is the only place a patch reaches the model.

**`python -m hk.integration ab`** is the end-to-end harness. It moved from
`tools/` into the package, so an installed wheel has it. It runs stock and
the user's arm alternately in fresh engines, and reports total, prefill and
decode-step time and whether greedy tokens match stock's. Arms are `stock`,
`hk`, `gemm`, `moe`, `int4` and `user`, joinable with `+`.

## The cookbook

`examples/11_vllm_integration.py` walks through the four steps for vLLM's
SiLU-and-mul:
1. a kernel that reads both halves of the `(tokens, 2I)` input in place;
2. registered with its torch reference and a route, then checked;
3. patched into `SiluAndMul.forward_native`, the method vLLM's default
   compile config (custom ops off) actually traces. Patching `forward_cuda`
   would have changed nothing.
4. measured end to end with one command.

| 2 x 12288 wide, bf16 | 16 tokens | 256 | 4096 |
|---|---|---|---|
| hk, as `torch.ops.hk_user.silu_mul`, x Inductor | 0.62x | 0.72x | 0.93x |
| torch eager, x Inductor | 1.35x | 0.51x | 0.57x |

The kernel loses to the SiLU-and-mul vLLM already compiles. That is
[chapter 12](12-strided-globals.md)'s result again, the one that dropped
this patch the first time. The example is there for the mechanics, and its
docstring says to measure before shipping a patch. At 16 tokens, 29 µs is
mostly the eager dispatch of a custom op through Python, which a captured
graph does not pay.

## The wheels

Version 0.3.0 is built for CPython 3.10 and 3.12 (`scripts/build_wheel.sh`;
11.3 and 11.4 MB) and attached to the GitHub release `v0.3.0`. Each wheel
carries the MLIR bindings for its interpreter and the HipKittens headers.

The cp312 wheel was installed into a clean venv outside the checkout. It
imported from site-packages, traced a kernel and emitted its C++ against
its own headers. The cp310 build needs setuptools ≥ 64, which the system's
59.6 is not, so it is built from a venv with a newer one. Publishing to
PyPI is a separate decision, not taken.

## SGLang: the time box, and what ended it

The plan gave SGLang a fixed time: run AMD-AIM/sglang-radeon on this card,
then route hk's linear, MoE and 4-bit kernels into it through
`hk.integration.sglang`, and record the result either way. Here is what
happened.

1. **The stack.** sglang-radeon installs SGLang 0.5.18 from PyPI and a
   prebuilt gfx1100 kernel, `sglang-kernel-rdna3`, so nothing is compiled.
   That kernel is one cp312 wheel built against torch 2.9.1 with HIP 7.2.
   download.pytorch.org has torch 2.9.1 for ROCm 6.4 only; its ROCm 7.0
   index starts at 2.10. A venv was built beside vLLM's in the vLLM pod:
   Python 3.12 from `uv`, then torch 2.9.1+rocm6.4, which loaded and saw the
   GPU. Whether the prebuilt kernel's HIP 7 build loads beside torch's HIP
   6.4 was never reached.
2. **The network.** `raw.githubusercontent.com` reset the connection from
   the pod, though `github.com` and `codeload.github.com` answered, so the
   installer was copied in instead of fetched.
3. **The disk.** The torch wheel unpacks to about 10 GB, and uv, unable to
   hardlink from its cache into the venv across filesystems, copied it a
   second time. The node's ephemeral storage then fell under kubelet's
   eviction threshold (68.4 GB free against 70.2). The pod was the largest
   user at 68.8 GB, counting the models downloaded for chapters 21 and 22,
   and it was evicted. The eviction took the pod's emptyDir with it: the
   vLLM 0.31 build, its venv, the models.

The evicted pod was deleted to give the node its disk back, since the
node's disk pressure put other tenants' pods at risk too. The pressure
cleared within a minute.

SGLang did not run, so `sglang.apply_linear()` is written and not verified
inside an SGLang server. A second attempt needs a pod whose models and
venvs live on a persistent volume rather than the node's ephemeral disk.
It also needs a torch whose HIP major version matches the prebuilt
kernel's, or a source build of the kernel against the torch installed.

The plan's other half of the second goal, vLLM, holds: hk's attention, GEMM,
MoE and 4-bit patches all run in vLLM 0.31 with greedy tokens identical to
stock's.
