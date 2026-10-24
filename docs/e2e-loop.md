# From "this model is slow" to "my engine runs hk's kernel": the loop

The steps a developer's harness -- a person or a coding agent -- takes to make
a model faster end to end with hk-dsl, in the order they pay off. Each step
names the tool and what to look at. The worked example is Qwen-Image 2.1 in
SGLang-Diffusion on a W7900D, 42.2 s -> 28.2 s an image
([chapter 46](history/46-qwen-image-on-rdna3.md)); the same steps apply to
vLLM, where `hk.integration` has more of them built in.

## 1. Measure the engine as it ships

One script that builds the engine once, runs a warm-up request, then times
several identical requests (same prompt, seed, size, steps), and writes the
wall times and the versions. `bench/e2e/qwen_image/qi_bench.py` is one; for
vLLM, `python -m hk.integration ab` does this and more. Check two things
before trusting a number:

- **It is the path that is served.** SGLang 0.5.18 silently ran Qwen-Image 2.1
  through a diffusers fallback at twice the time of the native pipeline; read
  the log for "falling back" before you read the clock.
- **Every request succeeded.** A failed request returns in milliseconds and
  looks like a fast one; make the script fail loudly (`qi_bench.py` raises).
  A request whose worker died can also never return: watch the log and treat
  minutes of silence as a hang (`podwait`'s silence timeout in E1's record).

## 2. Find where the device time goes

- vLLM: `python -m hk.integration headroom --model M --batch B --context N`
  attributes a decode step and a prefill to kernel families and ranks them
  against what hk has.
- Anything else: `rocprofv3 --kernel-trace --stats -f csv -- python script.py`
  traces every process the script spawns; sort `kernel_stats.csv` into
  families (`bench/e2e/qwen_image/families.py`). Qwen-Image's stock image was
  59% one hipBLASLt GEMM kernel and 31% AOTriton attention.

Count calls too: 1,920 attention calls in two images is one a layer a step,
which says which module issues them and at what shape.

## 3. Check hk's kernel at the model's exact shapes, in isolation

Before patching an engine, time hk's kernel against the engine's at the
shapes the trace showed, interleaved in one process:

```python
from hk.tune.measure import GemmBench                      # bench/e2e/qwen_image/gemm_ab.py
gb = GemmBench(4096, 12288, 4096, "gfx1100")
per = gb.time({"engine": lambda x, w, out: torch.matmul(x, w.t(), out=out),
               "hk": lambda x, w, out: hk.ops.matmul(x, w, out=out)}, rounds=7)
```

`python -m hk.check` does the same for a kernel of your own, and
`python -m hk.tune MxNxK` searches hk's schedule space for a GEMM shape (for
Qwen-Image's four it found the shipped route already best). If hk loses
here, stop: no integration will make it win.

## 4. Route the engine's calls to hk's kernels without editing the engine

Find the narrowest seam the engine calls at run time and rebind it:

- a module attribute looked up per call (`torch.nn.functional.scaled_dot_product_attention`
  -> `hk.ops.sdpa.patch()`);
- a method every layer of a kind goes through (`UnquantizedLinearMethod.apply`).

Then make the patch reach the process that runs the model. vLLM and SGLang's
LLM engine load plugins in their workers (`hk.integration`'s plugins).
SGLang-Diffusion does not, so `python -m hk.integration sglang-diffusion
install` puts a `.pth` boot in the environment that applies the patch, in
any process, when the patched module is imported, if `HK_SGLD` is set. The
same pattern works for any engine that spawns workers.

Make the patch say what it did: each patched process reports how many calls
took hk's kernel and how many fell back, and why (the drop-in's `STATS` and
`FALLBACKS`). A patch that silently falls back looks like a patch that works.

## 5. Check the output

- Run each arm twice: if the engine is deterministic, an arm's outputs are
  identical, and any difference between arms is the kernel's.
- Compare against stock and against a reference that is more exact than
  either (torch's math SDPA, an fp32 matmul). Qwen-Image's hk image is closer
  to the math-SDPA image than stock's is (MAE 1.37 against 1.83).
- Look at the output. PSNR and MAE rank images badly; side by side, text,
  fur and fine texture tell you whether anything that matters moved.

## 6. Profile again, and fix what the patch costs

The second profile shows what the first one hid. With hk's attention in,
copies grew: the engine passes transposed views and the kernel wanted
contiguous tensors. A kernel variant that reads the engine's layout in place
(`attn_kernel(bnhd=True)`) removed them -- 29.1 -> 28.2 s, the same image to
the byte. Then measure the whole request again, and more than once: under
MiniMax-H3's layer-wise offload, where each denoise step streams the DiT from
host memory, the same arm varied by 20% between batches on a shared node --
more than some of the effects being measured. Interleave the arms, one
request each, and repeat. And let each shape decide for itself:
`sglang_diffusion` times hk's GEMM against the engine's once per (M bucket,
N, K) on the call's own operands and keeps the faster (`HK_SGLD_LOG=1`).

## 7. Pitfalls met on the way

- **Platform shims that lie.** sglang-radeon 0.1.5 makes SGLang's global
  platform report CUDA on RDNA3 to get one model past a check; SGLang then
  picks CUDA-only kernels. Qwen-Image failed with "invalid input for
  qknorm_complex_rope"; MiniMax-H3's worker died compiling a Triton kernel with
  inline PTX and the parent waited forever. Scope such shims to the module
  that needs them (`bench/e2e/qwen_image/rocm_fix_sitecustomize.py`).
- **A worker that dies without a traceback.** LLVM's fatal errors call
  `exit(1)`; Python's faulthandler sees nothing. An `LD_PRELOAD` shim that
  prints a native and a Python backtrace in `exit()` found the kernel in one
  run.
- **Measure in one process where you can.** Two processes on these GPUs
  differ by more than many kernel wins (docs/history/32); end-to-end arms are
  separate processes by necessity, so run each arm several times and report
  every run.
