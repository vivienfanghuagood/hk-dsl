# Examples

Fifteen examples -- eleven kernels written from scratch, from a vector add to
flash attention, two files of template kernels, two about gfx950's XCDs --
and `validate.py`, which checks the shipped `hk.ops` kernels against torch.
What each one shows, which GPUs it runs on, and what they leave out:
[docs/examples.md](../docs/examples.md).

```bash
python3 examples/01_vector_add.py              # on the GPU: check against torch, then time
python3 examples/04_gemm.py --dump ir          # the hk MLIR module, no GPU needed
python3 examples/04_gemm.py --dump cpp         # the HipKittens C++ it translates to
python3 examples/05_attention.py --build       # compile with hipcc and print the resources
python3 examples/validate.py --quick           # every shipped op, smaller shapes
```

`--dump` and `--build` need no GPU; `--arch gfx1201` (or `gfx942`, `gfx950`)
builds for a part that is not in the machine; `--no-bench` checks
correctness without timing.
