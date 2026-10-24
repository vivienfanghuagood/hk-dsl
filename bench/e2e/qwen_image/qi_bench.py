"""qi_bench.py: Qwen-Image 2.1 in SGLang-Diffusion on one W7900D -- warm-up, then timed requests.

    python qi_bench.py --out DIR [--runs 3] [--steps 30] [--size 1024] [--backend torch_sdpa]
Writes DIR/result.json (wall seconds per run, versions, args) and DIR/img_<i>.png.
"""
import argparse, json, os, sys, time
os.environ.setdefault("SGLANG_USE_AITER", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

PROMPT = ("A cozy bookshop window at dusk, a hand-written chalk sign reading 'OPEN LATE', "
          "warm lamps, a ginger cat asleep on a stack of books, rain on the glass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--backend", default="torch_sdpa")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from sglang.multimodal_gen import DiffGenerator
    t0 = time.perf_counter()
    gen = DiffGenerator.from_pretrained(model_path=a.model, num_gpus=1, attention_backend=a.backend)
    load_s = time.perf_counter() - t0
    walls = []
    for i in range(a.runs + 1):
        t = time.perf_counter()
        r = gen.generate(sampling_params_kwargs=dict(
            prompt=PROMPT, height=a.size, width=a.size, num_inference_steps=a.steps, seed=a.seed,
            output_path=a.out, output_file_name=f"img_{i}", save_output=True))
        w = time.perf_counter() - t
        if r is None or (isinstance(r, list) and not r):
            raise SystemExit(f"run {i}: the request failed (see the log above)")
        print(f"run {i}{' (warm-up)' if i == 0 else ''}: {w:.2f} s", flush=True)
        if i:
            walls.append(w)
    import torch
    res = dict(args=vars(a), load_s=load_s, walls=walls, median=sorted(walls)[len(walls) // 2],
               torch=torch.__version__, hip=torch.version.hip, env={k: os.environ.get(k) for k in (
                   "SGLANG_USE_AITER", "MIOPEN_FIND_MODE", "HIP_VISIBLE_DEVICES", "HK_SGLD")})
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("RESULT", json.dumps(res), flush=True)
    gen.shutdown() if hasattr(gen, "shutdown") else None


if __name__ == "__main__":
    main()
