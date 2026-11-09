"""Record the README demo: stock vLLM vs vLLM + hk on one GPU, one long prompt, the answer streamed.

    python3 record.py SPEC.json OUTDIR

SPEC.json: {"model": path or hub id, "engine": {vllm.LLM keyword arguments}}. Both arms are built in ONE
process (VLLM_ENABLE_V1_MULTIPROCESSING=0), as docs/performance/scorecard.md's vLLM rows are measured:
"stock" with hk's routes reverted, "hk" with the installed wheel's default-on routes
(hk.integration._vllm_plugin.register()). Set PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_FILENAME
to a tuned results file to give both arms the scorecard's tuned GEMMs. Each arm is warmed first (8k and
32k-token prompts, a few decode batches), unrecorded. Then DEMO_ROUNDS rounds (default 3), arm order
alternating: each arm gets one fresh DEMO_TOKENS-token request (default 32768; a document with a unique first
line, so no prefix-cache hit, and a question about it) and streams DEMO_OUT tokens (default 48), greedy; every
engine step's arrival time and the text so far are recorded, with hk's call counters around the request.
Writes OUTDIR/demo.json; render.py turns a round into the asciicast the GIF is made from.
"""
import gc
import json
import os
import random
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_PLUGINS"] = "none"
os.environ.pop("PYTORCH_TUNABLEOP_RECORD_UNTUNED", None)

SPEC = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ARMS = ["stock", "hk"]
ROUNDS = int(os.environ.get("DEMO_ROUNDS", "3"))
TARGET = int(os.environ.get("DEMO_TOKENS", "32768"))
OUT_TOKENS = int(os.environ.get("DEMO_OUT", "48"))
REVERTS = ("vllm_prefill", "vllm_gemm", "vllm_int4", "vllm_gptq", "vllm_quark", "vllm_moe", "vllm_moe_wna16",
           "vllm_moe_a8w4", "vllm_mxfp8", "vllm_dsv41_index", "vllm_sparse_mla", "vllm")
CITIES = ["Lisbon", "Osaka", "Tunis", "Quito", "Bergen", "Pune", "Hobart", "Accra", "Tallinn", "Cusco"]
GOODS = ["copper wire", "rice", "glass panels", "wool", "solar cells", "olive oil", "paper", "steel bolts"]


def log(*a):
    print("[demo]", *a, flush=True)


def set_routes(arm):
    """hk's routes off (stock) or on (hk) in this process, where the model runs."""
    from hk.integration import _vllm_plugin
    for mod in REVERTS:
        try:
            __import__(f"hk.integration.{mod}", fromlist=["x"]).revert()
        except Exception:  # noqa: BLE001 -- a route this wheel or GPU does not have
            pass
    if arm == "hk":
        _vllm_plugin.register()


def counters():
    out = {}
    try:
        from hk.integration import vllm_prefill as vp
        out["vllm_prefill"] = dict(vp.PATCH.stats)
    except Exception as e:  # noqa: BLE001
        out["vllm_prefill"] = repr(e)
    for mod in ("vllm_gemm", "vllm_int4", "vllm_gptq", "vllm_quark", "vllm_moe", "vllm_moe_wna16"):
        try:
            m = __import__(f"hk.integration.{mod}", fromlist=["x"])
            st = getattr(m, "STATS", None) or getattr(getattr(m, "PATCH", None), "stats", None)
            if st is not None:
                out[mod] = dict(st)
        except Exception:  # noqa: BLE001
            pass
    return out


def prewarm_tunableop():
    """Run the tuned GEMMs once, eagerly, before an engine captures a graph (a tuned hipBLASLt solution's
    first use is a lookup, which is not allowed while a stream is capturing)."""
    import csv
    import torch
    f = os.environ.get("PYTORCH_TUNABLEOP_FILENAME")
    if not f or os.environ.get("PYTORCH_TUNABLEOP_ENABLED") != "1":
        return 0
    n = 0
    for r in csv.reader(open(f.replace(".csv", "0.csv"))):
        if r and r[0] == "GemmTunableOp_BFloat16_TN":
            nn, m, k = (int(x) for x in r[1].split("_")[1:4])
            torch.nn.functional.linear(torch.zeros(m, k, device="cuda", dtype=torch.bfloat16),
                                       torch.zeros(nn, k, device="cuda", dtype=torch.bfloat16))
            n += 1
    torch.cuda.synchronize()
    return n


def build(arm):
    from vllm import LLM
    set_routes(arm)
    os.environ["VLLM_PLUGINS"] = "hk_attention" if arm == "hk" else "none"
    t = time.time()
    llm = LLM(SPEC["model"], seed=0, **SPEC["engine"])
    log(f"built {arm} in {time.time() - t:.0f} s")
    return llm


def run(llm, prompts, out_len, ignore_eos=True):
    """Add the requests, step the engine to the end; per request [(t since add, tokens, text)]."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    eng = llm.llm_engine
    sp = SamplingParams(max_tokens=out_len, ignore_eos=ignore_eos, temperature=0.0)
    t0, ev = {}, {}
    for i, p in enumerate(prompts):
        rid = f"r{time.time_ns()}-{i}"
        t0[rid] = time.perf_counter()
        ev[rid] = []
        eng.add_request(rid, TokensPrompt(prompt_token_ids=p), sp)
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.request_id in ev and o.outputs and len(o.outputs[0].token_ids):
                ev[o.request_id].append([time.perf_counter() - t0[o.request_id], len(o.outputs[0].token_ids),
                                         o.outputs[0].text])
    return list(ev.values())


def warm(llm, rng):
    rand = lambda n: [rng.randrange(1000, 30000) for _ in range(n)]  # noqa: E731
    for c in (1, 4, 8):
        run(llm, [rand(1024) for _ in range(c)], 16)
    for n in (8192, 32768):
        run(llm, [rand(n)], 1)


def document(rng, nonce, tok, budget):
    paras, n, i = [], 0, 0
    while True:
        p = (f"Entry {i}: the warehouse in {rng.choice(CITIES)} shipped {rng.randrange(20, 900)} crates of "
             f"{rng.choice(GOODS)} on day {rng.randrange(1, 365)}, and {rng.randrange(2, 40)} more were held "
             f"back for inspection.\n")
        n += len(tok.encode(p, add_special_tokens=False))
        if n > budget:
            break
        paras.append(p)
        i += 1
    paras.insert(len(paras) * 2 // 3, "Note from the harbour master: the vault code for this season is 7319.\n")
    return f"Shipping log, session {nonce}.\n\n" + "".join(paras)


def prompt_ids(rng, nonce, tok):
    q = "\n\nQuestion: what is the vault code, and which note gives it? Answer in two short sentences."
    budget = TARGET - 200
    for _ in range(4):
        ids = tok.apply_chat_template([{"role": "user", "content": document(rng, nonce, tok, budget) + q}],
                                      add_generation_prompt=True, tokenize=True, enable_thinking=False)
        if hasattr(ids, "keys"):            # newer transformers return a BatchEncoding
            ids = ids["input_ids"]
        if abs(len(ids) - TARGET) <= 64:
            break
        budget += TARGET - len(ids)
    return ids


def main():
    import torch
    from transformers import AutoTokenizer
    import hk
    import vllm
    tok = AutoTokenizer.from_pretrained(SPEC["model"])
    rng = random.Random(20261109)
    tuned = prewarm_tunableop()
    engines = {a: build(a) for a in ARMS}
    for a in ARMS:
        set_routes(a)
        warm(engines[a], rng)
        run(engines[a], [prompt_ids(rng, f"warm-{a}", tok)], OUT_TOKENS, ignore_eos=False)
        log("warm", a, "done")
    res = {"spec": SPEC, "versions": {"hk": hk.__version__, "vllm": vllm.__version__, "torch": torch.__version__,
                                      "gpu": torch.cuda.get_device_name(0)},
           "tunableop_prewarmed": tuned, "rounds": []}
    for r in range(ROUNDS):
        order = ARMS if r % 2 == 0 else ARMS[::-1]
        row = {"order": order}
        for a in order:
            set_routes(a)
            ids = prompt_ids(rng, f"{r}-{a}-{rng.randrange(10**6)}", tok)
            c0 = counters()
            (ev,) = run(engines[a], [ids], OUT_TOKENS, ignore_eos=False)
            row[a] = {"prompt_tokens": len(ids), "events": ev, "counters_before": c0, "counters_after": counters()}
            log(f"round {r} {a}: {len(ids)} prompt tokens, first token {ev[0][0]:.3f} s, {ev[-1][1]} tokens by "
                f"{ev[-1][0]:.3f} s: {ev[-1][2][:100]!r}")
        res["rounds"].append(row)
        json.dump(res, open(f"{OUT}/demo.json", "w"))
    log("done", res["versions"])
    for e in engines.values():
        try:
            e.llm_engine.engine_core.shutdown()
        except Exception:  # noqa: BLE001
            pass
    gc.collect()


if __name__ == "__main__":
    main()
