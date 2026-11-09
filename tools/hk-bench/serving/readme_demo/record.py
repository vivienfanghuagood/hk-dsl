"""Record the README demo: stock vLLM vs vLLM + hk on one GPU, the same prompt to both, the answer streamed.

    python3 record.py SPEC.json OUTDIR

SPEC.json: {"model": path or hub id, "engine": {vllm.LLM keyword arguments}}. Both arms are built in ONE
process (VLLM_ENABLE_V1_MULTIPROCESSING=0), as docs/performance/scorecard.md's vLLM rows are measured:
"stock" with every hk vLLM route reverted, "hk" with the installed wheel's default-on routes
(hk.integration._vllm_plugin.register()). Set PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_FILENAME
to a tuned results file to give both arms the scorecard's tuned GEMMs. Each arm is warmed first, unrecorded.
Then DEMO_ROUNDS rounds (default 5), arm order alternating. A round builds ONE prompt of exactly DEMO_TOKENS
tokens (default 8192): a generated shipping log with a fresh first line and a request to copy it out, which a
greedy model answers the same way under both arms' rounding. Both arms get those same token ids; the spec
turns prefix caching off, so neither reuses the other's (or an earlier round's) KV cache. Each streams
exactly DEMO_OUT tokens (default 256): greedy, ignore_eos. Every engine step's arrival time, token count and
text are recorded, with the output token ids and hk's call counters around the request.
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
ROUNDS = int(os.environ.get("DEMO_ROUNDS", "5"))
TARGET = int(os.environ.get("DEMO_TOKENS", "8192"))
OUT_TOKENS = int(os.environ.get("DEMO_OUT", "256"))
CITIES = ["Lisbon", "Osaka", "Tunis", "Quito", "Bergen", "Pune", "Hobart", "Accra", "Tallinn", "Cusco"]
GOODS = ["copper wire", "rice", "glass panels", "wool", "solar cells", "olive oil", "paper", "steel bolts"]


def log(*a):
    print("[demo]", *a, flush=True)


def set_routes(arm):
    """hk's routes off (stock) or on (hk) in this process, where the model runs."""
    import pkgutil

    import hk.integration
    from hk.integration import _vllm_plugin
    for mod in sorted(m.name for m in pkgutil.iter_modules(hk.integration.__path__) if m.name.startswith("vllm")):
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
    for mod in ("vllm_norm", "vllm_gemm", "vllm_int4", "vllm_gptq", "vllm_quark", "vllm_moe", "vllm_moe_wna16"):
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


def run(llm, prompts, out_len, ignore_eos=True, ids=None):
    """Add the requests, step the engine to the end; per request [(t since add, tokens, text)]
    (and each request's output token ids appended to `ids`)."""
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
    last = {}
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.request_id in ev and o.outputs and len(o.outputs[0].token_ids):
                ev[o.request_id].append([time.perf_counter() - t0[o.request_id], len(o.outputs[0].token_ids),
                                         o.outputs[0].text])
                last[o.request_id] = list(o.outputs[0].token_ids)
    if ids is not None:
        ids += [last.get(r, []) for r in ev]
    return list(ev.values())


def warm(llm, rng):
    rand = lambda n: [rng.randrange(1000, 30000) for _ in range(n)]  # noqa: E731
    for c in (1, 4, 8):
        run(llm, [rand(1024) for _ in range(c)], 16)
    for n in (TARGET, TARGET):
        run(llm, [rand(n)], 32)


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
    return f"Shipping log, session {nonce}.\n\n" + "".join(paras)


QUESTION = ("\n\nCopy the log above, from Entry 0 onwards, exactly as written, one entry per line. "
            "Do not add anything.")


def prompt_ids(rng, nonce, tok):
    """Exactly TARGET token ids: the chat template around the log + request, the log's token ids cut to fit
    (its last entry may end mid-line; the copy starts at Entry 0)."""
    marker = "@@LOG@@"
    text = tok.apply_chat_template([{"role": "user", "content": marker + QUESTION}],
                                   add_generation_prompt=True, tokenize=False, enable_thinking=False)
    pre, post = text.split(marker)
    pre_ids = tok.encode(pre, add_special_tokens=False)
    post_ids = tok.encode(post, add_special_tokens=False)
    room = TARGET - len(pre_ids) - len(post_ids)
    doc_ids = tok.encode(document(rng, nonce, tok, room + 64), add_special_tokens=False)
    assert len(doc_ids) >= room, (len(doc_ids), room)
    ids = pre_ids + doc_ids[:room] + post_ids
    assert len(ids) == TARGET, len(ids)
    return ids


def main():
    import torch
    import hk
    import vllm
    rng = random.Random(20261109)
    tuned = prewarm_tunableop()
    engines = {a: build(a) for a in ARMS}
    tok = engines["hk"].get_tokenizer()
    for a in ARMS:
        set_routes(a)
        warm(engines[a], rng)
        run(engines[a], [prompt_ids(rng, f"warm-{a}", tok)], OUT_TOKENS)
        log("warm", a, "done")
    res = {"spec": SPEC, "versions": {"hk": hk.__version__, "vllm": vllm.__version__, "torch": torch.__version__,
                                      "gpu": torch.cuda.get_device_name(0)},
           "tunableop_prewarmed": tuned, "prompt_tokens": TARGET, "out_tokens": OUT_TOKENS, "rounds": []}
    for r in range(ROUNDS):
        order = ARMS if r % 2 == 0 else ARMS[::-1]
        ids = prompt_ids(rng, f"{r}-{rng.randrange(10**6)}", tok)
        row = {"order": order, "prompt_head": tok.decode(ids[:40])}
        for a in order:
            set_routes(a)
            c0 = counters()
            out = []
            (ev,) = run(engines[a], [ids], OUT_TOKENS, ids=out)
            row[a] = {"prompt_tokens": len(ids), "events": ev, "output_ids": out[0],
                      "counters_before": c0, "counters_after": counters()}
            log(f"round {r} {a}: {len(ids)} prompt tokens, first token {ev[0][0]:.3f} s, {ev[-1][1]} tokens by "
                f"{ev[-1][0]:.3f} s: {ev[-1][2][:100]!r}")
        row["same_output"] = row["stock"]["output_ids"] == row["hk"]["output_ids"]
        log(f"round {r}: outputs identical: {row['same_output']}")
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
