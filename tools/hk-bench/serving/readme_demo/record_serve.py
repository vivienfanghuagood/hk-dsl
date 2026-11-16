"""Record one arm of the README demo against a running `vllm serve` (OpenAI API), for a model too large to hold
two engines in one process (the scorecard's MiniCPM-V-4.7-35B-A3B, tensor parallel 2).

    python3 record_serve.py ARM MODEL OUT_DIR [--port 8000]      # ARM = stock or hk; writes OUT_DIR/ARM.json
    python3 record_serve.py merge OUT_DIR                         # both arms -> OUT_DIR/demo.json for render.py

Start the server for each arm the way scripts/scorecard/minicpmv-35b.sh does (stock: HK_DISABLE=1 and
--attention-backend ROCM_SEGMENTED_ATTN; hk: TRITON_ATTN; both CUDA graphs, the same TunableOp file), then run this.

The prompts are record.py's (repeated here): a generated shipping log plus a request to copy it, exactly DEMO_TOKENS token ids
(default 32,768), built from a fixed seed, so both arms get the same ids in the same order. Each request streams
exactly DEMO_OUT tokens (default 256, greedy, ignore_eos). Per round: when each token arrived (seconds since the
request was sent), the output text and token ids. Two warm-up requests come first; DEMO_ROUNDS rounds (default 3).
"""
import json
import os
import random
import sys
import time

import requests

TARGET = int(os.environ.get("DEMO_TOKENS", "32768"))
OUT_TOKENS = int(os.environ.get("DEMO_OUT", "256"))
ROUNDS = int(os.environ.get("DEMO_ROUNDS", "3"))
# record.py's prompt (record.py reads its arguments at import, so it is repeated here)
CITIES = ["Lisbon", "Osaka", "Tunis", "Quito", "Bergen", "Pune", "Hobart", "Accra", "Tallinn", "Cusco"]
GOODS = ["copper wire", "rice", "glass panels", "wool", "solar cells", "olive oil", "paper", "steel bolts"]
QUESTION = ("\n\nCopy the log above, from Entry 0 onwards, exactly as written, one entry per line. "
            "Do not add anything.")


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


def prompt_ids(rng, nonce, tok):
    """Exactly TARGET token ids: the chat template around the log + request, the log cut to fit."""
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


def stream(port, model, ids):
    body = {"model": model, "prompt": ids, "max_tokens": OUT_TOKENS, "min_tokens": OUT_TOKENS, "ignore_eos": True,
            "temperature": 0, "stream": True, "return_token_ids": True}
    t0 = time.perf_counter()
    ev, text, out = [], "", []
    with requests.post(f"http://127.0.0.1:{port}/v1/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            c = json.loads(line[6:])["choices"][0]
            new = c.get("token_ids") or ([None] if c.get("text") else [])   # None: server sent no ids
            if not new:
                continue
            text += c.get("text", "")
            out += new
            ev.append([time.perf_counter() - t0, len(out), text])
    return ev, out


def main():
    if sys.argv[1] == "merge":
        d = sys.argv[2]
        arms = {a: json.load(open(f"{d}/{a}.json")) for a in ("stock", "hk")}
        res = {k: arms["hk"][k] for k in ("model", "prompt_tokens", "out_tokens")}
        res["versions"] = {a: arms[a]["versions"] for a in arms}
        res["rounds"] = []
        for s, h in zip(arms["stock"]["rounds"], arms["hk"]["rounds"]):
            assert s["prompt_ids"] == h["prompt_ids"]
            same = s["output_ids"] == h["output_ids"] and s["events"][-1][2] == h["events"][-1][2]
            res["rounds"].append({"stock": s, "hk": h, "same_output": same})
            print(f"round {len(res['rounds']) - 1}: TTFT {s['events'][0][0]:.2f} / {h['events'][0][0]:.2f} s, "
                  f"done {s['events'][-1][0]:.2f} / {h['events'][-1][0]:.2f} s, same output {res['rounds'][-1]['same_output']}")
        json.dump(res, open(f"{d}/demo.json", "w"))
        return
    arm, model, out_dir = sys.argv[1:4]
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rng = random.Random(20261116)
    for w in range(2):
        stream(port, model, prompt_ids(rng, f"warm-{w}", tok))
    res = {"arm": arm, "model": os.path.basename(model.rstrip("/")), "prompt_tokens": TARGET,
           "out_tokens": OUT_TOKENS, "versions": requests.get(f"http://127.0.0.1:{port}/version").json(),
           "rounds": []}
    for r in range(ROUNDS):
        ids = prompt_ids(rng, f"{r}-{rng.randrange(10**6)}", tok)
        ev, out = stream(port, model, ids)
        res["rounds"].append({"prompt_tokens": len(ids), "prompt_ids": ids, "events": ev, "output_ids": out})
        print(f"{arm} round {r}: first token {ev[0][0]:.3f} s, {len(out)} tokens by {ev[-1][0]:.3f} s: "
              f"{ev[-1][2][:80]!r}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/{arm}.json", "w"))


if __name__ == "__main__":
    main()
