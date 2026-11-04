"""quality.py PORT MODEL OUT.json --data GSM8K.jsonl [--gsm N] [--chartqa N --chartqa-dir DIR] [--vqa N] [--conc 16]
Judge-free quality probe of a running OpenAI-compatible server (serve_bench.py runs it after the benches):
  - GSM8K test, N items sampled with a fixed seed from the split in its order, zero-shot chat, greedy,
    final number exact match;
  - ChartQA test, N items sampled with a fixed seed, greedy, the answer's exact or numeric match;
  - N synthetic images (random label text on a coloured background, 448-896 px), greedy,
    exact match of the label and of the colour word.
Per-item answers are saved, so two configurations can be compared item by item (paired)."""
import argparse, base64, os, http.client, io, json, random, re, threading
from concurrent.futures import ThreadPoolExecutor

COLORS = {"red": (200, 30, 30), "blue": (30, 60, 200), "green": (20, 140, 40), "black": (10, 10, 10),
          "orange": (240, 130, 20), "purple": (120, 40, 160)}


def chat(port, model, content, max_tokens):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    if "choices" not in d:
        return "ERROR " + json.dumps(d)[:200]
    return d["choices"][0]["message"].get("content") or ""


def _gsm_item(d):
    """A GSM8K line as {"q", "a"}: this form, or the dataset's own ("question", and "answer" ending in
    "#### N")."""
    if "q" in d:
        return d
    return {"q": d["question"], "a": float(d["answer"].split("####")[-1].strip().replace(",", ""))}


def num(s):
    m = re.findall(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", s, re.I)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def make_image(rnd, i):
    from PIL import Image, ImageDraw, ImageFont
    label = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)) + "-" + str(rnd.randrange(1000, 9999))
    color = rnd.choice(sorted(COLORS))
    size = rnd.choice([(448, 448), (896, 448), (672, 672), (896, 896)])
    im = Image.new("RGB", size, COLORS[color])
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size[1] // 7)
    ImageDraw.Draw(im).text((size[0] // 10, size[1] // 3), label, fill="white", font=font)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return label, color, base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--gsm", type=int, default=200)
    ap.add_argument("--vqa", type=int, default=0)
    ap.add_argument("--chartqa", type=int, default=200)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--data", required=True, help="GSM8K's test split as JSON lines (README.md: how to get it)")
    ap.add_argument("--chartqa-dir", help="ChartQA's test parquet files (lmms-lab/ChartQA's data/), for --chartqa")
    a = ap.parse_args()
    allg = [_gsm_item(json.loads(l)) for l in open(a.data)]
    items = [allg[i] for i in sorted(random.Random(1234).sample(range(len(allg)), a.gsm))]
    import glob
    import pyarrow.parquet as pq
    charts = []
    if a.chartqa:
        cq = pq.read_table(glob.glob(os.path.join(a.chartqa_dir, "test-*.parquet"))[0]).to_pylist()
        charts = [cq[i] for i in sorted(random.Random(5678).sample(range(len(cq)), a.chartqa))]
    rnd = random.Random(4747)
    imgs = [make_image(rnd, i) for i in range(a.vqa)]

    def gsm(it):
        q = it["q"] + "\nThink step by step, then end with: The answer is <number>."
        out = chat(a.port, a.model, q, 512)
        got = num(out)
        return {"kind": "gsm8k", "ref": it["a"], "got": got, "ok": got is not None and abs(got - float(it["a"])) < 1e-6,
                "text": out}

    def vqa(x):
        label, color, b64 = x
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                   {"type": "text", "text": "Copy the white text in the image exactly, then name the background "
                                            "colour in one English word. Answer as: TEXT; COLOUR"}]
        out = chat(a.port, a.model, content, 32)
        norm = re.sub(r"\s", "", out).upper()
        return {"kind": "vqa", "ref": f"{label}; {color}", "text": out,
                "ok_text": label.replace(" ", "").upper() in norm, "ok_color": color in out.lower(),
                "ok": label.upper() in norm and color in out.lower()}

    def fnum(x):
        x = x.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(x)
        except ValueError:
            return None

    def chart(it):
        b64 = base64.b64encode(it["image"]["bytes"]).decode()
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                   {"type": "text", "text": it["question"] + "\nAnswer the question with a single word or phrase."}]
        out = chat(a.port, a.model, content, 32)
        pred, ref = out.strip().splitlines()[0] if out.strip() else "", it["answer"]
        pn, rn = fnum(pred), fnum(ref)
        if pn is not None and rn is not None:
            ok = abs(pn - rn) <= 0.05 * abs(rn) if rn != 0 else pn == 0
        else:
            ok = pred.strip().rstrip(".").lower() == ref.strip().lower()
        return {"kind": "chartqa", "type": it["type"], "q": it["question"], "ref": ref, "text": out, "ok": ok}

    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(gsm, items)) + list(ex.map(chart, charts)) + list(ex.map(vqa, imgs))
    g = [r for r in res if r["kind"] == "gsm8k"]; v = [r for r in res if r["kind"] == "vqa"]
    summ = {"gsm8k_acc": sum(r["ok"] for r in g) / max(1, len(g)), "gsm8k_n": len(g),
            "vqa_acc": sum(r["ok"] for r in v) / max(1, len(v)), "vqa_text_acc": sum(r["ok_text"] for r in v) / max(1, len(v)),
            "vqa_color_acc": sum(r["ok_color"] for r in v) / max(1, len(v)), "vqa_n": len(v),
            "chartqa_acc": sum(r["ok"] for r in res if r["kind"] == "chartqa") / max(1, a.chartqa), "chartqa_n": a.chartqa,
            "errors": sum(r["text"].startswith("ERROR") for r in res)}
    json.dump({"summary": summ, "items": res}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("QUALITY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
