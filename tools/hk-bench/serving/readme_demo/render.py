"""Render demo.json (record.py) as an asciicast v2: two panes side by side, each replaying its arm's recorded
token arrivals at real time from t = 0 (the request sent). agg turns the cast into the GIF.

    python3 render.py demo.json ROUND out.cast --caption "..."

Each pane shows the input and output token counts, the time to the first token, the decode rate so far
(tokens after the first / time since the first), the elapsed time, and the text. When both arms are done the
footer gives hk's two ratios: time to first token (stock / hk) and decode rate (hk / stock).
"""
import argparse
import json
import re
import textwrap

W, PANE, ROWS = 100, 48, 17
RESET, BOLD, DIM, GREEN, RED, CYAN, YELLOW = "\x1b[0m", "\x1b[1m", "\x1b[2m", "\x1b[32m", "\x1b[31m", "\x1b[36m", "\x1b[33m"


def decode_rate(ev, upto=None):
    """Tokens after the first per second since the first, over the events up to `upto`."""
    seen = [e for e in ev if upto is None or e[0] <= upto]
    if len(seen) < 2 or seen[-1][1] < 2:
        return None
    return (seen[-1][1] - seen[0][1]) / (seen[-1][0] - seen[0][0])


def pane(title, color, arm, t, out_max):
    """The lines of one pane at time t (seconds since the request)."""
    ev = arm["events"]
    seen = [e for e in ev if e[0] <= t]
    end = ev[-1][0]
    n = seen[-1][1] if seen else 0
    done = t >= end
    out = [f"{BOLD}{color}{title}{RESET}",
           f"input  {arm['prompt_tokens']:,} tokens   output {n:3d} / {out_max} tokens"]
    if not seen:
        out.append(f"first token   {YELLOW}waiting ...{RESET}")
    else:
        out.append(f"first token   {BOLD}{GREEN}{ev[0][0]:5.2f} s{RESET}")
    r = decode_rate(ev, t)
    out.append(f"decode        {BOLD}{r:5.1f} tok/s{RESET}" if r else f"decode        {DIM}  -{RESET}")
    out.append(f"total         {min(t, end):5.2f} s" + (f"  {GREEN}done{RESET}" if done else ""))
    out.append("")
    body = textwrap.wrap(seen[-1][2].replace("\n", " ").strip(), PANE - 3) if seen else []
    out += ["  " + b for b in body[-(ROWS - 9):]]
    return out


def frame(stock, hk, t, out_max, caption):
    left = pane("stock vLLM", RED, stock, t, out_max)
    right = pane("vLLM + hk", CYAN, hk, t, out_max)
    rows = []
    for i in range(ROWS - 3):
        a = left[i] if i < len(left) else ""
        b = right[i] if i < len(right) else ""
        rows.append(a + " " * max(PANE - len(strip(a)), 1) + f"{DIM}│{RESET} " + b)
    rows.append(f"{DIM}{'─' * (W - 1)}{RESET}")
    if t >= max(stock["events"][-1][0], hk["events"][-1][0]):
        ttft = stock["events"][0][0] / hk["events"][0][0]
        dec = decode_rate(hk["events"]) / decode_rate(stock["events"])
        res = f"{BOLD}with hk: first token {ttft:.2f}x sooner, decode {dec:.2f}x faster{RESET}"
    else:
        res = f"{DIM}t = {t:4.1f} s, real time{RESET}"
    rows.append(res)
    foot = f"{DIM}{caption}{RESET}"
    for r in rows[-2:] + [foot]:
        assert len(strip(r)) < W, len(strip(r))       # one line each: a wrap would scroll the panes up
    rows.append(foot)
    return "\x1b[?25l\x1b[H\x1b[2J" + "\r\n".join(rows)


def strip(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("round", type=int)
    ap.add_argument("cast")
    ap.add_argument("--caption", required=True)
    ap.add_argument("--hold", type=float, default=2.5)
    ap.add_argument("--fps", type=float, default=10)
    a = ap.parse_args()
    d = json.load(open(a.demo))
    row = d["rounds"][a.round]
    stock, hk = row["stock"], row["hk"]
    assert stock["prompt_tokens"] == hk["prompt_tokens"] and stock["output_ids"] == hk["output_ids"], \
        "the two arms' prompts or outputs differ"
    out_max = d["out_tokens"]
    t_end = max(stock["events"][-1][0], hk["events"][-1][0]) + a.hold
    times = sorted({round(i / a.fps, 3) for i in range(int(t_end * a.fps) + 1)})
    with open(a.cast, "w") as f:
        f.write(json.dumps({"version": 2, "width": W, "height": ROWS, "timestamp": 0,
                            "env": {"TERM": "xterm-256color", "SHELL": "/bin/sh"}}) + "\n")
        for t in times:
            f.write(json.dumps([t, "o", frame(stock, hk, t, out_max, a.caption)]) + "\n")


if __name__ == "__main__":
    main()
