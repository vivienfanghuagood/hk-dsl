"""Render demo.json (demo_rec.py) as an asciicast v2: two panes side by side, each replaying its arm's recorded
token arrivals at real time from t = 0 (the request sent). agg turns the cast into the GIF.

    python3 render_cast.py demo.json ROUND out.cast --hk VERSION --caption "..."
"""
import argparse
import json
import textwrap

W, PANE, ROWS = 100, 48, 15
RESET, BOLD, DIM, GREEN, RED, CYAN, YELLOW = "\x1b[0m", "\x1b[1m", "\x1b[2m", "\x1b[32m", "\x1b[31m", "\x1b[36m", "\x1b[33m"


def pane(title, color, ptoks, t, ev, end):
    """The lines of one pane at time t (seconds since the request)."""
    seen = [e for e in ev if e[0] <= t]
    out = [f"{BOLD}{color}{title}{RESET}", f"{DIM}prompt: {ptoks:,} tokens{RESET}"]
    if not seen:
        out.append(f"waiting for the first token ... {YELLOW}{min(t, end):5.1f} s{RESET}")
        out.append("")
        body = []
    else:
        ttft = ev[0][0]
        out.append(f"first token: {BOLD}{GREEN}{ttft:5.2f} s{RESET}")
        n, tl = seen[-1][1], seen[-1][0]
        rate = (n - 1) / (tl - ttft) if n > 1 and tl > ttft else 0.0
        out.append(f"{n:3d} tokens{DIM}, {rate:5.0f} tok/s after the first{RESET}" if n > 1 else "")
        body = textwrap.wrap(seen[-1][2].replace("\n", " ").strip(), PANE - 2)
    out.append("")
    out += ["  " + b for b in body[-(ROWS - 7):]]
    return out


def frame(stock, hk, t, ends, caption):
    left = pane("stock vLLM", RED, stock["prompt_tokens"], t, stock["events"], ends[0])
    right = pane("vLLM + hk", CYAN, hk["prompt_tokens"], t, hk["events"], ends[1])
    rows = []
    for i in range(ROWS - 2):
        a = left[i] if i < len(left) else ""
        b = right[i] if i < len(right) else ""
        pad = PANE - len(strip(a))
        rows.append(a + " " * max(pad, 1) + f"{DIM}│{RESET} " + b)
    rows.append(f"{DIM}{'─' * (W - 1)}{RESET}")
    both = stock["events"][0][0] <= t and hk["events"][0][0] <= t
    ratio = (f"{BOLD}first token {stock['events'][0][0] / hk['events'][0][0]:.2f}x sooner with hk{RESET}{DIM}  ·  "
             if both else "")
    clock = f"t = {t:4.1f} s" + ("" if both else ", real time")
    foot = f"{ratio}{DIM}{caption}  ·  {clock}{RESET}"
    assert len(strip(foot)) < W, len(strip(foot))       # one line: a wrap would scroll the panes up
    rows.append(foot)
    return "\x1b[?25l\x1b[H\x1b[2J" + "\r\n".join(rows)


def strip(s):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("round", type=int)
    ap.add_argument("cast")
    ap.add_argument("--caption", required=True)
    ap.add_argument("--hold", type=float, default=0.6)
    ap.add_argument("--fps", type=float, default=10)
    a = ap.parse_args()
    d = json.load(open(a.demo))
    row = d["rounds"][a.round]
    stock, hk = row["stock"], row["hk"]
    ends = (stock["events"][-1][0], hk["events"][-1][0])
    t_end = max(ends) + a.hold
    times = sorted({round(i / a.fps, 3) for i in range(int(t_end * a.fps) + 1)}
                   | {e[0] for e in stock["events"]} | {e[0] for e in hk["events"]})
    with open(a.cast, "w") as f:
        f.write(json.dumps({"version": 2, "width": W, "height": ROWS, "timestamp": 0,
                            "env": {"TERM": "xterm-256color", "SHELL": "/bin/sh"}}) + "\n")
        for t in times:
            f.write(json.dumps([t, "o", frame(stock, hk, t, ends, a.caption)]) + "\n")


if __name__ == "__main__":
    main()
