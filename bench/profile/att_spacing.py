"""How far apart a thread trace puts back-to-back matrix instructions.

    python3 bench/profile/att_spacing.py KEEP_DIR

KEEP_DIR is a `python -m hk.profile ... --keep DIR`. Prints, per matrix
mnemonic, the distribution of the gap between a wave's consecutive matrix
instructions when nothing else issued between them: the matrix pipe's
occupancy per instruction in the trace's time unit, which is what
hk.profile.timeline.SIMD_FLOPS_PER_CYCLE models.
"""

from __future__ import annotations

import collections
import glob
import json
import sys


def main(keep: str) -> int:
    dirs = glob.glob(f"{keep}/**/ui_output_agent_*", recursive=True)
    if not dirs:
        print("no thread trace under", keep)
        return 2
    d = sorted(dirs)[-1]
    code = json.load(open(f"{d}/code.json"))["code"]
    gaps = collections.defaultdict(collections.Counter)
    for f in glob.glob(f"{d}/se*_wv*.json"):
        ins = json.load(open(f))["wave"]["instructions"]
        for a, b in zip(ins, ins[1:]):
            ma, mb = str(code[a[4]][0]).split()[0], str(code[b[4]][0]).split()[0]
            if ma.startswith(("v_mfma", "v_wmma")) and mb.startswith(("v_mfma", "v_wmma")):
                gaps[ma][b[0] - a[0]] += 1
    for mn, c in gaps.items():
        print(mn, c.most_common(6))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
