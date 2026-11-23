"""Mean +- std over seeds of the headline macro AUROCs, from results/metrics<tag>_s<seed>.json
(run scripts/evaluate.py for each seed first). Writes results/seeds<tag>.csv.

Usage: uv run python scripts/seeds.py [--tag _e30]
"""

import argparse
import json

import numpy as np
import pandas as pd

from discipulus.data import FINDINGS, ROOT

RES = ROOT / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="")
    tag = p.parse_args().tag
    runs = [json.loads(f.read_text()) for f in sorted(RES.glob(f"metrics{tag}_s*.json"))]
    held = [FINDINGS.index(f) for f in runs[0]["held_out"]]
    seen = [i for i in range(14) if i not in held]
    cols = {"all 14": list(range(14)), "seen": seen, "held-out": held}
    rows = []
    for m in runs[0]["point"]:
        vals = {k: [np.mean(np.array(r["point"][m])[idx]) for r in runs] for k, idx in cols.items()}
        vals["IU-Xray"] = [np.mean(r["point_iu"][m]) for r in runs]
        rows.append({"model": m, **{k: f"{np.mean(v):.3f} +- {np.std(v):.3f}" for k, v in vals.items()}})
    t = pd.DataFrame(rows)
    t.to_csv(RES / f"seeds{tag}.csv", index=False)
    print(f"{len(runs)} seeds\n")
    print(t.to_markdown(index=False))


if __name__ == "__main__":
    main()
