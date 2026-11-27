"""All quantitative results: NIH test (per-finding, seen / held-out, retention, paraphrases), IU-Xray external,
student-teacher agreement. Writes results/*.csv, results/metrics.json and figures, prints markdown tables.

Usage: uv run python scripts/evaluate.py [--seed 0] [--tag _e30]   (tag selects runs/<arm><tag>_s<seed>)
"""

import argparse
import json
from typing import SupportsFloat

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from medical_vlm_distillation.data import FINDINGS, IU_CSV, IU_NPY, ROOT, load_splits
from medical_vlm_distillation.device import DEVICE
from medical_vlm_distillation.evaluate import auroc_table, boot_auroc, boot_weights, ci
from medical_vlm_distillation.student import make_student
from medical_vlm_distillation.teacher import (
    CACHE,
    HELD_OUT,
    TEMPLATES,
    encode_images,
    load_cache,
    load_teacher,
    prompt_probs,
)
from medical_vlm_distillation.train import RUNS, predict

OUT = ROOT / "results"
ARMS = {"supervised": "Supervised", "A": "A: embedding", "B": "B: score", "AB": "A+B"}
IU_MIN_POS = 20  # external findings need at least this many positive images to be reported


def iu_teacher_emb() -> torch.Tensor:
    path = CACHE.with_name(CACHE.stem + "_iu.pt")
    if not path.exists():
        model = load_teacher()
        store = np.load(IU_NPY)
        emb = torch.cat([encode_images(model, torch.from_numpy(b).to(DEVICE)).cpu() for b in np.array_split(store, 32)])
        torch.save(emb, path)
    return torch.load(path, weights_only=True)


def scores(arm: str, out: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
    """(N, templates, F) P(finding). Supervised has one 'template' (its own sigmoid head)."""
    if arm == "supervised":
        return out.sigmoid()[:, None].numpy()
    return prompt_probs(out.to(DEVICE), bank.to(DEVICE)).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    seed, tag = args.seed, args.tag
    OUT.mkdir(exist_ok=True)

    splits = load_splits()
    cache = load_cache()
    bank = cache["bank"]
    test_rows = np.flatnonzero(splits["split"] == "test")
    y_test = splits.loc[test_rows, FINDINGS].to_numpy()
    iu = pd.read_csv(IU_CSV)
    y_iu = iu[FINDINGS].to_numpy()
    iu_f = [i for i, f in enumerate(FINDINGS) if y_iu[:, i].sum() >= IU_MIN_POS]
    seen = [i for i, f in enumerate(FINDINGS) if f not in HELD_OUT]
    held = [i for i, f in enumerate(FINDINGS) if f in HELD_OUT]

    # name -> (test scores (N,T,F), iu scores, raw test output or None)
    models: dict[str, tuple[np.ndarray, np.ndarray, torch.Tensor | None]] = {}
    t_test = cache["emb"][test_rows].float()
    models["Teacher (CheXzero)"] = (scores("A", t_test, bank), scores("A", iu_teacher_emb(), bank), t_test)
    for arm, label in ARMS.items():
        run = RUNS / f"{arm}{tag}_s{seed}"
        if not (run / "preds.pt").exists():
            print(f"skip {label}: no {run}/preds.pt")
            continue
        preds = torch.load(run / "preds.pt", weights_only=False)
        assert np.array_equal(preds["rows"]["test"], test_rows)
        model = make_student(14 if arm == "supervised" else 512).to(DEVICE).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(run / "final.pt", map_location="cpu", weights_only=True))
        iu_out = predict(model, np.arange(len(iu)), IU_NPY)
        raw = None if arm == "supervised" else preds["test"]
        models[label] = (scores(arm, preds["test"], bank), scores(arm, iu_out, bank), raw)

    w_test = boot_weights(splits.loc[test_rows, "patient"].to_numpy())
    w_iu = boot_weights(iu["patient"].to_numpy())
    boots = {m: boot_auroc(y_test, s[:, 0], w_test) for m, (s, _, _) in models.items()}
    boots_iu = {m: boot_auroc(y_iu[:, iu_f], s[:, 0][:, iu_f], w_iu) for m, (_, s, _) in models.items()}
    point = {m: auroc_table(y_test, s[:, 0]) for m, (s, _, _) in models.items()}
    point_iu = {m: auroc_table(y_iu[:, iu_f], s[:, 0][:, iu_f]) for m, (_, s, _) in models.items()}
    teacher = "Teacher (CheXzero)"

    def fmt(x: SupportsFloat, lo: SupportsFloat, hi: SupportsFloat) -> str:
        return f"{float(x):.3f} [{float(lo):.3f}, {float(hi):.3f}]"

    # Main table: macro AUROC over all / seen / held-out findings, retention vs teacher, external.
    rows = []
    for m in models:
        r = {"model": m}
        for name, idx in (("all 14", list(range(14))), ("seen", seen), ("held-out", held)):
            b = boots[m][:, idx].mean(1)
            lo, hi = ci(b)
            r[f"test macro AUROC ({name})"] = fmt(point[m][idx].mean(), lo, hi)
        ret = boots[m].mean(1) / boots[teacher].mean(1)
        lo, hi = ci(ret)
        r["retention (all 14)"] = fmt(point[m].mean() / point[teacher].mean(), lo, hi)
        b = boots_iu[m].mean(1)
        lo, hi = ci(b)
        r[f"IU-Xray macro AUROC ({len(iu_f)} findings)"] = fmt(point_iu[m].mean(), lo, hi)
        rows.append(r)
    main_t = pd.DataFrame(rows)
    main_t.to_csv(OUT / f"main{tag}_s{seed}.csv", index=False)

    # Per-finding AUROC with CIs on NIH test.
    per = pd.DataFrame({"finding": FINDINGS, "held out": [f in HELD_OUT for f in FINDINGS]})
    for m in models:
        lo, hi = ci(boots[m])
        per[m] = [fmt(point[m][i], lo[i], hi[i]) for i in range(14)]
    per.to_csv(OUT / f"per_finding{tag}_s{seed}.csv", index=False)

    # Per-finding external.
    per_iu = pd.DataFrame({"finding": [FINDINGS[i] for i in iu_f], "positives": y_iu[:, iu_f].sum(0)})
    for m in models:
        lo, hi = ci(boots_iu[m])
        per_iu[m] = [fmt(point_iu[m][k], lo[k], hi[k]) for k in range(len(iu_f))]
    per_iu.to_csv(OUT / f"per_finding_iu{tag}_s{seed}.csv", index=False)

    # Paraphrases: macro AUROC (all 14) per template, spread over the 5 withheld paraphrases + the training pair.
    para = []
    for m, (s, _, _) in models.items():
        if s.shape[1] == 1:
            continue
        a = np.array([auroc_table(y_test, s[:, t]) for t in range(len(TEMPLATES))])
        r = {"model": m, **{f"{TEMPLATES[t][0]!r} / {TEMPLATES[t][1]!r}": f"{a[t].mean():.3f}" for t in range(len(a))}}
        r["withheld mean"] = f"{a[1:].mean():.3f}"
        r["spread (max-min, macro)"] = f"{a.mean(1).max() - a.mean(1).min():.3f}"
        r["mean per-finding std"] = f"{a.std(0).mean():.3f}"
        para.append(r)
    para_t = pd.DataFrame(para)
    para_t.to_csv(OUT / f"paraphrase{tag}_s{seed}.csv", index=False)

    # Agreement with the teacher on NIH test (distilled students only).
    agree = []
    p_t = models[teacher][0][:, 0]
    for m, (s, _, raw) in models.items():
        if raw is None or m == teacher:
            continue
        cos = F.cosine_similarity(raw, t_test, dim=-1)
        corr = [np.corrcoef(s[:, 0, f], p_t[:, f])[0, 1] for f in range(14)]
        agree.append(
            {
                "model": m,
                "cosine to teacher emb (mean)": f"{cos.mean():.3f}",
                "score corr (seen)": f"{np.mean([corr[i] for i in seen]):.3f}",
                "score corr (held-out)": f"{np.mean([corr[i] for i in held]):.3f}",
            }
        )
    agree_t = pd.DataFrame(agree)
    agree_t.to_csv(OUT / f"agreement{tag}_s{seed}.csv", index=False)

    torch.save({"test": boots, "iu": boots_iu}, OUT / f"boot{tag}_s{seed}.pt")
    (OUT / f"metrics{tag}_s{seed}.json").write_text(json.dumps({
        "held_out": HELD_OUT, "iu_findings": [FINDINGS[i] for i in iu_f],
        "n_test": len(test_rows), "n_test_patients": int(splits.loc[test_rows, "patient"].nunique()),
        "n_iu": len(iu), "point": {m: v.tolist() for m, v in point.items()},
        "point_iu": {m: v.tolist() for m, v in point_iu.items()},
    }, indent=2))  # fmt: skip

    tables = {
        "Main": main_t,
        "Per finding (NIH test)": per,
        "IU-Xray per finding": per_iu,
        "Prompt paraphrases (NIH test, macro over 14)": para_t,
        "Agreement with teacher": agree_t,
    }
    for title, t in tables.items():
        print(f"\n### {title}\n")
        print(t.to_markdown(index=False))


if __name__ == "__main__":
    main()
