"""Train one arm. Every arm shares architecture, init, data order, epochs, optimizer and schedule;
only the head size and the loss differ. The last epoch is kept (no label-based model selection)."""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from medical_vlm_distillation.data import CXR, FINDINGS, IMAGES_NPY, ROOT, load_splits, normalize
from medical_vlm_distillation.device import DEVICE, PIN_MEMORY, empty_cache
from medical_vlm_distillation.evaluate import auroc_table
from medical_vlm_distillation.student import embedding_loss, make_student, score_loss
from medical_vlm_distillation.teacher import HELD_OUT, load_cache, prompt_probs

Arm = Literal["supervised", "A", "B", "AB"]
ARMS: tuple[Arm, ...] = ("supervised", "A", "B", "AB")
AB_WEIGHT = 1.0  # A+B loss = A + AB_WEIGHT * B, fixed before training
RUNS = ROOT / "runs"


def seen_mask() -> torch.Tensor:
    return torch.tensor([f not in HELD_OUT for f in FINDINGS])


@torch.no_grad()
def predict(model: torch.nn.Module, rows: np.ndarray, store: Path = IMAGES_NPY, batch: int = 512) -> torch.Tensor:
    """Raw head outputs (logits or embeddings), float32, in the order of `rows`."""
    model.eval()
    loader = DataLoader(CXR(rows, store), batch_size=batch, num_workers=8, pin_memory=PIN_MEMORY)
    out = []
    for x, _ in loader:
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            out.append(model(normalize(x.to(DEVICE, non_blocking=True)).contiguous(memory_format=torch.channels_last)))
    result = torch.cat(out).float().cpu()
    del out
    empty_cache()  # drop the batch-512 eval buffers before training resumes at batch 128
    return result


def train(
    arm: Arm,
    epochs: int,
    seed: int = 0,
    batch: int = 128,
    lr: float = 3e-4,
    limit: int | None = None,
    name: str | None = None,
) -> Path:
    out = RUNS / (name or f"{arm}_s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    log_f = (out / "train.log").open("a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    torch.manual_seed(seed)
    splits = load_splits()
    train_rows = np.flatnonzero(splits["split"] == "train")
    val_rows = np.flatnonzero(splits["split"] == "val")
    test_rows = np.flatnonzero(splits["split"] == "test")
    if limit:
        train_rows = np.sort(np.random.default_rng(seed).choice(train_rows, limit, replace=False))
        val_rows = train_rows  # overfit check: measure fit on the training images themselves

    cache = load_cache()
    t_emb = cache["emb"].to(DEVICE).float()
    bank = cache["bank"].to(DEVICE)
    scale = cache["logit_scale"].exp().to(DEVICE)
    seen = seen_mask().to(DEVICE)
    labels = torch.tensor(splits[FINDINGS].to_numpy(), dtype=torch.float32).to(DEVICE)

    model = make_student(14 if arm == "supervised" else 512).to(DEVICE).to(memory_format=torch.channels_last)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        CXR(train_rows),
        batch_size=batch,
        shuffle=True,
        drop_last=len(train_rows) > batch,
        num_workers=8,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        generator=gen,
    )
    steps = epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    cfg = {
        "arm": arm,
        "epochs": epochs,
        "seed": seed,
        "batch": batch,
        "lr": lr,
        "limit": limit,
        "n_train": len(train_rows),
        "held_out": HELD_OUT,
        "ab_weight": AB_WEIGHT,
        "teacher": cache["ckpt"],
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log(f"config {cfg}")

    step = 0
    for epoch in range(epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, rows in loader:
            x, rows = x.to(DEVICE, non_blocking=True), rows.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                y = model(normalize(x).contiguous(memory_format=torch.channels_last))
            y = y.float()
            if arm == "supervised":
                loss = F.binary_cross_entropy_with_logits(y, labels[rows])
            elif arm == "A":
                loss = embedding_loss(y, t_emb[rows])
            elif arm == "B":
                loss = score_loss(y, t_emb[rows], bank, seen, scale)
            else:
                loss = embedding_loss(y, t_emb[rows]) + AB_WEIGHT * score_loss(y, t_emb[rows], bank, seen, scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            total += loss.item() * len(rows)
            n += len(rows)
            if step % 100 == 0:
                log(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f}")
        msg = f"epoch {epoch} done: train loss {total / n:.4f}, {time.time() - t0:.0f}s"
        log(msg + ", " + monitor(arm, model, val_rows, t_emb, bank, labels))

    torch.save(model.state_dict(), out / "final.pt")
    preds = {s: predict(model, r) for s, r in (("val", val_rows), ("test", test_rows)) if not limit or s == "val"}
    torch.save({"rows": {"val": val_rows, "test": test_rows}, **preds}, out / "preds.pt")
    log(f"saved {out}")
    return out


def monitor(arm: Arm, model, rows: np.ndarray, t_emb, bank, labels) -> str:
    """Per-epoch numbers for the log only; never used to pick a checkpoint."""
    y = predict(model, rows)
    lab = labels[rows].cpu().numpy()
    both = (lab.min(0) == 0) & (lab.max(0) == 1)  # AUROC is undefined for a finding with one class in `rows`
    if arm == "supervised":
        return f"val macro AUROC {auroc_table(lab[:, both], y.sigmoid().numpy()[:, both]).mean():.4f}"
    cos = F.cosine_similarity(y.to(DEVICE), t_emb[rows], dim=-1).mean().item()
    p_s = prompt_probs(y.to(DEVICE), bank)[:, 0].cpu().numpy()
    p_t = prompt_probs(t_emb[rows], bank)[:, 0].cpu().numpy()
    a = auroc_table(lab[:, both], p_s[:, both]).mean()
    corr = np.mean([np.corrcoef(p_s[:, f], p_t[:, f])[0, 1] for f in range(p_s.shape[1])])
    return f"val cos-to-teacher {cos:.4f}, score corr {corr:.4f}, macro AUROC {a:.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=ARMS)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--limit", type=int, help="train on this many images (overfit check)")
    p.add_argument("--name")
    a = p.parse_args()
    train(a.arm, a.epochs, a.seed, a.batch, a.lr, a.limit, a.name)


if __name__ == "__main__":
    main()
