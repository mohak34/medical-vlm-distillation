"""Run the frozen teacher once over every NIH image; cache embeddings, prompt scores and the text bank.
Then report teacher zero-shot AUROC per finding on the validation split (CheXzero template)."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from discipulus.data import CXR, FINDINGS, load_splits
from discipulus.evaluate import auroc_table
from discipulus.teacher import CACHE, CKPT, TEMPLATES, encode_images, load_teacher, prompt_probs, text_bank


def main() -> None:
    splits = load_splits()
    model = load_teacher()
    loader = DataLoader(CXR(np.arange(len(splits))), batch_size=256, num_workers=8, pin_memory=True)
    emb = torch.empty(len(splits), 512, dtype=torch.float16)
    for i, (x, rows) in enumerate(loader):
        emb[rows] = encode_images(model, x.cuda(non_blocking=True)).half().cpu()
        if i % 50 == 0:
            print(f"{i * 256}/{len(splits)}", flush=True)
    bank = text_bank(model).cpu()
    probs = torch.cat([prompt_probs(e.cuda(), bank.cuda()).cpu() for e in emb.split(4096)])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"emb": emb, "probs": probs, "bank": bank, "logit_scale": model.logit_scale.detach().cpu(), "ckpt": CKPT.name},
        CACHE,
    )
    print(f"saved {CACHE}: emb {tuple(emb.shape)}, probs {tuple(probs.shape)}, bank {tuple(bank.shape)}")

    val = (splits["split"] == "val").to_numpy()
    y = splits.loc[val, FINDINGS].to_numpy()
    for t, pair in enumerate(TEMPLATES):
        a = auroc_table(y, probs.numpy()[val, t])
        print(f"val AUROC template {pair}: macro {a.mean():.4f}")
        if t == 0:
            for f, v in zip(FINDINGS, a, strict=True):
                print(f"  {f:20s} {v:.4f}")


if __name__ == "__main__":
    main()
