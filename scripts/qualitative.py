"""Poster figures: t-SNE of teacher and student embeddings, and Grad-CAM where teacher and student disagree.

Grad-CAM target for both models is the same prompt-score logit, cos(z, "finding") - cos(z, "no finding").
Student layer: ResNet-18 layer4 (7x7). Teacher layer: last ViT block tokens without CLS (7x7 patches).
Usage: uv run python scripts/qualitative.py [--arm A] [--seed 0]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from discipulus.data import FINDINGS, ROOT, load_splits, normalize, open_images
from discipulus.student import make_student
from discipulus.teacher import load_cache, load_teacher, prompt_probs
from discipulus.train import RUNS

OUT = ROOT / "results" / "figures"
GROUPS = ["No Finding", "Effusion", "Cardiomegaly", "Infiltration"]


def tsne_figure(seed: int, splits, cache) -> None:
    test = np.flatnonzero(splits["split"] == "test")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(test), 3000, replace=False)
    lab = splits.loc[test[pick], FINDINGS].to_numpy()
    # one colour per image: first matching group, single-label images only so colours are unambiguous
    group = np.full(len(pick), -1)
    group[lab.sum(1) == 0] = 0
    for g, f in enumerate(GROUPS[1:], start=1):
        group[(lab[:, FINDINGS.index(f)] == 1) & (lab.sum(1) == 1)] = g
    embs = {"Teacher": cache["emb"][test[pick]].float()}
    for arm, name in (("A", "A: embedding"), ("B", "B: score"), ("AB", "A+B")):
        p = RUNS / f"{arm}_s{seed}" / "preds.pt"
        if p.exists():
            embs[name] = torch.load(p, weights_only=False)["test"][pick]
    fig, axes = plt.subplots(1, len(embs), figsize=(4 * len(embs), 4))
    for ax, (name, e) in zip(np.atleast_1d(axes), embs.items(), strict=True):
        xy = TSNE(init="pca", random_state=0, perplexity=40).fit_transform(F.normalize(e, dim=-1).numpy())
        ax.scatter(xy[group == -1, 0], xy[group == -1, 1], s=2, c="#cccccc", label="other / multi-label")
        for g, gname in enumerate(GROUPS):
            m = group == g
            ax.scatter(xy[m, 0], xy[m, 1], s=3, label=gname)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(markerscale=4, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / f"tsne_s{seed}.png", dpi=200)
    print(f"saved {OUT / f'tsne_s{seed}.png'}")


def gradcam(acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
    """acts/grads (C,7,7) -> (224,224) map in [0,1]."""
    cam = F.relu((grads.mean((1, 2), keepdim=True) * acts).sum(0))
    cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def gradcam_figure(arm: str, seed: int, splits, cache, n: int = 4) -> None:
    bank = cache["bank"].cuda()
    test = np.flatnonzero(splits["split"] == "test")
    s_emb = torch.load(RUNS / f"{arm}_s{seed}" / "preds.pt", weights_only=False)["test"]
    p_s = prompt_probs(s_emb.cuda(), bank)[:, 0].cpu().numpy()
    p_t = prompt_probs(cache["emb"][test].float().cuda(), bank)[:, 0].cpu().numpy()
    # disagreement on the rank scale, positives only, for the finding where the two disagree most
    rank = lambda p: p.argsort(0).argsort(0) / len(p)  # noqa: E731
    gap = np.abs(rank(p_s) - rank(p_t))
    labels = splits.loc[test, FINDINGS].to_numpy()
    f = int((gap * labels).sum(0).argmax())
    cand = np.flatnonzero(labels[:, f] == 1)
    chosen = cand[np.argsort(-gap[cand, f])[:n]]

    teacher = load_teacher()
    student = make_student(512).cuda().eval()
    student.load_state_dict(torch.load(RUNS / f"{arm}_s{seed}" / "final.pt", weights_only=True))
    store = open_images()
    direction = bank[0, f, 0] - bank[0, f, 1]

    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.6 * n))
    for r, i in enumerate(chosen):
        x = normalize(torch.from_numpy(np.array(store[test[i]]))[None].cuda()).requires_grad_(False)
        maps = []
        layers = (teacher.get_submodule("visual.transformer.resblocks.11"), student.get_submodule("layer4"))
        for model, layer in zip((teacher, student), layers, strict=True):
            saved: dict[str, torch.Tensor] = {}
            h = layer.register_forward_hook(lambda _m, _i, o, saved=saved: saved.__setitem__("a", o))
            emb = model.encode_image(x) if model is teacher else model(x)
            h.remove()
            a = saved["a"]
            score = F.normalize(emb.float(), dim=-1)[0] @ direction
            (g,) = torch.autograd.grad(score, a)
            if model is teacher:  # tokens (L, N, C) -> (C, 7, 7), drop CLS
                a, g = (t[1:, 0].T.reshape(-1, 7, 7) for t in (a, g))
            else:
                a, g = a[0], g[0]
            maps.append(gradcam(a.float(), g.float()))
        img = np.array(store[test[i]])
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(f"{FINDINGS[f]} +  T {p_t[i, f]:.3f} / S {p_s[i, f]:.3f}", fontsize=7)
        for c, (title, m) in enumerate((("Teacher Grad-CAM", maps[0]), (f"Student {arm} Grad-CAM", maps[1])), 1):
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].imshow(m, cmap="jet", alpha=0.4)
            axes[r, c].set_title(title, fontsize=7)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    path = OUT / f"gradcam_{arm}_s{seed}.png"
    fig.savefig(path, dpi=200)
    print(f"saved {path} (finding {FINDINGS[f]}, test rows {test[chosen].tolist()})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="A")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    cache = load_cache()
    tsne_figure(a.seed, splits, cache)
    gradcam_figure(a.arm, a.seed, splits, cache)


if __name__ == "__main__":
    main()
