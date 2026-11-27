"""Frozen CheXzero teacher: model loading, prompt templates, text bank, prompt scores, cached outputs."""

from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from medical_vlm_distillation.chexzero_model import CLIP, build_model
from medical_vlm_distillation.data import DATA, FINDINGS, PROMPT_NAMES, ROOT, normalize
from medical_vlm_distillation.device import DEVICE

CKPT_DIR = ROOT / "checkpoints" / "CheXzero_Models"
# Pinned: highest validation score of the authors' released checkpoints (0.864, tied; first by name).
CKPT = CKPT_DIR / "best_64_5e-05_original_22000_0.864.pt"
CACHE = DATA / "cache" / f"teacher_{CKPT.stem}.pt"

# (positive, negative) templates. Index 0 is CheXzero's own pair and the only one used in training.
# The rest are withheld paraphrases, fixed before any student was trained.
TEMPLATES: list[tuple[str, str]] = [
    ("{}", "no {}"),
    ("there is {}", "there is no {}"),
    ("evidence of {}", "no evidence of {}"),
    ("{} is present", "{} is absent"),
    ("findings consistent with {}", "no findings of {}"),
    ("{} is seen", "no {} is seen"),
]


def load_teacher(ckpt: Path = CKPT, device: str = DEVICE) -> CLIP:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    return build_model(sd).float().eval().to(device)


@torch.no_grad()
def encode_images(model: CLIP, x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B,224,224) on device -> L2-normalized (B,512)."""
    return F.normalize(model.encode_image(normalize(x_uint8)), dim=-1)


@torch.no_grad()
def text_bank(model: CLIP, device: str = DEVICE) -> torch.Tensor:
    """(templates, findings, 2 = pos/neg, 512), each prompt embedded alone and L2-normalized (as CheXzero)."""
    texts = [t.format(PROMPT_NAMES[f]) for pos_neg in TEMPLATES for f in FINDINGS for t in pos_neg]
    tokens = open_clip.tokenize(texts, context_length=77).to(device)
    emb = F.normalize(model.encode_text(tokens).float(), dim=-1)
    return emb.view(len(TEMPLATES), len(FINDINGS), 2, -1)


def prompt_logits(emb: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of (B,512) embeddings with the bank -> (B,templates,findings,2)."""
    return torch.einsum("bd,tfkd->btfk", F.normalize(emb.float(), dim=-1), bank)


def prompt_probs(emb: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """CheXzero's score: softmax over (pos, neg) cosine similarities, temperature 1. Returns P(pos), (B,T,F)."""
    return prompt_logits(emb, bank).softmax(-1)[..., 0]


def load_cache() -> dict[str, torch.Tensor]:
    """Keys: emb (N,512) float16 for every row of the image store, bank (T,F,2,512), logit_scale ()."""
    return torch.load(CACHE, weights_only=True)


# Findings arm B never trains on. Chosen from teacher validation AUROC (>= 0.65) before any student run.
HELD_OUT: list[str] = ["Cardiomegaly", "Pneumothorax", "Emphysema", "Mass"]
