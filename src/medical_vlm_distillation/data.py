"""NIH ChestX-ray14: metadata, patient-disjoint splits, and the uint8 image store all arms read."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
NIH_RAW = DATA / "raw" / "nih"
IMAGES_NPY = DATA / "nih_224.npy"
SPLITS_CSV = DATA / "splits.csv"
IU_NPY = DATA / "iu_224.npy"
IU_CSV = DATA / "iu_labels.csv"

# NIH label names (column order everywhere) and the words used in prompts.
FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
PROMPT_NAMES = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural effusion",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural thickening",
    "Hernia": "hernia",
}

# CheXzero's input normalization, applied to raw 0..255 pixels (zero_shot.make in their repo).
CXR_MEAN = 101.48761
CXR_STD = 83.43944
RES = 224
VAL_PATIENT_FRAC = 0.1
SPLIT_SEED = 682


def chexzero_pad(img: Image.Image, size: int = 320) -> Image.Image:
    """CheXzero data_process.preprocess: scale the long side to `size` with LANCZOS, zero-pad to square."""
    ratio = size / max(img.size)
    new = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new, Image.Resampling.LANCZOS)
    out = Image.new("L", (size, size))
    out.paste(img, ((size - new[0]) // 2, (size - new[1]) // 2))
    return out


def resize_224(x: torch.Tensor) -> torch.Tensor:
    """Bicubic resize to 224 without antialiasing, as torchvision 0.11 Resize did on tensors in CheXzero."""
    return F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False)


def png_to_224(img: Image.Image) -> np.ndarray:
    """Full CheXzero image path up to normalization, stored as uint8 (normalize and resize commute)."""
    x = np.asarray(chexzero_pad(img.convert("L")), dtype=np.float32)
    y = resize_224(torch.from_numpy(x)[None, None])[0, 0]
    return y.round().clamp(0, 255).to(torch.uint8).numpy()


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B,H,W) -> float (B,3,H,W) in CheXzero's normalization."""
    x = (x_uint8.float() - CXR_MEAN) / CXR_STD
    return x[:, None].expand(-1, 3, -1, -1)


def load_metadata() -> pd.DataFrame:
    """One row per image, sorted by filename; row position is the index into the image store."""
    df = pd.read_csv(NIH_RAW / "Data_Entry_2017_v2020.csv")
    df = df.sort_values("Image Index").reset_index(drop=True)
    labels = df["Finding Labels"].str.split("|")
    for f in FINDINGS:
        df[f] = labels.map(lambda ls, f=f: int(f in ls))
    return df.rename(columns={"Image Index": "image", "Patient ID": "patient"})


def make_splits() -> pd.DataFrame:
    """Official NIH test list as test; train_val split 90/10 by patient. Asserts no patient crosses splits."""
    df = load_metadata()
    test_imgs = set((NIH_RAW / "test_list.txt").read_text().split())
    trainval_imgs = set((NIH_RAW / "train_val_list.txt").read_text().split())
    assert not test_imgs & trainval_imgs and len(test_imgs | trainval_imgs) == len(df)
    df["split"] = np.where(df["image"].isin(test_imgs), "test", "train")
    tv_patients = np.sort(df.loc[df["split"] == "train", "patient"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    val_patients = rng.choice(tv_patients, size=round(VAL_PATIENT_FRAC * len(tv_patients)), replace=False)
    df.loc[(df["split"] == "train") & df["patient"].isin(val_patients), "split"] = "val"
    assert_patient_disjoint(df)
    return df[["image", "patient", "split", *FINDINGS]]


def assert_patient_disjoint(df: pd.DataFrame) -> None:
    groups = {s: set(df.loc[df["split"] == s, "patient"]) for s in ("train", "val", "test")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(groups[a] & groups[b])
        assert n == 0, f"{n} patients in both {a} and {b}"


def load_splits() -> pd.DataFrame:
    df = pd.read_csv(SPLITS_CSV)
    assert_patient_disjoint(df)
    return df


def open_images(path: Path = IMAGES_NPY) -> np.ndarray:
    return np.load(path, mmap_mode="r")


class CXR(Dataset):
    """Rows of the image store. Returns (uint8 image, row); targets are gathered on the GPU by row."""

    def __init__(self, rows: np.ndarray, store: Path = IMAGES_NPY):
        self.rows = rows
        self.store = store
        self.images: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.images is None:  # open the memmap lazily so each worker gets its own handle
            self.images = open_images(self.store)
        j = int(self.rows[index])
        return torch.from_numpy(np.array(self.images[j])), j
