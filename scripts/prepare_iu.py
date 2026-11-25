"""External set: IU-Xray (Indiana University) frontal images with NIH findings mapped from MeSH terms.

Labels come from indiana_reports.csv / indiana_projections.csv (the Kaggle release of the Open-i reports,
mirrored on Hugging Face); images from the same mirror. Writes data/iu_224.npy and data/iu_labels.csv.
"""

import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from medical_vlm_distillation.data import DATA, FINDINGS, RES, png_to_224

RAW = DATA / "raw" / "iu"
MIRROR = "https://huggingface.co/datasets/sasi2004/chest-xrays-indiana-university/resolve/main/images/images_normalized"
# MeSH heading (text before the first "/") -> NIH finding. Pleural thickening needs the "pleura" qualifier.
MESH = {
    "Pulmonary Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Pleural Effusion": "Effusion",
    "Infiltrate": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Consolidation": "Consolidation",
    "Pulmonary Edema": "Edema",
    "Emphysema": "Emphysema",
    "Pulmonary Emphysema": "Emphysema",
    "Pulmonary Fibrosis": "Fibrosis",
    "Fibrosis": "Fibrosis",
    "Hernia, Hiatal": "Hernia",
    "Hernia, Diaphragmatic": "Hernia",
}


def labels_for(mesh: str) -> set[str]:
    out = set()
    for term in mesh.split(";"):
        parts = [p.strip() for p in term.split("/")]
        if parts[0] in MESH:
            out.add(MESH[parts[0]])
        if parts[0] == "Thickening" and len(parts) > 1 and parts[1] == "pleura":
            out.add("Pleural_Thickening")
    return out


def fetch(name: str) -> np.ndarray:
    path = RAW / "png" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        for attempt in range(8):
            try:
                with urllib.request.urlopen(f"{MIRROR}/{name}", timeout=120) as r:
                    tmp.write_bytes(r.read())
                break
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(2**attempt)
        tmp.rename(path)
    return png_to_224(Image.open(io.BytesIO(Path(path).read_bytes())))


def main() -> None:
    reports = pd.read_csv(RAW / "indiana_reports.csv")
    reports = reports[~reports["MeSH"].str.contains("No Indexing")]
    proj = pd.read_csv(RAW / "indiana_projections.csv")
    df = proj[proj["projection"] == "Frontal"].merge(reports[["uid", "MeSH"]], on="uid")
    df = df.sort_values("filename").reset_index(drop=True)
    labels = df["MeSH"].map(labels_for)
    for f in FINDINGS:
        df[f] = labels.map(lambda s, f=f: int(f in s))
    with ThreadPoolExecutor(8) as pool:
        images = list(pool.map(fetch, df["filename"]))
    store = np.lib.format.open_memmap(DATA / "iu_224.npy", mode="w+", dtype=np.uint8, shape=(len(df), RES, RES))
    store[:] = np.stack(images)
    store.flush()
    df.drop(columns="MeSH").rename(columns={"uid": "patient"}).to_csv(DATA / "iu_labels.csv", index=False)
    print(f"{len(df)} frontal images, {df['uid'].nunique()} reports")
    print(df[FINDINGS].sum().to_string())


if __name__ == "__main__":
    main()
