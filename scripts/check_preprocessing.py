"""Compare teacher embeddings from the uint8 224 store against CheXzero's exact float pipeline
(320 padded float image -> normalize -> bicubic 224 -> encoder) on images straight from a NIH zip."""

import io
import sys
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from discipulus.data import CXR_MEAN, CXR_STD, NIH_RAW, chexzero_pad, load_metadata, open_images, resize_224
from discipulus.teacher import encode_images, load_teacher

N = 256


def main(zip_name: str = "images_004.zip") -> None:
    z = zipfile.ZipFile(NIH_RAW / zip_name)
    members = [m for m in z.namelist() if m.startswith("images/") and m.endswith(".png")][:N]
    row = {name: i for i, name in enumerate(load_metadata()["image"])}
    store = open_images()
    model = load_teacher()
    exact, stored = [], []
    for m in members:
        x = np.asarray(chexzero_pad(Image.open(io.BytesIO(z.read(m))).convert("L")), dtype=np.float32)
        x = (torch.from_numpy(x) - CXR_MEAN) / CXR_STD
        exact.append(resize_224(x[None, None])[0].expand(3, -1, -1))
        stored.append(torch.from_numpy(np.array(store[row[m.rsplit("/", 1)[-1]]])))
    with torch.no_grad():
        e = F.normalize(model.encode_image(torch.stack(exact).cuda()), dim=-1)
    s = encode_images(model, torch.stack(stored).cuda())
    cos = (e * s).sum(-1)
    print(f"{N} images: cosine(exact, stored) min {cos.min():.6f} mean {cos.mean():.6f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
