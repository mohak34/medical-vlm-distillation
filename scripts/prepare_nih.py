"""Decode NIH PNGs from the downloaded zips into data/nih_224.npy (uint8, N x 224 x 224) and write splits.csv.

Incremental: processes every zip that finished downloading and was not processed yet, so it can run
while the download is still going. Run again until it reports all 112,120 images present.
"""

import io
import zipfile
from multiprocessing import Pool

import numpy as np
from PIL import Image

from discipulus.data import DATA, IMAGES_NPY, NIH_RAW, RES, SPLITS_CSV, load_metadata, make_splits, png_to_224

DONE_LIST = DATA / "prepared_zips.txt"
_zips: dict[str, zipfile.ZipFile] = {}


def decode(task: tuple[str, str]) -> tuple[str, np.ndarray]:
    path, member = task
    if path not in _zips:
        _zips[path] = zipfile.ZipFile(path)
    img = Image.open(io.BytesIO(_zips[path].read(member)))
    return member.rsplit("/", 1)[-1], png_to_224(img)


def main() -> None:
    meta = load_metadata()
    row = {name: i for i, name in enumerate(meta["image"])}
    if IMAGES_NPY.exists():
        store = np.load(IMAGES_NPY, mmap_mode="r+")
    else:
        store = np.lib.format.open_memmap(IMAGES_NPY, mode="w+", dtype=np.uint8, shape=(len(meta), RES, RES))
    done = set(DONE_LIST.read_text().split()) if DONE_LIST.exists() else set()

    for z in sorted(NIH_RAW.glob("images_*.zip")):
        if z.name in done or not z.with_name(z.name + ".done").exists():
            continue
        members = [m for m in zipfile.ZipFile(z).namelist() if m.startswith("images/") and m.endswith(".png")]
        with Pool(12) as pool:
            for name, arr in pool.imap_unordered(decode, [(str(z), m) for m in members], chunksize=64):
                store[row[name]] = arr
        store.flush()
        with DONE_LIST.open("a") as f:
            f.write(z.name + "\n")
        done.add(z.name)
        print(f"{z.name}: {len(members)} images", flush=True)

    filled = sum(1 for z in done)
    print(f"{filled}/12 zips processed")
    if filled == 12:
        splits = make_splits()
        splits.to_csv(SPLITS_CSV, index=False)
        print(splits["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
