#!/usr/bin/env bash
# Everything the pipeline reads: NIH ChestX-ray14 (HF mirror of the NIH release, ~45 GB), IU-Xray metadata,
# CheXzero checkpoints. Resumable; rerun until it prints ALL_DONE. IU images are fetched by prepare_iu.py.
set -euo pipefail
cd "$(dirname "$0")/.."
NIH=https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data
IU=https://huggingface.co/datasets/sasi2004/chest-xrays-indiana-university/resolve/main

mkdir -p data/raw/nih data/raw/iu checkpoints
for f in Data_Entry_2017_v2020.csv train_val_list.txt test_list.txt; do
  [ -e "data/raw/nih/$f" ] || curl -sfL -o "data/raw/nih/$f" "$NIH/$f"
done
for f in indiana_reports.csv indiana_projections.csv; do
  [ -e "data/raw/iu/$f" ] || curl -sfL -o "data/raw/iu/$f" "$IU/$f"
done
[ -d checkpoints/CheXzero_Models ] ||
  (cd checkpoints && uv run gdown --folder https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno)

# 12 image zips, 6 parallel connections (one connection is throttled to a few MB/s).
cd data/raw/nih
seq -w 1 12 | xargs -P 6 -I{} sh -c '
  f=images_0{}.zip
  [ -e $f.done ] && exit 0
  curl -sfL --retry 10 --retry-all-errors -C - -o $f "'"$NIH"'/images/$f" && touch $f.done && echo "$f done"'
echo ALL_DONE
