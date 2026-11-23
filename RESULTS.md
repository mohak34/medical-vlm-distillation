# Results

Every number below comes with the command that produced it. AUROC CIs are 95% patient-level bootstrap intervals (1000 resamples, same resamples for every model, so the comparisons are paired). The student tables fill in as runs finish.

## 1. Data and split audit

`uv run python scripts/prepare_nih.py` then the audit in `load_splits` + `assert_patient_disjoint`:

| split | images | patients |
|---|---:|---:|
| train | 78,025 | 25,207 |
| val | 8,499 | 2,801 |
| test (official NIH test list) | 25,596 | 2,797 |

- Patient overlap between the splits is zero (asserted).
- train + val = 86,524 = the official train_val list. All 112,120 images were decoded, and none is blank.
- Positive rate is higher in test than in train or val for every finding. For example, Pneumothorax is 10.4% in test vs 3.1% in train. This is a known property of the official split: test patients have more follow-up scans.

## 2. Preprocessing check

`uv run python scripts/check_preprocessing.py` compares CheXzero's own float pipeline with our stored uint8 224 images on 256 images. The teacher-embedding cosine is min 0.999864, mean 0.999972.

## 3. Teacher zero-shot (CheXzero, `best_64_5e-05_original_22000_0.864.pt`)

`uv run python scripts/cache_teacher.py`, NIH **val**, training prompt pair `"{}"` / `"no {}"`:

**Macro AUROC 0.701.** CARZero reports about 0.71 for CheXzero on ChestX-ray14, so the reproduction holds.

| finding | val AUROC | | finding | val AUROC |
|---|---:|---|---|---:|
| Atelectasis | 0.764 | | Consolidation | 0.797 |
| Cardiomegaly | 0.797 | | Edema | 0.908 |
| Effusion | 0.860 | | Emphysema | 0.740 |
| Infiltration | 0.612 | | Fibrosis | 0.525 |
| Mass | 0.698 | | Pleural_Thickening | 0.658 |
| Nodule | 0.535 | | Hernia | 0.464 |
| Pneumonia | 0.705 | | | |
| Pneumothorax | 0.750 | | | |

Val macro AUROC for the other prompt pairs: 0.705, 0.699, 0.664, 0.704 and 0.707.

**Held-out findings** (fixed in commit c5f50b5, before any student run): Cardiomegaly, Pneumothorax, Emphysema, Mass.

- The rule was teacher val AUROC >= 0.65 and four different kinds of finding: heart size, pleural air, diffuse lung damage, and a focal lesion.
- Three of the four are also scored on IU-Xray.

## 4. Overfit checks (256 training images, 30 epochs)

`uv run python -m discipulus.train {arm} --epochs 30 --limit 256 --name overfit_{arm}`

| arm | train loss, epoch 0 -> 29 | fit on those 256 images |
|---|---|---|
| supervised | 0.744 -> 0.039 | macro AUROC 1.000 |
| A | 0.926 -> 0.056 | cosine to teacher 0.937 |
| B | 0.439 -> 0.001 | score correlation with teacher 0.842 (all 14 findings) |

One full epoch takes 58 s: 609 steps at batch 128, bf16, RTX 3080 Laptop. One epoch of A already gives val cosine 0.975 to the teacher and val macro AUROC 0.690.

## 5. Main results, NIH test and IU-Xray (seed 0)

`uv run python scripts/evaluate.py --seed 0` writes `results/main_s0.csv`.

| model | test macro AUROC (all 14) | seen (10) | held-out (4) | retention | IU-Xray macro (12) |
|---|---|---|---|---|---|
| Teacher (CheXzero) | 0.653 [0.643, 0.662] | 0.643 [0.631, 0.654] | 0.679 [0.665, 0.694] | 1.000 | 0.831 [0.814, 0.846] |

- The teacher is lower on the official test list (0.653) than on val (0.701). The same preprocessing feeds both splits, and the val number matches the published one.
- The biggest test drops are Emphysema (0.740 -> 0.579) and Mass (0.698 -> 0.570). Both are held-out findings, so the held-out set is harder on test than its val AUROC suggested.
- IU-Xray uses 12 findings. Mass (17 positives) and Fibrosis (18) fall under the 20-positive minimum.

Per-finding tables are in `results/per_finding_s0.csv` and `results/per_finding_iu_s0.csv`.

## 6. Prompt paraphrases (NIH test, macro AUROC over 14)

| model | train pair | withheld mean | spread (max-min) | mean per-finding std |
|---|---:|---:|---:|---:|
| Teacher | 0.653 | 0.644 | 0.042 | 0.024 |

## 7. Efficiency

`uv run python scripts/efficiency.py` will run once training frees the GPU.
