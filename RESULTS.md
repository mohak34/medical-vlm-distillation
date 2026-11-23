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

## 5. Epoch budget

Every arm trains for **5 epochs** (OneCycle, last epoch kept). The first plan was 30 epochs, but the 30-epoch supervised run memorized the training set:

- Train loss fell from 0.25 to 0.0000.
- Val macro AUROC peaked at 0.799 in epoch 2, then fell to 0.704 by epoch 30.

5 epochs was chosen from that val curve alone, before any distillation run, and it applies to every arm. The 30-epoch runs are kept as an ablation (`runs/<arm>_e30_s0`, `EPOCHS=30 SEEDS=0 scripts/run_all.sh`).

5-epoch val curves (`runs/<arm>_s0/train.log`):

| arm | val per epoch |
|---|---|
| supervised | macro AUROC 0.771, 0.799, 0.809, 0.796, 0.779 |
| A | cosine to teacher 0.976 -> 0.984; macro AUROC 0.693 -> 0.696 |
| B | score corr 0.868 -> 0.881; cosine to teacher 0.19, as expected, since only the prompt directions are trained |
| A+B | cosine to teacher 0.969 -> 0.981; macro AUROC 0.689 -> 0.696 |

## 6. Main results, seed 0

`scripts/run_all.sh` then `uv run python scripts/evaluate.py --seed 0` writes `results/main_s0.csv`. Values are NIH official test unless marked; 95% patient-bootstrap CIs.

| model | macro AUROC (all 14) | seen (10) | held-out (4) | retention vs teacher | IU-Xray macro (12) |
|---|---|---|---|---|---|
| Teacher (CheXzero) | 0.653 [0.643, 0.662] | 0.643 [0.631, 0.654] | 0.679 [0.665, 0.694] | 1.000 | 0.831 [0.814, 0.846] |
| Supervised | 0.760 [0.750, 0.767] | 0.738 [0.726, 0.747] | 0.814 [0.802, 0.824] | 1.163 [1.145, 1.181] | 0.759 [0.740, 0.779] |
| A: embedding | 0.653 [0.643, 0.661] | 0.642 [0.631, 0.653] | 0.679 [0.664, 0.693] | 0.999 [0.996, 1.002] | 0.833 [0.816, 0.849] |
| B: score | 0.620 [0.609, 0.630] | 0.642 [0.630, 0.653] | 0.563 [0.546, 0.579] | 0.948 [0.942, 0.956] | 0.796 [0.778, 0.813] |
| A+B | 0.652 [0.642, 0.661] | 0.643 [0.631, 0.655] | 0.675 [0.659, 0.689] | 0.998 [0.995, 1.001] | 0.828 [0.812, 0.845] |

What it shows:

- **A keeps essentially all of the teacher**, on seen findings, held-out findings and the external set alike, with 11x fewer image-encoder parameters (section 9).
- **B matches the teacher only where it was trained.** On held-out findings it drops from 0.679 to 0.563. Emphysema goes to 0.480, below chance. On IU-Xray, held-out Cardiomegaly falls from 0.935 to 0.838 and Emphysema from 0.885 to 0.657, while seen findings stay at the teacher's level.
- **A+B is the same as A within the CIs.** Adding B neither helps nor hurts.
- **Supervised beats the teacher on NIH but loses on IU-Xray**: 0.759 vs 0.831 for the teacher and 0.833 for A. It learns NIH's own labels, including Hernia, Fibrosis and Nodule, where the teacher is near chance, but those labels transfer worse. Its "held-out" column is not held out, since it trains on all 14 labels.
- **The teacher is lower on the official test list (0.653) than on val (0.701).** The same preprocessing feeds both splits, and the val number matches the published one. The biggest drops are Emphysema (0.740 -> 0.579) and Mass (0.698 -> 0.570), both held out, so the held-out set is harder on test than its val AUROC suggested.
- IU-Xray uses 12 findings. Mass (17 positives) and Fibrosis (18) fall under the 20-positive minimum.

Per-finding tables: `results/per_finding_s0.csv` and `results/per_finding_iu_s0.csv`. Figures: `uv run python scripts/figures.py --seed 0` writes `results/figures/macro_auroc_s0.png`, `per_finding_s0.png` and `paraphrases_s0.png`.

Agreement with the teacher on NIH test (`results/agreement_s0.csv`):

| model | cosine to teacher embedding | score corr, seen | score corr, held-out |
|---|---:|---:|---:|
| A | 0.981 | 0.966 | 0.961 |
| B | 0.186 | 0.963 | 0.678 |
| A+B | 0.976 | 0.966 | 0.953 |

## 6b. Across seeds 0-2

Run `uv run python scripts/evaluate.py --seed {0,1,2}`, then `uv run python scripts/seeds.py`, which writes `results/seeds.csv`. Values are mean +- std of the point macro AUROC over 3 training seeds. The teacher is fixed, so its std is 0.

| model | NIH all 14 | seen (10) | held-out (4) | IU-Xray (12) |
|---|---|---|---|---|
| Teacher (CheXzero) | 0.653 | 0.643 | 0.679 | 0.831 |
| Supervised | 0.760 +- 0.002 | 0.737 +- 0.002 | 0.818 +- 0.004 | 0.765 +- 0.004 |
| A: embedding | 0.652 +- 0.001 | 0.641 +- 0.001 | 0.679 +- 0.001 | 0.831 +- 0.002 |
| B: score | 0.617 +- 0.002 | 0.642 +- 0.001 | 0.557 +- 0.009 | 0.796 +- 0.001 |
| A+B | 0.652 +- 0.000 | 0.642 +- 0.001 | 0.675 +- 0.001 | 0.828 +- 0.000 |

Seed variation is far smaller than the gaps between arms, so every seed-0 conclusion above holds.

## 7. Prompt paraphrases (NIH test, macro AUROC over 14, seed 0)

`results/paraphrase_s0.csv`. The training pair is `"{}"` / `"no {}"`, and the other 5 pairs were never used in training.

| model | train pair | withheld mean | spread (max-min) | mean per-finding std |
|---|---:|---:|---:|---:|
| Teacher | 0.653 | 0.644 | 0.042 | 0.024 |
| A | 0.653 | 0.643 | 0.043 | 0.023 |
| B | 0.620 | 0.616 | 0.016 | 0.017 |
| A+B | 0.652 | 0.641 | 0.042 | 0.024 |

A and A+B carry the teacher's prompt sensitivity over unchanged, including its weak pair, `"{} is present"` / `"{} is absent"`. B has a smaller spread, but that is not robustness. Its held-out findings sit near chance under every prompt, which flattens the spread.

## 8. Qualitative

`uv run python scripts/qualitative.py --arm B --seed 0` (and `--arm A`) writes `results/figures/tsne_s0.png` and `gradcam_{A,B}_s0.png`.

- **t-SNE** of 3,000 test embeddings: A and A+B keep the teacher's cluster layout. B's layout is different, because it only constrains 10 prompt directions.
- **Grad-CAM**, using the same prompt-score target for both models, on the test positives where teacher and student rank most differently:
  - For B, these are held-out Pneumothorax cases. The teacher ranks them at the 98-100th percentile and B at the 12-14th. B's maps sit on the shoulders and neck, the teacher's on the lung fields.
  - For A, the largest disagreements are on Infiltration.

## 9. Efficiency

`uv run python scripts/efficiency.py` will run once training frees the GPU.
