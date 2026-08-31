# Adapter evolution and model-selection logic

## Qwen LoRA baseline

The first strong Qwen3.5-4B QLoRA branch used LoRA `r=16`, `alpha=32`, dropout `0.05`, LR `1e-4`, one epoch and gradient accumulation 8. On the fixed 600-row validation, the recorded v1 scores were:

- BAD F1@0.5: **0.938856**
- FIRE F1@0.5: **0.823529**
- mean: **0.881193**
- tuned mean: about **0.882938**.

This immediately suggested that BAD and FIRE behaved differently enough to justify separate adapters and later separate calibration.

## Why we kept separate BAD/FIRE adapters

A key controlled ablation froze the exact successful runtime and swapped only adapter weights:

| Composition | Reported leaderboard score | Interpretation |
|---|---:|---|
| OLD BAD + OLD FIRE | 0.89281046 | frozen reference |
| NEW BAD + NEW FIRE | 0.866557 | combined new pair regressed |
| NEW BAD + OLD FIRE | 0.85337800 | NEW BAD was the main failure source |
| **OLD BAD + NEW FIRE** | **0.9024064171** | NEW FIRE was genuinely better |

This experiment changed the strategy. We stopped treating “newest adapter” as synonymous with “best system” and moved to **component-level model selection**.

## NEW FIRE

The strongest FIRE lineage was trained in `ecup2026_FIRE_qwen35_FSDP_FULLTEXT_2xT4_FINAL.ipynb`:

- FIRE-only training;
- Qwen3.5-4B;
- both T4 GPUs with FSDP FULL_SHARD;
- NF4 QLoRA;
- LoRA r=16, alpha=32, dropout=.05;
- full description + OCR text;
- visual budget around 448;
- balanced full-coverage schedule;
- last target-token full-vocabulary CE;
- resumable checkpoints.

The final successful FIRE weights are identified by SHA256 `6bd02b...55d01` and are shipped in the repository.

## BAD research after the hybrid breakthrough

We did not stop at OLD BAD. Several attempts tried to improve its known failure modes:

- resumed earlier BAD checkpoints exactly;
- full-data error mining;
- row-by-row OLD-vs-NEW comparison;
- robust finetune with conflict downweighting and OCR dropout;
- STRICT OLD + OCR on GPU and TPU;
- TPU SPMD / XLA / FSDP engineering experiments;
- STRICT OLD full/no-OCR training to remove OCR-training confounding;
- exact-conflict cleaning;
- CLEAN213 + HQ OCR training.

The important model-selection result was that none of those justified replacing OLD BAD in the final hybrid once leaderboard and threshold evidence were considered.

## Calibration was the last large gain

The STRICT OLD no-OCR BAD error-mining run on all 7,469 BAD rows recorded F1 **0.958164** at the default operating point and found a local threshold around **0.12** with F1 **0.967718**. Separately, leaderboard sweeps on CLEAN213 moved in the same direction: 0.35 -> 0.25 -> 0.20 improved the score.

This gave a coherent calibration hypothesis: the BAD adapter's ranking was useful, but `0.5` was not the best operating point for the competition metric/distribution. Applying `BAD=0.12` while keeping `FIRE=0.5` to the stronger OLD-BAD + NEW-FIRE system produced the private-best **0.9042222978**.
