# Data audit, conflicts and CLEAN213

## Original data

The verified competition table contained **12,971 rows**. The BAD subset contained **7,469 rows**:

- label 0: **1,905**
- label 1: **5,564**

The earliest fixed validation slice had **600 rows**: 346 BAD and 254 FIRE.

## Why duplicate analysis was necessary

Full-data error mining exposed a group of very difficult errors around duplicated or near-duplicated product cards. Some cards had extremely similar or identical normalized name+description but contradictory labels. Rather than blindly deduplicating, we separated:

- **exact normalized conflicts** — safe enough for an auditable hard rule;
- **near-exact / semantic duplicates** — reviewed but not automatically removed in the final conservative cleaner.

A visual review branch examined opposite-label high-similarity pairs and showed that some contradictions were literal listing duplicates, some could be explained by packaging/OCR evidence, and many remained genuinely ambiguous. This is why the final cleaning rule was intentionally conservative.

## CLEAN213 rule

On **2026-08-25** we froze exactly **213 BAD rows from 54 exact-conflict groups** for removal:

- 93 rows with original label 0;
- 120 rows with original label 1.

After cleaning:

- BAD: `7469 -> 7256`
- label 0: `1905 -> 1812`
- label 1: `5564 -> 5444`
- FIRE: unchanged.

No remaining labels were edited. Near-exact pairs were not automatically deleted.

The immutable ID list is included as `manifests/clean213_removed_ids.csv`.

## CLEAN213 training branch

The cleaned BAD table was joined one-to-one with `ocr_all_products_high_quality.csv`. The join was guarded so OCR could not add, reorder or alter training products. The V3 training recipe preserved the STRICT OLD structure:

- Qwen3.5-4B;
- single T4/GPU0;
- NF4 QLoRA, double quant, FP16 compute;
- LoRA r=16, alpha=32, dropout=0.05;
- 248 text-linear target modules;
- 576×576 contact sheets, up to 5 images;
- processor visual budget 448×448;
- description cap 2200, OLD 70/30 head-tail trim;
- inverse-class weighted sampling with replacement;
- 7,256 draws, seed 42;
- sampled labels `{0: 3702, 1: 3554}`;
- 4,199 unique source rows seen;
- 1 epoch, batch 1, grad accumulation 8;
- AdamW lr=1e-4, weight decay .01, cosine schedule;
- 907 optimizer updates, 45 warmup steps;
- ordinary full-vocabulary CE on the assistant target token;
- **full HQ OCR, no OCR dropout and no OCR truncation**.

V3 fixed a step-100 OOM by using Qwen `logits_to_keep` to materialize only the causal position that predicts the target `0/1` token. It did not change the mathematical target or truncate the OCR.

## What CLEAN213 taught us

The branch was valuable even though its adapter was not the final private-best BAD. Threshold sweeps showed that the classifier benefited strongly from a lower BAD decision threshold, which motivated applying the same calibration insight to the stronger OLD-BAD + NEW-FIRE hybrid lineage.
