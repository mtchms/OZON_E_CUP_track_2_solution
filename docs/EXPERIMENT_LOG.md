# Experiment log / project chronology

This is the condensed chronological narrative. The exhaustive notebook names are in `NOTEBOOK_INVENTORY.md`.

## 2026-08-11 — reproducible baseline and first Qwen QLoRA

- Audited the data and stabilized a 600-row validation slice.
- Built fast/OOM-safe/disk-safe inference variants.
- Moved from frozen-model prompting to Qwen3.5-4B QLoRA.
- Established separate BAD and FIRE behavior and direct `0/1` target-token classification.

## 2026-08-13 — alternative model and expert branches

- Tested embedding/expert variants.
- Benchmarked Gemma4 E4B QLoRA and TPU paths.
- Kept Qwen3.5-4B as the main practical direction.

## 2026-08-14 — full-data analysis, OOF and OCR ablation

- Ran full-dataset Qwen evaluation and FP16 speed experiments.
- Built 5-fold group-aware OOF variants on 2×T4 and TPU v5e-8.
- Started explicit BASE-vs-OCR evaluation.
- Stabilized an official PaddleOCR-VL-1.5 pipeline through multiple fixed revisions.

## 2026-08-15 — OCR-aware training and the FIRE specialization

- Tried legacy PPOCRv5/mobile OCR and moved away from it.
- Trained OCR-aware QLoRA and full-data OCR variants.
- Tested FSDP full-text training.
- Specialized the pipeline into a FIRE-only final FSDP training branch; this became the lineage of the successful NEW FIRE adapter.

## 2026-08-16 — error mining and controlled hybridization

- Resumed BAD training from checkpoints to eliminate accidental recipe differences.
- Ran full FIRE and BAD error mining.
- Built the first consolidated submission runtime with PaddleOCR-VL.
- Crucially built two controlled hybrids while freezing every non-adapter runtime byte:
  - A = OLD BAD + NEW FIRE
  - B = NEW BAD + OLD FIRE.

## 2026-08-17 — old-vs-new BAD diagnosis

- Per-row comparison confirmed that NEW BAD was responsible for the regression while NEW FIRE was beneficial.
- Hybrid A reached **0.9024064171122994**.

## 2026-08-18 to 2026-08-22 — attempts to improve BAD without breaking the strong old behavior

- Robust finetune with conflict-aware weighting/OCR dropout.
- STRICT OLD + OCR with controlled recipe changes.
- TPU v5e-8 SPMD, XLA/FSDP and CUDA DDP branches.
- Resume-safe engineering after memory/runtime failures.

## 2026-08-23 — STRICT OLD, no-OCR training

- Returned to an intentionally simple controlled BAD recipe: full 7,469 source pool, no OCR in training, old prompt/trim/sampling, single T4, NF4 LoRA.
- This created a clean diagnostic reference for understanding whether OCR training itself was helping.

## 2026-08-24 — BAD error mining and threshold discovery

- Full BAD error-mining run: 7,469 rows, F1 0.958164, 452 errors.
- 213 conflict rows accounted for a disproportionate share of errors.
- Local threshold analysis found a strong region near **0.12** (local F1 0.967718).

## 2026-08-25 — exact-conflict cleaning and CLEAN213

- Rejected an earlier too-aggressive near-duplicate deletion proposal.
- Switched to a conservative exact normalized name+description conflict policy.
- Removed exactly 213 verified BAD rows from 54 contradictory exact groups.
- Trained CLEAN213 + full HQ OCR with the STRICT OLD recipe; V3 fixed long-OCR OOM without changing the target.

## 2026-08-26 — CLEAN213 submission engineering

- Resumed training exactly from step 500 to completion.
- Built/verified CLEAN213 BAD + untouched successful FIRE archives.
- Initial leaderboard point at threshold .50: **0.9008031442**.

## 2026-08-27 — calibration sweep and private-best solution

CLEAN213 threshold sweep:

- .65 -> 0.90006463
- .575 -> 0.89924028
- .425 -> 0.90135895
- .35 -> 0.90235139
- .30 -> 0.90197202
- .25 -> 0.90287145
- .20 -> **0.90318486**.

This confirmed that lower BAD thresholds were advantageous. The calibration was then applied to the stronger Hybrid-A lineage rather than forcing CLEAN213 into the final system.

Final private-best operating point:

- OLD BAD
- successful NEW FIRE
- PaddleOCR-VL inference
- BAD threshold **0.12**
- FIRE threshold **0.50**
- score **0.9042222978144401**
- participant result: **2nd place on the private leaderboard**.

## 2026-08-31 — final archival consolidation

- Recovered the actual `run_old_012` private-best submission ZIP and the actual Hybrid-A ZIP.
- Verified whole-archive SHA256, every member SHA256, member sizes and ZIP timestamps.
- Confirmed that both final submissions use byte-identical OLD BAD / NEW FIRE adapters, metadata and PEFT wheel.
- Confirmed from the original `run.py` files that the only functional change is BAD threshold `0.50 -> 0.12`.
- Replaced reconstruction-based provenance in the jury repository with the original archived submissions.
