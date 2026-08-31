# Jury summary — E-CUP 2026 Quality Control

## Result

**2nd place on the private leaderboard**, best score **0.9042222978144401**.

## Final system in one sentence

Qwen3.5-4B multimodal inference with PaddleOCR-VL-1.5, **OLD BAD LoRA**, **successful NEW FIRE LoRA**, direct `0/1` token-logit scoring, and category-specific thresholds **BAD=0.12 / FIRE=0.50**.

## What materially improved the score

1. **Controlled adapter swap** isolated the improvement: NEW FIRE helped, NEW BAD hurt. This produced Hybrid A (**0.9024064171**).
2. **Error mining** showed a large BAD calibration issue and concentrated errors around contradictory duplicates.
3. **Conservative data audit** removed exactly 213 verified BAD rows from 54 exact-conflict groups for the CLEAN213 research branch; near-duplicates were not blindly removed.
4. **Threshold sweeps** confirmed that lower BAD thresholds improved leaderboard score.
5. We transferred the calibration insight back to the stronger Hybrid-A model pair instead of forcing the newer CLEAN213 adapter into the final system. BAD `0.12` produced **0.9042222978**.

## Reproducibility

The repository contains the exact two final adapters, exact runtime files, exact PEFT wheel, SHA256 manifest, immutable CLEAN213 removal IDs, leaderboard screenshots, a deterministic submission builder, and a dated inventory of all recovered E-CUP notebooks. Training notebooks themselves are intentionally excluded.


## Archive-level provenance

The final repository includes the two **original competition ZIP archives**, verified on 2026-08-31:

- private best `run_old_012.zip`: `ce87c10462da5ade2aff9d3b69c8b78cdaf4b276d0601a05b632c369f0b71636`;
- Hybrid A `ecup_HYBRID_A_OLD_BAD_NEW_FIRE.zip`: `63830b5412d8828f30dc4ad2019f608c8cda8b9072ee42609973b3eb05025611`.

This supersedes the earlier reconstruction-only provenance path. Jury verification can now be performed
at both the whole-archive level and the individual member level.
