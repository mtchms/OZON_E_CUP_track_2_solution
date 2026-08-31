# Leaderboard evidence

## Private-best

- archive name shown in the participant screenshot: `run_old_012.zip`
- score: **0.9042222978144401**
- participant-reported placement: **2nd place on the private leaderboard**
- screenshot: `assets/leaderboard/private_best_0.9042222978.png`

## Hybrid A

- archive: `ecup_HYBRID_A_OLD_BAD_NEW_FIRE...`
- score: **0.9024064171122994**
- screenshot: `assets/leaderboard/hybrid_a_0.9024064171.png`

## Important interpretation

The private-best score is not from a new model family. It is the result of **component selection + calibration**:

1. preserve OLD BAD because controlled swaps showed it generalized better;
2. adopt NEW FIRE because it improved the system;
3. preserve the stable PaddleOCR-VL runtime;
4. lower BAD threshold to 0.12 while keeping FIRE at 0.5.

## Original archive binding

The screenshots above are now paired with the exact archived submission files:

- `0.9042222978144401` -> `submissions/original/run_old_012.zip`
  - SHA256 `ce87c10462da5ade2aff9d3b69c8b78cdaf4b276d0601a05b632c369f0b71636`
- `0.9024064171122994` -> `submissions/original/ecup_HYBRID_A_OLD_BAD_NEW_FIRE.zip`
  - SHA256 `63830b5412d8828f30dc4ad2019f608c8cda8b9072ee42609973b3eb05025611`
