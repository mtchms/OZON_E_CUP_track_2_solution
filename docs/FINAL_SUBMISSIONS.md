# Final submissions and exact artifact lineage

This document is grounded in the **two original competition ZIP archives** supplied by the participant
and preserved under `submissions/original/`.

## 1. Private-best `run_old_012` — ORIGINAL ARCHIVE

Private leaderboard score: **0.9042222978144401**.  
Reported result: **2nd place on the private leaderboard**.

Original archive:

- path: `submissions/original/run_old_012.zip`
- SHA256: `ce87c10462da5ade2aff9d3b69c8b78cdaf4b276d0601a05b632c369f0b71636`
- bytes: `241490584`
- original `run.py` SHA256: `a0d89f62d57dedd2cb4e9eaf0eeea606c6b7c58e02336fb9f466d89e7a7b83a5`

Decision rule:

```python
threshold = 0.12 if category == BAD else 0.5

for j, p1 in zip(batch_idxs, probs):
    p1s[j] = float(p1)
    preds[j] = int(p1 >= threshold)
```

No reconstruction claim is needed anymore: **the actual submitted ZIP is archived in this repository**.

## 2. Hybrid A — OLD BAD + NEW FIRE — ORIGINAL ARCHIVE

Leaderboard score: **0.9024064171122994**.

Original archive:

- path: `submissions/original/ecup_HYBRID_A_OLD_BAD_NEW_FIRE.zip`
- SHA256: `63830b5412d8828f30dc4ad2019f608c8cda8b9072ee42609973b3eb05025611`
- bytes: `241489448`
- original `run.py` SHA256: `0fa9229810f90d26d8a1c923c4ed6c4b195e99ba3c7800c94c44f1f16987bd3e`

This was the successful controlled hybrid:

- BAD = **OLD BAD**
- FIRE = **NEW FIRE**
- BAD threshold = **0.50**
- FIRE threshold = **0.50**

## Exact adapter identity

Both original ZIPs contain the same adapter bytes.

| Component | SHA256 |
|---|---|
| OLD BAD config | `291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391` |
| OLD BAD weights | `5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d` |
| NEW FIRE config | `278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b` |
| NEW FIRE weights | `6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01` |
| metadata | `2e84e4db7be7e3f5d6b7b1fae53b0395eb1a6b913704db6dc4f0b39c51095812` |
| PEFT 0.20.0 wheel | `0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233` |

## Exact runtime delta

The complete diff is stored in `docs/PRIVATE_BEST_THRESHOLD_DIFF.patch`.
A direct diff of the two original `run.py` files shows only the threshold change described above.

Thus the private-best improvement is attributable to **decision calibration on the already successful
OLD BAD + NEW FIRE system**, not to a hidden adapter swap or runtime rewrite.

## Historical packaging

The original ZIP-container metadata is intentionally preserved. `manifests/original_submissions.json`
contains the exact whole-archive and member-level hashes and historical ZIP timestamps.

For more detail, see `docs/ORIGINAL_SUBMISSION_PROVENANCE.md`.
