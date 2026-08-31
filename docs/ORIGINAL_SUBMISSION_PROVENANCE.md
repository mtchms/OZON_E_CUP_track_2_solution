# Original submission provenance

## Status

On **2026-08-31**, the participant supplied the two actual ZIP archives used for the two final leaderboard submissions.
They are now preserved **byte-for-byte** in this repository. The previously generated `*_RECONSTRUCTED.zip`
artifacts are no longer used as provenance and are intentionally absent from this final repository.

## Original archive 1 — private leaderboard best

- Historical supplied filename: `run_old_012 (1).zip`
- Canonical repository filename: `submissions/original/run_old_012.zip`
- Score: **0.9042222978144401**
- Reported private leaderboard place: **2**
- BAD threshold: **0.12**
- FIRE threshold: **0.50**
- Archive bytes: **241490584**
- Archive SHA256: `ce87c10462da5ade2aff9d3b69c8b78cdaf4b276d0601a05b632c369f0b71636`
- `run.py` SHA256: `a0d89f62d57dedd2cb4e9eaf0eeea606c6b7c58e02336fb9f466d89e7a7b83a5`

Renaming the outer file from `run_old_012 (1).zip` to `run_old_012.zip` does not alter its bytes;
the SHA256 above is the hash of the exact uploaded archive bytes.

## Original archive 2 — Hybrid A

- Filename: `submissions/original/ecup_HYBRID_A_OLD_BAD_NEW_FIRE.zip`
- Score: **0.9024064171122994**
- BAD threshold: **0.50**
- FIRE threshold: **0.50**
- Archive bytes: **241489448**
- Archive SHA256: `63830b5412d8828f30dc4ad2019f608c8cda8b9072ee42609973b3eb05025611`
- `run.py` SHA256: `0fa9229810f90d26d8a1c923c4ed6c4b195e99ba3c7800c94c44f1f16987bd3e`

## Cross-archive byte verification

The following non-runtime members are byte-identical in both original archives:

- `metadata.json`
- `wheels/peft-0.20.0-py3-none-any.whl`
- `adapters/bad/adapter_config.json`
- `adapters/bad/adapter_model.safetensors`
- `adapters/fire/adapter_config.json`
- `adapters/fire/adapter_model.safetensors`

Therefore the two final submissions use the **same OLD BAD adapter, the same successful NEW FIRE adapter,
the same PEFT wheel and the same metadata**.

The only functional inference-code difference is the category-specific BAD threshold:

```diff
+ threshold = 0.12 if category == BAD else 0.5

  for j, p1 in zip(batch_idxs, probs):
      p1s[j] = float(p1)
-     preds[j] = int(p1 >= 0.5)
+     preds[j] = int(p1 >= threshold)
```

The complete exact `run.py` diff generated directly from the two original archives is in
`docs/PRIVATE_BEST_THRESHOLD_DIFF.patch`.

## ZIP-container differences that are not model changes

The private-best archive additionally contains explicit empty ZIP directory entries
(`wheels/`, `adapters/`, `adapters/bad/`, `adapters/fire/`). Hybrid A does not.
The archives also retain their historical per-member ZIP timestamps and member ordering.
Those packaging details explain why the two whole-archive hashes differ beyond the `run.py` change;
they do not change inference behavior.

## Machine-readable audit

See `manifests/original_submissions.json` for:

- whole-archive SHA256;
- archive byte size;
- every member SHA256;
- every member byte size;
- ZIP timestamps;
- directory entries;
- score and threshold mapping.
