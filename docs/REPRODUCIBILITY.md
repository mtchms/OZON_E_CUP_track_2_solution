# Reproducibility guide

## Competition environment

The preserved submission metadata is:

```json
{
  "image": "odsai/ecup26-quality-baseline:1.0",
  "entry_point": "python -u run.py"
}
```

Runtime model paths expected by `run.py`:

- `/shared_models/Qwen/Qwen3.5-4B`
- `/shared_models/PaddlePaddle/PaddleOCR-VL-1.5`

## Strongest form of reproducibility: use the archived originals

The full repository contains the two actual competition archives:

```text
submissions/original/run_old_012.zip
submissions/original/ecup_HYBRID_A_OLD_BAD_NEW_FIRE.zip
```

Their whole-archive SHA256 values are frozen in `manifests/original_submissions.json`.

To copy the exact historical archive to an output path:

```bash
python scripts/build_submissions.py --profile private_best_t012
python scripts/build_submissions.py --profile hybrid_a_t050
```

The default mode is `archived`, so the script performs a byte-for-byte copy of the original archive.
The resulting SHA256 is therefore identical to the preserved original.

## Rebuild from canonical extracted components

For an independently rebuilt ZIP with the same model/runtime member bytes:

```bash
python scripts/build_submissions.py --profile private_best_t012 --source rebuild
python scripts/build_submissions.py --profile hybrid_a_t050 --source rebuild
```

A rebuilt ZIP can have a different **whole-ZIP** SHA256 because ZIP ordering, timestamps and directory
entries are container metadata. Its inference-critical member SHA256 values are checked against the
original archives.

## Verify everything

```bash
python scripts/verify_artifacts.py
```

The verifier checks:

1. canonical extracted runtime/config/adapter/wheel hashes;
2. whole-archive hashes for both original competition ZIPs when they are present;
3. every original ZIP member hash;
4. that common non-runtime members are identical across the two final submissions.

## Inspect an archive

```bash
python scripts/inspect_submission.py submissions/original/run_old_012.zip
```

## Git / GitHub note

The adapter weights are about 130 MB each and the original submission ZIPs are about 241 MB each.
The full archival repository is therefore intended as a jury artifact bundle.

`.gitattributes` marks:

- `*.safetensors`
- `*.whl`
- `submissions/original/*.zip`

for Git LFS.

For a conventional GitHub repository, use the supplied **SOURCE_ONLY** package and attach the two
original archives separately (or publish them through Git LFS / release assets).

## What is intentionally not included

Training/research notebooks are not shipped. Their recovered names, dates, roles and lineage are documented
in `docs/NOTEBOOK_INVENTORY.md` and `manifests/notebook_history.csv`.
