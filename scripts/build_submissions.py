#!/usr/bin/env python3
import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "hybrid_a_t050": {
        "run": ROOT / "runtime/hybrid_a_t050/run.py",
        "original": ROOT / "submissions/original/ecup_HYBRID_A_OLD_BAD_NEW_FIRE.zip",
    },
    "private_best_t012": {
        "run": ROOT / "runtime/private_best_t012/run.py",
        "original": ROOT / "submissions/original/run_old_012.zip",
    },
}

def rebuild(profile, out):
    members = {
        "run.py": PROFILES[profile]["run"],
        "metadata.json": ROOT / "runtime/common/metadata.json",
        "wheels/peft-0.20.0-py3-none-any.whl": ROOT / "wheels/peft-0.20.0-py3-none-any.whl",
        "adapters/bad/adapter_config.json": ROOT / "adapters/old_bad/adapter_config.json",
        "adapters/bad/adapter_model.safetensors": ROOT / "adapters/old_bad/adapter_model.safetensors",
        "adapters/fire/adapter_config.json": ROOT / "adapters/new_fire/adapter_config.json",
        "adapters/fire/adapter_model.safetensors": ROOT / "adapters/new_fire/adapter_model.safetensors",
    }
    missing = [str(p) for p in members.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing canonical artifacts: " + ", ".join(missing))

    # A rebuild preserves inference-critical member bytes, but intentionally does not
    # claim whole-ZIP byte identity with the historical competition archive.
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arcname, path in members.items():
            z.write(path, arcname)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument(
        "--source",
        choices=["archived", "rebuild"],
        default="archived",
        help="archived = byte-for-byte copy of actual submitted ZIP; rebuild = repack canonical members",
    )
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    suffix = "ORIGINAL.zip" if args.source == "archived" else "REBUILT.zip"
    out = args.output or ROOT / "submission_out" / f"{args.profile}_{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "archived":
        original = PROFILES[args.profile]["original"]
        if not original.exists():
            raise FileNotFoundError(
                f"Original archive is absent: {original}. "
                "Use --source rebuild only if canonical extracted components are available."
            )
        shutil.copyfile(original, out)
    else:
        rebuild(args.profile, out)

    print(out)

if __name__ == "__main__":
    main()
