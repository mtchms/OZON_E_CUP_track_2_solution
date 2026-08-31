from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_final.zip"

FILES = [
    "run.py",
    "metadata.json",
    "adapters/bad/adapter_config.json",
    "adapters/bad/adapter_model.safetensors",
    "adapters/fire/adapter_config.json",
    "adapters/fire/adapter_model.safetensors",
    "wheels/peft-0.20.0-py3-none-any.whl",
]


def main():
    missing = [rel for rel in FILES if not (ROOT / rel).exists()]
    if missing:
        print("Missing files:")
        for rel in missing:
            print(" -", rel)
        raise SystemExit(1)

    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in FILES:
            zf.write(ROOT / rel, arcname=rel)

    print(f"Created: {OUT}")


if __name__ == "__main__":
    main()
