from pathlib import Path
import argparse
import shutil
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_out"

VARIANTS = {
    "bad_t012_fire_t050": ROOT / "runtime/bad_t012_fire_t050/run.py",
    "bad_t050_fire_t050": ROOT / "runtime/bad_t050_fire_t050/run.py",
}

FILES = {
    "metadata.json": ROOT / "metadata.json",
    "wheels/peft-0.20.0-py3-none-any.whl": ROOT / "wheels/peft-0.20.0-py3-none-any.whl",
    "adapters/bad/adapter_config.json": ROOT / "adapters/bad/adapter_config.json",
    "adapters/bad/adapter_model.safetensors": ROOT / "adapters/bad/adapter_model.safetensors",
    "adapters/fire/adapter_config.json": ROOT / "adapters/fire/adapter_config.json",
    "adapters/fire/adapter_model.safetensors": ROOT / "adapters/fire/adapter_model.safetensors",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build(variant: str):
    run_path = VARIANTS[variant]
    required = {"run.py": run_path, **FILES}

    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    stage = OUT / variant
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for arcname, src in required.items():
        dst = stage / arcname
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    zip_path = OUT / f"submission_{variant}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())

    print(f"Built: {zip_path}")
    print(f"SHA256: {sha256(zip_path)}")
    print("Note: outer ZIP hash is not expected to match the historical upload,")
    print("because ZIP packing metadata/timestamps differ. Runtime/adapters are exact.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    build(args.variant)

if __name__ == "__main__":
    main()
