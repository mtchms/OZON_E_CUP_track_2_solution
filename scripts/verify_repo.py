from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "adapters/bad/adapter_model.safetensors": "5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d",
    "adapters/bad/adapter_config.json": "291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391",
    "adapters/fire/adapter_model.safetensors": "6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01",
    "adapters/fire/adapter_config.json": "278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b",
    "run.py": "a0d89f62d57dedd2cb4e9eaf0eeea606c6b7c58e02336fb9f466d89e7a7b83a5",
    "metadata.json": "2e84e4db7be7e3f5d6b7b1fae53b0395eb1a6b913704db6dc4f0b39c51095812",
    "wheels/peft-0.20.0-py3-none-any.whl": "0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


ok = True

for rel, expected in EXPECTED.items():
    path = ROOT / rel

    if not path.exists():
        print(f"MISSING  {rel}")
        ok = False
        continue

    actual = sha256(path)

    if actual == expected:
        print(f"OK       {rel}  {actual}")
    else:
        print(f"MISMATCH {rel}")
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")
        ok = False


print()
if ok:
    print("RESULT: ALL OK")
    sys.exit(0)
else:
    print("RESULT: FAILED")
    sys.exit(1)
