from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parents[1]
a_path = ROOT / "runtime/bad_t050_fire_t050/run.py"
b_path = ROOT / "runtime/bad_t012_fire_t050/run.py"

a = a_path.read_text(encoding="utf-8").splitlines(keepends=True)
b = b_path.read_text(encoding="utf-8").splitlines(keepends=True)

diff = list(difflib.unified_diff(
    a, b,
    fromfile=str(a_path.relative_to(ROOT)),
    tofile=str(b_path.relative_to(ROOT)),
))

print("".join(diff))
print(f"Unified diff lines: {len(diff)}")
