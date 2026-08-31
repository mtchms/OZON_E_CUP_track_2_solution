#!/usr/bin/env python3
import difflib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
hybrid = ROOT / "submissions/original/ecup_HYBRID_A_OLD_BAD_NEW_FIRE.zip"
private = ROOT / "submissions/original/run_old_012.zip"

with zipfile.ZipFile(hybrid) as zh, zipfile.ZipFile(private) as zp:
    h = zh.read("run.py").decode("utf-8").splitlines(keepends=True)
    p = zp.read("run.py").decode("utf-8").splitlines(keepends=True)

print("".join(difflib.unified_diff(
    h, p,
    fromfile="Hybrid A original/run.py",
    tofile="Private best original/run.py",
)))
