#!/usr/bin/env python3
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

ok = True

final_manifest = json.loads(
    (ROOT / "manifests/final_artifacts.json").read_text(encoding="utf-8")
)

print("== Canonical extracted artifacts ==")
for rel, meta in final_manifest["artifacts"].items():
    p = ROOT / rel
    got = sha_file(p) if p.exists() else "<missing>"
    good = p.exists() and got == meta["sha256"] and p.stat().st_size == meta["bytes"]
    print(("OK  " if good else "FAIL"), rel, got)
    ok &= good

orig_manifest = json.loads(
    (ROOT / "manifests/original_submissions.json").read_text(encoding="utf-8")
)

print("\n== Original competition archives ==")
for profile, meta in orig_manifest["profiles"].items():
    p = ROOT / meta["repository_path"]
    if not p.exists():
        print("SKIP", profile, "(archive absent; expected in FULL archival repository)")
        continue

    archive_hash = sha_file(p)
    archive_good = (
        archive_hash == meta["archive_sha256"]
        and p.stat().st_size == meta["archive_bytes"]
    )
    print(("OK  " if archive_good else "FAIL"), profile, "archive", archive_hash)
    ok &= archive_good

    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        for member, mm in meta["members"].items():
            if member not in names:
                print("FAIL", profile, member, "<missing>")
                ok = False
                continue
            data = z.read(member)
            got = sha_bytes(data)
            good = got == mm["sha256"] and len(data) == mm["bytes"]
            print(("OK  " if good else "FAIL"), profile, member, got)
            ok &= good

# Strong cross-archive identity assertion when both originals are present.
p_private = ROOT / orig_manifest["profiles"]["private_best_t012"]["repository_path"]
p_hybrid = ROOT / orig_manifest["profiles"]["hybrid_a_t050"]["repository_path"]
if p_private.exists() and p_hybrid.exists():
    print("\n== Cross-archive identity ==")
    common = orig_manifest["cross_archive_verification"]["common_members_checked"]
    with zipfile.ZipFile(p_private) as zp, zipfile.ZipFile(p_hybrid) as zh:
        for member in common:
            same = zp.read(member) == zh.read(member)
            print(("OK  " if same else "FAIL"), member, "byte-identical")
            ok &= same

raise SystemExit(0 if ok else 1)
