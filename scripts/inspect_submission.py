#!/usr/bin/env python3
import hashlib
import sys
import zipfile
from pathlib import Path

def sha(data):
    return hashlib.sha256(data).hexdigest()

p = Path(sys.argv[1])
print(f"archive\t{p.stat().st_size}\t{sha(p.read_bytes())}")
with zipfile.ZipFile(p) as z:
    for i in z.infolist():
        if i.is_dir():
            print(f"DIR\t{i.filename}\t-\t-\t{i.date_time}")
            continue
        data = z.read(i.filename)
        print(f"FILE\t{i.filename}\t{len(data)}\t{sha(data)}\t{i.date_time}")
