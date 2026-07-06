"""Merge per-event result JSONs into a single results.json for easy scp,
and report which (index, class) pairs are missing so failures can be rerun.

Run on the cluster:  python studies/collect_results.py slurm/out/rn_L1
Then copy down just  slurm/out/rn_L1/results.json
"""

import glob
import json
import re
import sys
from pathlib import Path

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")

outdir = Path(sys.argv[1])
files = sorted(glob.glob(str(outdir / "results" / "e*_*.json")))
rows = [json.loads(Path(f).read_text()) for f in files]
(outdir / "results.json").write_text(json.dumps(rows))

done = {(r["index"], r["cls"]) for r in rows}
indices = {i for i, _ in done}
missing = [(i, c) for i in sorted(indices) for c in CLASSES if (i, c) not in done]

print(f"{len(rows)} rows -> {outdir / 'results.json'}")
for c in CLASSES:
    print(f"  {c:>11s}: {sum(r['cls'] == c for r in rows)}")
if missing:
    print(f"{len(missing)} missing (index, cls) pairs:")
    print(" ".join(f"e{i}:{c}" for i, c in missing))
    print("rerun indices: " + ",".join(str(i) for i in sorted({i for i, _ in missing})))
