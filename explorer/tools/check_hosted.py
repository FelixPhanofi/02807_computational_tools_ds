"""Smoke-test the preselected hosted rule on digital excerpts and disjoint absent tracks."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpu_match import catalogue, identify


def main():
    """Score one off-grid ten-second crop per song and retain all decisions."""
    processed = HERE.parent / "data/medium/data/processed"
    meta = pd.read_csv(processed / "manifest.csv").set_index("track_id")
    known = [int(row[0]) for row in catalogue()]
    absent = meta.drop(index=known).query("duration_s >= 20").sample(48, random_state=280711).index
    rows = []
    for ids, present in ((known, True), (absent, False)):
        for tid in ids:
            pcm = np.load(processed / meta.loc[tid, "path"], mmap_mode="r")
            result = identify(pcm[137203:377203].astype("<i2").tobytes())
            rows.append(dict(source=int(tid), present=present, **result))
    target = HERE / "hosted-check.json"
    for row in rows:
        row.pop("lookup_ms")
    target.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf8")
    print(
        "Correct accepted:",
        sum(r["present"] and r["accepted"] and r["track_id"] == r["source"] for r in rows),
        "/",
        len(known),
    )
    print("Absent accepted:", sum(not r["present"] and r["accepted"] for r in rows), "/", len(absent))


if __name__ == "__main__":
    main()
