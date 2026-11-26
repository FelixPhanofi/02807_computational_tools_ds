"""Build the interactive search lesson from the released Medium index and three archived queries."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from afp.lsh import AngularLSH

PROCESSED = ROOT / "data/medium/data/processed"
PUBLIC = ROOT / "explorer/public"
ROWS = (2, 5, 6)
BOXES_PER_FILE = 64


def save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


def build_boxes(index, starts, labels):
    """Group every window in the first map by its source recording; fetch 64 boxes at a time."""
    order, offsets = index.buckets[0]
    counts = np.diff(offsets).tolist()
    directory = PUBLIC / "ruler-boxes"
    directory.mkdir(exist_ok=True)
    for first in range(0, len(counts), BOXES_PER_FILE):
        block = []
        for box in range(first, first + BOXES_PER_FILE):
            groups = {}
            for row in order[offsets[box]:offsets[box + 1]]:
                groups.setdefault(int(index.track_ids[row]), []).append(int(starts[row]))
            songs = sorted(groups, key=lambda tid: (-len(groups[tid]), labels[tid]["title"].casefold(), tid))
            block.append([[tid, labels[tid]["title"], labels[tid]["artist"], round(float(labels[tid]["duration_s"]), 4), *sorted(groups[tid])] for tid in songs])
            assert sum(len(g) - 4 for g in block[-1]) == counts[box]
        save(directory / f"{first // BOXES_PER_FILE}.json", block)
    return counts


def make_case(index, starts, queries, row, labels, shard):
    query = queries["vectors"][row]
    source = int(queries["track_id"][row])
    assert shard["track_id"][row] == source and shard["start_s"][row] == queries["start_s"][row]
    parts = query[:, None] * index.planes[:, :index.bits]
    pulls = np.column_stack((np.maximum(parts, 0).sum(0), -np.minimum(parts, 0).sum(0)))
    projections = (query @ index.planes).reshape(index.tables, index.bits)
    assert np.allclose(pulls[:, 0] - pulls[:, 1], projections[0], atol=1e-6)
    keys = (projections >= 0) @ index.weights
    uncertain = np.argsort(abs(projections), axis=1, kind="stable")[:, :2]
    shortlist = []
    for table, (key, (a, b), (order, bounds)) in enumerate(zip(keys, uncertain, index.buckets, strict=True)):
        boxes = [int(key ^ mask) for mask in (0, 1 << a, 1 << b, (1 << a) | (1 << b))]
        if table == 0:
            first_boxes = boxes
        shortlist.append(np.concatenate([order[bounds[k]:bounds[k + 1]] for k in boxes]))
    candidates = np.unique(np.concatenate(shortlist))
    scores = index.references[candidates] @ query
    best = int(np.argmax(scores))
    winner = int(index.track_ids[candidates[best]])
    other = np.flatnonzero(index.track_ids[candidates] != winner)
    second = other[np.argmax(scores[other])]
    runner = int(index.track_ids[candidates[second]])
    by_row = dict(zip(candidates.tolist(), scores.tolist(), strict=True))
    timelines = {}
    for tid in dict.fromkeys((source, winner, runner)):
        windows = []
        for ref in np.flatnonzero(index.track_ids == tid):
            maps = [i for i, found in enumerate(shortlist, 1) if np.any(found == ref)]
            assert bool(maps) == (ref in by_row)
            windows.append(dict(start=float(starts[ref]), score=by_row.get(ref), maps=maps,
                                best=bool(ref == candidates[best])))
        timelines[tid] = dict(title=labels[tid]["title"], artist=labels[tid]["artist"],
                              duration=float(labels[tid]["duration_s"]), windows=windows)
    # The source waveform shows the saved recording. The fingerprint belongs to the altered query.
    raw = np.load(PROCESSED / labels[source]["path"], mmap_mode="r")
    envelope = [round(float(np.abs(chunk.astype("f4")).max()) / 32768, 4) for chunk in np.array_split(raw, 192)]
    score, gap = float(scores[best]), float(scores[best] - scores[second])
    return dict(row=row, source=source, winner=winner, runner=runner, start=float(queries["start_s"][row]),
                change=str(queries["augmentation"][row]), waveform=envelope, fingerprint=query.tolist(),
                features=shard["distorted"][row].tolist(), pulls=pulls.tolist(), projections=projections[0].tolist(), probes=first_boxes, candidates=len(candidates),
                score=score, gap=gap, accepted=score >= .749896646 and gap >= .117484212, timelines=timelines)


def main():
    with (PROCESSED / "manifest.csv").open(encoding="utf-8", newline="") as file:
        labels = {int(r["track_id"]): r for r in csv.DictReader(file)}
    with np.load(PROCESSED / "fingerprints/references.npz") as archive:
        starts = archive["start_s"]
        assert np.all(starts == starts.astype(int))
        index = AngularLSH(archive["vectors"], archive["track_id"], bits=10, tables=64, seed=2807)
    counts = build_boxes(index, starts, labels)
    with np.load(PROCESSED / "fingerprints/queries.npz") as queries:
        with np.load(PROCESSED / "embeddings/queries/shard_0000.npz") as shard:
            cases = [make_case(index, starts, queries, row, labels, shard) for row in ROWS]
    assert sum(counts) == len(starts) and len(counts) == 1024
    for case in cases:
        assert sum((p >= 0) << bit for bit, p in enumerate(case["projections"])) == case["probes"][0]
        assert len(set(case["probes"])) == 4
        assert sum(w["best"] for t in case["timelines"].values() for w in t["windows"]) == 1
    save(PUBLIC / "ruler.json", dict(counts=counts, windows=len(starts), tracks=len(np.unique(index.track_ids)),
                                     boxesPerFile=BOXES_PER_FILE, windowSeconds=10, maps=index.tables, cases=cases))


if __name__ == "__main__":
    main()
