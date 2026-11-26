"""Publish compact maps in the release's reference-row order and three recorded search paths."""
import csv
import io
import sys
import zipfile
from urllib.parse import urlsplit

import numpy as np

from build_ruler import PROCESSED, PUBLIC, ROOT, ROWS, save

sys.path.insert(0, str(ROOT / "explorer"))
from search_trace import pack_trace
from afp.lsh import AngularLSH


def source_metadata():
    with zipfile.ZipFile(PROCESSED.parent / "raw/fma_metadata.zip") as archive:
        with archive.open("fma_metadata/raw_tracks.csv") as source:
            return {
                int(r["track_id"]): (r["track_url"], r["license_title"], r["license_url"])
                for r in csv.DictReader(io.TextIOWrapper(source, encoding="utf8"))
            }


def main():
    out = PUBLIC / "search-atlas"
    out.mkdir(exist_ok=True)
    with (PROCESSED / "manifest.csv").open(encoding="utf8", newline="") as f:
        labels = {int(r["track_id"]): r for r in csv.DictReader(f)}
    with np.load(PROCESSED / "fingerprints/references.npz") as f:
        vectors, ids, starts = (f[k] for k in ("vectors", "track_id", "start_s"))
    assert np.all(np.diff(ids) >= 0)
    sources, licenses, songs = source_metadata(), [], []
    for tid, begin, length in zip(*np.unique(ids, return_index=True, return_counts=True), strict=True):
        meta = labels[tid]
        source, rights, url = sources[tid]
        host = urlsplit(url).hostname or ""
        playable = (
            host in {"creativecommons.org", "www.creativecommons.org", "artlibre.org"}
            or rights == "Electronic Frontier Foundation Open Audio License"
        )
        license_row = [rights, url, playable]
        if license_row not in licenses:
            licenses.append(license_row)
        songs.append(
            [
                int(tid),
                meta["title"],
                meta["artist"],
                meta["genre_top"],
                round(float(meta["duration_s"]), 4),
                starts[begin : begin + length].astype(int).tolist(),
                source,
                licenses.index(license_row),
            ]
        )
    save(out / "songs.json", dict(library="medium-v1", songs=songs, licenses=licenses, windows=len(ids)))
    index = AngularLSH(vectors, ids, bits=10, tables=64, seed=2807)
    index.planes.T.astype("<f4").tofile(out / "checks.bin")
    for table, (order, offsets) in enumerate(index.buckets):
        keys = np.empty(len(ids), dtype="<u2")
        keys[order] = np.repeat(np.arange(1024), np.diff(offsets))
        keys.tofile(out / f"map-{table}.bin")
    with np.load(PROCESSED / "fingerprints/queries.npz") as f:
        for row in ROWS:
            query = f["vectors"][row]
            projections = (query @ index.planes).reshape(64, 10)
            addresses = (projections >= 0) @ index.weights
            bits = np.argsort(abs(projections), axis=1, kind="stable")[:, :2]
            boxes, parts = [], []
            for address, (a, b), (order, offsets) in zip(addresses, bits, index.buckets, strict=True):
                checked = [int(address ^ mask) for mask in (0, 1 << a, 1 << b, (1 << a) | (1 << b))]
                boxes.append(checked)
                parts.extend(order[offsets[k] : offsets[k + 1]] for k in checked)
            rows = np.unique(np.concatenate(parts))
            scores = vectors[rows] @ query
            best = int(scores.argmax())
            others = np.flatnonzero(ids[rows] != ids[rows[best]])
            second = others[scores[others].argmax()]
            score = float(scores[best])
            gap = float(score - scores[second])
            raw = np.load(PROCESSED / labels[f["track_id"][row]]["path"], mmap_mode="r")
            wave = [round(float(np.abs(chunk.astype("f4")).max()) / 32768, 4) for chunk in np.array_split(raw, 96)]
            save(
                out / f"example-{row}.json",
                dict(
                    wave=wave,
                    track_id=int(ids[rows[best]]),
                    source=int(f["track_id"][row]),
                    start=float(f["start_s"][row]),
                    accepted=score >= 0.749896646 and gap >= 0.117484212,
                    score=score,
                    gap=gap,
                    trace=pack_trace(query, projections, boxes, rows, scores, len(ids), rows[best], rows[second]),
                ),
            )


if __name__ == "__main__":
    main()
