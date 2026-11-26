"""Cross-check the merged CPU clustering algorithms with the published FMA fingerprints."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from afp.cluster import IVFIndex, exhaustive_search
from afp.related import dbscan, describe_pairs, shared_track_pairs


def check_ivf(root: Path) -> dict:
    """Fit on 16,384 windows, assign the full catalogue, and query a frozen sample."""
    with np.load(root / "references.npz") as data:
        vectors, tracks = data["vectors"], data["track_id"]
    index = IVFIndex.build(vectors, 256, sample=16384, seed=2807, max_iter=20)
    with np.load(root / "queries.npz") as archive:
        songs = np.unique(archive["track_id"])
    dev = np.random.default_rng(2211).choice(songs, len(songs) // 4, replace=False)
    result = {
        "windows": len(tracks),
        "tracks": len(np.unique(tracks)),
        "largest_cluster_share": float(index.cluster_sizes().max() / len(tracks)),
        "families": {},
    }
    for family in ("queries", "queries_tiktok"):
        with np.load(root / f"{family}.npz") as archive:
            held = np.flatnonzero(~np.isin(archive["track_id"], dev))
            sample = np.sort(np.random.default_rng(280711).choice(held, 128, replace=False))
            queries, truth = archive["vectors"][sample], archive["track_id"][sample]
        exact = exhaustive_search(vectors, queries, chunk=16)
        hits, scanned = index.search(queries, 16)
        all_hits, _ = index.search(queries[:8], 256)
        if (hits < 0).any() or (all_hits < 0).any():
            raise ValueError("No candidate returned")
        np.testing.assert_allclose(
            (vectors[all_hits] * queries[:8]).sum(1), (vectors[exact[:8]] * queries[:8]).sum(1), atol=1e-5
        )
        result["families"][family] = {
            "queries": len(truth),
            "exact_correct": int(sum(tracks[exact] == truth)),
            "ivf_correct": int(sum(tracks[hits] == truth)),
            "track_agreement": int(sum(tracks[hits] == tracks[exact])),
            "mean_window_share": float(np.mean(scanned) / len(tracks)),
            "all_probe_checks": len(all_hits),
        }
    return result


def check_dbscan(root: Path) -> dict:
    """Count Small strict-ID errors linked by DBSCAN and retain one concrete window match."""
    fp = root / "fingerprints"
    with np.load(fp / "references.npz") as data:
        vectors, tracks, starts = data["vectors"], data["track_id"], data["start_s"]
    labels, _ = dbscan(vectors, min_similarity=0.9, min_points=2)
    pairs = describe_pairs(shared_track_pairs(vectors, tracks, labels), pd.read_csv(root / "manifest.csv"))
    lookup = set(zip(pairs.track_a, pairs.track_b, strict=True))
    families, examples = {}, []
    for family in ("queries", "queries_tiktok"):
        with np.load(fp / f"{family}.npz") as data:
            truth, query_start, queries = data["track_id"], data["start_s"], data["vectors"]
        best = exhaustive_search(vectors, queries)
        predicted = tracks[best]
        missed = np.flatnonzero(truth != predicted)
        linked = [i for i in missed if tuple(sorted((truth[i], predicted[i]))) in lookup]
        families[family] = {"strict_errors": len(missed), "linked_errors": len(linked), "queries": len(truth)}
        for i in linked:
            a, b = sorted((truth[i], predicted[i]))
            pair = pairs[(pairs.track_a == a) & (pairs.track_b == b)].iloc[0]
            examples.append(
                {
                    "family": family,
                    "query_row": int(i),
                    "source_id": int(truth[i]),
                    "returned_id": int(predicted[i]),
                    "query_start_s": float(query_start[i]),
                    "reference_start_s": float(starts[best[i]]),
                    "artist_a": str(pair.artist_a),
                    "title_a": str(pair.title_a),
                    "artist_b": str(pair.artist_b),
                    "title_b": str(pair.title_b),
                    "track_a": int(a),
                    "track_b": int(b),
                    "strongest_window_cosine": float(pair.max_similarity),
                }
            )
    return {
        "linked_pairs": len(pairs),
        "noise_share": float(np.mean(labels == -1)),
        "families": families,
        "example": max(examples, key=lambda row: row["strongest_window_cosine"]) if examples else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--small-processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("cluster-check.json"))
    parser.add_argument("--audio-dir", type=Path, help="Optional preprocessed 24 kHz audio folder for the linked pair")
    args = parser.parse_args()
    result = {
        "settings": {
            "k": 256,
            "probes": 16,
            "fit_sample": 16384,
            "fit_seed": 2807,
            "max_iter": 20,
            "query_seed": 280711,
        }
    }
    for name, directory in (("medium", "medium/data"), ("large", "large")):
        result[name] = check_ivf(args.data_root / directory / "processed/fingerprints")
        print(name, result[name], flush=True)
    result["small_dbscan"] = check_dbscan(args.small_processed)
    case = result["small_dbscan"]["example"]
    if args.audio_dir and case:
        hashes = {}
        for tid in (case["source_id"], case["returned_id"]):
            pcm = np.load(args.audio_dir / f"{tid // 1000:03d}/{tid:06d}.npy")
            hashes[str(tid)] = hashlib.sha256(pcm.tobytes()).hexdigest()
        case["pcm_sha256"] = hashes
        case["identical_pcm"] = len(set(hashes.values())) == 1
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
