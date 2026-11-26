"""Repeat the closed-set comparison on the v0.3 release with the merged LSH and restored GPU IVF."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from afp.gpu_retrieval import GPUAngularLSH, gpu_exact
from afp.ivf import SphericalIVF


def digest(path: Path) -> str:
    """Compute SHA-256 without loading a second copy of an archive."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def held_out(folder: Path) -> dict:
    """Apply the original development-track exclusion to both query families."""
    with np.load(folder / "queries.npz") as a:
        songs = np.unique(a["track_id"])
    dev = np.random.default_rng(2211).choice(songs, len(songs) // 4, replace=False)
    queries = {}
    for family, filename in (("standard", "queries.npz"), ("tiktok", "queries_tiktok.npz")):
        with np.load(folder / filename) as a:
            mask = ~np.isin(a["track_id"], dev)
            queries[family] = (a["vectors"][mask], a["track_id"][mask])
    return queries


def benchmark(path: Path, queries: dict) -> tuple[dict, dict]:
    """Score both families, then time the same resident queries in shuffled method order."""
    with np.load(path) as a:
        v, t = cp.asarray(a["vectors"]), cp.asarray(a["track_id"])
    lsh = GPUAngularLSH(v, t, bits=10, tables=64, seed=2807)
    ivf = SphericalIVF(v, t, n_clusters=256, seed=2807, max_iter=20)
    methods = {
        "exact": lambda q: gpu_exact(v, t, q)[0],
        "lsh": lambda q: lsh.search(q, probes=3)[0],
        "ivf": lambda q: ivf.search(q, nprobe=16, validate=False)[0],
    }
    result = {"windows": len(t), "tracks": int(cp.unique(t).size), "reference_sha256": digest(path), "families": {}}
    predictions = {}
    for family, (vectors, truth) in queries.items():
        q = cp.asarray(vectors)
        for name, search in methods.items():
            preds = cp.concatenate([search(q[i : i + 128]) for i in range(0, len(q), 128)]).get()
            predictions[f"{family}_{name}"] = preds
        result["families"][family] = {name: int((predictions[f"{family}_{name}"] == truth).sum()) for name in methods}
    q = cp.asarray(np.concatenate([v for v, _ in queries.values()]))
    selection = np.random.default_rng(42007).choice(len(q), 192, replace=False)
    order = np.random.default_rng(42008)
    sync = cp.cuda.get_current_stream().synchronize
    samples = {name: [] for name in methods}
    for method in methods.values():
        method(q[:1])
    sync()
    for i in selection:
        query = q[i : i + 1]
        for name in order.permutation(list(methods)):
            sync()
            start = time.perf_counter_ns()
            methods[name](query)
            sync()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    result["median_lookup_ms"] = {name: float(np.median(x)) for name, x in samples.items()}
    seconds = {name: [] for name in methods}
    for repeat in range(3):
        for name in np.random.default_rng(42107 + repeat).permutation(list(methods)):
            sync()
            start = time.perf_counter()
            for i in range(0, len(q), 128):
                methods[name](q[i : i + 128])
            sync()
            seconds[name].append(time.perf_counter() - start)
    result["batch128_queries_per_s"] = {name: float(len(q) / np.median(x)) for name, x in seconds.items()}
    return result, predictions


def run(root: Path, output: Path) -> None:
    """Save portable per-query predictions and counters for the report notebook."""
    medium = root / "medium/data/processed/fingerprints"
    queries = held_out(medium)
    summary = {
        "gpu": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        "queries": {k: len(t) for k, (_, t) in queries.items()},
    }
    arrays = {f"{k}_truth": t for k, (_, t) in queries.items()}
    output.mkdir(parents=True, exist_ok=True)
    for name, path in (("medium", medium), ("large", root / "large/processed/fingerprints")):
        for filename in ("queries.npz", "queries_tiktok.npz"):
            if digest(path / filename) != digest(medium / filename):
                raise ValueError(f"The query archive differs between catalogues: {filename}")
        summary[name], predictions = benchmark(path / "references.npz", queries)
        arrays.update({f"{name}_{k}": a for k, a in predictions.items()})
        print(name, summary[name], flush=True)
    np.savez_compressed(output / "predictions.npz", **arrays)
    (output / "retrieval-rerun.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/report-rerun"))
    args = parser.parse_args()
    run(args.data_root, args.output)
