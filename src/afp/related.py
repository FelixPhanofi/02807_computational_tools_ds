"""Find songs that share audio (alternative takes, live and album versions, covers) with DBSCAN on the fingerprints.

Every reference window is a point. DBSCAN groups windows that lie in a dense region of near-identical fingerprints and
labels everything else as noise, so most windows (unique music) end up as noise. A cluster that contains windows of
two or more different tracks means those tracks share audio. Working on windows instead of one vector per track also
catches partial overlaps such as a shared intro.

    uv run python -m afp.related --min-similarity 0.9 --min-points 2

The neighbourhood radius is given as a cosine similarity ``s``. For unit vectors this is the Euclidean radius
``eps = sqrt(2 - 2 s)`` of the textbook definition.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from afp.cluster import davies_bouldin, exhaustive_search
from afp.paths import FINGERPRINTS_DIR, QUERY_KINDS


def radius_neighbours(x: np.ndarray, min_similarity: float, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For every point, all points (itself included) with cosine similarity at least ``min_similarity``.

    Args:
        x: Unit-norm vectors, shape ``[n, dim]``.
        min_similarity: Neighbourhood threshold.
        chunk: Rows compared at a time, to bound memory.

    Returns:
        A tuple ``(indptr, indices)`` in compressed sparse row form: the neighbours of point ``i`` are
        ``indices[indptr[i]:indptr[i + 1]]``.
    """
    counts, indices = [], []
    for i in range(0, len(x), chunk):
        block = x[i : i + chunk]
        rows, cols = np.nonzero(block @ x.T >= min_similarity)
        counts.append(np.bincount(rows, minlength=len(block)))
        indices.append(cols)
    indptr = np.concatenate([[0], np.cumsum(np.concatenate(counts))])
    return indptr, np.concatenate(indices)


def dbscan(x: np.ndarray, min_similarity: float, min_points: int) -> tuple[np.ndarray, np.ndarray]:
    """DBSCAN with cosine similarity: grow clusters from core points through chains of directly reachable points.

    A point is a core point if at least ``min_points`` points (itself included) have similarity at least
    ``min_similarity`` to it. A cluster is everything reachable from a core point. A non-core point reachable from
    several clusters joins the first one that reaches it. Points reachable from no core point are noise.

    Args:
        x: Unit-norm vectors, shape ``[n, dim]``.
        min_similarity: Neighbourhood threshold (cosine similarity).
        min_points: Neighbours needed for a core point.

    Returns:
        A tuple ``(labels, core)``: the cluster of every point (``-1`` for noise) and a mask of the core points.
    """
    indptr, indices = radius_neighbours(np.asarray(x, dtype=np.float32), min_similarity)
    core = np.diff(indptr) >= min_points
    labels = np.full(len(x), -1, dtype=np.int64)
    cluster = 0
    for seed in np.flatnonzero(core):
        if labels[seed] != -1:
            continue
        labels[seed] = cluster
        stack = [seed]
        while stack:
            point = stack.pop()
            if not core[point]:
                continue
            for neighbour in indices[indptr[point] : indptr[point + 1]]:
                if labels[neighbour] == -1:
                    labels[neighbour] = cluster
                    stack.append(neighbour)
        cluster += 1
    return labels, core


def shared_track_pairs(vectors: np.ndarray, track_id: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """List the pairs of different tracks that have windows in the same DBSCAN cluster.

    Args:
        vectors: Unit-norm window fingerprints, shape ``[n, dim]``.
        track_id: Track of each window.
        labels: DBSCAN cluster of each window (``-1`` for noise).

    Returns:
        One row per track pair (``track_a < track_b``) with the number of clusters they share and the highest cosine
        similarity between any window of one and any window of the other, sorted by that similarity.
    """
    shared: dict[tuple[int, int], int] = {}
    clustered = labels >= 0
    frame = pd.DataFrame({"cluster": labels[clustered], "track": track_id[clustered]}).drop_duplicates()
    for tracks in frame.groupby("cluster")["track"].apply(np.sort):
        for i, a in enumerate(tracks):
            for b in tracks[i + 1 :]:
                shared[(int(a), int(b))] = shared.get((int(a), int(b)), 0) + 1
    rows = []
    for (a, b), count in shared.items():
        similarity = float((vectors[track_id == a] @ vectors[track_id == b].T).max())
        rows.append({"track_a": a, "track_b": b, "shared_clusters": count, "max_similarity": similarity})
    columns = ["track_a", "track_b", "shared_clusters", "max_similarity"]
    return pd.DataFrame(rows, columns=columns).sort_values("max_similarity", ascending=False, ignore_index=True)


def describe_pairs(pairs: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Add artist, title and genre of both tracks and whether they share an artist.

    Args:
        pairs: Output of :func:`shared_track_pairs`.
        manifest: Table with ``track_id``, ``artist``, ``title`` and ``genre_top``.

    Returns:
        ``pairs`` with ``artist_a``, ``title_a``, ``genre_a``, the same for ``b``, and ``same_artist``.
    """
    info = manifest.set_index("track_id")[["artist", "title", "genre_top"]]
    out = pairs.copy()
    for side in ("a", "b"):
        part = info.reindex(out[f"track_{side}"]).reset_index(drop=True)
        out[f"artist_{side}"] = part["artist"]
        out[f"title_{side}"] = part["title"]
        out[f"genre_{side}"] = part["genre_top"]
    out["same_artist"] = out["artist_a"].str.casefold() == out["artist_b"].str.casefold()
    return out


def sweep(
    vectors: np.ndarray,
    track_id: np.ndarray,
    manifest: pd.DataFrame,
    similarities: list[float],
    min_points: list[int],
) -> pd.DataFrame:
    """Run DBSCAN for every parameter pair and summarise the clusters and the track pairs they link.

    Args:
        vectors: Unit-norm window fingerprints, shape ``[n, dim]``.
        track_id: Track of each window.
        manifest: Table with ``track_id`` and ``artist``.
        similarities: Values of ``min_similarity`` to try.
        min_points: Values of ``min_points`` to try.

    Returns:
        One row per setting: clusters, share of windows labelled noise, clusters spanning several tracks, linked
        track pairs, share of those pairs by the same artist, and the Davies-Bouldin index of the non-noise windows.
    """
    rows = []
    for similarity in similarities:
        for points in min_points:
            labels, _ = dbscan(vectors, similarity, points)
            clustered = labels >= 0
            spans = pd.Series(track_id[clustered]).groupby(labels[clustered]).nunique()
            pairs = describe_pairs(shared_track_pairs(vectors, track_id, labels), manifest)
            rows.append(
                {
                    "min_similarity": similarity,
                    "min_points": points,
                    "clusters": int(labels.max()) + 1,
                    "noise": float(np.mean(~clustered)),
                    "multi_track_clusters": int((spans > 1).sum()),
                    "track_pairs": len(pairs),
                    "same_artist": float(pairs["same_artist"].mean()) if len(pairs) else float("nan"),
                    "davies_bouldin": davies_bouldin(vectors[clustered], labels[clustered]) if clustered.any() else 0.0,
                }
            )
    return pd.DataFrame(rows)


def version_aware_top1(predicted: np.ndarray, source: np.ndarray, pairs: pd.DataFrame) -> dict[str, float]:
    """Top-1 accuracy, strict and with a linked version of the source track also counted as correct.

    FMA contains duplicate uploads and alternative versions of the same recording, so a strict "miss" can be the
    right song under another track id.

    Args:
        predicted: Predicted track id per query.
        source: Source track id per query.
        pairs: Linked track pairs, as returned by :func:`shared_track_pairs`.

    Returns:
        A dict with ``queries``, ``misses``, ``linked_misses`` (misses that predicted a linked version), ``top1`` and
        ``top1_versions``.
    """
    linked = set(zip(pairs["track_a"], pairs["track_b"])) | set(zip(pairs["track_b"], pairs["track_a"]))
    miss = predicted != source
    version = np.array([(int(s), int(p)) in linked for s, p in zip(source, predicted)], dtype=bool)
    n = len(predicted)
    return {
        "queries": n,
        "misses": int(miss.sum()),
        "linked_misses": int((miss & version).sum()),
        "top1": float(1 - miss.mean()) if n else float("nan"),
        "top1_versions": float(1 - (miss & ~version).mean()) if n else float("nan"),
    }


def track_groups(pairs: pd.DataFrame) -> pd.DataFrame:
    """Group linked tracks into connected components, to show how far DBSCAN chained them together.

    Args:
        pairs: Linked track pairs with ``track_a`` and ``track_b``.

    Returns:
        A table aligned with ``pairs`` with the ``group`` id of each pair and the ``group_size`` in tracks.
    """
    parent: dict[int, int] = {}

    def find(track: int) -> int:
        parent.setdefault(track, track)
        while parent[track] != track:
            parent[track] = parent[parent[track]]
            track = parent[track]
        return track

    for a, b in zip(pairs["track_a"], pairs["track_b"]):
        parent[find(int(a))] = find(int(b))
    sizes = pd.Series([find(track) for track in list(parent)]).value_counts()
    group = pairs["track_a"].map(lambda track: find(int(track)))
    return pd.DataFrame({"group": group, "group_size": group.map(sizes)}, index=pairs.index)


def group_summary(pairs: pd.DataFrame, bounds: tuple[int, ...] = (2, 10)) -> pd.DataFrame:
    """Linked pairs and their same-artist share by size of the track group they belong to.

    A large group means DBSCAN chained many tracks together through pairwise links, which with ``min_points = 2``
    behaves like single-linkage clustering.

    Args:
        pairs: Output of :func:`describe_pairs`.
        bounds: Upper bounds of the group-size bands; a last band holds every larger group.

    Returns:
        One row per band with ``group_size``, ``groups``, ``pairs`` and ``same_artist``.
    """
    groups = track_groups(pairs)
    edges = [1, *bounds, np.inf]
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        band = (groups["group_size"] > low) & (groups["group_size"] <= high)
        rows.append(
            {
                "group_size": f"{low + 1}-{int(high)}" if np.isfinite(high) else f">{low}",
                "groups": int(groups.loc[band, "group"].nunique()),
                "pairs": int(band.sum()),
                "same_artist": float(pairs.loc[band, "same_artist"].mean()) if band.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Command line entry point: DBSCAN parameter sweep, then the linked track pairs for the chosen setting."""
    parser = argparse.ArgumentParser(description="Find tracks that share audio with DBSCAN on the fingerprints.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--min-similarity", type=float, default=0.9, help="cosine threshold for neighbours")
    parser.add_argument("--min-points", type=int, default=2, help="neighbours needed for a core point")
    parser.add_argument("--sweep", type=float, nargs="*", default=[0.85, 0.9, 0.95], help="thresholds to compare")
    parser.add_argument("--show", type=int, default=40, help="linked pairs to print")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    processed = args.data_dir / "processed"
    references = np.load(processed / FINGERPRINTS_DIR / "references.npz")
    vectors, track_id = references["vectors"], references["track_id"]
    manifest = pd.read_csv(processed / "manifest.csv")

    if args.sweep:
        table = sweep(vectors, track_id, manifest, args.sweep, [2, 3, 5])
        table.to_csv(processed / FINGERPRINTS_DIR / "dbscan_sweep.csv", index=False)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print()

    labels, _ = dbscan(vectors, args.min_similarity, args.min_points)
    pairs = describe_pairs(shared_track_pairs(vectors, track_id, labels), manifest)
    pairs.to_csv(processed / FINGERPRINTS_DIR / "related_pairs.csv", index=False)
    columns = ["max_similarity", "shared_clusters", "artist_a", "title_a", "artist_b", "title_b"]
    if args.show > 0:
        with pd.option_context("display.max_colwidth", 35, "display.width", 200):
            print(pairs[columns].head(args.show).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print()
    print(f"{len(pairs)} linked track pairs; by size of the chained track group:")
    print(group_summary(pairs).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    rows = []
    for kind in QUERY_KINDS:
        path = processed / FINGERPRINTS_DIR / f"{kind}.npz"
        if path.exists():
            queries = np.load(path)
            predicted = track_id[exhaustive_search(vectors, queries["vectors"])]
            rows.append({"queries_set": kind, **version_aware_top1(predicted, queries["track_id"], pairs)})
    if rows:
        print("\nExact-search top-1, strict and counting a linked version of the source track as correct:")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
