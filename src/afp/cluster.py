"""Cluster the fingerprints with k-means and use the clusters as an inverted-file (IVF) index for fast search.

Exhaustive search compares a query with every reference window. An IVF index splits the references into ``k``
k-means clusters (Voronoi cells) and only compares the query with the windows in the ``nprobe`` cells whose centroids
are closest to it. This trades a little accuracy for far fewer comparisons, and it is a data-dependent alternative to
locality-sensitive hashing.

    uv run python -m afp.cluster --k 64 256 1024 --nprobe 1 4 16

k-means (Lloyd's algorithm with k-means++ seeding) and the Davies-Bouldin index are implemented here rather than
taken from scikit-learn. Because fingerprints are unit vectors compared by cosine similarity, the index uses
spherical k-means: centroids are rescaled to unit length after every update. Like CURE, k-means can be fitted on a
random sample that fits in memory and the remaining points are then assigned to the closest centroid.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from afp.paths import FINGERPRINTS_DIR, QUERY_KINDS


@dataclass
class KMeansResult:
    """Output of :func:`kmeans`.

    Attributes:
        centroids: Cluster centres, shape ``[k, dim]``.
        labels: Cluster of each point, shape ``[n]``.
        inertia: Sum of squared distances from each point to its centroid.
        n_iter: Number of assignment/update rounds that were run.
    """

    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int


def _as_float(x: np.ndarray) -> np.ndarray:
    """Return ``x`` as a floating point array, keeping ``float32`` and ``float64`` as they are."""
    x = np.asarray(x)
    return x if x.dtype in (np.float32, np.float64) else x.astype(np.float64)


def _normalise(x: np.ndarray) -> np.ndarray:
    """Scale every row to unit length. All-zero rows are left unchanged."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1)


def assign(x: np.ndarray, centroids: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Assign every point to its closest centroid (Euclidean distance).

    Args:
        x: Points, shape ``[n, dim]``.
        centroids: Centroids, shape ``[k, dim]``.
        chunk: Points handled at a time, to bound memory.

    Returns:
        A tuple ``(labels, squared_distances)`` with the closest centroid of each point and the squared distance to it.
    """
    labels = np.empty(len(x), dtype=np.int64)
    squared = np.empty(len(x), dtype=x.dtype)
    centroid_norms = (centroids**2).sum(axis=1)
    for i in range(0, len(x), chunk):
        block = x[i : i + chunk]
        partial = centroid_norms[None, :] - 2 * block @ centroids.T
        labels[i : i + chunk] = partial.argmin(axis=1)
        squared[i : i + chunk] = partial[np.arange(len(block)), labels[i : i + chunk]] + (block**2).sum(axis=1)
    return labels, np.maximum(squared, 0)


def kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``k`` initial centroids with k-means++ seeding (Arthur and Vassilvitskii, 2007).

    The first centroid is a uniformly random point. Each next one is a point drawn with probability proportional to
    its squared distance from the closest centroid chosen so far, so the seeds tend to be spread out.

    Args:
        x: Points, shape ``[n, dim]``.
        k: Number of centroids.
        rng: Random generator.

    Returns:
        The initial centroids, shape ``[k, dim]``.
    """
    norms = (x**2).sum(axis=1)
    centroids = np.empty((k, x.shape[1]), dtype=x.dtype)
    centroids[0] = x[rng.integers(len(x))]
    closest = np.maximum(norms - 2 * x @ centroids[0] + (centroids[0] ** 2).sum(), 0)
    for j in range(1, k):
        total = closest.sum()
        index = rng.choice(len(x), p=closest / total) if total > 0 else rng.integers(len(x))
        centroids[j] = x[index]
        closest = np.minimum(closest, np.maximum(norms - 2 * x @ centroids[j] + (centroids[j] ** 2).sum(), 0))
    return centroids


def cluster_means(x: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean of every cluster.

    Args:
        x: Points, shape ``[n, dim]``.
        labels: Cluster of each point, values in ``[0, k)``.
        k: Number of clusters.

    Returns:
        A tuple ``(means, counts)``. The mean of an empty cluster is all zeros.
    """
    membership = sparse.csr_matrix((np.ones(len(x), dtype=x.dtype), (labels, np.arange(len(x)))), shape=(k, len(x)))
    counts = np.bincount(labels, minlength=k)
    sums = np.asarray(membership @ x)
    return sums / np.maximum(counts, 1)[:, None], counts


def kmeans(
    x: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 0.0,
    init: str | np.ndarray = "k-means++",
    spherical: bool = False,
    seed: int = 0,
) -> KMeansResult:
    """Lloyd's k-means: alternate between assigning points to the closest centroid and moving centroids to the mean.

    Stops when at most a fraction ``tol`` of the points changed cluster in a round (``tol=0`` waits until no point
    moves), or after ``max_iter`` rounds. A cluster that becomes empty is restarted at the point that is farthest from
    its own centroid.

    Args:
        x: Points, shape ``[n, dim]``.
        k: Number of clusters.
        max_iter: Maximum number of rounds.
        tol: Fraction of reassigned points below which the algorithm counts as converged.
        init: ``"k-means++"``, ``"random"`` (``k`` distinct random points) or an array of initial centroids.
        spherical: Rescale centroids to unit length after each update, for unit-norm data compared by cosine.
        seed: Seed for the initialisation.

    Returns:
        The fitted :class:`KMeansResult`.

    Raises:
        ValueError: If ``k`` is not between 1 and the number of points.
    """
    x = _as_float(x)
    if not 1 <= k <= len(x):
        raise ValueError(f"k must be between 1 and {len(x)}, got {k}")
    rng = np.random.default_rng(seed)
    if isinstance(init, np.ndarray):
        centroids = init.astype(x.dtype, copy=True)
    elif init == "random":
        centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
    else:
        centroids = kmeans_plus_plus(x, k, rng)
    if spherical:
        centroids = _normalise(centroids)

    labels, squared = assign(x, centroids)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        centroids, counts = cluster_means(x, labels, k)
        for empty in np.flatnonzero(counts == 0):
            farthest = int(squared.argmax())
            centroids[empty] = x[farthest]
            squared[farthest] = 0
        if spherical:
            centroids = _normalise(centroids)
        new_labels, squared = assign(x, centroids)
        moved = np.count_nonzero(new_labels != labels)
        labels = new_labels
        if moved <= tol * len(x):
            break
    return KMeansResult(centroids, labels, float(squared.sum()), n_iter)


def davies_bouldin(x: np.ndarray, labels: np.ndarray, chunk: int = 8192) -> float:
    """Davies-Bouldin index: mean over clusters of the worst ratio ``(r_i + r_j) / d(c_i, c_j)``. Lower is better.

    ``c_i`` is the mean of cluster ``i`` and ``r_i`` the average Euclidean distance of its points to ``c_i``. Empty
    labels are ignored.

    Args:
        x: Points, shape ``[n, dim]``.
        labels: Cluster of each point.
        chunk: Points handled at a time, to bound memory.

    Returns:
        The index, or 0.0 when there are fewer than two clusters.
    """
    x = _as_float(x)
    _, labels = np.unique(labels, return_inverse=True)
    k = int(labels.max()) + 1
    if k < 2:
        return 0.0
    centroids, counts = cluster_means(x, labels, k)
    distances = np.concatenate(
        [np.linalg.norm(x[i : i + chunk] - centroids[labels[i : i + chunk]], axis=1) for i in range(0, len(x), chunk)]
    )
    radius = np.bincount(labels, weights=distances, minlength=k) / counts
    norms = (centroids**2).sum(axis=1)
    gap = np.sqrt(np.maximum(norms[:, None] + norms[None, :] - 2 * centroids @ centroids.T, 0))
    gap[gap <= 1e-12 * max(norms.max(), 1)] = np.inf
    ratio = (radius[:, None] + radius[None, :]) / gap
    np.fill_diagonal(ratio, -np.inf)
    return float(ratio.max(axis=1).mean())


class IVFIndex:
    """Inverted-file index: the references grouped by k-means cluster, searched only in the closest clusters.

    Args:
        vectors: Unit-norm reference fingerprints, shape ``[n, dim]``.
        centroids: Cluster centres, shape ``[k, dim]``.
        labels: Cluster of each reference.
    """

    def __init__(self, vectors: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> None:
        self.centroids = centroids
        self.labels = labels
        self.order = np.argsort(labels, kind="stable")
        self.sorted_vectors = np.ascontiguousarray(vectors[self.order])
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(labels, minlength=len(centroids)))])

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        k: int,
        sample: int | None = None,
        seed: int = 0,
        **kmeans_kwargs,
    ) -> "IVFIndex":
        """Fit spherical k-means on (a sample of) the references and assign every reference to a cluster.

        Args:
            vectors: Unit-norm reference fingerprints, shape ``[n, dim]``.
            k: Number of clusters.
            sample: Fit k-means on this many random references instead of all of them.
            seed: Seed for the sample and the k-means initialisation.
            **kmeans_kwargs: Passed to :func:`kmeans` (``max_iter``, ``tol``, ``init``).

        Returns:
            The index.
        """
        vectors = _as_float(vectors)
        train = vectors
        if sample is not None and sample < len(vectors):
            train = vectors[np.random.default_rng(seed).choice(len(vectors), size=sample, replace=False)]
        kmeans_kwargs.setdefault("spherical", True)
        result = kmeans(train, k, seed=seed, **kmeans_kwargs)
        labels = result.labels if train is vectors else assign(vectors, result.centroids)[0]
        return cls(vectors, result.centroids, labels)

    @property
    def k(self) -> int:
        """Number of clusters."""
        return len(self.centroids)

    def cluster_sizes(self) -> np.ndarray:
        """Number of references in each cluster."""
        return np.diff(self.offsets)

    def search(self, queries: np.ndarray, nprobe: int) -> tuple[np.ndarray, np.ndarray]:
        """Find the most similar reference of each query, looking only in the ``nprobe`` closest clusters.

        Args:
            queries: Unit-norm query fingerprints, shape ``[m, dim]``.
            nprobe: Number of clusters searched per query.

        Returns:
            A tuple ``(nearest, candidates)``: the index of the best reference found for each query (-1 if the probed
            clusters were empty) and the number of references compared with it.
        """
        queries = _as_float(queries)
        nprobe = min(nprobe, self.k)
        centroid_distance = (self.centroids**2).sum(axis=1)[None, :] - 2 * queries @ self.centroids.T
        probes = np.argpartition(centroid_distance, nprobe - 1, axis=1)[:, :nprobe]
        nearest = np.full(len(queries), -1, dtype=np.int64)
        candidates = np.zeros(len(queries), dtype=np.int64)
        for q, clusters in enumerate(probes):
            best = -np.inf
            for c in clusters:
                start, stop = self.offsets[c], self.offsets[c + 1]
                if stop == start:
                    continue
                scores = self.sorted_vectors[start:stop] @ queries[q]
                top = int(scores.argmax())
                if scores[top] > best:
                    best, nearest[q] = scores[top], self.order[start + top]
                candidates[q] += stop - start
        return nearest, candidates


def exhaustive_search(references: np.ndarray, queries: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Index of the most similar reference (largest inner product) for each query.

    Args:
        references: Unit-norm reference fingerprints, shape ``[n, dim]``.
        queries: Unit-norm query fingerprints, shape ``[m, dim]``.
        chunk: Queries searched at a time. Use 1 to time one query at a time.

    Returns:
        The index of the nearest reference for each query.
    """
    return np.concatenate(
        [(queries[i : i + chunk] @ references.T).argmax(axis=1) for i in range(0, len(queries), chunk)]
    )


def evaluate_ivf(
    references: np.ndarray,
    reference_track: np.ndarray,
    queries: np.ndarray,
    query_track: np.ndarray,
    ks: list[int],
    nprobes: list[int],
    sample: int | None = None,
    seed: int = 0,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare IVF search for several ``k`` and ``nprobe`` with exhaustive search.

    Both searches handle one query at a time, as a live identification service would, so the timings compare.

    Args:
        references: Unit-norm reference fingerprints, shape ``[n, dim]``.
        reference_track: Track id of each reference.
        queries: Unit-norm query fingerprints, shape ``[m, dim]``.
        query_track: Source track id of each query.
        ks: Numbers of clusters to try.
        nprobes: Numbers of probed clusters to try (values above ``k`` are skipped).
        sample: Fit k-means on this many references instead of all of them.
        seed: Seed for sampling and initialisation.
        max_iter: Maximum k-means rounds.
        tol: k-means convergence threshold (fraction of reassigned points).

    Returns:
        A tuple ``(clusters, search)``. ``clusters`` has one row per ``k``: fitting time, Davies-Bouldin index and the
        share of references in the largest cluster. ``search`` has one row per ``(k, nprobe)`` plus an exhaustive row
        (``k = 0``): top-1 track hit rate, agreement with exhaustive search, fraction of the references compared
        (counting the ``k`` centroid comparisons) and milliseconds per query.
    """
    references = _as_float(references)
    queries = _as_float(queries)
    start = time.perf_counter()
    exact = exhaustive_search(references, queries, chunk=1)
    exact_ms = (time.perf_counter() - start) * 1000 / len(queries)
    search_rows = [
        {
            "k": 0,
            "nprobe": 0,
            "top1": float(np.mean(reference_track[exact] == query_track)),
            "agree": 1.0,
            "compared": 1.0,
            "ms_per_query": exact_ms,
        }
    ]
    cluster_rows = []
    for k in ks:
        start = time.perf_counter()
        index = IVFIndex.build(references, k, sample=sample, seed=seed, max_iter=max_iter, tol=tol)
        cluster_rows.append(
            {
                "k": k,
                "fit_s": time.perf_counter() - start,
                "davies_bouldin": davies_bouldin(references, index.labels),
                "largest_share": index.cluster_sizes().max() / len(references),
            }
        )
        for nprobe in (p for p in nprobes if p <= k):
            start = time.perf_counter()
            nearest, candidates = index.search(queries, nprobe)
            elapsed = (time.perf_counter() - start) * 1000 / len(queries)
            found = nearest >= 0
            hits = found & (reference_track[np.where(found, nearest, 0)] == query_track)
            search_rows.append(
                {
                    "k": k,
                    "nprobe": nprobe,
                    "top1": float(hits.mean()),
                    "agree": float(np.mean(nearest == exact)),
                    "compared": float((candidates.mean() + k) / len(references)),
                    "ms_per_query": elapsed,
                }
            )
    return pd.DataFrame(cluster_rows), pd.DataFrame(search_rows)


def main(argv: list[str] | None = None) -> None:
    """Command line entry point: evaluate IVF search on the saved fingerprints and write the tables as CSV."""
    parser = argparse.ArgumentParser(description="k-means IVF index over the fingerprints, compared to exact search.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--k", type=int, nargs="+", default=[64, 128, 256, 512, 1024], help="numbers of clusters")
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32], help="clusters searched")
    parser.add_argument("--sample", type=int, default=None, help="fit k-means on this many references")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3, help="stop when this fraction of points moved")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    fingerprints = args.data_dir / "processed" / FINGERPRINTS_DIR
    references = np.load(fingerprints / "references.npz")
    tables = []
    for kind in QUERY_KINDS:
        if not (fingerprints / f"{kind}.npz").exists():
            continue
        queries = np.load(fingerprints / f"{kind}.npz")
        clusters, search = evaluate_ivf(
            references["vectors"],
            references["track_id"],
            queries["vectors"],
            queries["track_id"],
            args.k,
            args.nprobe,
            sample=args.sample,
            seed=args.seed,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        tables.append(search.assign(queries=kind))
    clusters.to_csv(fingerprints / "ivf_clusters.csv", index=False)
    search = pd.concat(tables, ignore_index=True)
    search.to_csv(fingerprints / "ivf_search.csv", index=False)
    print(clusters.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(search.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
