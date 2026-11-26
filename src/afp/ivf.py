"""Spherical k-means inverted-file search for unit audio fingerprints."""

from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np


@lru_cache(maxsize=1)
def _search_kernel():
    """Compile the CUDA kernel that scores the selected cluster lists."""
    source = Path(__file__).with_name("ivf.cu").read_text(encoding="utf-8")
    module = cp.RawModule(code=source)
    return module.get_function("select_lists"), module.get_function("scan_lists"), module.get_function("finish")


class SphericalIVF:
    """GPU inverted-file index trained by spherical k-means.

    A query probes its most similar centroids and reranks every window in those
    lists by cosine similarity. Search returns track IDs, scores and comparison
    counts, all on the GPU.
    """

    def __init__(
        self,
        references: cp.ndarray,
        tracks: cp.ndarray,
        *,
        n_clusters: int,
        seed: int = 2807,
        max_iter: int = 20,
        fit_chunk_size: int = 8192,
    ) -> None:
        """Fit centroids and construct sorted window lists on the current CUDA device."""
        if references.ndim != 2 or references.shape[1] != 256 or len(references) == 0:
            raise ValueError("Expected nonempty 256-dimensional references")
        if tracks.shape != (len(references),):
            raise ValueError("Expected one track ID for every reference window")
        if not 1 <= n_clusters <= len(references) or max_iter < 1 or fit_chunk_size < 1:
            raise ValueError("Cluster count and fit parameters must be positive and valid")
        self.references = cp.ascontiguousarray(references, dtype=cp.float32)
        self.tracks = cp.ascontiguousarray(tracks, dtype=cp.int32)
        self.n_clusters = n_clusters
        self.seed = seed
        self.max_iter = max_iter
        self.fit_chunk_size = fit_chunk_size
        self.centroids, self.iterations = self._fit()
        labels = self._assign(self.references)
        self.order = cp.argsort(labels).astype(cp.int32)
        counts = cp.bincount(labels, minlength=n_clusters).astype(cp.int32)
        self.offsets = cp.empty(n_clusters + 1, dtype=cp.int32)
        self.offsets[0] = 0
        self.offsets[1:] = cp.cumsum(counts, dtype=cp.int32)
        self._select, self._scan, self._finish = _search_kernel()

    def _assign(self, vectors: cp.ndarray) -> cp.ndarray:
        """Assign normalized vectors to their most similar centroid in bounded batches."""
        return self._assign_with(vectors, self.centroids)

    def _fit(self) -> tuple[cp.ndarray, int]:
        """Run seeded spherical Lloyd iterations with empty-cluster recovery."""
        if not cp.isfinite(self.references).all().item():
            raise ValueError("Reference vectors must be finite")
        norms = cp.linalg.norm(self.references, axis=1)
        if not cp.allclose(norms, 1.0, rtol=0, atol=2e-4).item():
            raise ValueError("Spherical k-means requires unit-normalized reference vectors")
        rng = np.random.default_rng(self.seed)
        initial_rows = rng.choice(len(self.references), self.n_clusters, replace=False)
        centroids = self.references[cp.asarray(initial_rows)].copy()
        previous = None
        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_with(self.references, centroids)
            sums = cp.zeros((self.n_clusters, self.references.shape[1]), dtype=cp.float32)
            for start in range(0, len(self.references), self.fit_chunk_size):
                stop = min(start + self.fit_chunk_size, len(self.references))
                cp.add.at(sums, labels[start:stop], self.references[start:stop])
            counts = cp.bincount(labels, minlength=self.n_clusters)
            populated = counts > 0
            means = cp.zeros_like(sums)
            means[populated] = sums[populated] / counts[populated, None]
            norms = cp.linalg.norm(means, axis=1)
            valid = populated & (norms > 1e-12)
            centroids = cp.zeros_like(means)
            centroids[valid] = means[valid] / norms[valid, None]
            empty = ~valid
            if empty.any().item():
                replacement_rows = rng.choice(len(self.references), int(empty.sum().item()), replace=False)
                centroids[empty] = self.references[cp.asarray(replacement_rows)]
            if previous is not None and cp.array_equal(labels, previous).item():
                return centroids, iteration
            previous = labels
        return centroids, self.max_iter

    def _assign_with(self, vectors: cp.ndarray, centroids: cp.ndarray) -> cp.ndarray:
        """Assign a matrix to a supplied set of centroids in bounded batches."""
        labels = []
        for start in range(0, len(vectors), self.fit_chunk_size):
            scores = vectors[start : start + self.fit_chunk_size] @ centroids.T
            labels.append(scores.argmax(axis=1).astype(cp.int32))
        return cp.concatenate(labels)

    def search(
        self, queries: cp.ndarray, *, nprobe: int, validate: bool = True
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Search the nprobe closest lists for each query.

        Set validate to false only when finite unit-norm queries were checked
        before the timed or repeated search operation.
        """
        if queries.ndim != 2 or queries.shape[1] != 256:
            raise ValueError("Expected queries with shape [N, 256]")
        if not 1 <= nprobe <= self.n_clusters:
            raise ValueError("nprobe must be between 1 and the number of clusters")
        queries = cp.ascontiguousarray(queries, dtype=cp.float32)
        if validate:
            if not cp.isfinite(queries).all().item():
                raise ValueError("Query vectors must be finite")
            if len(queries) and not cp.allclose(cp.linalg.norm(queries, axis=1), 1.0, rtol=0, atol=2e-4).item():
                raise ValueError("Queries must be unit-normalized")
        n = len(queries)
        track = cp.empty(n, dtype=cp.int32)
        score = cp.empty(n, dtype=cp.float32)
        visits = cp.empty(n, dtype=cp.int32)
        if not n:
            return track, score, visits
        centroid_scores = queries @ self.centroids.T
        if self.n_clusters <= 1024 and nprobe <= 16:
            selected = cp.empty((n, nprobe), dtype=cp.int32)
            self._select(
                (n,),
                (32,),
                (centroid_scores, np.int32(self.n_clusters), np.int32(nprobe), selected),
                shared_mem=4 * self.n_clusters,
            )
        else:
            selected = cp.argpartition(centroid_scores, -nprobe, axis=1)[:, -nprobe:].astype(cp.int32)
        stripes = 4 if n < 4 else 1
        parts = nprobe * stripes
        partial_scores = cp.empty((n, parts), dtype=cp.float32)
        partial_rows = cp.empty((n, parts), dtype=cp.int32)
        partial_visits = cp.empty((n, parts), dtype=cp.int32)
        self._scan(
            (parts, n),
            (128,),
            (
                queries,
                self.references,
                self.order,
                self.offsets,
                selected,
                np.int32(nprobe),
                np.int32(stripes),
                partial_scores,
                partial_rows,
                partial_visits,
            ),
        )
        self._finish(
            (n,),
            (32,),
            (
                partial_scores,
                partial_rows,
                partial_visits,
                self.tracks,
                np.int32(parts),
                track,
                score,
                visits,
            ),
        )
        return track, score, visits

    @property
    def index_bytes(self) -> int:
        """Return centroid and posting-list bytes, excluding shared reference vectors."""
        return int(self.centroids.nbytes + self.order.nbytes + self.offsets.nbytes)
