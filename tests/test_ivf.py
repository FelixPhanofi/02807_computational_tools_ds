"""Verify the GPU spherical-k-means index against exhaustive cosine search."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    device_count = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    device_count = 0
pytestmark = pytest.mark.skipif(device_count == 0, reason="CUDA device required")

from afp.ivf import SphericalIVF


def unit_vectors(seed: int, rows: int) -> np.ndarray:
    """Make deterministic unit vectors for an independent search fixture."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((rows, 256), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_probing_every_list_matches_exhaustive_search_with_ties():
    """Full probing covers each window once and keeps the earliest equal-scoring row."""
    references = unit_vectors(21, 48)
    references[1] = references[0]
    tracks = np.arange(100, 148, dtype=np.int32)
    queries = np.stack((references[0], references[12], -references[31]))
    index = SphericalIVF(cp.asarray(references), cp.asarray(tracks), n_clusters=8, seed=9, max_iter=8)
    found, scores, visits = index.search(cp.asarray(queries), nprobe=index.n_clusters)
    dense = queries @ references.T
    exact_rows = dense.argmax(axis=1)
    np.testing.assert_array_equal(found.get(), tracks[exact_rows])
    np.testing.assert_allclose(scores.get(), dense.max(axis=1), atol=2e-6)
    np.testing.assert_array_equal(visits.get(), np.full(len(queries), len(references)))
    assert found[0].item() == tracks[0]


def test_fit_is_seeded_and_search_reports_selected_rows():
    """Repeated fitting uses the same assignments and visit counts."""
    references = unit_vectors(42, 72)
    tracks = np.arange(72, dtype=np.int32)
    queries = references[[2, 8, 19]]
    first = SphericalIVF(cp.asarray(references), cp.asarray(tracks), n_clusters=6, seed=12, max_iter=6)
    second = SphericalIVF(cp.asarray(references), cp.asarray(tracks), n_clusters=6, seed=12, max_iter=6)
    np.testing.assert_allclose(first.centroids.get(), second.centroids.get(), atol=1e-6, rtol=1e-5)
    found, scores, visits = first.search(cp.asarray(queries), nprobe=2)
    dense = queries @ references.T
    assert np.all(np.isin(found.get(), tracks))
    assert np.all(scores.get() <= dense.max(axis=1) + 2e-6)
    assert np.all(visits.get() > 0) and np.all(visits.get() <= len(references))
    assert first.index_bytes == first.centroids.nbytes + first.order.nbytes + first.offsets.nbytes


def test_invalid_vectors_and_probe_counts_are_rejected():
    """Validate the unit-vector and nprobe requirements."""
    tracks = cp.arange(3, dtype=cp.int32)
    with pytest.raises(ValueError, match="unit-normalized"):
        SphericalIVF(cp.ones((3, 256), dtype=cp.float32), tracks, n_clusters=2)
    references = cp.asarray(unit_vectors(4, 3))
    index = SphericalIVF(references, tracks, n_clusters=2, max_iter=2)
    with pytest.raises(ValueError, match="nprobe"):
        index.search(references[:1], nprobe=3)


@pytest.mark.parametrize("clusters,probes", [(1, 1), (63, 4), (256, 16), (1024, 16)])
def test_shortlist_kernel_agrees_with_stable_sort(clusters, probes):
    """Compare every selected centroid to a full stable sort on tied, negative and random scores."""
    from afp.ivf import _search_kernel

    rng = np.random.default_rng(190)
    scores = rng.integers(-20, 0, (7, clusters)).astype(np.float32)
    scores[0] = -2
    result = cp.empty((len(scores), probes), dtype=cp.int32)
    _search_kernel()[0](
        (len(scores),),
        (32,),
        (cp.asarray(scores), np.int32(clusters), np.int32(probes), result),
        shared_mem=4 * clusters,
    )
    np.testing.assert_array_equal(result.get(), np.argsort(-scores, axis=1, kind="stable")[:, :probes])
