import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from afp.related import (
    dbscan,
    describe_pairs,
    group_summary,
    shared_track_pairs,
    track_groups,
    version_aware_top1,
)


def test_version_aware_top1_counts_a_linked_version_as_correct():
    pairs = pd.DataFrame({"track_a": [1, 5], "track_b": [2, 6]})
    source = np.array([1, 2, 3, 5, 7])
    predicted = np.array([1, 1, 4, 6, 7])
    result = version_aware_top1(predicted, source, pairs)
    assert result == {"queries": 5, "misses": 3, "linked_misses": 2, "top1": 0.4, "top1_versions": 0.8}


def test_track_groups_and_summary_measure_chaining():
    pairs = pd.DataFrame(
        {
            "track_a": [1, 2, 3, 10, 20],
            "track_b": [2, 3, 4, 11, 21],
            "same_artist": [False, False, True, True, True],
        }
    )
    groups = track_groups(pairs)
    assert list(groups["group_size"]) == [4, 4, 4, 2, 2]
    assert groups["group"].iloc[:3].nunique() == 1 and groups["group"].nunique() == 3
    summary = group_summary(pairs, bounds=(2,)).set_index("group_size")
    assert summary.loc["2-2", "groups"] == 2 and summary.loc["2-2", "same_artist"] == 1.0
    assert summary.loc[">2", "pairs"] == 3 and np.isclose(summary.loc[">2", "same_artist"], 1 / 3)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_dbscan_matches_scikit_learn_on_core_points_and_noise():
    rng = np.random.default_rng(0)
    centres = _unit(rng.normal(size=(5, 16)))
    x = _unit(
        np.concatenate([c + rng.normal(scale=0.08, size=(40, 16)) for c in centres] + [rng.normal(size=(60, 16))])
    )
    similarity, points = 0.9, 5
    labels, core = dbscan(x, similarity, points)
    reference = DBSCAN(eps=np.sqrt(2 - 2 * similarity), min_samples=points).fit(x)
    expected_core = np.zeros(len(x), dtype=bool)
    expected_core[reference.core_sample_indices_] = True
    np.testing.assert_array_equal(core, expected_core)
    np.testing.assert_array_equal(labels == -1, reference.labels_ == -1)
    assert adjusted_rand_score(labels[core], reference.labels_[core]) == 1.0
    assert labels.max() + 1 == 5


def test_shared_track_pairs_links_tracks_with_a_common_window():
    rng = np.random.default_rng(1)
    base = _unit(rng.normal(size=(6, 8)))
    shared = _unit(base[:1] + 0.01 * rng.normal(size=(1, 8)))
    vectors = np.vstack([base[:3], shared, base[3:]])
    track_id = np.array([10, 10, 10, 20, 30, 30, 30])
    labels, _ = dbscan(vectors, 0.95, 2)
    pairs = shared_track_pairs(vectors, track_id, labels)
    assert list(zip(pairs["track_a"], pairs["track_b"])) == [(10, 20)]
    assert pairs.loc[0, "max_similarity"] > 0.95

    manifest = pd.DataFrame(
        {
            "track_id": [10, 20, 30],
            "artist": ["A", "a", "B"],
            "title": ["x", "x (live)", "y"],
            "genre_top": ["Rock"] * 3,
        }
    )
    described = describe_pairs(pairs, manifest)
    assert described.loc[0, "same_artist"] and described.loc[0, "title_b"] == "x (live)"
