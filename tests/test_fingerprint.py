import pytest

pytest.importorskip("torch")

import numpy as np
import pandas as pd
import torch

from afp.augment import (
    AUGMENTATIONS,
    CATEGORIES,
    TEST_AUGMENTATIONS,
    TEST_CATEGORIES,
    VOICE_CATEGORIES,
    _pitch_shift,
    augment,
)
from afp.data import SAMPLE_RATE, track_path
from afp.fingerprint import cache_clean, cache_clips, encode, search, split_tracks, train, unknown_song_table


class FakeBackbone:
    """Stand-in for MuQ: 32 log-spectrum band energies per window."""

    def encode(self, waveforms: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(waveforms, axis=1))[:, :120_000]
        return np.log1p(spectrum.reshape(len(waveforms), 32, -1).mean(axis=2)).astype(np.float32)


def test_augmentations_are_finite_and_only_time_stretch_changes_length():
    rng = np.random.default_rng(0)
    waveform = (0.3 * np.sin(2 * np.pi * 440 * np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32)
    for name in AUGMENTATIONS:
        out = augment(waveform, name, rng, SAMPLE_RATE)
        assert np.isfinite(out).all() and np.abs(out).max() <= 1.0, name
        assert (len(out) == len(waveform)) == (name != "time_stretch"), name
    shifted = _pitch_shift(torch.from_numpy(waveform), 12.0).numpy()
    peak_hz = np.fft.rfftfreq(len(shifted), 1 / SAMPLE_RATE)[np.abs(np.fft.rfft(shifted)).argmax()]
    assert abs(peak_hz - 880.0) < 2.0


def test_test_augmentations_speed_up_raises_pitch_and_others_keep_length():
    rng = np.random.default_rng(0)
    waveform = (0.3 * np.sin(2 * np.pi * 440 * np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32)
    speech = rng.uniform(-0.2, 0.2, size=(4, SAMPLE_RATE)).astype(np.float32)
    for name in TEST_AUGMENTATIONS:
        out = augment(waveform, name, rng, SAMPLE_RATE, speech)
        assert np.isfinite(out).all() and np.abs(out).max() <= 1.0, name
        assert (len(out) == len(waveform)) == (name != "speed_up"), name
    record = {}
    augment(waveform, "voice_over", rng, SAMPLE_RATE, speech, record)
    assert -5.0 <= record["voice_over_db"] <= 10.0
    sped = augment(waveform, "speed_up", np.random.default_rng(1), SAMPLE_RATE)
    rate = len(waveform) / len(sped)
    peak_hz = np.fft.rfftfreq(len(sped), 1 / SAMPLE_RATE)[np.abs(np.fft.rfft(sped)).argmax()]
    assert 1.1 <= rate <= 1.5 and abs(peak_hz - 440.0 * rate) < 5.0


def test_fingerprint_pipeline_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for track_id in range(6):
        t = np.arange(25 * SAMPLE_RATE) / SAMPLE_RATE
        tone = sum(np.sin(2 * np.pi * f * t) for f in rng.uniform(100, 4000, size=3)) / 3
        path = track_path(tmp_path, track_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, (tone * 20_000).astype(np.int16))
        rows.append({"track_id": track_id, "path": path.relative_to(tmp_path).as_posix()})
    manifest = pd.DataFrame(rows)
    train_ids, eval_ids = split_tracks(manifest["track_id"], 0.5, seed=0)
    assert set(train_ids).isdisjoint(eval_ids)

    backbone = FakeBackbone()
    cache_clean(manifest, tmp_path, backbone, batch_size=4, tracks_per_shard=3)
    speech = rng.uniform(-0.2, 0.2, size=(3, 4 * SAMPLE_RATE)).astype(np.float32)
    cache_clips(
        "pairs",
        manifest,
        train_ids,
        12,
        tmp_path,
        backbone,
        True,
        seed=0,
        batch_size=4,
        clips_per_shard=6,
        categories=CATEGORIES + VOICE_CATEGORIES,
        speech=speech,
    )
    cache_clips("queries", manifest, eval_ids, 6, tmp_path, backbone, False, seed=1, batch_size=4, clips_per_shard=6)
    cache_clips(
        "queries_tiktok",
        manifest,
        eval_ids,
        4,
        tmp_path,
        backbone,
        False,
        seed=2,
        batch_size=4,
        categories=TEST_CATEGORIES,
        speech=speech,
    )
    history = train(
        tmp_path, tmp_path / "projection_head.pt", hidden_dim=64, out_dim=16, epochs=2, batch_size=4, device="cpu"
    )
    assert len(history) == 2

    tables = encode(tmp_path, tmp_path / "projection_head.pt", device="cpu", out_name="fingerprints")
    assert tables["queries"].loc["overall", "n"] == 6 and 0.0 <= tables["queries"].loc["overall", "top1"] <= 1.0
    assert tables["queries_tiktok"].loc["overall", "n"] == 4
    assert list(tables["queries_unknown"].columns) == ["found", "false_match"]
    tiktok = np.load(tmp_path / "fingerprints" / "queries_tiktok.npz")
    assert tiktok["vectors"].shape == (4, 16)
    is_voice = np.char.find(tiktok["augmentation"], "voice_over") >= 0
    assert np.isnan(tiktok["voice_over_db"][~is_voice]).all() and np.isfinite(tiktok["voice_over_db"][is_voice]).all()
    references = np.load(tmp_path / "fingerprints" / "references.npz")
    assert references["vectors"].shape == (24, 16)
    np.testing.assert_allclose(np.linalg.norm(references["vectors"], axis=1), 1.0, atol=1e-5)


def test_search_rejects_below_threshold_and_unknown_table_counts_removed_tracks():
    references = np.eye(4, dtype=np.float32)
    reference_track = np.array([10, 10, 20, 30])
    queries = np.array([[1, 0, 0, 0], [0, 0, 0.6, 0.8]], dtype=np.float32)
    track, score = search(references, reference_track, queries, threshold=0.9)
    assert track.tolist() == [10, -1] and np.allclose(score, [1.0, 0.8])
    table = unknown_song_table(references, reference_track, queries, np.array([10, 30]), thresholds=(0.5, 0.9))
    assert set(table.columns) == {"found", "false_match"} and table.index.tolist() == [0.5, 0.9]


def test_cache_clean_adds_rms_to_old_shards_without_the_backbone(tmp_path):
    t = np.arange(12 * SAMPLE_RATE) / SAMPLE_RATE
    audio = np.concatenate([np.zeros(10 * SAMPLE_RATE), np.sin(2 * np.pi * 440 * t)])
    path = track_path(tmp_path, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, (audio * 20_000).astype(np.int16))
    manifest = pd.DataFrame([{"track_id": 1, "path": path.relative_to(tmp_path).as_posix()}])
    cache_clean(manifest, tmp_path, FakeBackbone())
    shard = tmp_path / "embeddings" / "clean" / "shard_0000.npz"
    old = {key: value for key, value in np.load(shard).items() if key != "rms"}
    np.savez(shard, **old)
    cache_clean(manifest, tmp_path, backbone=None)
    rms = np.load(shard)["rms"]
    assert len(rms) == len(old["embeddings"]) and rms[0] < 1e-3 < rms[-1]


def test_audio_cache_and_partial_resume_preserve_augmented_pairs(tmp_path):
    """Check that a bounded waveform cache and resuming between shards preserve seeded outputs."""
    rng = np.random.default_rng(91)
    rows = []
    for track in (7, 8):
        path = track_path(tmp_path, track)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, rng.integers(-5000, 5000, 12 * SAMPLE_RATE, dtype=np.int16))
        rows.append({"track_id": track, "path": path.relative_to(tmp_path).as_posix()})
    manifest = pd.DataFrame(rows)
    args = {
        "manifest": manifest,
        "track_ids": np.array([7, 8]),
        "processed_dir": tmp_path,
        "backbone": FakeBackbone(),
        "with_clean": True,
        "seed": 12,
        "batch_size": 2,
        "clips_per_shard": 2,
        "categories": (("noise",),),
    }
    cache_clips("uncached", n_clips=5, **args)
    one_track_gib = (12 * SAMPLE_RATE * 2) / (1 << 30)
    cache_clips("cached", n_clips=2, audio_cache_gb=one_track_gib, **args)
    first_path = tmp_path / "embeddings/cached/shard_0000.npz"
    first_write = first_path.stat().st_mtime_ns
    cache_clips("cached", n_clips=5, audio_cache_gb=one_track_gib, prep_workers=3, **args)
    assert first_path.stat().st_mtime_ns == first_write
    for original in sorted((tmp_path / "embeddings/uncached").glob("shard_*.npz")):
        with np.load(original) as expected, np.load(first_path.with_name(original.name)) as resumed:
            assert expected.files == resumed.files
            for key in expected.files:
                np.testing.assert_array_equal(expected[key], resumed[key])
