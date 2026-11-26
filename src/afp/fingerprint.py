"""Produce fingerprints from the processed audio: cache MuQ features, train the projection head, encode.

Each step is a subcommand and is resumable:

    uv run python -m afp.fingerprint cache    # MuQ features: clean windows, training pairs, evaluation queries
    uv run python -m afp.fingerprint train    # contrastive training of the head on the cached pairs
    uv run python -m afp.fingerprint encode   # fingerprints for every window and query, plus exact-search top-1

Optionally, ``cache --voice-over`` (after ``python -m afp.data --speech train``) adds voice-over to the training pairs,
and ``cache-tiktok`` (after ``python -m afp.data --speech test``) adds held-out queries with TikTok-like distortions
(sped-up playback, voice-over by speakers never heard in training, MP3 compression).

Tracks are split into a training part (used only to build the pairs the head learns from) and an evaluation part
(the source of the queries), so the head is never scored on tracks it was trained on. All tracks are in the
reference database.

Outputs under ``data/processed``: ``embeddings/{clean,pairs,queries,queries_tiktok}/shard_*.npz`` (cached MuQ
features) and ``fingerprints/{references,queries,queries_tiktok}.npz`` (the fingerprints). The trained head goes to ``models/projection_head.pt``.
"""

import argparse
import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from afp.augment import CATEGORIES, TEST_CATEGORIES, VOICE_CATEGORIES, random_augment
from afp.data import SAMPLE_RATE, SEGMENT_S, SPEECH_FILES, load_processed, load_speech_bank, segment
from afp.paths import EMBEDDINGS_DIR, FINGERPRINTS_DIR, QUERY_KINDS
from afp.embed import EMBED_DIM, HIDDEN_DIM, MUQ_REPO, MuQBackbone, ProjectionHead, train_head
from afp.retrieval import search

MIN_RMS = 1e-3
UNKNOWN_THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)


def split_tracks(track_ids: pd.Series, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split track ids into a training part and an evaluation part.

    Args:
        track_ids: All track ids.
        train_frac: Fraction of tracks used to build training pairs.
        seed: Seed for the shuffle.

    Returns:
        A tuple ``(train_ids, eval_ids)`` of sorted arrays.
    """
    shuffled = np.random.default_rng(seed).permutation(np.asarray(track_ids))
    n_train = int(len(shuffled) * train_frac)
    return np.sort(shuffled[:n_train]), np.sort(shuffled[n_train:])


def _embed(backbone: MuQBackbone, windows: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the backbone over ``windows`` in batches. Any object with a compatible ``encode`` method works."""
    batches = [backbone.encode(windows[i : i + batch_size]) for i in range(0, len(windows), batch_size)]
    return np.concatenate(batches)


def _embed_cache(backbone: MuQBackbone, windows: np.ndarray, batch_size: int) -> np.ndarray:
    """Encode a shard, convert it to the on-disk dtype, and reject non-finite features."""
    embeddings = _embed(backbone, windows, batch_size).astype(np.float16)
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("Refusing to save non-finite MuQ features")
    return embeddings


def _save_shard(path: Path, **arrays: np.ndarray) -> None:
    """Write arrays to ``path`` atomically so an interrupted run never leaves a half-written shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with open(temporary, "wb") as f:
        np.savez(f, **arrays)
    temporary.replace(path)


def load_shards(directory: Path) -> dict[str, np.ndarray]:
    """Load and concatenate every ``shard_*.npz`` in a directory.

    Args:
        directory: Folder written by :func:`cache_clean` or :func:`cache_clips`.

    Returns:
        A dict mapping each array name to the concatenation across shards.

    Raises:
        FileNotFoundError: If the directory holds no shards.
    """
    files = sorted(Path(directory).glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"no shards in {directory}, run the cache step first")
    parts = [dict(np.load(f)) for f in files]
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def cache_clean(
    manifest: pd.DataFrame,
    processed_dir: Path,
    backbone: MuQBackbone,
    batch_size: int = 4,
    tracks_per_shard: int = 50,
) -> None:
    """Embed every clean 10 s window (5 s hop) of every track and save the features in shards.

    Each window's RMS level is stored too, so near-silent windows can be dropped later. Finished shards are skipped;
    a finished shard without RMS levels gets them added from the audio, without running the backbone again. Delete
    ``embeddings/clean`` to start over with different settings.

    Args:
        manifest: Table with ``track_id`` and ``path`` (relative to ``processed_dir``).
        processed_dir: Directory holding the processed audio.
        backbone: Feature extractor.
        batch_size: Windows per backbone call.
        tracks_per_shard: Tracks per output file.
    """
    out_dir = Path(processed_dir) / EMBEDDINGS_DIR / "clean"
    for shard, first in enumerate(tqdm(range(0, len(manifest), tracks_per_shard), desc="clean windows")):
        path = out_dir / f"shard_{shard:04d}.npz"
        existing = dict(np.load(path)) if path.exists() else None
        if existing is not None and "rms" in existing:
            continue
        windows, track_id, segment_idx, start_s = [], [], [], []
        for row in manifest.iloc[first : first + tracks_per_shard].itertuples():
            segments, starts = segment(load_processed(Path(processed_dir) / row.path))
            windows.append(segments)
            track_id += [row.track_id] * len(segments)
            segment_idx += range(len(segments))
            start_s += starts.tolist()
        windows = np.concatenate(windows)
        rms = np.sqrt(np.mean(windows**2, axis=1)).astype(np.float32)
        if existing is not None:
            _save_shard(path, **existing, rms=rms)
            continue
        _save_shard(
            path,
            embeddings=_embed_cache(backbone, windows, batch_size),
            track_id=np.array(track_id, dtype=np.int32),
            segment_idx=np.array(segment_idx, dtype=np.int16),
            start_s=np.array(start_s, dtype=np.float32),
            rms=rms,
        )


def cache_clips(
    kind: str,
    manifest: pd.DataFrame,
    track_ids: np.ndarray,
    n_clips: int,
    processed_dir: Path,
    backbone: MuQBackbone,
    with_clean: bool,
    seed: int,
    batch_size: int = 4,
    clips_per_shard: int = 100,
    categories: tuple[tuple[str, ...], ...] = CATEGORIES,
    speech: np.ndarray | None = None,
    audio_cache_gb: float = 0.0,
    prep_workers: int = 1,
) -> None:
    """Cut random 10 s clips, distort each one and save the MuQ features of the distorted (and optionally clean) clip.

    The whole track is distorted before the clip is cut, so filters and reverb have no edge effects and time
    stretching maps the clip back to its position in the original track. Shards are seeded individually, so an
    interrupted run can be repeated. Delete ``embeddings/<kind>`` to start over with different settings.

    Args:
        kind: Output folder name below ``embeddings``: ``"pairs"`` for training, ``"queries"`` or
            ``"queries_tiktok"`` for evaluation.
        manifest: Table with ``track_id`` and ``path`` (relative to ``processed_dir``).
        track_ids: Tracks the clips are drawn from.
        n_clips: Number of clips.
        processed_dir: Directory holding the processed audio.
        backbone: Feature extractor.
        with_clean: Also embed the undistorted clip at the same position (needed to train the head).
        seed: Base seed.
        batch_size: Windows per backbone call.
        clips_per_shard: Clips per output file.
        categories: Distortion categories to draw from (see :func:`afp.augment.random_augment`).
        speech: Speech chunks, needed when a category includes ``voice_over``.
        audio_cache_gb: Maximum int16 waveform audio held in an LRU cache; zero disables it.
        prep_workers: Number of bounded CPU shard-preparation workers.
    """
    paths = manifest.set_index("track_id")["path"]
    window = round(SEGMENT_S * SAMPLE_RATE)
    out_dir = Path(processed_dir) / EMBEDDINGS_DIR / kind
    if audio_cache_gb < 0:
        raise ValueError("audio_cache_gb cannot be negative")
    if prep_workers < 1:
        raise ValueError("prep_workers must be at least 1")
    cache_limit_bytes = round(audio_cache_gb * (1 << 30))
    audio_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
    audio_cache_lock = Lock()
    audio_cache_bytes = 0
    audio_cache_peak_bytes = 0
    audio_cache_hits = 0
    audio_cache_misses = 0

    def load_cached(path: Path) -> np.ndarray:
        nonlocal audio_cache_bytes, audio_cache_peak_bytes, audio_cache_hits, audio_cache_misses
        if cache_limit_bytes == 0:
            return load_processed(path)
        with audio_cache_lock:
            raw = audio_cache.get(path)
            if raw is not None:
                audio_cache.move_to_end(path)
                audio_cache_hits += 1
            else:
                audio_cache_misses += 1
                raw = np.load(path)
                if raw.dtype == np.int16 and raw.nbytes <= cache_limit_bytes:
                    while audio_cache and audio_cache_bytes + raw.nbytes > cache_limit_bytes:
                        _, evicted = audio_cache.popitem(last=False)
                        audio_cache_bytes -= evicted.nbytes
                    audio_cache[path] = raw
                    audio_cache_bytes += raw.nbytes
                    audio_cache_peak_bytes = max(audio_cache_peak_bytes, audio_cache_bytes)
        return raw.astype(np.float32) / 32768.0

    def prepare_shard(
        shard: int,
    ) -> tuple[Path, dict[str, np.ndarray], np.ndarray | None, np.ndarray, float]:
        prepare_started = time.perf_counter()
        rng = np.random.default_rng([seed, shard])
        count = min(clips_per_shard, n_clips - shard * clips_per_shard)
        clean, distorted, track_id, start_s, labels, voice_db = [], [], [], [], [], []
        while len(distorted) < count:
            chosen = int(rng.choice(track_ids))
            track = load_cached(Path(processed_dir) / paths[chosen])
            start = int(rng.integers(0, len(track) - window + 1))
            record: dict[str, float] = {}
            audio, rate, label = random_augment(track, rng, SAMPLE_RATE, categories, speech, record)
            distorted_start = round(start / rate)
            if distorted_start + window > len(audio):
                continue
            clean.append(track[start : start + window])
            distorted.append(audio[distorted_start : distorted_start + window])
            track_id.append(chosen)
            start_s.append(start / SAMPLE_RATE)
            labels.append(label)
            voice_db.append(record.get("voice_over_db", np.nan))
        arrays = {
            "track_id": np.array(track_id, dtype=np.int32),
            "start_s": np.array(start_s, dtype=np.float32),
            "augmentation": np.array(labels),
            "voice_over_db": np.array(voice_db, dtype=np.float32),
        }
        path = out_dir / f"shard_{shard:04d}.npz"
        return path, arrays, np.stack(clean) if with_clean else None, np.stack(
            distorted
        ), time.perf_counter() - prepare_started

    total_shards = math.ceil(n_clips / clips_per_shard)
    pending = [shard for shard in range(total_shards) if not (out_dir / f"shard_{shard:04d}.npz").exists()]
    futures = {}
    next_pending = 0
    with ThreadPoolExecutor(max_workers=prep_workers) as executor:

        def submit_bounded() -> None:
            nonlocal next_pending
            while next_pending < len(pending) and len(futures) < prep_workers:
                shard = pending[next_pending]
                futures[shard] = executor.submit(prepare_shard, shard)
                next_pending += 1

        submit_bounded()
        for index, shard in enumerate(tqdm(pending, total=len(pending), desc=kind)):
            path, arrays, clean_windows, distorted_windows, prepare_seconds = futures.pop(shard).result()
            submit_bounded()
            inference_started = time.perf_counter()
            arrays["distorted"] = _embed_cache(backbone, distorted_windows, batch_size)
            if with_clean:
                assert clean_windows is not None
                arrays["clean"] = _embed_cache(backbone, clean_windows, batch_size)
            inference_seconds = time.perf_counter() - inference_started
            _save_shard(path, **arrays)
            if index < 5:
                print(
                    f"{kind} shard {path.stem}: CPU prepare {prepare_seconds:.2f}s, "
                    f"feature inference {inference_seconds:.2f}s"
                )
    if cache_limit_bytes:
        print(
            f"{kind} audio LRU: {audio_cache_hits} hits, {audio_cache_misses} misses, "
            f"peak {audio_cache_peak_bytes / (1 << 30):.2f}/{audio_cache_gb:.2f} GiB"
        )


def train(
    processed_dir: Path,
    model_path: Path,
    hidden_dim: int = HIDDEN_DIM,
    out_dim: int = EMBED_DIM,
    **train_kwargs,
) -> list[tuple[float, float]]:
    """Train the projection head on the cached pairs and save it.

    Args:
        processed_dir: Directory holding ``embeddings/pairs``.
        model_path: Where to save the head (a dict with the weights and layer sizes).
        hidden_dim: Hidden layer width.
        out_dim: Fingerprint size.
        **train_kwargs: Passed to :func:`afp.embed.train_head` (epochs, batch_size, lr, temperature, seed, device).

    Returns:
        A list of ``(train_loss, val_loss)`` per epoch.
    """
    pairs = load_shards(Path(processed_dir) / EMBEDDINGS_DIR / "pairs")
    in_dim = pairs["clean"].shape[1]
    torch.manual_seed(train_kwargs.get("seed", 0))
    head = ProjectionHead(in_dim, hidden_dim, out_dim)
    history = train_head(head, pairs["clean"], pairs["distorted"], **train_kwargs)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": head.state_dict(), "in_dim": in_dim, "hidden_dim": hidden_dim, "out_dim": out_dim}, model_path
    )
    return history


def load_head(model_path: Path) -> ProjectionHead:
    """Load a head saved by :func:`train` in evaluation mode."""
    saved = torch.load(model_path, map_location="cpu")
    head = ProjectionHead(saved["in_dim"], saved["hidden_dim"], saved["out_dim"])
    head.load_state_dict(saved["state_dict"])
    return head.eval()


def _apply_head(head: ProjectionHead, embeddings: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    """Turn backbone features into unit-norm ``float32`` fingerprints."""
    head.to(device)
    with torch.no_grad():
        batches = [
            head(torch.from_numpy(embeddings[i : i + batch_size].astype(np.float32)).to(device)).cpu().numpy()
            for i in range(0, len(embeddings), batch_size)
        ]
    return np.concatenate(batches)


def top1_hit_rate(
    references: np.ndarray,
    reference_track: np.ndarray,
    queries: np.ndarray,
    query_track: np.ndarray,
    labels: np.ndarray,
    chunk: int = 256,
) -> pd.DataFrame:
    """Exact nearest-neighbour search: how often is the closest reference window from the query's source track?

    Args:
        references: Unit-norm reference fingerprints, shape ``[n, dim]``.
        reference_track: Track id of each reference fingerprint.
        queries: Unit-norm query fingerprints, shape ``[m, dim]``.
        query_track: Source track id of each query.
        labels: Distortion category of each query.
        chunk: Queries searched at a time, to bound memory.

    Returns:
        A table indexed by distortion category plus ``overall`` with the query count ``n`` and the hit rate ``top1``.
    """
    predicted, _ = search(references, reference_track, queries, chunk=chunk)
    hits = pd.DataFrame({"label": labels, "hit": predicted == query_track})
    table = hits.groupby("label")["hit"].agg(n="size", top1="mean")
    table.loc["overall"] = [len(hits), hits["hit"].mean()]
    return table


def unknown_song_table(
    references: np.ndarray,
    reference_track: np.ndarray,
    queries: np.ndarray,
    query_track: np.ndarray,
    thresholds: tuple[float, ...] = UNKNOWN_THRESHOLDS,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate songs missing from the database by removing half of the query tracks from the references.

    Args:
        references: Unit-norm reference fingerprints, shape ``[n, dim]``.
        reference_track: Track id of each reference fingerprint.
        queries: Unit-norm query fingerprints, shape ``[m, dim]``.
        query_track: Source track id of each query.
        thresholds: Similarity thresholds to evaluate.
        seed: Seed for choosing the removed tracks.

    Returns:
        A table indexed by threshold with ``found`` (share of queries whose track is still in the references that are
        matched correctly and accepted) and ``false_match`` (share of the other queries that are wrongly accepted).
    """
    query_tracks = np.unique(query_track)
    removed = np.random.default_rng(seed).choice(query_tracks, len(query_tracks) // 2, replace=False)
    keep = ~np.isin(reference_track, removed)
    predicted, score = search(references[keep], reference_track[keep], queries)
    known = ~np.isin(query_track, removed)
    rows = {
        t: {
            "found": np.mean((predicted == query_track)[known] & (score[known] >= t)),
            "false_match": np.mean(score[~known] >= t),
        }
        for t in thresholds
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("threshold")


def encode(
    processed_dir: Path,
    model_path: Path,
    device: str | None = None,
    out_name: str = FINGERPRINTS_DIR,
    min_rms: float = MIN_RMS,
    skip_evaluation: bool = False,
) -> dict[str, pd.DataFrame]:
    """Turn the cached features into fingerprints, save them, and report exact-search results per query set.

    Near-silent reference windows are dropped: every silent window gets the same fingerprint, so they would match each
    other across tracks. Every query set in ``QUERY_KINDS`` that has cached features is encoded and scored, both as
    top-1 per distortion and with half of its tracks removed from the references (see :func:`unknown_song_table`).

    Args:
        processed_dir: Directory holding ``embeddings/clean`` and at least ``embeddings/queries``.
        model_path: Head saved by :func:`train`.
        device: Device for the head. Defaults to CUDA when available.
        out_name: Folder below ``processed_dir`` for the fingerprints, so heads can be compared side by side.
        min_rms: Reference windows with a lower RMS level are dropped. Ignored for shards without RMS levels.
        skip_evaluation: Save vectors without running the CPU exact-search metric tables.

    Returns:
        A dict mapping ``<query set>`` to its :func:`top1_hit_rate` table and ``<query set>_unknown`` to its
        :func:`unknown_song_table`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head = load_head(model_path)
    out_dir = Path(processed_dir) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = load_shards(Path(processed_dir) / EMBEDDINGS_DIR / "clean")
    if "rms" in clean:
        loud = clean["rms"] >= min_rms
        print(f"dropped {np.sum(~loud)} near-silent reference windows (rms < {min_rms:g})")
        clean = {key: value[loud] for key, value in clean.items()}
    references = _apply_head(head, clean["embeddings"], device)
    np.savez(
        out_dir / "references.npz",
        vectors=references,
        **{key: clean[key] for key in ("track_id", "segment_idx", "start_s", "rms") if key in clean},
    )
    tables = {}
    for kind in QUERY_KINDS:
        if kind != "queries" and not (Path(processed_dir) / EMBEDDINGS_DIR / kind).exists():
            continue
        queries = load_shards(Path(processed_dir) / EMBEDDINGS_DIR / kind)
        query_vectors = _apply_head(head, queries["distorted"], device)
        extra = {"voice_over_db": queries["voice_over_db"]} if "voice_over_db" in queries else {}
        np.savez(
            out_dir / f"{kind}.npz",
            vectors=query_vectors,
            track_id=queries["track_id"],
            start_s=queries["start_s"],
            augmentation=queries["augmentation"],
            **extra,
        )
        if not skip_evaluation:
            tables[kind] = top1_hit_rate(
                references, clean["track_id"], query_vectors, queries["track_id"], queries["augmentation"]
            )
            tables[f"{kind}_unknown"] = unknown_song_table(
                references, clean["track_id"], query_vectors, queries["track_id"]
            )
    return tables


def main(argv: list[str] | None = None) -> None:
    """Command line entry point with the subcommands ``cache``, ``train`` and ``encode``."""
    parser = argparse.ArgumentParser(description="Produce fingerprints from the processed audio.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/projection_head.pt"))
    parser.add_argument("--seed", type=int, default=0)
    commands = parser.add_subparsers(dest="command", required=True)

    cache = commands.add_parser("cache", help="extract and store MuQ features")
    cache.add_argument("--limit-tracks", type=int, default=None, help="use only the first N tracks of the manifest")
    cache.add_argument("--train-frac", type=float, default=0.2, help="fraction of tracks used for training pairs")
    cache.add_argument("--pairs", type=int, default=20_000, help="number of training pairs")
    cache.add_argument("--queries", type=int, default=3_000, help="number of evaluation queries")
    cache.add_argument("--batch-size", type=int, default=4, help="windows per MuQ call")
    cache.add_argument("--voice-over", action="store_true", help="add voice-over (training speakers) to the pairs")
    cache.add_argument("--device", default=None)
    cache.add_argument("--backbone", default=MUQ_REPO, help="Hugging Face MuQ id or local model directory")
    cache.add_argument("--precision", choices=("fp32", "bf16"), default="fp32", help="MuQ inference precision")
    cache.add_argument("--audio-cache-gb", type=float, default=0, help="cap raw int16 track LRU (GiB; 0 disables)")
    cache.add_argument("--prep-workers", type=int, default=1, help="bounded CPU shard-preparation workers")

    tiktok = commands.add_parser("cache-tiktok", help="extract MuQ features of held-out TikTok-like queries")
    tiktok.add_argument("--limit-tracks", type=int, default=None, help="use only the first N tracks of the manifest")
    tiktok.add_argument("--train-frac", type=float, default=0.2, help="must match the value used for cache")
    tiktok.add_argument("--queries", type=int, default=2_000, help="number of evaluation queries")
    tiktok.add_argument("--batch-size", type=int, default=4, help="windows per MuQ call")
    tiktok.add_argument("--device", default=None)
    tiktok.add_argument("--backbone", default=MUQ_REPO, help="Hugging Face MuQ id or local model directory")
    tiktok.add_argument("--precision", choices=("fp32", "bf16"), default="fp32", help="MuQ inference precision")
    tiktok.add_argument("--audio-cache-gb", type=float, default=0, help="cap raw int16 track LRU (GiB; 0 disables)")
    tiktok.add_argument("--prep-workers", type=int, default=1, help="bounded CPU shard-preparation workers")

    fit = commands.add_parser("train", help="train the projection head on the cached pairs")
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--temperature", type=float, default=0.1)
    fit.add_argument("--device", default=None)

    encoding = commands.add_parser("encode", help="write fingerprints and report exact-search top-1")
    encoding.add_argument("--device", default=None)
    encoding.add_argument("--out-dir", default=FINGERPRINTS_DIR, help="folder below data/processed for the output")
    encoding.add_argument("--min-rms", type=float, default=MIN_RMS, help="drop quieter reference windows")
    encoding.add_argument(
        "--skip-evaluation", action="store_true", help="write GPU-encoded vectors without running CPU search metrics"
    )

    args = parser.parse_args(argv)
    processed_dir = args.data_dir / "processed"

    if args.command == "cache":
        manifest = pd.read_csv(processed_dir / "manifest.csv")
        if args.limit_tracks:
            manifest = manifest.head(args.limit_tracks)
        train_ids, eval_ids = split_tracks(manifest["track_id"], args.train_frac, args.seed)
        autocast_dtype = torch.bfloat16 if args.precision == "bf16" else None
        backbone = MuQBackbone(args.device, repo=args.backbone, autocast_dtype=autocast_dtype)
        print(f"MuQ inference: {args.precision}, batch size {args.batch_size}, device {backbone.device}")
        cache_clean(manifest, processed_dir, backbone, args.batch_size)
        pair_categories, pair_speech = CATEGORIES, None
        if args.voice_over:
            pair_categories = CATEGORIES + VOICE_CATEGORIES
            pair_speech = load_speech_bank(processed_dir / SPEECH_FILES["train"])
        cache_clips(
            "pairs",
            manifest,
            train_ids,
            args.pairs,
            processed_dir,
            backbone,
            True,
            args.seed,
            args.batch_size,
            categories=pair_categories,
            speech=pair_speech,
            audio_cache_gb=args.audio_cache_gb,
            prep_workers=args.prep_workers,
        )
        cache_clips(
            "queries",
            manifest,
            eval_ids,
            args.queries,
            processed_dir,
            backbone,
            False,
            args.seed + 1,
            args.batch_size,
            audio_cache_gb=args.audio_cache_gb,
            prep_workers=args.prep_workers,
        )
        print(f"FP32 fallback batches: {backbone.fp32_fallbacks}")
    elif args.command == "cache-tiktok":
        manifest = pd.read_csv(processed_dir / "manifest.csv")
        if args.limit_tracks:
            manifest = manifest.head(args.limit_tracks)
        _, eval_ids = split_tracks(manifest["track_id"], args.train_frac, args.seed)
        speech = load_speech_bank(processed_dir / SPEECH_FILES["test"])
        backbone = MuQBackbone(
            args.device,
            repo=args.backbone,
            autocast_dtype=torch.bfloat16 if args.precision == "bf16" else None,
        )
        print(f"MuQ inference: {args.precision}, batch size {args.batch_size}, device {backbone.device}")
        cache_clips(
            "queries_tiktok",
            manifest,
            eval_ids,
            args.queries,
            processed_dir,
            backbone,
            False,
            args.seed + 2,
            args.batch_size,
            categories=TEST_CATEGORIES,
            speech=speech,
            audio_cache_gb=args.audio_cache_gb,
            prep_workers=args.prep_workers,
        )
        print(f"FP32 fallback batches: {backbone.fp32_fallbacks}")
    elif args.command == "train":
        history = train(
            processed_dir,
            args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            seed=args.seed,
            device=args.device,
        )
        for epoch, (train_loss, val_loss) in enumerate(history, start=1):
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
    else:
        for kind, table in encode(
            processed_dir, args.model, args.device, args.out_dir, args.min_rms, args.skip_evaluation
        ).items():
            print(f"{kind}:\n{table.to_string(float_format=lambda x: f'{x:.3f}')}\n")


if __name__ == "__main__":
    main()
