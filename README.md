# Neural audio fingerprinting

DTU 02807 Computational Tools for Data Science, deadline 4 December 2026. Identify a song from a short, possibly distorted TikTok-like clip, following [Robust Neural Audio Fingerprinting using Music Foundation Models](https://arxiv.org/abs/2511.05399): a frozen pretrained MuQ model (the paper's best backbone) turns 10 s windows into 256-number fingerprints that stay similar under noise, compression, speed changes and similar distortions. Locality-sensitive hashing (LSH) and a k-means inverted-file index search the fingerprints, measuring the accuracy/speed trade-off against exhaustive search; k-means and DBSCAN expose non-obvious connections between tracks (duplicates, remixes, covers, samples). Evaluation uses synthetic distortions, including TikTok-like edits (sped-up playback, voice-over and compression). Voice-over training and evaluation use disjoint speakers.

## Report findings

**Start here:** [short report (PDF)](reports/report-draft.pdf) · [runnable report notebook](notebooks/report_findings.ipynb) · [experiment inputs and full rerun commands](reports/report-evidence/README.md). Run the notebook top to bottom on CPU. It performs a real lookup on a small fingerprint sample and recomputes the full accuracy table from saved per-query predictions.

On 8,992 held-out Medium TikTok-like queries, the trained 256-dimensional projection increased exact-search accuracy from **58.02% to 93.54%** relative to raw MuQ. Adding 80,735 reference tracks with that head and those queries fixed cost exhaustive search **316 correct answers**. The table shows the Large catalogue results from the original paired RTX 4090 run. Every accuracy count reproduced on the released fingerprints.

| Large search | Standard top-1 (7,458) | TikTok-like top-1 (8,992) | Median single lookup | Queries/s in batches of 128 |
|---|---:|---:|---:|---:|
| Exhaustive cosine | 96.34% | 90.02% | 0.585 ms | **95,267** |
| Angular LSH | **96.47%** | **90.20%** | 0.253 ms | 7,057 |
| Spherical k-means IVF | 95.99% | 89.46% | **0.124 ms** | 89,420 |

LSH gave near-exhaustive recognition at 2.31× the single-query lookup speed. IVF gave the fastest single lookup; dense exhaustive search won in batches. The timer starts with a fingerprint in GPU memory. On Small, DBSCAN linked 69 track pairs. For one strict-ID miss, two IDs for Marco Trovatello’s *Be Sweet* had identical decoded audio. The notebook traces that example to the query and reference timestamps.

## Setup

[uv](https://docs.astral.sh/uv/), Python 3.12-3.13, Linux or Windows. Torch is the CUDA 11.8 build (MuQ on CPU takes about 1.6 s per window).

```bash
uv sync                    # core
uv sync --extra notebook   # + Jupyter, matplotlib (notebooks)
uv sync --extra gpu        # + CuPy/CUDA 12 (GPU LSH, scripts/compare_catalogues.py); extras can be combined
uv run pytest tests/       # GPU tests skip without CuPy; tests need no data or model
uv run invoke --list       # test, data, fmt
```

## Data

[Free Music Archive](https://github.com/mdeff/fma) (FMA; Defferrard et al., ISMIR 2017): 30 s mp3 clips plus metadata. Audio is under each artist's licence and meant for research, so it is downloaded, not redistributed.

| FMA subset | Source tracks | Searchable tracks | Searchable windows | Fingerprint download |
|---|---:|---:|---:|---|
| Small | 8,000 | 7,989 | 35,292 | [v0.2 (Small head and queries)](https://github.com/nickiliak/02807_computational_tools_ds/releases/tag/v0.2-fingerprints) |
| Medium | 25,000 | 24,970 | 110,320 | [Medium ZIP](https://github.com/nickiliak/02807_computational_tools_ds/releases/download/v0.3-fingerprints/fma-medium-fingerprints.zip) |
| Large | 106,574 | 105,705 | 464,464 | [Large ZIP](https://github.com/nickiliak/02807_computational_tools_ds/releases/download/v0.3-fingerprints/fma-large-fingerprints.zip) |

Extract the Medium and Large ZIPs in the repository root. They include the shared Medium query fingerprints, window and track IDs, and metadata. All 110,320 Medium reference windows are preserved in Large. Small’s separately trained head and query set make its original scores a separate experiment. [Shared head, pre-projection features, optional open-set queries and checksums](https://github.com/nickiliak/02807_computational_tools_ds/releases/tag/v0.3-fingerprints).

```bash
uv run python -m afp.data --subset small             # download + process everything (resumable, SHA-1 checked)
uv run python -m afp.data --subset small --limit 50  # quick test on 50 tracks
```

Each clip is decoded straight from the zip to 24 kHz mono int16, `data/processed/audio/<NNN>/<track_id>.npy`; re-running skips finished tracks. `manifest.csv` lists every processed track (id, path, duration, title, artist, genre); `skipped.csv` lists tracks that failed to decode or are under 10 s.

GitHub release `v0.2-fingerprints` has the `small` fingerprints, trained head (with voice-over) and `manifest.csv` (`v0.1-fingerprints`: first head), so audio is only needed to fingerprint new clips or re-run the pipeline.

## Fingerprints (`src/afp/fingerprint.py`)

```bash
uv run python -m afp.fingerprint cache     # MuQ features: every window, training pairs, evaluation queries
uv run python -m afp.fingerprint train     # contrastive training of the 1024 -> 4096 -> 256 projection head
uv run python -m afp.fingerprint encode    # write fingerprints, print exact-search top-1 per distortion
uv run python -m afp.fingerprint cache --limit-tracks 200 --pairs 500 --queries 100   # quick trial
```

All steps are resumable. Clips are cut into 10 s windows with a 5 s hop (35,326 windows for the 7,994 `small` tracks). MuQ is frozen; only the head trains, on (clean window, distorted copy) pairs. Tracks split 20/80: the 20% only supply training pairs, evaluation queries come from the other 80%, so the head is never scored on tracks it learned from.

- **Output:** `data/processed/fingerprints/references.npz` (`vectors` float32 `[N, 256]` unit norm, `track_id`, `segment_idx`, `start_s`, `rms`) and `queries.npz` (same plus `augmentation`); head in `models/projection_head.pt`. `encode` drops near-silent reference windows (`--min-rms`, default 0.001; 34 in `small`) because all silent windows share one fingerprint and match each other across tracks.
- **Search:** `afp.fingerprint.search(references, track_id, queries, threshold)` (defined in `src/afp/retrieval.py`) returns the best track per query, or `-1` below the cosine `threshold` ("not in the database"). `encode` also reports, over a range of thresholds, the share of queries still found and the share of queries of songs removed from the references (half the query tracks) wrongly matched. About 0.7 is a reasonable default.
- **Speed:** needs a GPU. On a DTU HPC V100 about 0.07 s per window; the full `small` run (`hpc/setup_small.sh`, then `hpc/run_small.sh`) took about 1.5 h. The default `cache` is about 153,000 windows (roughly three days on CPU).
- **Distortions** (`src/afp/augment.py`, used for training and standard queries): time stretch (pitch kept), pitch shift (up to 5 semitones, speed kept), both, coloured noise, reverb, reverb + noise, band-/high-/low-pass, echo.
- **TikTok-like test queries**, alone and all three combined: sped-up playback (1.1x-1.5x, pitch rises), voice-over (LibriSpeech, CC BY 4.0, -5 to +10 dB music-to-speech, level saved as `voice_over_db`), MP3 compression (about 27-40 kbit/s). They do not cover real phone microphones or TikTok's own processing.
- **Voice-over training** (`cache --voice-over`): training speech from LibriSpeech `train-clean-100`, test speech from `dev-clean` (disjoint speakers, so no test voice is heard in training). Sped-up, compression and the combined category stay unseen.

```bash
uv run python -m afp.data --speech train   # 600 speech chunks, 100 train-clean-100 speakers
uv run python -m afp.data --speech test    # 240 speech chunks, 40 dev-clean speakers
uv run python -m afp.fingerprint cache --voice-over --pairs 10000 --queries 2000
uv run python -m afp.fingerprint cache-tiktok   # features of 2,000 held-out TikTok-like queries
uv run python -m afp.fingerprint encode         # also scores queries_tiktok when present
```

**Differences from the paper:** noise and reverb are synthetic (paper: MUSAN, Aachen RIRs), no Encodec compression, and the open MuQ checkpoint was trained on the Million Song Dataset, so the paper's numbers may not reproduce. MuQ weights are CC-BY-NC 4.0.

**Exact-search top-1, FMA `small`, 2,000 queries per set from unseen tracks:**

| Query set | Head v1 | Head v2 (voice-over) |
|---|---|---|
| standard distortions | 0.981 | 0.983 |
| TikTok-like: compression | 0.986 | 0.989 |
| TikTok-like: sped up | 0.962 | 0.951 |
| TikTok-like: voice-over | 0.393 | 0.973 |
| TikTok-like: all three | 0.234 | 0.645 |
| TikTok-like: overall | 0.651 | 0.899 |

At threshold 0.7, head v2 finds 0.925 of standard and 0.668 of TikTok-like queries whose song is in the references, and wrongly accepts 0.050 and 0.034 of queries whose song is not. Real-world recordings are not covered.

## LSH retrieval (`src/afp/lsh.py`)

Angular (random-hyperplane) LSH with multi-probe; optional numba reranker (`rerank.py`) and CUDA version (`gpu_retrieval.py` + `gpu_lsh.cu`, needs `--extra gpu`). Run notebooks with `uv sync --extra notebook` (add `--extra gpu` for the GPU one), then `uv run jupyter lab notebooks/<name>.ipynb` and Run All, or regenerate headless: `uv run jupyter nbconvert --execute --inplace --to notebook --ExecutePreprocessor.timeout=600 notebooks/<name>.ipynb`.

- **`notebooks/lsh_retrieval.ipynb`** (CPU): track-disjoint parameter sweep, exact vs LSH by distortion, CPU timings; figure `reports/figures/lsh-retrieval.{pdf,png}`. It verifies the three v0.2 fingerprint archives in `data/processed/fingerprints/` against published digests and downloads missing ones.
- **`notebooks/gpu_retrieval.ipynb`**: LSH vs exhaustive FP32 cosine on the same CUDA device. Picks a configuration on development tracks, then measures held-out identification, false acceptance after removing songs, and single/batched speed. Timing starts with embeddings in VRAM (model inference, disk, host-to-device copies and index build excluded). Set `AFP_DATASET_TAG` (`small` or `medium`); results go to dataset-specific JSON and PDF/PNG under `reports/` (`gpu-retrieval-{small,medium}.json`, `reports/figures/gpu-retrieval-*`, `gpu-operating-points-medium.pdf`). `AFP_RETUNE=1` reselects settings using latency on another GPU. Also: `reports/medium-pipeline.json` (hardware, source hashes, preprocessing, worker measurements) and `reports/medium-catalogue-audit.json` (split check; effect of adding reference tracks with head and queries fixed). Fingerprints: `gh release download v0.2-fingerprints --repo nickiliak/02807_computational_tools_ds --pattern '*.npz' --dir data/processed/fingerprints`.

FMA `medium` on an RTX 4090: 110,320 windows from 24,970 tracks; 7,458 standard and 8,992 TikTok-like queries; synthetic distortions; the test includes tracks made unsearchable by the silence filter.

| Method | Standard top-1 | TikTok-like top-1 | Median lookup |
|---|---:|---:|---:|
| Exhaustive cosine | 98.08% | 93.54% | 0.196 ms |
| Fastest qualifying dev setting (14 bits, 32 tables, 3 extra probes) | 97.25% | 89.52% | 0.052 ms |
| Highest dev accuracy (10 bits, 64 tables, 3 extra probes) | 98.08% | 93.54% | 0.097 ms |

The accuracy-first row was added after seeing the primary result: it matches exact search's correct counts and returns the same track for 16,446 of 16,450 queries (its own interleaved exact timing: 0.207 ms). The primary index processes 1.10 million queries/s in batches of 128 versus 366,000 for exact. At cosine 0.70 it correctly accepts 83.58% of known queries and falsely accepts 5.99% of unknowns (exact: 84.61% and 6.35%).

### Rebuilding the medium fingerprints (GPU, PowerShell)

Fits a new head on 50,000 pairs; writes archives for the clean reference catalogue, 10,000 standard and 10,000 TikTok-like queries. All tracks can be references; a seeded 20% supplies training pairs, the other 80% queries; train and test voice-overs use separate LibriSpeech speakers. Data and model go under `data/medium/`. Needs `uv sync --locked` and the 22.19 GiB archive plus space for waveforms, speech, weights and features.

```powershell
$root = 'data/medium'; $data = "$root/data"; $model = "$root/models/projection_head.pt"; $backbone = "$root/muq"
$env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
uv run python -m afp.data --data-dir $data --subset medium
uv run python -m afp.data --data-dir $data --speech train
uv run python -m afp.data --data-dir $data --speech test
uv run hf download OpenMuQ/MuQ-large-msd-iter --revision 0562a57814f6f8bbd9fdea0a25921a2fce1a841a --local-dir $backbone
uv run python -m afp.fingerprint --data-dir $data --seed 0 cache --backbone $backbone --train-frac 0.2 --pairs 50000 --queries 10000 --voice-over --batch-size 16 --device cuda --precision bf16 --audio-cache-gb 8 --prep-workers 8
uv run python -m afp.fingerprint --data-dir $data --seed 0 cache-tiktok --backbone $backbone --train-frac 0.2 --queries 10000 --batch-size 64 --device cuda --precision bf16 --audio-cache-gb 8 --prep-workers 12
uv run python -m afp.fingerprint --data-dir $data --model $model --seed 0 train --epochs 30 --batch-size 512 --device cuda
uv run python -m afp.fingerprint --data-dir $data --model $model encode --device cuda --skip-evaluation
```

MuQ stays in GPU memory per cache stage and every batch is checked for finite values; with BF16 (supported GPUs) non-finite batches are retried in FP32. Audio preparation overlaps inference; the 8 GiB host cache holds reusable int16 waveforms (lower `--audio-cache-gb` with less RAM). Finished feature shards are reused after an interruption. Use separate data directories for different seeds, model revisions, audio or augmentation settings.

**Catalogue size check** (needs `--extra gpu`; run after encoding medium): validates shapes and norms, checks that evaluation tracks are absent from the training pairs, prints checksums and paired counts per query set. It uses the medium head for both searches, restricts the small index to tracks in the v0.2 small manifest (download: `gh release download v0.2-fingerprints --repo nickiliak/02807_computational_tools_ds --pattern manifest.csv --dir data/processed`, SHA-256 `8fe1d544d30f2a161aa4109451750621f4d6558629f4c9293f129644101ca643`) and evaluates the same queries against both catalogues, isolating the effect of adding candidate songs. The original small release uses its own head and query draws.

```bash
uv run python scripts/compare_catalogues.py data/medium/data/processed
```

## Clustering (`src/afp/cluster.py`, `src/afp/related.py`)

General clustering from the course, written by hand (k-means, DBSCAN, Davies-Bouldin) and tested against scikit-learn; CPU, seconds on the released `small` fingerprints.

**k-means IVF index.** Spherical k-means (Lloyd, k-means++ seeding, unit-length centroids) splits the windows into `k` clusters; a query is compared only with windows in its `nprobe` closest clusters, stored as contiguous blocks.

```bash
uv run python -m afp.cluster --k 128 256 512 --nprobe 4 8 16 32   # writes ivf_clusters.csv, ivf_search.csv
```

`small`, 35,292 windows, one query at a time, laptop CPU (numpy):

| Index | Standard top-1 | TikTok-like top-1 | Windows compared | ms/query |
|---|---:|---:|---:|---:|
| exact | 0.983 | 0.899 | 100% | 0.11 |
| k=128, nprobe=8 | 0.971 | 0.866 | 6.7% | 0.04 |
| k=128, nprobe=32 | 0.980 | 0.896 | 25% | 0.14 |
| k=512, nprobe=16 | 0.976 | 0.883 | 4.7% | 0.05 |

Davies-Bouldin keeps falling as `k` grows (4.0 at 32, 2.8 at 1024): the fingerprints have no natural coarse cluster structure, so `k` is set by the speed/accuracy trade-off. Song-level k-means barely follows genre (NMI about 0.27 vs `genre_top`), as expected for features trained to tell recordings apart, not styles. Timings are hardware-dependent; compare accuracy and share of windows compared across machines.

**DBSCAN: tracks that share audio.** Cosine neighbourhood (`eps = sqrt(2 - 2 s)` for unit vectors); unique music is noise, a cluster with windows of two or more tracks links them.

```bash
uv run python -m afp.related --min-similarity 0.9 --min-points 2   # writes dbscan_sweep.csv, related_pairs.csv
```

At 0.9 and `min_points` 2: 82.6% of windows are noise, 69 track pairs are linked, 97% by the same artist (duplicate uploads, album/single, instrumental/vocal versions, a radio edit, a live version, remixes, two across artists). Some exact-search misses are such versions: 5 of 34 standard and 7 of 201 TikTok-like misses return a linked version of the right song.

## Layout

```txt
src/afp/    data.py (FMA download, 24 kHz decode, manifest, segments)   augment.py (distortions)
            embed.py (MuQ, projection head, training)   fingerprint.py (cache/train/encode CLI)
            retrieval.py (exact search)   lsh.py, rerank.py (LSH, numba)   gpu_retrieval.py, gpu_lsh.cu (GPU LSH)
            cluster.py (k-means, Davies-Bouldin, IVF)   related.py (DBSCAN)
hpc/        LSF jobs: setup_small (download FMA small + MuQ), run_small (cache/train/encode), run_voice (head v2),
            run_tiktok (held-out queries), run_encode (CPU encode of heads v1/v2), gpucheck (node check)
scripts/    compare_catalogues.py       notebooks/  lsh_retrieval, gpu_retrieval       reports/  results and figures
tests/      no data or model needed     data/{raw,processed}/  git-ignored contents
```

## Experiment appendix

[Run the appendix notebook](notebooks/experiment_appendix.ipynb) or inspect the [source and measurement index](reports/appendix/SOURCES.md). It replays archived measurements on CPU from the files in this checkout. Start with the question you want to answer:

| Question | What the test found |
|---|---|
| Can we recover a track missing from the first five LSH candidates? | On 22 previously missed clips, independent acoustic search recovered 13. Widening LSH to ten recovered one. At the 1% development target the acoustic path gained nine correct accepted answers, with one additional absent clip accepted and 2.6 ms higher median decoded-audio latency. |
| When should the recogniser decline to name a song? | On Large, recalibrating the LSH policy reduced absent accepts from 23 to 8 per 1,024, and correct accepts from 691 to 337 per 1,024. The notebook shows both sides of each threshold decision. |
| Do extra windows, a different projection or more cluster probes help? | Temporal voting gained two net identifications on 384 synthetic clips, with a paired interval spanning zero. The notebook also covers the Medium representation control, IVF tuning and agreement between the top two distinct songs. |

The notebook distinguishes threshold fitting on development songs, final held-out tests and retrospective analysis of known failures. Physical phone recordings are still an open test.
