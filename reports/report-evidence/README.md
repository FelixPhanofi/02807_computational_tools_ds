# Reproducing the report

The [report notebook](../../notebooks/report_findings.ipynb) runs on CPU with NumPy, pandas and Matplotlib. From the repository root run `uv sync --extra notebook`, then open it with `uv run jupyter lab` and choose **Restart Kernel and Run All Cells**. It calculates the accuracy table from paired query predictions, creates [the figure](../figures/report-catalogues.pdf) and inspects the linked-track example. The small executable lookup uses one distorted query from each of 32 sampled held-out Medium tracks and their reference windows. The complete held-out evaluation uses 7,458 standard and 8,992 TikTok-like clips per method and catalogue.

## Full reruns

Download both fingerprint ZIPs and the Medium embeddings ZIP from [v0.3](https://github.com/nickiliak/02807_computational_tools_ds/releases/tag/v0.3-fingerprints). Extract all three into the repository root. For the Small DBSCAN check, put the three Small NPZ files from [v0.2](https://github.com/nickiliak/02807_computational_tools_ds/releases/tag/v0.2-fingerprints) in `data/processed/fingerprints/` and its manifest in `data/processed/manifest.csv`. Use a CUDA-capable environment for the first two commands. Each writes a fresh measurement outside the checked-in evidence.

```sh
uv run --extra gpu python scripts/check_projection.py . --output projection-ablation.json
uv run --extra gpu python scripts/check_report_retrieval.py --data-root data --output data/report-rerun
uv run python scripts/check_report_clusters.py --data-root data --small-processed data/processed --output cluster-check.json
```

The clustering check uses the merged CPU implementation at 256 centroids, 16 probes and 20 maximum iterations. It fits a reproducible 16,384-window sample and assigns every reference. Query selection uses seed 280711, 128 held-out queries per family; an all-probe exact-cosine check uses eight per family. Supplying `--audio-dir data/medium/data/processed/audio` adds a hash comparison of the decoded samples for the DBSCAN example. The ZIPs alone support all fingerprint comparisons.

## Which evidence supports each claim?

- **Representation.** `projection.json` was reproduced from the released embeddings and fingerprints and matched [the #5 measurements](https://github.com/ExoHumann/02807_computational_tools_ds/blob/1964e602f3c7f00aef5a31928129269e95353eae/experiments/medium_ablations/projection-ablation.json). Confidence intervals resample songs, keeping repeated clips together.
- **Search.** `original-retrieval.json` contains the report's counts, paired medians and throughput selected from [the #7 benchmark](https://github.com/ExoHumann/02807_computational_tools_ds/blob/d0be3706a8b4a0b7c62e5c60b631fd3fd1abf618/reports/catalogue-growth.json). `predictions.npz` and `retrieval-rerun.json` come from an independent run using the merged LSH and the [GPU IVF contributed in #4](https://github.com/ExoHumann/02807_computational_tools_ds/tree/b35e9525385f7855b7858af046f9540c4dcb5e87/src/afp). The notebook checks all 12 counts. Timings are separate measurements on the RTX 4090. Both runs time warm resident fingerprints, 192 shuffled paired single lookups and three passes in batches of 128. Earlier #1's 2.13× is a separate paired Medium run.
- **Identity.** `clustering.json` records the merged CPU implementation and a repeat of [#11's Small DBSCAN experiment](https://github.com/nickiliak/02807_computational_tools_ds/pull/11). The 69 linked pairs, 82.6% noise and five linked standard errors reproduced. The current strict-ID run found six linked errors among 200 distorted misses; the original reported seven among 201. The historical query-level winners are unavailable in the merged report, so the individual discrepancy is unresolved. The two IDs in the notebook have identical fingerprint windows and identical decoded PCM. Duplicate windows make exact-ID ties possible.
- **Scope.** The catalogue comparison freezes the Medium head and queries. It measures synthetic identification under catalogue growth. Real phone captures are pending. The PDF is a group draft; add the final author list and the agreed contribution outline before submission.
