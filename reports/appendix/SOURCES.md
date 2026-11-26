# Measurement sources

Run the experiment appendix from the repository root in the default environment:

```sh
uv run python scripts/experiment_appendix.py
```

The [file manifest](sources.csv) records the original repository path, full source commit and SHA-256 of every archived input. The notebook checks these hashes using Git-style LF line endings for text files. All input files are local. The links below document the original measurement procedures, including prerequisites for generating new fingerprints or GPU timings.

| PR | Experiment and original procedure | Evidence replayed here |
|---|---|---|
| 2 | [Medium absent-song baseline](https://github.com/ExoHumann/02807_computational_tools_ds/blob/d15e8e937472d5cb7b0f3597f58407aee9fc7d62/reports/unknown-songs.md) | Calibrate the score cutoff on development rows and count held-out outcomes. |
| 3 | [Time-offset voting](https://github.com/ExoHumann/02807_computational_tools_ds/blob/4f58cea6d75b0f7c29f44a13b26260826291facf/reports/temporal-study.md) | Compare paired clip predictions, bootstrap by song and read the recorded decoded-clip timing summary. |
| 4 | [GPU spherical IVF](https://github.com/ExoHumann/02807_computational_tools_ds/blob/b35e9525385f7855b7858af046f9540c4dcb5e87/reports/ivf.md) | Select from the archived development sweep and display measured held-out accuracy, timing and index size. |
| 5 | [Raw versus trained projection and exploratory margin](https://github.com/ExoHumann/02807_computational_tools_ds/blob/1964e602f3c7f00aef5a31928129269e95353eae/notebooks/medium_ablations.ipynb) | Display recorded exact-search projection counts, song intervals and the separate IVF top-two comparison. |
| 6 | [Fresh Medium score-and-gap policy](https://github.com/ExoHumann/02807_computational_tools_ds/blob/c897de3728d60c47fda5271449b8166aa09cd01b/reports/runner-up-margin.md) | Apply the saved development cutoffs to held-out scores from two distinct track IDs. |
| 8 | [Large recalibration](https://github.com/ExoHumann/02807_computational_tools_ds/blob/e86ef0a1b5305d120bfc41b13f1f24448dd7a09d/reports/large-recalibration.md) | Apply both LSH policies to the same query rows and count retained, lost and gained identifications. |
| 9 | [Group-aware landmark verification](https://github.com/ExoHumann/02807_computational_tools_ds/blob/b3d6a10c8efa4b653a5d90c2bade38f052400d87/reports/recording-landmarks.md) | Rerank five saved candidates per query, score groups and strict archive IDs, and summarize paired timings. |
| 10 | [Independent acoustic nomination](https://github.com/ExoHumann/02807_computational_tools_ds/blob/46b339f1520a5629d039f01b3934ca237beef8d0/reports/acoustic-search.md) | Rerank each saved pool, apply frozen policies and count paired changes and shortlist recoveries. Summarize 64 timing rows. |

## Connections between the measurements

The original GPU studies used the [accuracy-first LSH implementation](https://github.com/ExoHumann/02807_computational_tools_ds/tree/5e9427ee5f83cd32fdd9b5a1c8f40396c3772989/src/afp). PRs 5 and 7 used the [PR 4 GPU IVF implementation](https://github.com/ExoHumann/02807_computational_tools_ds/tree/b35e9525385f7855b7858af046f9540c4dcb5e87/src/afp). The [PR 7 catalogue-growth driver](https://github.com/ExoHumann/02807_computational_tools_ds/blob/d0be3706a8b4a0b7c62e5c60b631fd3fd1abf618/scripts/benchmark_catalogue_growth.py) pins both implementations and the PR 6 policy. PR 8 used the PR 7 Large catalogue and the PR 6 query archive. PR 10 reused the PR 9 groups, query labels and five-candidate verifier. The published PR 9 and PR 10 timings come from separate live runs even where the recognition system is shared.

The Medium threshold studies use synthetic queries with two distortions per song. The temporal study uses separate songs and three conditions. The Large landmark studies define target identity using groups of linked excerpts and remove every member of an absent group from the reference index. Raw counts therefore have meaning within the specified cohort, identity definition and timing boundary.

## Next test with a phone

A small paired test could use the existing FMA playback excerpts and a second device as the microphone. Fix the catalogue, recognition rule and known/absent IDs before recording. Play the same ten-second passages in quiet and under one specified interference condition. Keep a digital copy of each passage. Report correct titles, wrong titles and abstentions with denominators for each condition. Record device, distance and clip-to-answer time. Public submissions without a verified source ID are useful demo traffic and cannot establish test accuracy.

For a public demo, a Vercel or Netlify HTTPS page could record ten seconds after the listener grants microphone access and send the clip to a separately hosted inference API. That service would keep MuQ, the shared projection head and the FMA index loaded, then return a title and artist or an unknown result. Display the catalogue name so listeners know which recordings are searchable. Measure upload plus server response separately from warm GPU lookup. This is a proposed deployment and recording protocol. [Microphone API](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) · [Netlify deployment](https://docs.netlify.com/deploy/create-deploys/) · [Vercel deployment](https://vercel.com/docs/deployments)
