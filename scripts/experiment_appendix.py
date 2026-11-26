r"""Script version of the former notebooks/experiment_appendix.ipynb (the notebook was replaced by this script).

Run from anywhere inside the repository: ``uv run python scripts/experiment_appendix.py``.

# Supplementary experiments in audio identification

Runs on a CPU. The script reads the [bundled measurements and source revisions](../reports/appendix/SOURCES.md). It recomputes decisions and timing summaries where individual observations are available. The projection comparison, IVF sweep and exploratory margin study use published aggregate measurements.

**Units.** A known clip has a searchable source. A false accept means an absent clip received a title. A correct accept requires both the correct identity and a score above the selected cutoffs. A 1% target limits false accepts during development. The tables report what happened on held-out clips. The Medium experiments score track IDs. The last two experiments score recording groups independently of archive ID.
"""


# %%
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

data = next(
    p / "reports/appendix" for p in (Path.cwd(), *Path.cwd().parents) if (p / "reports/appendix/sources.csv").is_file()
)
sources = pd.read_csv(data / "sources.csv")
for source in sources.itertuples():
    file = data / Path(source.source_path).name
    content = file.read_bytes()
    if file.suffix in {".csv", ".json"}:
        content = content.replace(b"\r\n", b"\n")  # Git's canonical text on Windows
    assert hashlib.sha256(content).hexdigest() == source.sha256, file.name


def report(name):
    """Read the archived result for a study."""
    return json.loads((data / name).read_text(encoding="utf-8"))


def scores(name):
    """Read candidate-level scores and identity labels."""
    with np.load(data / name, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive}


print(f"Verified {len(sources)} measurement files from {sources.pr.nunique()} studies.")

# %% [markdown]
# ## 1. Can the system decline an absent Medium song? (PRs 2, 5 and 6)
#
# PR 2 selected separate maximum-score cutoffs for exact and LSH on development songs. Each test group contains 512 songs with two distortions per song. PR 5 reused this cohort to ask whether approximate search preserves the two highest-scoring distinct track IDs. Its exploratory gap-only rule thresholds their score difference. PR 6 then tested a joint score-and-gap rule on fresh songs that excluded the PR 2 cohort. Each Medium study removed every reference window belonging to its designated absent songs.


# %%
def outcome(known, correct, accept):
    """Count listener outcomes from aligned query-level decisions."""
    return {
        "known n": int(known.sum()),
        "absent n": int((~known).sum()),
        "correct accepted": int((known & correct & accept).sum()),
        "wrong known": int((known & ~correct & accept).sum()),
        "false absent": int((~known & accept).sum()),
    }


medium = pd.read_csv(data / "unknown-song-scores.csv.gz", dtype={"exact_score": "float32", "lsh_score": "float32"})
test = medium.query("phase == 'test'")
baseline = pd.DataFrame(report("unknown-songs.json")["evaluation"])
rows = {}
for method in ("exact", "lsh"):
    absent_dev = medium.loc[(medium.phase == "development") & ~medium.known, f"{method}_score"]
    cutoff = np.nextafter(np.sort(absent_dev)[::-1][len(absent_dev) // 100], np.float32(np.inf))
    expected = baseline[(baseline.family == "all") & (baseline.method == method)].set_index("metric")
    assert np.all(expected.threshold == float(cutoff))
    row = outcome(test.known, test[f"{method}_track"] == test.track_id, test[f"{method}_score"] >= cutoff)
    assert row["correct accepted"] == expected.loc["correct known acceptance", "events"]
    assert row["false absent"] == expected.loc["unknown false acceptance", "events"]
    rows[f"PR 2 · {method} score"] = row

fresh = pd.read_csv(data / "runner-up-margin-scores.csv.gz")
assert set(medium.track_id).isdisjoint(fresh.track_id)
test = fresh.query("phase == 'test'")
for result in report("runner-up-margin.json")["test"]:
    accept = test.exact_score >= result["score_threshold"]
    if result["gap_threshold"] is not None:
        accept &= test.score_margin >= result["gap_threshold"]
    row = outcome(test.known, test.exact_track == test.track_id, accept)
    assert row["correct accepted"] == result["known_correct_accepted"]
    assert row["false absent"] == result["unknown_false_accepts"]
    rows[f"PR 6 · {result['rule_name']}"] = row
print(pd.DataFrame.from_dict(rows, orient="index"))

margin = report("cluster-margins.json")
print(
    f"PR 5 · exact and IVF agreed on {margin['same_top']}/{margin['test_queries']} first IDs"
    f" and {margin['same_ordered_top_two']}/{margin['test_queries']} ordered pairs."
)
print(
    pd.DataFrame(margin["rows"])[
        ["retrieval", "calibration", "signal", "threshold", "correct_known", "known_n", "false_unknown", "unknown_n"]
    ].round(6)
)

# %% [markdown]
# ## 2. Does agreement across time identify more clips? (PR 3)
#
# Each of 128 fresh Medium songs supplies three 18-second digital controls. The strongest match and temporal voting use the same three overlapping windows. The bootstrap samples songs with all three conditions together. The RTX 4090 timing starts with decoded audio and includes encoding, retrieval and selection. A separate 12-track phone-capture protocol was planned. Physical recordings are still unmeasured.

# %%
clips = pd.read_csv(data / "temporal-controls.csv")
assert not clips.duplicated(["track_id", "condition"]).any() and clips.track_id.nunique() == 128
hits = clips[["strongest", "temporal"]].eq(clips.track_id, axis="index")
print(hits.groupby(clips.condition).sum().assign(clips=clips.groupby("condition").size()))
delta = hits.temporal.astype(int) - hits.strongest.astype(int)
song_delta = delta.groupby(clips.track_id).mean().to_numpy()
draws = np.random.default_rng(422).choice(song_delta, size=(5000, len(song_delta))).mean(axis=1)
lo, hi = 100 * np.quantile(draws, [0.025, 0.975])
print(
    f"{hits.strongest.sum()} → {hits.temporal.sum()} correct / {len(clips)} clips; "
    f"{(delta == 1).sum()} gained, {(delta == -1).sum()} lost. "
    f"Change {100*delta.mean():+.2f} pp (95% paired interval {lo:+.2f} to {hi:+.2f})."
)
print(pd.DataFrame(report("temporal-controls-timing.json")).T.round(2))

# %% [markdown]
# ## 3. How much does the Medium GPU index search? (PR 4)
#
# The GPU spherical IVF study used 110,320 reference windows. It chose the fastest setting within 0.5 percentage points of exact accuracy on 2,542 development queries, then evaluated disjoint held-out songs. The tables replay GPU-resident lookup timings from this study. They exclude encoding, transfer and index building. PR 7 measured its own timing series for catalogue growth. The merged CPU spherical k-means implementation in PR 11 uses a separate code path.

# %%
ivf = report("ivf-medium.json")
sweep = pd.DataFrame(ivf["development_sweep"])
floor = ivf["baselines"]["development_exact_accuracy"] - ivf["protocol"]["development_accuracy_tolerance"]
chosen = sweep[sweep.dev_accuracy >= floor].sort_values("dev_ivf_median_ms").iloc[0]
assert (chosen.clusters, chosen.probes) == (ivf["selected_ivf"]["clusters"], ivf["selected_ivf"]["probes"])
print(
    f"{chosen.clusters:.0f} clusters, {chosen.probes:.0f} probes, {100*chosen.visit_fraction:.2f}% of windows visited."
)
closed = pd.DataFrame(ivf["baselines"]["heldout_closed"])
print((100 * closed.pivot(index=["query_set", "queries"], columns="method", values="track_accuracy")).round(3))
cost = pd.DataFrame(ivf["single_query_timing"]).T
cost["batch queries/s"] = pd.DataFrame(ivf["resident_batch_throughput"]).T.median_queries_per_second.round()
print(cost.round(4))
memory = ivf["index_footprint"]
print(
    f"Additional persistent index bytes: IVF {memory['ivf_bytes']:,}, LSH {memory['frozen_lsh_bytes']:,}; "
    f"shared reference bytes {memory['shared_reference_bytes']:,}."
)

# %% [markdown]
# **Projection ablation (PR 5).** Exact cosine searched identical windows and query clips for raw MuQ (1,024 dimensions) and the trained 256-dimensional projection. The paired song-bootstrap intervals below quantify the effect of the full change in representation.

# %%
projection = pd.DataFrame(report("projection-ablation.json")["families"]).T
projection.index = projection.index.map({"queries": "Standard", "queries_tiktok": "TikTok-like"})
print(projection[["queries", "songs", "raw_correct", "trained_correct", "gained", "lost"]].astype(int))
for family, row in projection.iterrows():
    lo, hi = row.paired_song_bootstrap95_pp
    print(f"{family}: {row.delta_pp:+.2f} pp (95% paired interval {lo:+.2f} to {hi:+.2f}).")

# %% [markdown]
# ## 4. What changes when the catalogue grows? (PR 8)
#
# The same PR 6 songs and queries now face 459,911 Large windows from 104,681 tracks after the absent songs were removed. The first LSH rule uses the frozen Medium score and gap cutoffs. The second rule was selected on Large development tracks under the same ten-false-accept budget. Its optimum used a score cutoff alone. Retained, lost and gained below refer to the individual known clips correctly accepted by each rule.

# %%
growth = pd.read_csv(data / "large-recalibration-scores.csv.gz", converters={"top_score": float, "gap": float})
keys = ["track_id", "known", "phase", "family", "distortion"]
lsh = growth.query("method == 'lsh'").reset_index(drop=True)
pd.testing.assert_frame_equal(fresh[keys].reset_index(drop=True), lsh[keys])
test = lsh.query("phase == 'test'")
rules = report("large-recalibration.json")["methods"]["lsh"]["test"]
rows, accepted_correct = {}, {}
for name, policy in rules.items():
    rule = policy["score_and_gap"]
    accept = test.top_score >= rule["score_threshold"]
    if rule["gap_threshold"] is not None:
        accept &= test.gap >= rule["gap_threshold"]
    correct = test.top_track == test.track_id
    rows[name] = outcome(test.known, correct, accept)
    assert rows[name]["correct accepted"] == rule["known_correct_accepted"]
    assert rows[name]["false absent"] == rule["unknown_false_accepts"]
    accepted_correct[name] = test.known & correct & accept
before, after = accepted_correct["frozen"], accepted_correct["large_development_recalibrated"]
print(pd.DataFrame.from_dict(rows, orient="index"))
print(f"Retained {(before & after).sum()}, lost {(before & ~after).sum()}, gained {(~before & after).sum()}.")

# %% [markdown]
# ## 5. Does acoustic evidence change the answer? (PRs 9 and 10)
#
# Both experiments use the same group-disjoint Large cohort. Each held-out cell contains 256 recording groups, each supplying two ten-second digital queries. A waveform check linked excerpts with matching passages before splitting. The searchable catalogue excluded all IDs in absent groups and retained 105,180 tracks. Group matches can credit another ID containing the verified passage.
#
# PR 9 reranks the five neural LSH candidates using spectral-peak alignment. PR 10 compares widening the neural shortlist to ten with adding five independently nominated acoustic candidates. All three PR 10 systems use the same verifier. The closed-set and 1%-target decision rules were fitted separately on development groups. Here we apply those frozen rules to the saved per-candidate scores.

# %%
landmark = scores("landmark-scores.npz")
acoustic = scores("acoustic-nominations.npz")
for key in ("phase", "known", "group_id", "track_id", "family", "distortion"):
    np.testing.assert_array_equal(landmark[key], acoustic[key])
held = acoustic["phase"] == "test"
known = acoustic["known"][held]
truth = acoustic["group_id"][held]
assert known.sum() == (~known).sum() == 512
assert set(truth).isdisjoint(acoustic["group_id"][~held])


def rank(groups, evidence, rule):
    """Rank held-out candidates and apply the published winner and runner-up cutoffs."""
    values = np.where(groups[held] >= 0, evidence[held], -1e6)
    order = np.argsort(-values, axis=1, kind="stable")
    top = np.take_along_axis(values, order[:, :2], axis=1).astype("float64")
    accept = (top[:, 0] >= rule["score"]) & (top[:, 0] - top[:, 1] >= rule["gap"])
    ranked = np.take_along_axis(groups[held], order, axis=1)
    return ranked == truth[:, None], accept, order[:, 0]


rows = {}
for run in report("landmark-results.json")["runs"]:
    if run["target"] not in (None, 0.01):
        continue
    rule = run["rule"]
    weight = rule["landmark_weight"]
    evidence = landmark["landmarks"] if weight == "landmarks" else landmark["scores"] + weight * landmark["landmarks"]
    match, accept, best = rank(landmark["candidate_groups"], evidence, rule)
    row = rows.setdefault(run["method"], {})
    if run["target"] is None:
        row["first"] = int((known & match[:, 0]).sum())
        assert row["first"] == run["test"]["closed_top1"]
    else:
        row.update(outcome(known, match[:, 0], accept))
        strict = landmark["candidates"][held][np.arange(len(best)), best] == landmark["track_id"][held]
        row["strict ID accepted"] = int((known & strict & accept).sum())
        assert row["strict ID accepted"] == run["strict_id_correct_accepted"]
        assert row["correct accepted"] == run["test"]["correct"]
        assert row["false absent"] == run["test"]["false_accepts"]
print(pd.DataFrame.from_dict(rows, orient="index"))

# %%
methods = report("acoustic-results.json")["methods"]
rows, answered = {}, {}
for method, runs in methods.items():
    tag, width = ("lsh10", 5) if method == "lsh5" else (method, 10)
    groups, cosine, aligned = (acoustic[f"{tag}_{key}"][:, :width] for key in ("groups", "cos", "lm"))
    closed = runs["closed"]["rule"]
    match, _, _ = rank(groups, cosine + closed["weight"] * aligned, closed)
    rule = runs["0.01"]["rule"]
    open_match, accept, _ = rank(groups, cosine + rule["weight"] * aligned, rule)
    row = outcome(known, open_match[:, 0], accept)
    row["in pool"] = int((known & match.any(axis=1)).sum())
    row["final five"] = int((known & match[:, :5].any(axis=1)).sum())
    row["first"] = int((known & match[:, 0]).sum())
    assert [row[k] for k in ("in pool", "final five", "first")] == [
        runs["closed"]["test"][k] for k in ("pool", "top5", "top1")
    ]
    assert [row[k] for k in ("correct accepted", "wrong known", "false absent")] == [
        runs["0.01"]["test"][k] for k in ("correct", "wrong_known", "fa")
    ]
    rows[method] = row
    answered[method] = known & open_match[:, 0] & accept
print(
    pd.DataFrame.from_dict(rows, orient="index")[
        ["in pool", "final five", "first", "correct accepted", "wrong known", "false absent", "known n", "absent n"]
    ]
)

neural = acoustic["lsh10_groups"][held] == truth[:, None]
nominated = (acoustic["acoustic_groups"][held] == truth[:, None]).any(axis=1)
miss = known & ~neural[:, :5].any(axis=1)
before, after = answered["lsh5"], answered["dual"]
print(
    f"Among {miss.sum()} prior shortlist misses from {len(np.unique(truth[miss]))} groups, "
    f"acoustic nomination recovered {(miss & nominated).sum()}; LSH 10 recovered {(miss & neural.any(axis=1)).sum()}."
)
print(
    f"The dual system correctly accepted {(miss & after).sum()} of those misses. Across all known clips, "
    f"it gained {(after & ~before).sum()} accepted answers and lost {(before & ~after).sum()}."
)
paired = report("acoustic-results.json")["paired"]["0.01"]
lo, hi = paired["correct_delta_95_pp"]
print(
    f"Change in correct acceptance {100*(after.sum()-before.sum())/known.sum():+.2f} pp "
    f"(archived paired recording-group 95% interval {lo:+.2f} to {hi:+.2f})."
)

# %% [markdown]
# ### Decoded-clip latency
#
# These paired warm RTX 4090 trials start with a decoded ten-second waveform and stop when the system returns its decision. They include feature extraction, lookup and verification where used. Capture, file decoding and offline reference indexing are outside the timer. PR 9 and PR 10 timed their own implementations. Use within-study comparisons. The 22 previous misses were examined retrospectively on the same clips.

# %%
timings = {}
for study, filename, query, columns in (
    ("PR 9", "landmark-latency.csv", "query_index", ["cosine_ms", "fusion_ms"]),
    ("PR 10", "acoustic-latency.csv", "query", ["lsh5", "lsh10", "dual", "dual_serial"]),
):
    trials = pd.read_csv(data / filename).set_index(query)
    assert trials.index.is_unique and len(trials) == 64
    for method in columns:
        timings[(study, method)] = [len(trials), trials[method].median(), trials[method].quantile(0.95)]
timing_table = pd.DataFrame.from_dict(timings, orient="index", columns=["paired clips", "median ms", "p95 ms"])
print(timing_table.round(2))

# %% [markdown]
# ## 6. Which genres contribute to the Medium result?
#
# We join the searchable Medium IDs from the released reference fingerprints to the manifest's top-level genre. The bundled two-column extract contains each searchable ID once. The query rows below come from PR 2's held-out, known-song test and use its development-frozen LSH score cutoff. Each song contributes two clips. Count songs as well as clips when judging support for a genre.
#
# The queries are generated from known FMA audio so that the source ID, distortion and catalogue membership are controlled. Standard clips test routine signal changes; TikTok-like clips add effects such as speed change, voice-over and compression. They answer whether the method survives those transformations. A phone adds a speaker, room, microphone and device processing. Results on other researchers' captured audio motivate that paired test; different catalogues and query conditions prevent a direct leaderboard comparison. See [Araz et al., Table 6](https://arxiv.org/html/2506.22661v1) and the [original genre metadata](https://github.com/mdeff/fma).

# %%
genre_path = data / "medium-track-genres.csv.gz"
assert (
    hashlib.sha256(genre_path.read_bytes()).hexdigest()
    == "d704cfe7949b00fca35803296413f0f8a5d35c826e566555734d28afc8025f81"
)
catalogue = pd.read_csv(genre_path)
assert catalogue.track_id.is_unique
known = medium.loc[(medium.phase == "test") & medium.known].merge(
    catalogue, on="track_id", how="left", validate="many_to_one"
)
assert known.genre.notna().all() and not known.duplicated(["track_id", "family"]).any()
rule = baseline.query("method == 'lsh' and family == 'all' and metric == 'correct known acceptance'").iloc[0]
known["correct"] = known.track_id.eq(known.lsh_track) & known.lsh_score.ge(rule.threshold)
coverage = (
    catalogue.genre.value_counts()
    .rename("catalogue_tracks")
    .to_frame()
    .join(
        known.groupby("genre").agg(
            known_test_tracks=("track_id", "nunique"),
            known_queries=("track_id", "size"),
            correct_accepted=("correct", "sum"),
        )
    )
    .fillna(0)
    .astype(int)
)
assert tuple(coverage.sum()) == (24970, 512, int(rule.queries), int(rule.events))
print(coverage)

# %% [markdown]
# Rock and Electronic supply 13,407 of 24,970 searchable tracks and 309 of the 512 tested songs. Blues and Spoken have two tested songs each. These rows describe coverage; a genre ranking would need more independent songs per stratum.
