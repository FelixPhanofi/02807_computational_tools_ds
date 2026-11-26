"""Rebuild the public query explorer from the two reviewed evidence revisions and v0.3."""
import io
import json
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "explorer/public"
APPENDIX = "5c83e09"
REPORT = "98aac223ea8ea0d32282cd43d9e5de95a22efda0"


def archived(commit, path):
    """Read a pinned Git blob without switching branches."""
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def main():
    """Export held-out decisions, genre support and same-machine retrieval measurements."""
    meta = pd.read_csv(ROOT / "data/medium/data/processed/manifest.csv").fillna("").set_index("track_id")
    def appendix(name):
        return archived(APPENDIX, "reports/appendix/" + name)

    scores = pd.read_csv(io.BytesIO(appendix("runner-up-margin-scores.csv.gz")), compression="gzip")
    results = json.loads(appendix("runner-up-margin.json"))
    rule = results["threshold_selection"]["score_and_margin"]
    queries = []
    for row in scores[scores.phase == "test"].itertuples():
        src, pred, runner = (meta.loc[k] for k in (row.track_id, row.exact_track, row.runner_up_track))
        accepted = row.exact_score >= rule["score_threshold"] and row.score_margin >= rule["gap_threshold"]
        queries.append(dict(id=f"{row.track_id:06}-{row.family}", track=str(row.track_id), expected=str(row.track_id),
                            title=src.title, artist=src.artist, genre=src.genre_top, family=row.family,
                            condition=row.distortion, predicted=str(row.exact_track), predicted_title=pred.title,
                            predicted_artist=pred.artist, runner_up=str(row.runner_up_track), runner_up_title=runner.title,
                            known=row.known, score=row.exact_score, gap=row.score_margin, accepted=accepted,
                            correct=accepted and row.known and row.track_id == row.exact_track))
    expected = next(r for r in results["test"] if r["rule_name"] == "score_and_margin")
    assert sum(q["correct"] for q in queries) == expected["known_correct_accepted"]
    assert sum(q["accepted"] and not q["known"] for q in queries) == expected["unknown_false_accepts"]
    genres = pd.read_csv(io.BytesIO(appendix("medium-track-genres.csv.gz")), compression="gzip")
    original = pd.read_csv(io.BytesIO(appendix("unknown-song-scores.csv.gz")), compression="gzip")
    known = original[original.known & (original.phase == "test")].merge(genres, validate="many_to_one")
    cutoff = next(r for r in json.loads(appendix("unknown-songs.json"))["evaluation"]
                  if r["method"] == "lsh" and r["family"] == "all" and r["metric"] == "correct known acceptance")
    known["correct"] = known.track_id.eq(known.lsh_track) & known.lsh_score.ge(cutoff["threshold"])
    coverage = genres.groupby("genre").size().rename("tracks").to_frame().join(
        known.groupby("genre").agg(songs=("track_id","nunique"),clips=("track_id","size"),correct=("correct","sum"))
    ).fillna(0).astype(int).reset_index()
    assert coverage.correct.sum() == cutoff["events"]
    rerun = json.loads(archived(REPORT, "reports/report-evidence/retrieval-rerun.json"))
    methods = [dict(catalogue=size, family=family, method=method, correct=count, total=rerun["queries"][family],
                    ms=rerun[size]["median_lookup_ms"][method], qps=rerun[size]["batch128_queries_per_s"][method])
               for size in ("medium","large") for family, counts in rerun[size]["families"].items()
               for method,count in counts.items()]
    (OUT/"evidence.json").write_text(json.dumps(dict(queries=queries,methods=methods,genres=coverage.to_dict("records")),
                                             separators=(",",":"),allow_nan=False),encoding="utf8")


if __name__ == "__main__":
    main()

