"""Portable replay of the exact shortlist scored for one Medium fingerprint."""
import base64

import numpy as np


def pack_trace(query, projections, boxes, rows, scores, count, best_row, runner_row):
    # Keep the real GPU ordering for the winners. Other scores are shown to three decimals.
    packed = np.full(count, -32768, dtype="<i2")
    packed[rows] = np.rint(np.clip(scores, -1, 1) * 32767).astype("<i2")
    return dict(
        kind="lsh",
        library="medium-v1",
        vector=query.tolist(),
        projections=projections.tolist(),
        boxes=boxes,
        candidates=len(rows),
        bestRow=int(best_row),
        runnerRow=int(runner_row) if runner_row is not None else None,
        scores=base64.b64encode(packed.tobytes()).decode("ascii"),
    )
