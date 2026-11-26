"""Small hosted catalogue using the spectral peak extractor of PR 9 and fixed-speed offset voting."""
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage, signal
from scipy.fft import rfft

HERE = Path(__file__).resolve().parent


def peaks(samples):
    """Select up to six spectral landmarks per 256 ms, as in landmark_verify.py."""
    wave = signal.resample_poly(samples, 1, 3)
    frames = np.lib.stride_tricks.sliding_window_view(wave, 1024)[::128]
    mag = abs(rfft(frames * np.hanning(1024).astype("f4"), axis=1))
    log = np.log(np.maximum(mag, 1e-7))
    contrast = log - ndimage.uniform_filter1d(log, 23, axis=1)
    mask = (log == ndimage.maximum_filter(log, size=(9, 9))) & (contrast > .65)
    mask &= mag > max(1e-7, mag.max() * .001)
    mask[:, :12] = False
    mask[:, 495:] = False
    times, bins = np.where(mask)
    if not len(times):
        return np.empty((0, 2), dtype="f4")
    left, mid, right = log[times, bins-1], log[times, bins], log[times, bins+1]
    fraction = np.clip(.5*(left-right)/np.minimum(left-2*mid+right, -1e-7), -.5, .5)
    freq = 48*np.log2((bins+fraction)*8000/1024/64)
    strength = contrast[times, bins] + .36*mid
    order = np.lexsort((-strength,times//16))
    groups = times[order]//16
    starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(groups)!=0], np.arange(len(groups)), 0))
    take = order[np.arange(len(groups))-starts < 6]
    return np.column_stack((freq[take],times[take])).astype("f4")


@lru_cache(maxsize=1)
def catalogue():
    """Load the public demo catalogue's peak coordinates and track labels."""
    with np.load(HERE/"landmarks.npz") as z:
        return list(zip(z["ids"], z["titles"], z["artists"], np.split(z["features"],z["ends"][:-1]), strict=True))


def alignment(query, reference):
    """Measure support for the best time offset at the original playback rate."""
    if not len(query) or not len(reference):
        return 0., None
    qi, ri = np.where(np.abs(query[:,0,None]-reference[:,0]) <= .85)
    if not len(qi):
        return 0., None
    offsets = np.rint((reference[ri,1]-query[qi,1])*.5).astype(int)
    votes = np.bincount(offsets-offsets.min())
    smoothed = np.convolve(votes, np.ones(3))
    best = int(smoothed.argmax())
    return float(smoothed[best]/len(query)), float((best - 1 + offsets.min()) * .032)


def identify(payload):
    """Recognise an excerpt against the CC0 subset or abstain."""
    q = peaks(np.frombuffer(payload, dtype="<i2").astype("f4")/32768)
    t = perf_counter()
    hits = []
    for tid, title, artist, reference in catalogue():
        score, start = alignment(q, reference)
        hits.append((score, int(tid), str(title), str(artist), start))
    hits.sort(reverse=True)
    score, tid, title, artist, start = hits[0]
    gap = score-hits[1][0]
    return dict(accepted=len(q)>=24 and score>=.32 and gap>=.12, title=title, artist=artist,
                track_id=tid, score=score, gap=gap, lookup_ms=(perf_counter()-t)*1000,
                trace=dict(kind="landmarks", start=start, peaks=len(q)),
                engine="Spectral landmarks · 24-track demo")

