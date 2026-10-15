"""Voice activity detection and speech emotion recognition (SER).

Two complementary SER models run over sliding windows of detected speech:

* a **categorical** wav2vec2 classifier (neutral / happy / angry / sad,
  fine-tuned on IEMOCAP via SUPERB), and
* a **dimensional** wav2vec2-large regressor (arousal / dominance / valence,
  fine-tuned on MSP-Podcast, Wagner et al. 2023) — one of the strongest
  publicly released models for valence from voice.
"""

from __future__ import annotations

import logging

import numpy as np

from . import models
from .audio import Audio
from .emotions import IDX, empty_probs
from .tracks import Grid, Track

log = logging.getLogger("tonely")

SER_MAP = {
    "neu": "neutral", "neutral": "neutral",
    "hap": "joy", "happy": "joy", "joy": "joy",
    "ang": "anger", "angry": "anger", "anger": "anger",
    "sad": "sadness", "sadness": "sadness",
    "fea": "fear", "fear": "fear", "fearful": "fear",
    "dis": "disgust", "disgust": "disgust",
    "sur": "surprise", "surprise": "surprise", "surprised": "surprise",
    "calm": "calm",
}
# 4-class IEMOCAP models cannot name every emotion; each coarse label is
# spread over the fine emotions it subsumes (by circumplex proximity), and
# the fusion stage lets the other channels disambiguate.
SER_EXPAND = {
    "neutral": {"neutral": 0.7, "calm": 0.3},
    "joy": {"joy": 0.8, "surprise": 0.2},
    "anger": {"anger": 0.7, "disgust": 0.15, "fear": 0.15},
    "sadness": {"sadness": 0.85, "fear": 0.15},
}


def _expand(dist: dict[str, float], fine_labels: set[str]) -> np.ndarray:
    out = empty_probs()
    for label, p in dist.items():
        coarse = SER_MAP.get(label.lower())
        if coarse is None:
            continue
        # only expand when the model lacks the finer classes itself
        spread = {k: v for k, v in SER_EXPAND.get(coarse, {coarse: 1.0}).items() if k == coarse or k not in fine_labels}
        total = sum(spread.values())
        for target, w in spread.items():
            out[IDX[target]] += p * w / total
    s = out.sum()
    return out / s if s > 0 else out


# --------------------------------------------------------------------- VAD
def speech_intervals(audio: Audio) -> list[tuple[float, float]] | None:
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception as exc:  # noqa: BLE001
        log.warning("VAD unavailable: %s", exc)
        return None
    opts = VadOptions(min_silence_duration_ms=300, speech_pad_ms=150)
    ts = get_speech_timestamps(audio.samples, opts)
    return [(t["start"] / audio.sr, t["end"] / audio.sr) for t in ts]


def speech_fraction(intervals: list[tuple[float, float]], grid: Grid, win: float | None = None) -> np.ndarray:
    out = np.zeros(len(grid))
    for i in range(len(grid)):
        a, b = grid.bounds(i, win)
        covered = sum(max(0.0, min(b, e) - max(a, s)) for s, e in intervals)
        out[i] = covered / max(b - a, 1e-6)
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------ helpers
def _model_windows(audio: Audio, duration: float, hop: float, win: float):
    centers = np.arange(hop / 2, max(duration, hop / 2 + 1e-3), hop)
    half = int(win / 2 * audio.sr)
    for c in centers:
        mid = int(c * audio.sr)
        a, b = max(0, mid - half), min(len(audio.samples), mid + half)
        yield c, audio.samples[a:b]


def _batched(items, size):
    batch = []
    for it in items:
        batch.append(it)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _interp(grid: Grid, centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(grid.times, centers, values)
    return np.stack([np.interp(grid.times, centers, values[:, j]) for j in range(values.shape[1])], axis=1)


# --------------------------------------------------------- categorical SER
def ser_categorical(audio: Audio, grid: Grid, speech_frac_m, cfg, device, progress=None) -> Track | None:
    loaded = models.ser(cfg.ser_model, device)
    if loaded is None:
        return None
    fe, model, torch = loaded
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    fine = {SER_MAP[l.lower()] for l in id2label.values() if l.lower() in SER_MAP}

    centers, probs, peaks = [], [], []
    windows = list(_model_windows(audio, grid.duration, cfg.model_hop, cfg.win))
    total = len(windows)
    done = 0
    for batch in _batched(windows, 16):
        clips = [w for _, w in batch]
        inputs = fe(clips, sampling_rate=audio.sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            p = torch.softmax(model(**inputs).logits, dim=-1).float().cpu().numpy()
        for (c, _), row in zip(batch, p):
            centers.append(c)
            probs.append(_expand({id2label[j]: float(v) for j, v in enumerate(row)}, fine))
            peaks.append(float(row.max()))
        done += len(batch)
        if progress:
            progress(done / total)

    centers = np.array(centers)
    P = _interp(grid, centers, np.array(probs))
    P /= P.sum(1, keepdims=True) + 1e-9
    track = Track.empty("ser", "Voice emotion (wav2vec2)", len(grid))
    track.probs = P
    # Confidence follows how much of the window is speech and how peaked
    # the raw (pre-expansion) prediction is.
    k = len(id2label)
    peak = np.interp(grid.times, centers, np.array(peaks))
    sharp = np.clip((peak - 1 / k) / (1 - 1 / k), 0, 1)
    track.conf_cat = speech_frac_m * (0.45 + 0.4 * sharp)
    return track


# --------------------------------------------------------- dimensional SER
def ser_dimensional(audio: Audio, grid: Grid, speech_frac_m, cfg, device, progress=None) -> Track | None:
    loaded = models.ser_dim(cfg.ser_dim_model, device)
    if loaded is None:
        return None
    fe, model, torch = loaded

    centers, outs = [], []
    windows = list(_model_windows(audio, grid.duration, cfg.model_hop, cfg.win))
    total = len(windows)
    done = 0
    for batch in _batched(windows, 8):
        clips = [w for _, w in batch]
        # this checkpoint was trained on normalised, unpadded audio
        n = max(len(c) for c in clips)
        x = np.stack([np.pad(c, (0, n - len(c))) for c in clips])
        x = fe(list(x), sampling_rate=audio.sr, return_tensors="pt").input_values.to(model.device)
        with torch.no_grad():
            y = model(x).float().cpu().numpy()  # arousal, dominance, valence ≈ [0, 1]
        for (c, _), row in zip(batch, y):
            centers.append(c)
            outs.append(row)
        done += len(batch)
        if progress:
            progress(done / total)

    centers = np.array(centers)
    Y = _interp(grid, centers, np.array(outs))
    # Map [0,1] → [-1,1]. MSP-Podcast labels centre around ~0.5 with modest
    # spread, so stretch slightly to use the full circumplex.
    to_bipolar = lambda v: np.clip((v - 0.5) * 2.6, -1, 1)  # noqa: E731
    track = Track.empty("ser_dim", "Voice affect (wav2vec2-large)", len(grid), categorical=False)
    track.arousal = to_bipolar(Y[:, 0])
    track.dominance = to_bipolar(Y[:, 1])
    track.valence = to_bipolar(Y[:, 2])
    track.conf_va = 0.85 * speech_frac_m
    track.extra["valence_weight"] = 0.85
    return track
