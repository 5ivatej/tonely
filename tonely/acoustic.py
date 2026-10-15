"""Model-free acoustic/prosodic affect estimation.

Works on any audio (speech, music, ambience) and is always available, so it
anchors the fusion even when no neural model is installed.

Arousal is well-predicted by low-level descriptors (loudness, spectral
brightness, spectral flux, event density, tempo, pitch level/variability) —
see Juslin & Laukka (2003), Eyben et al. (2016, eGeMAPS). Valence is much
harder from acoustics alone; we use tonal mode/key clarity, consonance
(spectral flatness of the harmonic part), and tempo, and report a
correspondingly lower confidence so better-informed models dominate fusion.
"""

from __future__ import annotations

import librosa
import numpy as np

from .audio import Audio
from .tracks import Grid, Track

HOP = 512
N_FFT = 2048

# Krumhansl–Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _profiles() -> tuple[np.ndarray, np.ndarray]:
    def norm(p):
        p = p - p.mean()
        return p / np.linalg.norm(p)

    major = np.stack([np.roll(norm(_MAJOR), k) for k in range(12)])
    minor = np.stack([np.roll(norm(_MINOR), k) for k in range(12)])
    return major, minor


_MAJ, _MIN = _profiles()
_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _robust_z(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    ref = x[mask] if mask is not None and mask.any() else x
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros_like(x)
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826 + 1e-9
    z = (x - med) / mad
    return np.clip(np.nan_to_num(z), -4, 4)


def frame_features(audio: Audio) -> dict[str, np.ndarray]:
    y, sr = audio.samples, audio.sr
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    H, P = librosa.decompose.hpss(S, margin=1.0)
    power = S**2

    rms = librosa.feature.rms(S=S, frame_length=N_FFT)[0]
    feats = {
        "rms_db": librosa.amplitude_to_db(rms, ref=1.0, top_db=None),
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],
        "flatness": librosa.feature.spectral_flatness(S=H + 1e-10)[0],
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0],
        "flux": librosa.onset.onset_strength(S=librosa.power_to_db(power), sr=sr),
        "harm": (H**2).sum(0),
        "perc": (P**2).sum(0),
    }
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    feats["hf_ratio"] = power[freqs >= 1000].sum(0) / (power.sum(0) + 1e-10)
    feats["chroma"] = librosa.feature.chroma_stft(S=H**2, sr=sr, hop_length=HOP)

    # Pitch: YIN is ~50x faster than pYIN; voicing is estimated from harmonicity
    # + energy, which is plenty for window-level statistics.
    f0 = librosa.yin(y, fmin=65, fmax=800, sr=sr, frame_length=N_FFT, hop_length=HOP)
    n = min(len(f0), S.shape[1])
    harmonicity = feats["harm"][:n] / (feats["harm"][:n] + feats["perc"][:n] + 1e-10)
    loud_gate = feats["rms_db"][:n] > np.percentile(feats["rms_db"], 30)
    voiced = (harmonicity > 0.55) & loud_gate & (feats["flatness"][:n] < 0.2)
    feats["f0"] = np.where(voiced, f0[:n], np.nan)

    oenv = feats["flux"]
    feats["tempo"] = librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=HOP, aggregate=None)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=HOP, units="frames")
    feats["onset_mask"] = np.zeros(len(oenv))
    feats["onset_mask"][onsets[onsets < len(oenv)]] = 1.0
    feats["frame_times"] = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)
    return feats


def _window_stats(feats: dict[str, np.ndarray], grid: Grid) -> dict[str, np.ndarray]:
    ft = feats["frame_times"]
    n = len(grid)
    out = {k: np.full(n, np.nan) for k in (
        "loud", "loud_var", "centroid", "rolloff", "flatness", "zcr", "flux", "hf_ratio",
        "f0_mean", "f0_std", "voiced", "tempo", "onset_rate", "perc_ratio", "mode", "key_clarity",
    )}
    keys: list[str | None] = [None] * n
    for i in range(n):
        a, b = grid.bounds(i)
        lo, hi = np.searchsorted(ft, [a, b])
        hi = max(hi, lo + 1)
        sl = slice(lo, hi)
        db = feats["rms_db"][sl]
        out["loud"][i] = np.percentile(db, 75)
        out["loud_var"][i] = np.std(db)
        for k in ("centroid", "rolloff", "flatness", "zcr", "flux", "hf_ratio"):
            out[k][i] = np.mean(feats[k][sl])
        f0 = feats["f0"][sl]
        f0 = f0[np.isfinite(f0)]
        out["voiced"][i] = len(f0) / (hi - lo)
        if len(f0) >= 5:
            semis = 12 * np.log2(f0 / 100.0)
            out["f0_mean"][i] = np.median(semis)
            out["f0_std"][i] = np.std(semis)
        t = feats["tempo"][sl.start:min(sl.stop, len(feats["tempo"]))]
        if t.size:
            out["tempo"][i] = np.median(t)
        out["onset_rate"][i] = feats["onset_mask"][sl].sum() / max(b - a, 1e-3)
        h, p = feats["harm"][sl].sum(), feats["perc"][sl].sum()
        out["perc_ratio"][i] = p / (h + p + 1e-10)

        chroma = feats["chroma"][:, sl].mean(1)
        if chroma.sum() > 1e-6:
            c = chroma - chroma.mean()
            c /= np.linalg.norm(c) + 1e-9
            maj, mnr = _MAJ @ c, _MIN @ c
            out["mode"][i] = maj.max() - mnr.max()
            out["key_clarity"][i] = max(maj.max(), mnr.max())
            keys[i] = (_KEYS[int(maj.argmax())] + " major") if maj.max() >= mnr.max() else (_KEYS[int(mnr.argmax())] + " minor")
    out["_keys"] = keys  # type: ignore[assignment]
    return out


def analyze(audio: Audio, grid: Grid) -> tuple[Track, dict]:
    feats = frame_features(audio)
    w = _window_stats(feats, grid)
    n = len(grid)

    # Silence gate relative to the file's loud passages.
    loud = w["loud"]
    silence_thresh = max(-60.0, np.percentile(loud, 95) - 35.0)
    silent = loud < silence_thresh
    active = ~silent

    z = {k: _robust_z(w[k], active) for k in (
        "loud", "loud_var", "centroid", "flatness", "flux", "hf_ratio", "onset_rate", "tempo", "perc_ratio",
    )}
    f0_ok = np.isfinite(w["f0_mean"])
    z["f0_mean"] = np.where(f0_ok, _robust_z(w["f0_mean"], f0_ok & active), 0.0)
    z["f0_std"] = np.where(f0_ok, _robust_z(w["f0_std"], f0_ok & active), 0.0)

    # --- arousal: relative (within-file) + absolute (loudness/brightness anchor)
    rel_arousal = (
        0.26 * z["loud"] + 0.16 * z["centroid"] + 0.14 * z["flux"] + 0.10 * z["onset_rate"]
        + 0.08 * z["tempo"] + 0.08 * z["hf_ratio"] + 0.10 * z["f0_mean"] + 0.08 * z["f0_std"]
    )
    abs_loud = np.clip((loud + 42.0) / 15.0, -1.5, 1.5)
    abs_bright = np.clip((w["centroid"] - 1800.0) / 900.0, -1.5, 1.5)
    arousal = np.tanh(0.55 * rel_arousal + 0.35 * abs_loud + 0.20 * abs_bright)

    # --- valence: tonal mode, consonance, tempo, pitch variability
    mode = np.nan_to_num(w["mode"]) * np.nan_to_num(w["key_clarity"])
    tempo_pref = -((np.nan_to_num(w["tempo"], nan=110) - 120.0) / 45.0) ** 2 + 0.5  # moderate-fast tempos read as positive
    rel_valence = (
        5.0 * mode
        - 0.22 * z["flatness"]
        - 0.12 * z["hf_ratio"] * (z["loud"] > 1.0)  # loud + harsh → negative
        + 0.10 * z["f0_std"]
        + 0.08 * z["centroid"]
        - 0.10 * z["loud_var"] * (z["loud"] > 1.5)
    )
    valence = np.tanh(0.8 * rel_valence + 0.25 * tempo_pref)

    # Confidence: shrink in silence; valence is intrinsically weaker.
    activity = np.clip((loud - silence_thresh) / 12.0, 0, 1)
    conf_va = 0.55 * activity
    track = Track(
        name="acoustic",
        label="Acoustic / prosody",
        valence=np.where(silent, 0.0, valence),
        arousal=np.where(silent, -0.8, arousal),
        conf_va=np.maximum(conf_va, 0.15),  # silence is itself informative (low arousal)
    )
    track.extra["valence_weight"] = 0.3  # relative trust of valence vs arousal

    descriptors = {
        "loudness_db": np.round(loud, 2).tolist(),
        "brightness_hz": np.round(w["centroid"], 1).tolist(),
        "tempo_bpm": np.round(np.nan_to_num(w["tempo"]), 1).tolist(),
        "pitch_semitones": [None if not np.isfinite(v) else round(float(v), 2) for v in w["f0_mean"]],
        "voiced_ratio": np.round(w["voiced"], 3).tolist(),
        "onset_rate": np.round(w["onset_rate"], 2).tolist(),
        "key": w["_keys"],
        "silent": silent.tolist(),
    }
    global_tempo = float(librosa.feature.tempo(onset_envelope=feats["flux"], sr=audio.sr, hop_length=HOP)[0])
    summary = {
        "tempo_bpm": round(global_tempo, 1),
        "key": _global_key(feats["chroma"]),
        "silence_ratio": round(float(silent.mean()), 3),
        "loudness_db": round(float(np.median(loud[active])) if active.any() else float(np.median(loud)), 2),
    }
    return track, {"descriptors": descriptors, "summary": summary, "silent": silent}


def _global_key(chroma: np.ndarray) -> str:
    c = chroma.mean(1)
    c = c - c.mean()
    c /= np.linalg.norm(c) + 1e-9
    maj, mnr = _MAJ @ c, _MIN @ c
    if maj.max() >= mnr.max():
        return _KEYS[int(maj.argmax())] + " major"
    return _KEYS[int(mnr.argmax())] + " minor"
