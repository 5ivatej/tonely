"""Late fusion of all analyzer tracks into one coherent emotional timeline.

Dimensional (valence / arousal / dominance)
    A 1-D linear-Gaussian state-space model per dimension. Each analyzer is a
    noisy sensor whose observation variance is inversely proportional to its
    confidence; a Kalman filter plus Rauch–Tung–Striebel backward pass yields
    the posterior mean *and* uncertainty at every time step. Multiple sensors
    at the same step are absorbed sequentially, i.e. precision-weighted.

Categorical (8 emotions)
    Per-step logarithmic opinion pooling (a confidence-weighted product of
    experts) of every analyzer's distribution plus a circumplex-kernel
    likelihood from the fused valence/arousal. The pooled frame posteriors
    are then used as scaled emissions of a sticky hidden Markov model; the
    forward–backward algorithm gives smoothed per-step posteriors and Viterbi
    decoding gives clean emotion segments.

Story shape
    The fused valence trajectory is matched against the six canonical
    emotional arcs of Reagan et al. (2016).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .emotions import EMOTIONS, IDX, va_to_probs
from .tracks import Grid, Track

N_E = len(EMOTIONS)


# ------------------------------------------------------------ Kalman / RTS
def kalman_smooth(
    obs: list[tuple[np.ndarray, np.ndarray]],
    q: float,
    rho: float = 0.995,
    r0: float = 0.06,
    prior_var: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """obs: list of (values, weights) arrays per sensor; NaN/0-weight = missing."""
    n = len(obs[0][0])
    m_f, p_f = np.zeros(n), np.zeros(n)
    m_p, p_p = np.zeros(n), np.zeros(n)
    m, p = 0.0, prior_var
    for t in range(n):
        if t:
            m, p = rho * m, rho * rho * p + q
        m_p[t], p_p[t] = m, p
        for vals, w in obs:
            z, wt = vals[t], w[t]
            if not np.isfinite(z) or wt <= 1e-4:
                continue
            r = r0 / wt
            k = p / (p + r)
            m += k * (z - m)
            p *= 1 - k
        m_f[t], p_f[t] = m, p
    m_s, p_s = m_f.copy(), p_f.copy()
    for t in range(n - 2, -1, -1):
        c = p_f[t] * rho / p_p[t + 1]
        m_s[t] = m_f[t] + c * (m_s[t + 1] - m_p[t + 1])
        p_s[t] = p_f[t] + c * c * (p_s[t + 1] - p_p[t + 1])
    return np.clip(m_s, -1, 1), np.sqrt(np.maximum(p_s, 0))


# ------------------------------------------------------------------- HMM
def _transition(stay: float) -> np.ndarray:
    T = np.full((N_E, N_E), (1 - stay) / (N_E - 1))
    np.fill_diagonal(T, stay)
    # neighbouring emotions on the circumplex are easier to move between
    from .emotions import PROTO

    d = np.linalg.norm(PROTO[:, None] - PROTO[None], axis=-1)
    ease = np.exp(-d)
    np.fill_diagonal(ease, 0)
    off = (1 - stay) * ease / ease.sum(1, keepdims=True)
    np.fill_diagonal(off, stay)
    return 0.5 * T + 0.5 * off


def forward_backward(emis: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = len(emis)
    alpha = np.zeros((n, N_E))
    a = np.full(N_E, 1 / N_E) * emis[0]
    alpha[0] = a / a.sum()
    for t in range(1, n):
        a = (alpha[t - 1] @ T) * emis[t]
        alpha[t] = a / a.sum()
    beta = np.ones(N_E)
    post = np.zeros_like(alpha)
    post[-1] = alpha[-1]
    for t in range(n - 2, -1, -1):
        beta = T @ (emis[t + 1] * beta)
        beta /= beta.sum()
        g = alpha[t] * beta
        post[t] = g / g.sum()
    return post


def viterbi(emis: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = len(emis)
    logT = np.log(T)
    le = np.log(emis + 1e-12)
    delta = np.log(1 / N_E) + le[0]
    back = np.zeros((n, N_E), dtype=np.int64)
    for t in range(1, n):
        cand = delta[:, None] + logT
        back[t] = cand.argmax(0)
        delta = cand.max(0) + le[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ------------------------------------------------------------ story arcs
def _arc_templates(n: int) -> dict[str, np.ndarray]:
    x = np.linspace(0, 1, n)
    zigzag = np.interp(x, [0, 1 / 3, 2 / 3, 1], [0, 1, 0, 1])
    return {
        "Rags to riches (rise)": x,
        "Tragedy (fall)": -x,
        "Man in a hole (fall → rise)": -np.sin(np.pi * x),
        "Icarus (rise → fall)": np.sin(np.pi * x),
        "Cinderella (rise → fall → rise)": zigzag,
        "Oedipus (fall → rise → fall)": -zigzag,
    }


def story_arc(valence: np.ndarray) -> dict:
    if len(valence) < 12:
        return {"shape": "Too short to call", "fit": 0.0}
    k = max(3, len(valence) // 10)
    v = np.convolve(valence, np.ones(k) / k, mode="valid")
    span = float(v.max() - v.min())
    if span < 0.12:
        return {"shape": "Steady", "fit": round(1 - span / 0.12, 3)}
    v = (v - v.mean()) / (v.std() + 1e-9)
    best, best_r = "Steady", -1.0
    for name, tpl in _arc_templates(len(v)).items():
        tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-9)
        r = float(np.mean(v * tpl))
        if r > best_r:
            best, best_r = name, r
    return {"shape": best if best_r > 0.45 else "Rollercoaster (irregular)", "fit": round(best_r, 3)}


# --------------------------------------------------------------- fusion
def fuse(tracks: list[Track], grid: Grid, silent: np.ndarray) -> dict:
    n = len(grid)
    hop_scale = grid.hop / 0.5

    def obs_for(attr: str, weight_key: str):
        out = []
        for tr in tracks:
            vals = getattr(tr, attr)
            if vals is None:
                continue
            w = tr.conf_va * tr.extra.get(weight_key, 1.0)
            out.append((vals, w))
        return out

    v_obs = obs_for("valence", "valence_weight")
    a_obs = obs_for("arousal", "arousal_weight")
    valence, v_sd = kalman_smooth(v_obs, q=0.012 * hop_scale)
    arousal, a_sd = kalman_smooth(a_obs, q=0.02 * hop_scale)
    d_obs = obs_for("dominance", "dominance_weight")
    dominance = kalman_smooth(d_obs, q=0.015 * hop_scale)[0] if d_obs else None

    # ---- logarithmic opinion pool
    eps = 0.02
    logp = np.zeros((n, N_E))
    for tr in tracks:
        if tr.probs is None:
            continue
        P = np.nan_to_num(tr.probs, nan=1 / N_E)
        # Classifiers trained on acted corpora fall back to "neutral" when
        # they lack evidence; treat that mass as partial abstention so a
        # confident signal elsewhere is not vetoed by a shrug.
        w = tr.conf_cat * (1 - 0.6 * P[:, IDX["neutral"]])
        P = (1 - eps) * P + eps / N_E
        logp += w[:, None] * np.log(P)
    # fused circumplex position as one more expert, weighted by its certainty
    kernel = va_to_probs(valence, arousal, sharpness=4.0)
    va_cert = np.clip(1 - (v_sd + a_sd) / 0.8, 0.1, 1)
    logp += 0.9 * va_cert[:, None] * np.log((1 - eps) * kernel + eps / N_E)
    # silence prior
    sil = np.zeros(N_E)
    sil[IDX["neutral"]], sil[IDX["calm"]] = 0.6, 0.3
    sil += 0.1 / N_E
    logp += np.where(silent, 1.5, 0)[:, None] * np.log(sil / sil.sum())

    logp -= logp.max(1, keepdims=True)
    frame = np.exp(logp)
    frame /= frame.sum(1, keepdims=True)

    stay = 0.9 ** hop_scale
    T = _transition(stay)
    post = forward_backward(frame, T)
    path = viterbi(frame, T)

    # ---- segments from Viterbi path
    segments = []
    start = 0
    for t in range(1, n + 1):
        if t == n or path[t] != path[start]:
            seg_p = post[start:t, path[start]].mean()
            segments.append({
                "start": round(float(max(0, grid.times[start] - grid.hop / 2)), 3),
                "end": round(float(min(grid.duration, grid.times[t - 1] + grid.hop / 2)), 3),
                "emotion": EMOTIONS[path[start]],
                "confidence": round(float(seg_p), 3),
            })
            start = t
    segments = _merge_short(segments, min_len=1.5)

    # ---- derived curves
    intensity = np.clip(np.sqrt(valence**2 + arousal**2) / 1.2, 0, 1)
    sentiment = np.where(valence > 0.15, "positive", np.where(valence < -0.15, "negative", "neutral"))

    active = ~silent if (~silent).any() else np.ones(n, bool)
    mean_probs = post[active].mean(0)
    time_share = {e: 0.0 for e in EMOTIONS}
    for s in segments:
        time_share[s["emotion"]] += s["end"] - s["start"]
    total = sum(time_share.values()) or 1
    time_share = {k: round(v / total, 4) for k, v in time_share.items()}

    summary = {
        "dominant_emotion": EMOTIONS[int(mean_probs.argmax())],
        "emotion_mix": {e: round(float(p), 4) for e, p in zip(EMOTIONS, mean_probs)},
        "time_share": time_share,
        "valence": round(float(valence[active].mean()), 3),
        "arousal": round(float(arousal[active].mean()), 3),
        "sentiment": _label_sentiment(float(valence[active].mean())),
        "positivity": round(float((valence[active] > 0.15).mean()), 3),
        "negativity": round(float((valence[active] < -0.15).mean()), 3),
        "volatility": round(float(np.std(np.diff(valence)) + np.std(np.diff(arousal))) * 10, 3) if n > 2 else 0.0,
        "arc": story_arc(valence[active]),
        "transitions": max(0, len(segments) - 1),
    }

    sources = []
    for tr in tracks:
        sources.append({
            "name": tr.name,
            "label": tr.label,
            "valence": _r(tr.valence),
            "arousal": _r(tr.arousal),
            "weight": _r(tr.conf_va),
            "cat_weight": _r(tr.conf_cat) if tr.conf_cat is not None else None,
            "top": [EMOTIONS[int(i)] if np.isfinite(row).all() else None for i, row in zip(np.nan_to_num(tr.probs, nan=0).argmax(1), tr.probs)] if tr.probs is not None else None,
        })

    return {
        "valence": _r(valence),
        "arousal": _r(arousal),
        "dominance": _r(dominance) if dominance is not None else None,
        "valence_sd": _r(v_sd),
        "arousal_sd": _r(a_sd),
        "intensity": _r(intensity),
        "sentiment": sentiment.tolist(),
        "probs": np.round(post, 4).tolist(),
        "frame_probs": np.round(frame, 4).tolist(),
        "segments": segments,
        "moments": _moments(grid, valence, arousal, segments, silent),
        "summary": summary,
        "sources": sources,
    }


def _r(a, nd=4):
    a = np.asarray(a, dtype=np.float64)
    return [None if not np.isfinite(x) else round(float(x), nd) for x in a]


def _label_sentiment(v: float) -> str:
    if v > 0.35:
        return "very positive"
    if v > 0.12:
        return "positive"
    if v < -0.35:
        return "very negative"
    if v < -0.12:
        return "negative"
    return "neutral / mixed"


def _merge_short(segments: list[dict], min_len: float) -> list[dict]:
    if len(segments) <= 1:
        return segments
    out = [dict(segments[0])]
    for s in segments[1:]:
        if s["end"] - s["start"] < min_len:
            out[-1]["end"] = s["end"]
        elif out[-1]["emotion"] == s["emotion"]:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    # first segment may itself be short
    if len(out) > 1 and out[0]["end"] - out[0]["start"] < min_len:
        out[1]["start"] = out[0]["start"]
        out.pop(0)
    return out


def _moments(grid: Grid, valence, arousal, segments, silent) -> list[dict]:
    moments = []
    dist = max(1, int(8 / grid.hop))
    t = grid.times

    def add(i, kind, title):
        moments.append({"t": round(float(t[i]), 2), "kind": kind, "title": title,
                        "valence": round(float(valence[i]), 3), "arousal": round(float(arousal[i]), 3)})

    for i in find_peaks(arousal, prominence=0.25, distance=dist)[0][:6]:
        if not silent[i]:
            add(i, "peak", "Energy peak")
    for i in find_peaks(valence, prominence=0.25, distance=dist)[0][:5]:
        add(i, "high", "Most positive stretch" if valence[i] == valence.max() else "Positive swing")
    for i in find_peaks(-valence, prominence=0.25, distance=dist)[0][:5]:
        add(i, "low", "Most negative stretch" if valence[i] == valence.min() else "Negative dip")
    for a, b in zip(segments[:-1], segments[1:]):
        moments.append({"t": round(b["start"], 2), "kind": "shift",
                        "title": f"{a['emotion'].capitalize()} → {b['emotion']}",
                        "valence": None, "arousal": None})
    moments.sort(key=lambda m: m["t"])
    # de-duplicate moments that land on top of each other
    out = []
    for m in moments:
        if out and abs(out[-1]["t"] - m["t"]) < 1.0 and out[-1]["kind"] == m["kind"]:
            continue
        out.append(m)
    return out[:40]
