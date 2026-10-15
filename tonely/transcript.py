"""Linguistic channel: Whisper ASR with word timestamps, then transformer
emotion + sentiment classification of what is being said.

Non-English speech is additionally translated to English by Whisper itself
so the (English) text classifiers work for ~100 languages.
"""

from __future__ import annotations

import logging

import numpy as np

from . import models
from .audio import Audio
from .emotions import EMOTIONS, probs_to_va, remap
from .tracks import Grid, Track

log = logging.getLogger("tonely")

TEXT_MAP = {e: e for e in EMOTIONS}
TEXT_MAP.update({"happiness": "joy", "love": "joy", "optimism": "joy"})


def transcribe(audio: Audio, cfg, device, progress=None) -> dict | None:
    model = models.whisper(cfg.whisper_model, device)
    if model is None:
        return None
    segments_iter, info = model.transcribe(
        audio.samples, word_timestamps=True, vad_filter=True, beam_size=5,
        condition_on_previous_text=False,
    )
    segments = []
    for s in segments_iter:
        text = s.text.strip()
        if not text or s.no_speech_prob > 0.8:
            continue
        segments.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": text,
            "conf": float(np.exp(min(0.0, s.avg_logprob))),
            "words": [
                {"s": round(w.start, 3), "e": round(w.end, 3), "w": w.word, "p": round(w.probability, 3)}
                for w in (s.words or [])
            ],
        })
        if progress and audio.duration:
            progress(min(1.0, s.end / audio.duration) * 0.85)

    translated = None
    if info.language != "en" and segments:
        tr_iter, _ = model.transcribe(audio.samples, task="translate", vad_filter=True, beam_size=5)
        translated = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in tr_iter if s.text.strip()]
    if progress:
        progress(1.0)
    return {
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "segments": segments,
        "translated": translated,
    }


def _classify(clf, texts: list[str]) -> list[dict[str, float]]:
    if not texts:
        return []
    out = clf(texts, batch_size=16)
    return [{d["label"].lower(): float(d["score"]) for d in row} for row in out]


def text_track(transcript: dict, grid: Grid, cfg, device) -> Track | None:
    emo_clf = models.text_classifier(cfg.text_emotion_model, device)
    sent_clf = models.text_classifier(cfg.text_sentiment_model, device)
    if emo_clf is None and sent_clf is None:
        return None

    units = transcript["translated"] or transcript["segments"]
    if not units:
        return None
    texts = [u["text"] for u in units]
    # Each unit is classified alone and with its predecessor as context; the
    # blend stabilises short utterances ("yeah.", "okay") whose emotion is
    # carried by what came before.
    ctx = [(texts[i - 1] + " " + texts[i]) if i else texts[i] for i in range(len(texts))]

    n_e = len(EMOTIONS)
    P = np.full((len(units), n_e), 1.0 / n_e)
    if emo_clf is not None:
        alone, with_ctx = _classify(emo_clf, texts), _classify(emo_clf, ctx)
        P = np.array([0.7 * remap(a, TEXT_MAP) + 0.3 * remap(c, TEXT_MAP) for a, c in zip(alone, with_ctx)])
    v_emo, a_emo = probs_to_va(P)

    polarity = np.zeros(len(units))
    if sent_clf is not None:
        sent = _classify(sent_clf, texts)
        polarity = np.array([s.get("positive", 0) - s.get("negative", 0) for s in sent])
        valence = 0.6 * polarity + 0.4 * v_emo
    else:
        valence = v_emo

    n = len(grid)
    acc_v, acc_a, acc_p, acc_w = np.zeros(n), np.zeros(n), np.zeros((n, n_e)), np.zeros(n)
    conf_unit = []
    for k, u in enumerate(units):
        n_words = max(1, len(u["text"].split()))
        c = float(np.clip(n_words / 6, 0.35, 1.0)) * float(u.get("conf", 0.8))
        conf_unit.append(c)
        a, b = u["start"] - 0.25, u["end"] + 0.25
        sel = (grid.times >= a) & (grid.times <= b)
        acc_v[sel] += c * valence[k]
        acc_a[sel] += c * a_emo[k]
        acc_p[sel] += c * P[k]
        acc_w[sel] += c

    has = acc_w > 0
    track = Track.empty("text", "Words (Whisper + RoBERTa)", n)
    track.valence[has] = acc_v[has] / acc_w[has]
    track.arousal[has] = acc_a[has] / acc_w[has]
    track.probs[has] = acc_p[has] / acc_w[has][:, None]
    count = np.zeros(n)
    for u in units:
        count[(grid.times >= u["start"] - 0.25) & (grid.times <= u["end"] + 0.25)] += 1
    mean_conf = np.where(has, acc_w / np.maximum(count, 1), 0)
    track.conf_va = 0.8 * mean_conf
    track.conf_cat = 0.75 * mean_conf
    track.extra["valence_weight"] = 1.0
    track.extra["arousal_weight"] = 0.35  # words say little about arousal

    # annotate transcript segments for the UI
    for seg in transcript["segments"]:
        mid = (seg["start"] + seg["end"]) / 2
        k = int(np.argmin([abs((u["start"] + u["end"]) / 2 - mid) for u in units]))
        seg["emotion"] = EMOTIONS[int(P[k].argmax())]
        seg["emotion_p"] = round(float(P[k].max()), 3)
        seg["sentiment"] = round(float(polarity[k]), 3)
        seg["valence"] = round(float(valence[k]), 3)
        if transcript["translated"]:
            seg["translation"] = units[k]["text"]
    return track
