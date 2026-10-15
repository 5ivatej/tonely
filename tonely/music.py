"""Zero-shot audio mood via CLAP (contrastive language–audio pretraining).

Every window's audio embedding is compared against ensembles of natural
language prompts — per emotion, per circumplex pole, and per content type
(music / speech / ambience). This gives a semantic read of *music and
non-speech sound* that the speech models cannot provide.
"""

from __future__ import annotations

import logging

import numpy as np

from . import models
from .emotions import EMOTIONS
from .tracks import Grid, Track

log = logging.getLogger("tonely")

CLAP_SR = 48_000
WIN, HOP = 6.0, 2.0

EMOTION_PROMPTS = {
    "joy": ["happy, joyful, cheerful music", "a person laughing happily", "an uplifting, celebratory sound"],
    "calm": ["calm, peaceful, relaxing music", "a soft, gentle ambient sound", "a soothing, quiet voice"],
    "sadness": ["sad, melancholic music", "a person crying", "a mournful, slow, sorrowful sound"],
    "anger": ["angry, aggressive music", "a person shouting angrily", "a harsh, furious, heavy sound"],
    "fear": ["scary, tense, suspenseful music", "a frightened scream", "an eerie horror soundtrack"],
    "surprise": ["a sudden, dramatic, surprising sound", "a person gasping in surprise", "an unexpected exciting crescendo"],
    "disgust": ["a person expressing disgust", "an unpleasant, grating, dissonant noise"],
    "neutral": ["a neutral background sound", "a person speaking in a flat, neutral tone", "a plain, ordinary sound"],
}
POLES = {
    "valence+": ["a pleasant, positive, happy sound", "bright, uplifting, feel-good music"],
    "valence-": ["an unpleasant, negative, dark sound", "gloomy, distressing, sad music"],
    "arousal+": ["an energetic, intense, loud and exciting sound", "fast, powerful, high-energy music"],
    "arousal-": ["a calm, quiet, slow and sleepy sound", "soft, slow, gentle music"],
}
CONTENT = {
    "music": ["music", "a song with instruments"],
    "speech": ["speech", "a podcast conversation"],
    "ambience": ["background noise", "ambient environmental sound"],
    "silence": ["silence"],
}


def _text_bank(proc, model, torch, groups: dict[str, list[str]]):
    names, embs = [], []
    for name, prompts in groups.items():
        t = proc(text=prompts, return_tensors="pt", padding=True)
        t = {k: v.to(model.device) for k, v in t.items()}
        with torch.no_grad():
            e = model.get_text_features(**t)
        e = getattr(e, "pooler_output", e)
        e = torch.nn.functional.normalize(e.float(), dim=-1).mean(0)
        embs.append(torch.nn.functional.normalize(e, dim=-1))
        names.append(name)
    return names, torch.stack(embs)


def _audio_embeddings(samples48: np.ndarray, duration: float, proc, model, torch, progress=None):
    centers = np.arange(HOP / 2, max(duration, HOP / 2 + 1e-3), HOP)
    half = int(WIN / 2 * CLAP_SR)
    clips = []
    for c in centers:
        mid = int(c * CLAP_SR)
        a, b = max(0, mid - half), min(len(samples48), mid + half)
        clips.append(samples48[a:b])
    out = []
    for i in range(0, len(clips), 8):
        batch = clips[i:i + 8]
        try:
            a = proc(audio=batch, sampling_rate=CLAP_SR, return_tensors="pt")
        except TypeError:
            a = proc(audios=batch, sampling_rate=CLAP_SR, return_tensors="pt")
        a = {k: v.to(model.device) for k, v in a.items()}
        with torch.no_grad():
            e = model.get_audio_features(**a)
        e = getattr(e, "pooler_output", e)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1))
        if progress:
            progress(min(1.0, (i + len(batch)) / len(clips)))
    return centers, torch.cat(out)


def analyze(samples48: np.ndarray, grid: Grid, speech_frac: np.ndarray, cfg, device, progress=None):
    loaded = models.clap(cfg.clap_model, device)
    if loaded is None:
        return None, None
    proc, model, torch = loaded
    scale = float(model.logit_scale_a.detach().exp().clamp(max=100))

    centers, A = _audio_embeddings(samples48, grid.duration, proc, model, torch, progress)
    emo_names, E = _text_bank(proc, model, torch, EMOTION_PROMPTS)
    pole_names, Pl = _text_bank(proc, model, torch, POLES)
    content_names, C = _text_bank(proc, model, torch, CONTENT)

    def soft(sim):
        return torch.softmax(scale * sim, dim=-1).cpu().numpy()

    emo = soft(A @ E.T)
    poles = (A @ Pl.T).cpu().numpy()
    content = soft(A @ C.T)

    pi = {n: i for i, n in enumerate(pole_names)}
    pair = lambda p, m: np.tanh(scale * 0.5 * (poles[:, pi[p]] - poles[:, pi[m]]))  # noqa: E731
    valence, arousal = pair("valence+", "valence-"), pair("arousal+", "arousal-")

    order = [emo_names.index(e) for e in EMOTIONS]
    emo = emo[:, order]

    def interp(v):
        if v.ndim == 1:
            return np.interp(grid.times, centers, v)
        return np.stack([np.interp(grid.times, centers, v[:, j]) for j in range(v.shape[1])], 1)

    P = interp(emo)
    P /= P.sum(1, keepdims=True)
    content_g = interp(content)
    music_p = content_g[:, content_names.index("music")]

    entropy = -(P * np.log(P + 1e-12)).sum(1) / np.log(P.shape[1])
    sharp = 1 - entropy
    # CLAP is most trustworthy on music / non-speech; speech is covered by
    # dedicated models, so its weight there is reduced.
    domain = np.clip(1 - 0.75 * speech_frac + 0.4 * music_p * speech_frac, 0.2, 1)

    track = Track.empty("clap", "Sound mood (CLAP)", len(grid))
    track.probs = P
    track.conf_cat = domain * (0.30 + 0.5 * sharp)
    track.valence = interp(valence)
    track.arousal = interp(arousal)
    track.conf_va = 0.45 * domain
    track.extra["valence_weight"] = 0.8
    content_out = {n: np.round(content_g[:, i], 3).tolist() for i, n in enumerate(content_names)}
    return track, content_out
