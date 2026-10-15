"""End-to-end analysis: decode → analyzers → fusion → JSON-ready result."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, acoustic, audio as audio_mod, fusion, models, music, speech, transcript
from .config import Config
from .emotions import EMOTIONS, PROTOTYPES
from .tracks import Grid

log = logging.getLogger("tonely")

Progress = Callable[[str, float, str], None]  # (stage, overall 0..1, message)

# (stage id, label, share of the progress bar)
STAGES = [
    ("decode", "Decoding audio", 0.03),
    ("acoustic", "Acoustic & prosodic features", 0.10),
    ("vad", "Detecting speech", 0.04),
    ("asr", "Transcribing words", 0.25),
    ("text", "Reading emotion in words", 0.06),
    ("ser", "Voice emotion model", 0.16),
    ("ser_dim", "Voice valence / arousal model", 0.18),
    ("clap", "Sound & music mood (CLAP)", 0.14),
    ("fusion", "Fusing signals", 0.04),
]


class _Progress:
    def __init__(self, cb: Progress | None, enabled: set[str]):
        self.cb = cb
        self.stages = [s for s in STAGES if s[0] in enabled]
        total = sum(s[2] for s in self.stages)
        self.offsets, acc = {}, 0.0
        for sid, _, share in self.stages:
            self.offsets[sid] = (acc / total, share / total)
            acc += share
        self.labels = {s[0]: s[1] for s in STAGES}

    def __call__(self, stage: str, frac: float = 0.0):
        if self.cb is None or stage not in self.offsets:
            return
        off, share = self.offsets[stage]
        self.cb(stage, min(1.0, off + share * frac), self.labels[stage])

    def sub(self, stage: str):
        return lambda f: self(stage, f)


def cache_key(path: Path, cfg: Config) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(json.dumps(cfg.__dict__, sort_keys=True, default=str).encode())
    h.update(__version__.encode())
    return h.hexdigest()[:24]


def analyze(path: str | Path, cfg: Config | None = None, progress: Progress | None = None, name: str | None = None) -> dict:
    cfg = cfg or Config()
    path = Path(path)
    device = models.torch_device(cfg.device)
    enabled = {"decode", "acoustic", "vad", "fusion"}
    if cfg.use_asr:
        enabled.add("asr")
    if cfg.use_asr and cfg.use_text:
        enabled.add("text")
    if cfg.use_ser:
        enabled.add("ser")
    if cfg.use_ser_dim:
        enabled.add("ser_dim")
    if cfg.use_clap:
        enabled.add("clap")
    prog = _Progress(progress, enabled)
    report: list[dict] = []

    def run(stage, fn):
        prog(stage, 0.0)
        t0 = time.perf_counter()
        try:
            out = fn()
            status = "ok" if out is not None else "unavailable"
        except Exception as exc:  # noqa: BLE001 - one analyzer must never sink the run
            log.exception("%s failed", stage)
            out, status = None, f"failed: {exc}"
        report.append({"stage": stage, "label": prog.labels[stage], "status": status,
                       "seconds": round(time.perf_counter() - t0, 2)})
        prog(stage, 1.0)
        return out

    audio = run("decode", lambda: audio_mod.load(path))
    if audio is None:
        raise ValueError("Could not decode this file as audio.")
    if audio.duration < 0.5:
        raise ValueError("Audio is too short to analyze (< 0.5 s).")
    grid = Grid.build(audio.duration, hop=cfg.hop, win=cfg.win)

    ac = run("acoustic", lambda: acoustic.analyze(audio, grid))
    if ac is None:
        raise RuntimeError("acoustic analysis failed")
    ac_track, ac_info = ac
    silent = ac_info["silent"]
    tracks = [ac_track]

    intervals = run("vad", lambda: speech.speech_intervals(audio))
    if intervals is not None:
        speech_frac = speech.speech_fraction(intervals, grid)
    else:
        voiced = np.array(ac_info["descriptors"]["voiced_ratio"])
        speech_frac = np.clip(voiced * 1.5, 0, 1)
        intervals = []

    has_speech = speech_frac.max() > 0.2 if len(speech_frac) else False
    tr = None
    if "asr" in enabled and has_speech:
        tr = run("asr", lambda: transcript.transcribe(audio, cfg, device, prog.sub("asr")))
        if tr is not None and "text" in enabled and tr["segments"]:
            t_track = run("text", lambda: transcript.text_track(tr, grid, cfg, device))
            if t_track is not None:
                tracks.append(t_track)

    if "ser" in enabled and has_speech:
        t = run("ser", lambda: speech.ser_categorical(audio, grid, speech_frac, cfg, device, prog.sub("ser")))
        if t is not None:
            tracks.append(t)
    if "ser_dim" in enabled and has_speech:
        t = run("ser_dim", lambda: speech.ser_dimensional(audio, grid, speech_frac, cfg, device, prog.sub("ser_dim")))
        if t is not None:
            tracks.append(t)

    content = None
    if "clap" in enabled:
        def _clap():
            a48 = audio_mod.load(path, sr=music.CLAP_SR)
            return music.analyze(a48.samples, grid, speech_frac, cfg, device, prog.sub("clap"))

        res = run("clap", _clap)
        if res is not None and res[0] is not None:
            tracks.append(res[0])
            content = res[1]

    fused = run("fusion", lambda: fusion.fuse(tracks, grid, silent))
    if fused is None:
        raise RuntimeError("fusion failed")

    kind = _content_kind(content, speech_frac, silent)
    return {
        "tonely_version": __version__,
        "file": {"name": name or path.name, "duration": round(audio.duration, 3)},
        "emotions": EMOTIONS,
        "prototypes": {k: list(v) for k, v in PROTOTYPES.items()},
        "grid": {"hop": grid.hop, "win": grid.win, "times": np.round(grid.times, 3).tolist()},
        "waveform": audio_mod.waveform_peaks(audio),
        "content": {"kind": kind, "probs": content, "speech_fraction": np.round(speech_frac, 3).tolist(),
                    "speech_intervals": [[round(a, 3), round(b, 3)] for a, b in intervals]},
        "acoustic": {**ac_info["descriptors"], "summary": ac_info["summary"]},
        "transcript": tr,
        "fused": fused,
        "analyzers": report,
        "device": device,
    }


def _content_kind(content, speech_frac, silent) -> str:
    sp = float(np.mean(speech_frac))
    mu = float(np.mean(content["music"])) if content else 0.0
    if content and sp > 0.35:
        # music under speech: judge only by the speech-free parts and the strongest music windows
        m = np.array(content["music"])
        mu = float(np.percentile(m, 75)) if (speech_frac < 0.3).sum() < 4 else float(m[speech_frac < 0.3].mean())
    if silent.mean() > 0.8:
        return "mostly silence"
    if sp > 0.35 and mu > 0.55:
        return "speech over music"
    if sp > 0.35:
        return "speech"
    if mu > 0.35 or (content is None and sp < 0.15):
        return "music"
    return "mixed audio"
