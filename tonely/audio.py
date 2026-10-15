"""Audio decoding. PyAV bundles FFmpeg, so any common format works without a
system-wide ffmpeg install (mp3, wav, flac, ogg, m4a/aac, opus, webm, mp4 …)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

SR = 16_000  # every model in the pipeline runs at 16 kHz mono


@dataclass
class Audio:
    samples: np.ndarray  # float32 mono in [-1, 1]
    sr: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr


def load(path: str | Path, sr: int = SR) -> Audio:
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("file contains no audio stream")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("could not decode any audio")
    samples = np.concatenate(chunks).astype(np.float32)
    peak = float(np.abs(samples).max())
    if peak > 1.0:
        samples /= peak
    return Audio(samples=samples, sr=sr)


def waveform_peaks(audio: Audio, n: int = 1600) -> list[list[float]]:
    """Min/max envelope for drawing a waveform overview."""
    x = audio.samples
    n = max(1, min(n, len(x)))
    edges = np.linspace(0, len(x), n + 1, dtype=np.int64)
    peaks = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[a:max(b, a + 1)]
        peaks.append([round(float(seg.min()), 4), round(float(seg.max()), 4)])
    return peaks
