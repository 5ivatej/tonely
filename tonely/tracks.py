"""Common data structures shared by analyzers and the fusion stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .emotions import EMOTIONS


@dataclass
class Grid:
    """Uniform analysis timeline. Point ``i`` describes the window centred on
    ``times[i]`` with total width ``win`` seconds."""

    times: np.ndarray
    hop: float
    win: float
    duration: float

    @classmethod
    def build(cls, duration: float, hop: float = 0.5, win: float = 3.0) -> "Grid":
        n = max(1, int(np.ceil(duration / hop)))
        times = np.minimum((np.arange(n) + 0.5) * hop, duration)
        return cls(times=times, hop=hop, win=win, duration=duration)

    def __len__(self) -> int:
        return len(self.times)

    def bounds(self, i: int, win: float | None = None) -> tuple[float, float]:
        w = (win or self.win) / 2
        return max(0.0, self.times[i] - w), min(self.duration, self.times[i] + w)


@dataclass
class Track:
    """One analyzer's opinion along the grid.

    Values are NaN where the analyzer has nothing to say; confidences are in
    [0, 1] and drive the fusion weights.
    """

    name: str
    label: str
    valence: np.ndarray
    arousal: np.ndarray
    conf_va: np.ndarray
    probs: np.ndarray | None = None  # (n, len(EMOTIONS))
    conf_cat: np.ndarray | None = None
    dominance: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, name: str, label: str, n: int, categorical: bool = True) -> "Track":
        nan = np.full(n, np.nan)
        return cls(
            name=name,
            label=label,
            valence=nan.copy(),
            arousal=nan.copy(),
            conf_va=np.zeros(n),
            probs=np.full((n, len(EMOTIONS)), np.nan) if categorical else None,
            conf_cat=np.zeros(n) if categorical else None,
        )
