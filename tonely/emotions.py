"""The shared emotion space every analyzer maps into.

Two complementary representations are used throughout:

* **Dimensional** — Russell's circumplex: valence (unpleasant → pleasant) and
  arousal (calm → excited), both in [-1, 1], plus dominance where available.
* **Categorical** — a probability distribution over eight discrete emotions.

Each categorical emotion has a prototype location in valence/arousal space so
the two views can be converted into one another (see ``va_to_probs``).
"""

from __future__ import annotations

import numpy as np

# Order matters: it is the stacking order used by the UI (validated for
# colour-vision-deficiency separation of adjacent bands).
EMOTIONS: list[str] = [
    "sadness",
    "surprise",
    "calm",
    "joy",
    "neutral",
    "disgust",
    "fear",
    "anger",
]
IDX = {name: i for i, name in enumerate(EMOTIONS)}

# Prototype (valence, arousal) per emotion — after Russell (1980) and
# Scherer's (2005) Geneva Emotion Wheel placements.
PROTOTYPES: dict[str, tuple[float, float]] = {
    "sadness": (-0.65, -0.45),
    "surprise": (0.25, 0.75),
    "calm": (0.50, -0.55),
    "joy": (0.75, 0.40),
    "neutral": (0.0, -0.05),
    "disgust": (-0.65, 0.20),
    "fear": (-0.55, 0.65),
    "anger": (-0.70, 0.75),
}
PROTO = np.array([PROTOTYPES[e] for e in EMOTIONS], dtype=np.float64)


def empty_probs() -> np.ndarray:
    return np.zeros(len(EMOTIONS), dtype=np.float64)


def remap(dist: dict[str, float], mapping: dict[str, str]) -> np.ndarray:
    """Project a model's label distribution into the shared emotion space."""
    out = empty_probs()
    for label, p in dist.items():
        target = mapping.get(label.lower())
        if target is not None:
            out[IDX[target]] += p
    s = out.sum()
    return out / s if s > 0 else out


def va_to_probs(valence: np.ndarray, arousal: np.ndarray, sharpness: float = 5.0) -> np.ndarray:
    """Soft-assign valence/arousal points to emotions with a Gaussian kernel.

    Returns an array of shape (n, len(EMOTIONS)).
    """
    pts = np.stack([np.asarray(valence), np.asarray(arousal)], axis=-1)[:, None, :]
    d2 = ((pts - PROTO[None, :, :]) ** 2).sum(-1)
    logits = -sharpness * d2
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def probs_to_va(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected valence/arousal of a categorical distribution (n, E)."""
    va = probs @ PROTO
    return va[:, 0], va[:, 1]
