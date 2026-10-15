"""Synthesize a three-part test track: bright major → slow minor → harsh noise."""

from __future__ import annotations

import sys

import numpy as np
import soundfile as sf

SR = 22_050


def tone(freqs, dur, amp, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return amp * sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)


def arpeggio(root, minor, bpm, seconds, amp):
    third = 3 if minor else 4
    notes = [0, third, 7, 12, 7, third]
    beat = 60 / bpm / 2
    out = []
    for i in range(int(seconds / beat)):
        f = root * 2 ** (notes[i % len(notes)] / 12)
        n = tone([f, f * 2, f * 3], beat, amp)
        env = np.exp(-np.linspace(0, 4, len(n)))
        out.append(n * env)
    pad = tone([root, root * 2 ** (third / 12), root * 1.5], seconds, amp * 0.4)
    arp = np.concatenate(out)
    return pad[: len(arp)] + arp


def build(seconds_each=15.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    happy = arpeggio(523.25, False, 150, seconds_each, 0.6)
    sad = arpeggio(110.0, True, 60, seconds_each, 0.15)
    n = int(seconds_each * SR)
    harsh = 0.7 * np.sign(np.sin(2 * np.pi * 93 * np.arange(n) / SR)) + 0.5 * rng.standard_normal(n)
    harsh *= (np.arange(n) % (SR // 4) < SR // 8)  # pulsing
    return np.concatenate([happy, sad, harsh.astype(np.float64) * 0.6]).astype(np.float32)


if __name__ == "__main__":
    sf.write(sys.argv[1] if len(sys.argv) > 1 else "synth.wav", build(), SR)
