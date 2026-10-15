"""Fast tests: fusion maths and the model-free (acoustic) pipeline end to end."""

import json

import numpy as np
import soundfile as sf

from tonely.config import Config
from tonely.emotions import EMOTIONS, IDX, va_to_probs
from tonely.fusion import forward_backward, kalman_smooth, story_arc, viterbi, _transition
from tonely.pipeline import analyze
from tests.synth import SR, build


def test_va_kernel_peaks_at_prototypes():
    p = va_to_probs(np.array([0.75, -0.7]), np.array([0.4, 0.75]))
    assert EMOTIONS[p[0].argmax()] == "joy"
    assert EMOTIONS[p[1].argmax()] == "anger"


def test_kalman_prefers_confident_sensor():
    n = 50
    a = (np.full(n, 0.8), np.full(n, 0.9))
    b = (np.full(n, -0.8), np.full(n, 0.1))
    m, sd = kalman_smooth([a, b], q=0.01)
    assert m[25] > 0.4
    assert (sd > 0).all()


def test_kalman_bridges_missing_observations():
    vals = np.full(40, np.nan)
    vals[:5], vals[-5:] = 0.5, 0.5
    m, sd = kalman_smooth([(vals, np.ones(40))], q=0.005)
    assert 0.2 < m[20] <= 0.5
    assert sd[20] > sd[2]


def test_hmm_smooths_single_frame_blips():
    n = 30
    emis = np.full((n, len(EMOTIONS)), 0.02)
    emis[:, IDX["joy"]] = 0.9
    emis[15] = 0.02
    emis[15, IDX["anger"]] = 0.6
    T = _transition(0.9)
    assert (viterbi(emis, T) == IDX["joy"]).all()
    post = forward_backward(emis, T)
    assert np.allclose(post.sum(1), 1)


def test_story_arc_detects_rise_and_fall():
    x = np.linspace(0, 1, 200)
    assert story_arc(x * 1.5 - 0.75)["shape"].startswith("Rags")
    assert story_arc(np.sin(np.pi * x) - 0.3)["shape"].startswith("Icarus")
    assert story_arc(np.full(100, 0.1))["shape"] == "Steady"


def test_acoustic_pipeline_on_synthetic_music(tmp_path):
    path = tmp_path / "synth.wav"
    sf.write(path, build(seconds_each=10.0), SR)
    stages = []
    r = analyze(path, Config.profile("acoustic"), lambda s, f, l: stages.append(s))
    json.dumps(r)  # serialisable
    f = r["fused"]
    n = len(r["grid"]["times"])
    assert len(f["valence"]) == len(f["arousal"]) == len(f["probs"]) == n
    assert {"decode", "acoustic", "fusion"} <= set(stages)
    t = np.array(r["grid"]["times"])
    aro = np.array(f["arousal"])
    # the soft, slow middle section must read calmer than the harsh last one
    assert aro[(t > 12) & (t < 18)].mean() < aro[t > 23].mean() - 0.3
    assert f["segments"] and f["segments"][0]["start"] == 0
