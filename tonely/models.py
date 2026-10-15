"""Lazy, cached model loading. Every loader returns ``None`` (and logs why)
when its dependency or checkpoint is unavailable, so the pipeline degrades
gracefully instead of failing."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

log = logging.getLogger("tonely")
_lock = threading.Lock()


@lru_cache(maxsize=None)
def torch_device(pref: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _guard(name: str):
    def deco(fn):
        @lru_cache(maxsize=None)
        def wrapper(*args):
            with _lock:
                try:
                    return fn(*args)
                except Exception as exc:  # noqa: BLE001 - any failure → analyzer disabled
                    log.warning("%s unavailable: %s", name, exc)
                    return None

        return wrapper

    return deco


@_guard("whisper")
def whisper(size: str, device: str):
    from faster_whisper import WhisperModel

    # CTranslate2 has no MPS backend; CPU int8 is fast on Apple silicon.
    dev = "cuda" if device == "cuda" else "cpu"
    return WhisperModel(size, device=dev, compute_type="float16" if dev == "cuda" else "int8")


@_guard("speech emotion (categorical)")
def ser(name: str, device: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    fe = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModelForAudioClassification.from_pretrained(name).to(device).eval()
    return fe, model, torch


@_guard("speech emotion (dimensional)")
def ser_dim(name: str, device: str):
    import torch
    from torch import nn
    from transformers import Wav2Vec2FeatureExtractor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            return self.classifier(hidden.mean(dim=1))

    fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
    model = EmotionModel.from_pretrained(name).to(device).eval()
    return fe, model, torch


@_guard("text classifier")
def text_classifier(name: str, device: str):
    from transformers import pipeline

    return pipeline("text-classification", model=name, top_k=None, device=device, truncation=True)


@_guard("CLAP")
def clap(name: str, device: str):
    import torch
    from transformers import ClapModel, ClapProcessor

    proc = ClapProcessor.from_pretrained(name)
    model = ClapModel.from_pretrained(name).to(device).eval()
    return proc, model, torch
