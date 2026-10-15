"""Runtime configuration: which analyzers run and which checkpoints they use."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Config:
    # Analysis grid
    hop: float = 0.5
    win: float = 3.0
    model_hop: float = 1.0  # neural analyzers run on a coarser hop, then get interpolated

    # Analyzers
    use_asr: bool = True
    use_text: bool = True
    use_ser: bool = True
    use_ser_dim: bool = True
    use_clap: bool = True

    # Checkpoints (all pulled from the Hugging Face Hub on first use)
    whisper_model: str = "small"
    ser_model: str = "superb/wav2vec2-base-superb-er"
    ser_dim_model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    text_emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    text_sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    clap_model: str = "laion/clap-htsat-unfused"

    device: str = field(default_factory=lambda: os.environ.get("TONELY_DEVICE", "auto"))

    @classmethod
    def profile(cls, name: str) -> "Config":
        name = name.lower()
        if name == "full":
            return cls()
        if name == "lite":
            return cls(whisper_model="base", use_ser_dim=False, use_clap=False)
        if name == "acoustic":
            return cls(use_asr=False, use_text=False, use_ser=False, use_ser_dim=False, use_clap=False)
        raise ValueError(f"unknown profile {name!r} (expected full, lite or acoustic)")

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
