"""Runtime configuration, read from environment variables.

The variable names follow the Hack Apertus template convention
(LLM_NAME / LLM_BASE_URL / LLM_API_KEY) so judges can point the coach
at any OpenAI-compatible Apertus endpoint (vLLM, MLX, CSCS, HF ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load a local .env when running outside Docker
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _default_data_dir() -> Path:
    # src/smartstart/config.py -> track_2a/data/fhgr
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "data" / "fhgr"
        if cand.exists():
            return cand
    return Path("/app/data/fhgr")


@dataclass(frozen=True)
class Settings:
    llm_name: str = field(default_factory=lambda: os.getenv("LLM_NAME", "swiss-ai/Apertus-v1.5-8B"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://host.docker.internal:8080/v1").rstrip("/")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "not-needed"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "300")))
    llm_seed: int = field(default_factory=lambda: int(os.getenv("LLM_SEED", "42")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SMARTSTART_DATA_DIR", str(_default_data_dir()))))
    # "end": realistic interview, feedback at the end (default)
    # "per_answer": additionally a one-line coaching tip after each answer (no extra LLM call)
    feedback_mode: str = field(default_factory=lambda: os.getenv("SMARTSTART_FEEDBACK_MODE", "end"))
    max_followups_per_question: int = 1
    max_followups_total: int = 4
    n_exemplars: int = 3

    @property
    def mock(self) -> bool:
        return self.llm_base_url.lower() in {"mock", "mock://", "mock:"}


SETTINGS = Settings()
