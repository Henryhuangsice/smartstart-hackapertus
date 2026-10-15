"""Shared helpers for the evaluation scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartstart.config import SETTINGS  # noqa: E402

JUDGE = {
    "base_url": os.getenv("JUDGE_BASE_URL") or SETTINGS.llm_base_url,
    "model": os.getenv("JUDGE_NAME") or SETTINGS.llm_name,
    "api_key": os.getenv("JUDGE_API_KEY") or SETTINGS.llm_api_key,
}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lang_group(answer: dict) -> str:
    return "gsw" if answer.get("lang") == "gsw" else answer.get("lang", "de")
