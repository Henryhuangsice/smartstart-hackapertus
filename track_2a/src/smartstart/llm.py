"""Minimal OpenAI-compatible chat client for Apertus (vLLM / MLX / CSCS / HF).

Every call is counted so we can prove the "< 5 LLM calls per candidate
answer" efficiency gate. A deterministic mock backend (LLM_BASE_URL=mock)
is provided for tests and smoke runs without a GPU.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import urllib.error
import urllib.request

from .config import SETTINGS

_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    pass


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
    tag: str = "",
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Send a chat completion request and return the assistant text.

    base_url / model / api_key override the Apertus defaults (used for an optional
    separate judge model during evaluation)."""
    base = (base_url or SETTINGS.llm_base_url).rstrip("/")
    if base.lower().startswith("mock"):
        return _mock(messages, tag)
    payload: dict[str, Any] = {
        "model": model or SETTINGS.llm_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": SETTINGS.llm_seed,
    }
    headers = {"Authorization": f"Bearer {api_key or SETTINGS.llm_api_key}"}
    url = f"{base}/chat/completions"
    last: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={**headers, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=SETTINGS.llm_timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"] or ""
            return _THINK.sub("", text).strip()
        except urllib.error.HTTPError as e:  # keep the server's error message
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:800]
            except Exception:
                pass
            last = f"{e} {body}"
            if 400 <= e.code < 500:
                break  # client errors will not succeed on retry
        except Exception as e:  # network hiccups, 5xx, ...
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"LLM call failed ({tag}): {last}")


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response (robust to prose / code fences)."""
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    start = t.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start : i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    # tolerate trailing commas / single quotes
                    fixed = re.sub(r",\s*([}\]])", r"\1", chunk)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        return {}
    return {}


# ----------------------------------------------------------------------
# Mock backend — deterministic, language-aware, used by tests / CI.
# ----------------------------------------------------------------------
def _mock(messages: list[dict[str, str]], tag: str) -> str:
    user = messages[-1]["content"]
    if tag == "assess":
        n = len(user.split("CANDIDATE ANSWER:", 1)[-1].split("\n", 1)[0].split())
        level = 1 if n < 6 else 2 if n < 15 else 3 if n < 40 else 4
        return json.dumps(
            {
                "quality_level": level,
                "scores": {"clarity": min(4, level + 1), "relevance": level},
                "evidence": "…",
                "strength": "mock strength",
                "improvement": "mock improvement",
                "coach_action": "probe" if level <= 2 else "continue",
                "problem_flags": ["one_word"] if n < 3 else [],
                "follow_up": "Kannst du mir ein Beispiel geben?" if level <= 2 else "",
                "candidate_question": "",
                "tip": "Nenne ein konkretes Beispiel.",
            }
        )
    if tag == "feedback" or tag == "rewrite":
        return (
            "## Deine Stärken\nDu hast gesagt: «ich arbeite gern genau». Das passt gut zum Beruf.\n\n"
            "## Daran kannst du arbeiten\nBei «weiss nicht» fehlt noch ein Beispiel. "
            "Erzähl beim nächsten Mal von deiner Schnupperlehre.\n\n"
            "## Dein nächster Schritt\nSchreib dir zwei Beispiele auf.\n\n"
            "Das war eine gute Übung – bleib dran!"
        )
    if tag == "candidate":
        return "Ich habe in der Schnupperlehre viel gelernt, zum Beispiel wie man genau misst."
    if tag == "judge":
        return json.dumps({"scores": {}, "criteria_met": [], "comment": "mock"})
    return "Danke. " + (user.split("NEXT QUESTION:", 1)[-1].strip().splitlines() or [""])[0]
