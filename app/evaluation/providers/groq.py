"""Groq Cloud multimodal and LLM inference provider for Stage 4B AI Highlight Judge.

Provides zero-cost, ultra-high-speed editorial evaluation using Groq's free API tier:
- Llama 3.3 70B (llama-3.3-70b-versatile) for deep reasoning & humor detection
- Llama 3.2 Vision (llama-3.2-11b-vision-preview) for multimodal frame evaluation
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence
import urllib.parse
import winreg

import httpx

from app.evaluation.inference import HighlightJudgment
from app.evaluation.models import CandidateContext
from app.evaluation.schema import AIJudgmentRecord

LOGGER = logging.getLogger(__name__)

DEFAULT_GROQ_ENDPOINT = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_VISION_MODEL = "openai/gpt-oss-20b"


def parse_header_time(val: Any) -> float:
    """Parse time expression from Groq rate limit headers (e.g. '5.437s', '832.5ms', '1m12s')."""
    if not val:
        return 0.0
    text = str(val).strip()
    m_ms = re.search(r"(\d+(?:\.\d+)?)\s*ms", text, re.I)
    if m_ms:
        return float(m_ms.group(1)) / 1000.0
    m_s = re.search(
        r"(?:(\d+(?:\.\d+)?)\s*m(?:in)?(?![a-z]))?\s*(?:(\d+(?:\.\d+)?)\s*s(?:ec)?)?",
        text,
        re.I,
    )
    if m_s:
        mins = float(m_s.group(1)) if m_s.group(1) else 0.0
        secs = float(m_s.group(2)) if m_s.group(2) else 0.0
        total = mins * 60.0 + secs
        if total > 0:
            return total
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_wait_seconds(body_text: str) -> float:
    """Parse retry wait seconds from Groq 429 error messages."""
    if not body_text:
        return 2.0
    # 1. Direct ms check
    m_ms = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*ms", body_text, re.I)
    if m_ms:
        return float(m_ms.group(1)) / 1000.0
    # 2. Minutes and seconds check
    m_s = re.search(
        r"try again in\s+(?:(\d+(?:\.\d+)?)\s*m(?:in)?(?![a-z]))?\s*(?:(\d+(?:\.\d+)?)\s*s(?:ec)?)?",
        body_text,
        re.I,
    )
    if m_s:
        mins = float(m_s.group(1)) if m_s.group(1) else 0.0
        secs = float(m_s.group(2)) if m_s.group(2) else 0.0
        total = mins * 60.0 + secs
        if total > 0:
            return total
    return 2.0


def resolve_groq_api_key(explicit_key: str | None = None) -> str | None:
    """Resolve Groq API key from explicit argument, environment variable, or Windows registry."""
    if explicit_key is not None:
        return explicit_key.strip() if explicit_key.strip() else None

    env_key = os.environ.get("GROQ_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # Fallback to Windows HKCU User Environment if on Windows
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            val, _ = winreg.QueryValueEx(key, "GROQ_API_KEY")
            if val and str(val).strip():
                return str(val).strip()
    except Exception:
        pass

    return None


@dataclass
class GroqProviderConfig:
    api_key: str | None = None
    endpoint_url: str = DEFAULT_GROQ_ENDPOINT
    model_name: str = DEFAULT_GROQ_MODEL
    vision_model_name: str = DEFAULT_GROQ_VISION_MODEL
    temperature: float = 0.1
    max_output_tokens: int = 2048
    timeout_seconds: float = 45.0
    max_frames: int = 4
    max_retries: int = 3
    use_vision: bool = False
    inter_request_delay_seconds: float = 0.5


class GroqHighlightJudgeProvider:
    """Editorial judge powered by Groq's high-speed free LPU inference engine."""

    def __init__(self, config: GroqProviderConfig | None = None) -> None:
        self.config = config or GroqProviderConfig()
        resolved_key = resolve_groq_api_key(self.config.api_key)
        self._api_key = resolved_key
        self._http_client = httpx.Client(timeout=self.config.timeout_seconds)

    @property
    def model_name(self) -> str:
        if self.config.use_vision:
            return self.config.vision_model_name
        return self.config.model_name

    def has_valid_key(self) -> bool:
        return bool(self._api_key and len(self._api_key) > 10)

    def evaluate_candidate(
        self, context: CandidateContext, timeout_seconds: float = 45.0
    ) -> HighlightJudgment:
        record = self.judge_candidate_record(context, timeout_seconds=timeout_seconds)
        ent = float(record.editorial_rating) * 2.0
        return HighlightJudgment(
            window_id=context.candidate_id,
            decision="APPROVE" if record.editorial_rating >= 3 else "REJECT",
            confidence=record.confidence,
            editorial_scores={
                "entertainment_value": ent,
                "hook_strength": ent,
                "narrative_coherence": 8.0 if record.context_complete else 4.0,
                "visual_clarity": 8.0,
                "pacing": 8.0 if (record.setup_present and record.payoff_present) else 5.0,
            },
            primary_category=record.event_category,
            reasoning=record.summary,
            suggested_title=record.summary,
            hook_text=record.hook_suggestion,
            recommended_layout=record.recommended_layout,
            suggested_bounds={
                "recommended_start_seconds": record.recommended_start_seconds,
                "recommended_end_seconds": record.recommended_end_seconds,
            },
            model_metadata=record.model_metadata,
            is_mock=False,
        )

    def judge_candidate(
        self, context: CandidateContext, timeout_seconds: float = 45.0
    ) -> HighlightJudgment:
        return self.evaluate_candidate(context, timeout_seconds=timeout_seconds)

    def judge_candidate_record(
        self, context: CandidateContext, timeout_seconds: float = 45.0
    ) -> AIJudgmentRecord:
        if not self.has_valid_key():
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free API key at https://console.groq.com/keys "
                "and set it with: setx GROQ_API_KEY 'your-key' or pass it in configuration."
            )

        start_time = time.perf_counter()
        messages = self._build_prompt_messages(context)

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw_json, telemetry = self._call_completion(messages, timeout_seconds=timeout_seconds)

                text_to_parse = raw_json.strip()
                if text_to_parse.startswith("```"):
                    lines = text_to_parse.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    text_to_parse = "\n".join(lines).strip()

                parsed_data = json.loads(text_to_parse)
                parsed_data["candidate_id"] = context.candidate_id
                telemetry["attempt"] = attempt
                telemetry["total_latency_seconds"] = round(time.perf_counter() - start_time, 3)
                telemetry["model"] = self.model_name
                telemetry["provider"] = "groq"
                parsed_data["model_metadata"] = telemetry

                return AIJudgmentRecord.model_validate(parsed_data)
            except Exception as exc:
                LOGGER.warning(
                    "Groq inference attempt %d failed for candidate %s: %s",
                    attempt,
                    context.candidate_id,
                    exc,
                )
                last_error = exc
                if attempt < self.config.max_retries:
                    # Clean re-prompt without unbounded token growth
                    messages = self._build_prompt_messages(context)
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Schema error on previous attempt: {exc}. Return ONLY the strict JSON object matching the required schema.",
                        }
                    )
                    time.sleep(1.0)

        raise RuntimeError(
            f"Failed to generate valid AIJudgmentRecord from Groq for {context.candidate_id}: {last_error}"
        )

    def _call_completion(
        self, messages: list[dict[str, Any]], timeout_seconds: float
    ) -> tuple[str, dict[str, Any]]:
        url = f"{self.config.endpoint_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        max_rate_limit_retries = 8
        for rl_attempt in range(1, max_rate_limit_retries + 1):
            call_start = time.perf_counter()
            response = self._http_client.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            latency = round(time.perf_counter() - call_start, 3)

            if response.status_code == 429:
                wait_time = 2.0
                retry_header = response.headers.get("retry-after")
                if retry_header:
                    wait_time = parse_header_time(retry_header)
                elif "x-ratelimit-reset-tokens" in response.headers:
                    wait_time = parse_header_time(response.headers["x-ratelimit-reset-tokens"])
                else:
                    wait_time = parse_wait_seconds(response.text)

                wait_time = max(wait_time, 0.5) + 0.5  # Add 500ms safety cushion
                LOGGER.warning(
                    "Groq TPM/RPM rate limit encountered (429). Backing off for %.2fs (attempt %d/%d)...",
                    wait_time,
                    rl_attempt,
                    max_rate_limit_retries,
                )
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:300]}")

            resp_data = response.json()
            choices = resp_data.get("choices", [])
            if not choices:
                raise ValueError(f"No choices returned by Groq: {resp_data}")

            content = choices[0].get("message", {}).get("content", "")
            usage = resp_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            telemetry = {
                "latency_seconds": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_second": round(completion_tokens / max(latency, 0.001), 2),
            }

            # Proactive token pacing: check remaining tokens in current window
            rem_tokens_str = response.headers.get("x-ratelimit-remaining-tokens")
            if rem_tokens_str and rem_tokens_str.isdigit():
                rem_tokens = int(rem_tokens_str)
                if rem_tokens < 1500:
                    reset_tokens_str = response.headers.get("x-ratelimit-reset-tokens")
                    reset_time = parse_header_time(reset_tokens_str) if reset_tokens_str else 2.0
                    if 0 < reset_time <= 10.0:
                        LOGGER.info(
                            "Groq remaining tokens low (%d < 1500). Proactively pausing %.2fs for TPM window reset...",
                            rem_tokens,
                            reset_time + 0.5,
                        )
                        time.sleep(reset_time + 0.5)

            return content, telemetry

        raise RuntimeError(f"Exceeded {max_rate_limit_retries} retries due to persistent Groq rate limits (429).")

    def _build_prompt_messages(self, context: CandidateContext) -> list[dict[str, Any]]:
        system_prompt = (
            "You are an expert viral content editor evaluating candidate moments for YouTube Shorts & Instagram Reels from gaming streams.\n"
            "Your goal is pinpoint accuracy: identifying hilarious player banter, funny NPC encounters, chaotic gameplay fails, and authentic reactions.\n\n"
            "EDITORIAL SCORING RULES:\n"
            "1. COMEDIC BANTER & SURPRISE (Rating 3 to 5): Streamer teasing, funny character appearances (e.g. guy with suspenders, weird NPCs), loud laughter, unexpected physics/fails, frantic shouting, or intense action. Rate these 3, 4, or 5 stars.\n"
            "2. MUNDANE / BORING (Rating 0 to 1): Silent driving, routine navigation with no jokes or banter, ordinary transit with no action.\n"
            "3. Ground all judgments in observable evidence: visuals, spoken dialogue, and audio energy. Never hallucinate events.\n"
            "4. BOUNDARY TIMESTAMPS:\n"
            "   - Inspect the speech transcript timestamps below.\n"
            "   - Set recommended_start_seconds to the exact beginning of the spoken thought or joke (never cut mid-sentence).\n"
            "   - Set recommended_end_seconds to the clean end of the punchline, laughter, or exclamation.\n\n"
            "Output MUST be a strictly valid JSON object matching this schema:\n"
            "{\n"
            '  "schema_version": "1.0",\n'
            f'  "candidate_id": "{context.candidate_id}",\n'
            '  "event_category": "<funny|unexpected|impressive|dramatic|chaotic|skillful|interaction|fail|story_moment|ordinary|unknown>",\n'
            '  "summary": "<1-2 sentence description of what happens>",\n'
            '  "editorial_rating": <integer 0 to 5>,\n'
            '  "confidence": <float 0.0 to 1.0 representing certainty>,\n'
            '  "context_complete": <true/false>,\n'
            '  "setup_present": <true/false>,\n'
            '  "payoff_present": <true/false>,\n'
            '  "entertainment_factors": ["<factor1>", "<factor2>"],\n'
            '  "evidence": [\n'
            '    {"timestamp": <float>, "source": "<visual|transcript|chat|audio|inference>", "observation": "<description>"}\n'
            '  ],\n'
            '  "hook_suggestion": "<punchy on-screen caption hook>",\n'
            f'  "recommended_start_seconds": {context.start_seconds:.2f},\n'
            f'  "recommended_end_seconds": {context.end_seconds:.2f},\n'
            '  "recommended_layout": "blurred_background",\n'
            '  "rejection_reason": null\n'
            "}"
        )

        user_content: list[dict[str, Any]] = []

        valid_segments = [
            s for s in context.speech.segments
            if s.avg_logprob >= -1.2 and s.no_speech_prob <= 0.65
        ] if context.speech.has_speech else []
        transcript_lines = []
        for s in valid_segments:
            transcript_lines.append(f"  [{s.start:.2f}s - {s.end:.2f}s]: \"{s.text}\"")
        transcript_text = "\n".join(transcript_lines) if transcript_lines else "(No dialogue detected)"
        signals_summary = ", ".join(f"{k}={v:.2f}" for k, v in context.components.items())

        text_intro = (
            f"EVALUATION REQUEST:\n"
            f"- Candidate ID: {context.candidate_id}\n"
            f"- Time Interval: {context.start_seconds:.2f}s to {context.end_seconds:.2f}s (Duration: {context.duration_seconds:.2f}s, Peak: {context.peak_seconds:.2f}s)\n"
            f"- Signal Features: {signals_summary} (Raw Score: {context.candidate_score:.2f})\n"
            f"- Speech Transcript with Timestamps:\n{transcript_text}\n"
        )

        if not self.config.use_vision:
            full_prompt = text_intro + "\n\nAnalyze the candidate moment above and return strictly a valid JSON object matching the schema."
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ]

        user_content.append({"type": "text", "text": text_intro})

        # Attach frames if using vision model
        if self.config.use_vision:
            frames_to_send = context.frames[: self.config.max_frames]
            for f in frames_to_send:
                f_path = Path(f.frame_path)
                if not f_path.is_file():
                    fname = f_path.name
                    search_candidates = [
                        Path("data/synthetic_vod/output/context") / context.candidate_id / "frames" / fname,
                        Path("artifacts/context") / context.candidate_id / "frames" / fname,
                    ]
                    search_candidates.extend(list(Path("data/processing").glob(f"*/context/{context.candidate_id}/frames/{fname}")))
                    search_candidates.extend(list(Path("data/completed").glob(f"*/runs/*/contexts/{context.candidate_id}/frames/{fname}")))
                    for sc in search_candidates:
                        if sc.is_file():
                            f_path = sc
                            break

                if f_path.is_file():
                    try:
                        b64_data = base64.b64encode(f_path.read_bytes()).decode("utf-8")
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                            }
                        )
                    except Exception as exc:
                        LOGGER.warning("Could not encode frame %s: %s", f_path, exc)

        user_content.append(
            {
                "type": "text",
                "text": "Analyze the moment above and provide your structured editorial judgment JSON.",
            }
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
