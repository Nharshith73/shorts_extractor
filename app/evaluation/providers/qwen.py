"""Qwen2.5-VL Multimodal Inference Provider via Standalone llama.cpp Server."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence
import urllib.parse

import httpx

from ..inference import HighlightJudgment
from ..models import CandidateContext
from ..schema import AIJudgmentRecord

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QwenProviderConfig:
    """Runtime configuration for Qwen2.5-VL local inference."""

    model_path: Path = Path("D:/models/qwen2.5-vl-7b-gguf/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
    projector_path: Path = Path("D:/models/qwen2.5-vl-7b-gguf/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf")
    server_binary: Path = Path("D:/tools/llama.cpp/llama-server.exe")
    endpoint_url: str = "http://127.0.0.1:8080/v1"
    model_name: str = "Qwen2.5-VL-7B-Instruct-Q4_K_M"
    n_gpu_layers: int = 99  # 99 = full GPU offloading; 0 = CPU fallback
    n_ctx: int = 6144
    max_output_tokens: int = 1024
    temperature: float = 0.1
    timeout_seconds: float = 60.0
    max_frames: int = 8
    max_retries: int = 2
    auto_start_server: bool = True
    server_host: str = "127.0.0.1"
    server_port: int = 8080


class QwenVisionLanguageProvider:
    """Real local multimodal Highlight Judge provider powered by Qwen2.5-VL-7B via llama.cpp."""

    def __init__(self, config: QwenProviderConfig | None = None) -> None:
        self.config = config or QwenProviderConfig()
        self._server_process: subprocess.Popen[str] | None = None
        self._http_client = httpx.Client(timeout=self.config.timeout_seconds)

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def is_server_running(self) -> bool:
        """Check if local llama-server is healthy and responsive."""
        base_url = self.config.endpoint_url.rstrip("/").rstrip("/v1")
        try:
            r = self._http_client.get(f"{base_url}/health", timeout=3.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass

        # Try /v1/models endpoint as alternative check
        try:
            r = self._http_client.get(f"{self.config.endpoint_url}/models", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def start_server(self, wait_timeout: float = 45.0) -> None:
        """Launch local llama-server as a persistent process and await readiness."""
        if self.is_server_running():
            LOGGER.info("llama-server is already running at %s", self.config.endpoint_url)
            return

        server_bin = self.config.server_binary.resolve()
        model_file = self.config.model_path.resolve()
        proj_file = self.config.projector_path.resolve()

        if not server_bin.is_file():
            raise FileNotFoundError(f"llama-server binary not found at {server_bin}")
        if not model_file.is_file():
            raise FileNotFoundError(f"Qwen model weights not found at {model_file}")
        if not proj_file.is_file():
            raise FileNotFoundError(f"Multimodal projector not found at {proj_file}")

        cmd = [
            str(server_bin),
            "-m",
            str(model_file),
            "--mmproj",
            str(proj_file),
            "-ngl",
            str(self.config.n_gpu_layers),
            "-c",
            str(self.config.n_ctx),
            "--host",
            self.config.server_host,
            "--port",
            str(self.config.server_port),
            "--log-disable",
        ]

        LOGGER.info(
            "Launching local llama-server: model=%s, mmproj=%s, ngl=%d, port=%d",
            model_file.name,
            proj_file.name,
            self.config.n_gpu_layers,
            self.config.server_port,
        )

        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # Poll until healthy or timed out
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            if self.is_server_running():
                LOGGER.info("llama-server successfully initialized and ready")
                return
            if self._server_process.poll() is not None:
                exit_code = self._server_process.poll()
                raise RuntimeError(
                    f"llama-server exited prematurely with return code {exit_code}"
                )
            time.sleep(1.0)

        raise TimeoutError(
            f"Timed out waiting for llama-server to become ready ({wait_timeout}s)"
        )

    def ensure_ready(self) -> None:
        """Ensure model server is responsive, launching if configured."""
        if not self.is_server_running():
            if self.config.auto_start_server:
                self.start_server()
            else:
                raise ConnectionError(
                    f"llama-server is not reachable at {self.config.endpoint_url}"
                )

    def close(self) -> None:
        """Terminate managed server process if started by this instance."""
        if self._server_process is not None:
            LOGGER.info("Terminating managed llama-server process")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None
        self._http_client.close()

    def __enter__(self) -> QwenVisionLanguageProvider:
        self.ensure_ready()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def evaluate_candidate(
        self, context: CandidateContext, timeout_seconds: float = 60.0
    ) -> HighlightJudgment:
        """Process CandidateContext through Qwen2.5-VL and emit validated HighlightJudgment."""
        self.ensure_ready()
        record = self.judge_candidate_record(context, timeout_seconds=timeout_seconds)

        # Convert AIJudgmentRecord to HighlightJudgment for pipeline backward compatibility
        return HighlightJudgment(
            window_id=record.candidate_id,
            decision="APPROVE" if record.is_approved else "REJECT",
            confidence=record.confidence,
            editorial_scores={
                "hook_strength": float(min(10.0, record.editorial_rating * 2.0)),
                "narrative_coherence": 8.0 if record.context_complete else 4.0,
                "entertainment_value": float(record.editorial_rating * 2.0),
                "visual_clarity": 8.0,
                "pacing": 7.5,
            },
            primary_category=record.event_category,
            reasoning=f"{record.summary} Evidence: {len(record.evidence)} items. Factors: {', '.join(record.entertainment_factors)}.",
            suggested_title=record.summary[:60],
            hook_text=record.hook_suggestion,
            recommended_layout=record.recommended_layout,
            suggested_bounds={
                "refined_start_seconds": record.recommended_start_seconds,
                "refined_end_seconds": record.recommended_end_seconds,
                "facecam_detected": False,
                "facecam_region": None,
            },
            model_metadata=record.model_metadata,
            is_mock=False,
        )

    def judge_candidate(
        self, context: CandidateContext, timeout_seconds: float = 60.0
    ) -> HighlightJudgment:
        return self.evaluate_candidate(context, timeout_seconds=timeout_seconds)

    def judge_candidate_record(
        self, context: CandidateContext, timeout_seconds: float = 60.0
    ) -> AIJudgmentRecord:
        """Process CandidateContext and emit strictly validated AIJudgmentRecord."""
        self.ensure_ready()
        start_time = time.perf_counter()
        messages = self._build_prompt_messages(context)

        # Bounded retry loop for schema resilience
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw_json, telemetry = self._call_completion(
                    messages, timeout_seconds=timeout_seconds
                )
                parsed_data = json.loads(raw_json)

                # Ensure candidate_id matches
                parsed_data["candidate_id"] = context.candidate_id

                # Attach runtime telemetry
                telemetry["attempt"] = attempt
                telemetry["total_latency_seconds"] = round(
                    time.perf_counter() - start_time, 3
                )
                telemetry["model"] = self.model_name
                telemetry["n_gpu_layers"] = self.config.n_gpu_layers
                parsed_data["model_metadata"] = telemetry

                # Strict Pydantic validation
                return AIJudgmentRecord.model_validate(parsed_data)

            except Exception as exc:
                LOGGER.warning(
                    "Inference attempt %d failed for candidate %s: %s",
                    attempt,
                    context.candidate_id,
                    exc,
                )
                last_error = exc
                if attempt < self.config.max_retries:
                    # Append corrective message for retry
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response had a schema error: {exc}. "
                                "Please return ONLY valid JSON matching the exact schema."
                            ),
                        }
                    )
                    time.sleep(1.0)

        raise RuntimeError(
            f"Failed to generate valid AIJudgmentRecord for candidate {context.candidate_id} "
            f"after {self.config.max_retries} attempts: {last_error}"
        )

    def _call_completion(
        self, messages: list[dict[str, Any]], timeout_seconds: float
    ) -> tuple[str, dict[str, Any]]:
        """Call OpenAI-compatible /v1/chat/completions endpoint on local server."""
        url = f"{self.config.endpoint_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        call_start = time.perf_counter()
        response = self._http_client.post(url, json=payload, timeout=timeout_seconds)
        latency = round(time.perf_counter() - call_start, 3)

        if response.status_code != 200:
            raise RuntimeError(
                f"Server error {response.status_code}: {response.text[:300]}"
            )

        resp_data = response.json()
        choices = resp_data.get("choices", [])
        if not choices:
            raise ValueError(f"No choices returned by model: {resp_data}")

        content = choices[0].get("message", {}).get("content", "")
        usage = resp_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        telemetry = {
            "latency_seconds": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": (
                round(completion_tokens / latency, 2) if latency > 0 else 0.0
            ),
        }
        return content, telemetry

    def _build_prompt_messages(
        self, context: CandidateContext
    ) -> list[dict[str, Any]]:
        """Construct multimodal messages array with embedded frame images and metadata."""
        system_prompt = (
            "You are StreamClip AI's expert gameplay highlight judge.\n"
            "Your task is to evaluate whether a gameplay clip has editorial value for short-form video (YouTube Shorts, TikTok).\n"
            "CRITICAL GUIDELINES:\n"
            "1. Ground all judgments in real evidence. Distinguish strictly between:\n"
            "   - 'visual': Observable in the supplied video frames.\n"
            "   - 'transcript': Spoken dialogue from the audio transcription.\n"
            "   - 'chat': Audience reactions and emote density from chat metrics.\n"
            "   - 'audio': Acoustic signals (explosions, volume spikes).\n"
            "   - 'inference': Your own deductive interpretation.\n"
            "2. Never fabricate dialogue, spoken words, or player speech not in the transcript.\n"
            "3. If nothing exciting occurs, assign a low rating (0-2) and state the rejection reason.\n"
            "4. Output MUST be a strictly valid JSON object matching this schema:\n"
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

        # 1. Textual contextual overview
        transcript_text = " ".join(s.text for s in context.speech.segments) if context.speech.has_speech else "(No dialogue detected)"
        chat_summary = (
            f"Chat present: {context.chat.message_count} messages, "
            f"{context.chat.unique_chatters} chatters, emote density: {context.chat.emote_density:.2f}"
            if context.chat.present
            else "(No chat data)"
        )
        signals_summary = ", ".join(f"{k}={v:.2f}" for k, v in context.components.items())

        text_intro = (
            f"EVALUATION REQUEST:\n"
            f"- Candidate ID: {context.candidate_id}\n"
            f"- Time Interval: {context.start_seconds:.2f}s to {context.end_seconds:.2f}s (Duration: {context.duration_seconds:.2f}s, Peak: {context.peak_seconds:.2f}s)\n"
            f"- Signal Detector Features: {signals_summary} (Raw Score: {context.candidate_score:.2f})\n"
            f"- Speech Transcript: {transcript_text}\n"
            f"- Twitch Chat Activity: {chat_summary}\n\n"
            f"Below are {min(len(context.frames), self.config.max_frames)} chronologically ordered visual frames sampled from the moment:"
        )
        user_content.append({"type": "text", "text": text_intro})

        # 2. Sampled visual frame attachments
        frames_to_send = context.frames[: self.config.max_frames]
        for f in frames_to_send:
            f_path = Path(f.frame_path)
            # Handle relative vs absolute frame paths
            if not f_path.is_file():
                search_candidates = [
                    Path("data/synthetic_vod/output/context") / context.candidate_id / "frames" / f_path.name,
                    Path("data/synthetic_vod/output/context") / context.candidate_id / f_path.name,
                    Path("artifacts/context") / context.candidate_id / "frames" / f_path.name,
                    Path("artifacts/context") / context.candidate_id / f_path.name,
                ]
                for sc in search_candidates:
                    if sc.is_file():
                        f_path = sc
                        break

            if not f_path.is_file():
                LOGGER.warning("Frame image not found: %s", f.frame_path)
                continue
            try:
                b64_data = base64.b64encode(f_path.read_bytes()).decode("utf-8")
                user_content.append(
                    {
                        "type": "text",
                        "text": f"[Frame at {f.timestamp:.2f}s (Role: {f.frame_type})]:",
                    }
                )
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_data}"
                        },
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
