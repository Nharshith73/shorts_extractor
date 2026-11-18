"""Live Twitch & Kick stream watcher & autonomous short clip generator.

Listens to Twitch or Kick chat anonymously (zero tokens required), records the
live stream with Streamlink, detects rolling median message spikes, snaps speech
boundaries to prevent abrupt cuts, optionally evaluates with AI, and renders
ready-to-publish 9:16 vertical Shorts with styled subtitles.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any

from app.clips.render import EncoderSelector
from app.detection.chat_spike import ChatSpikeDetector, ChatSpikeEvent
from app.live.kick_chat import KickChatClient
from app.live.twitch_irc import ChatMessage, TwitchIRCClient
from app.live.twitch_streamlink import StreamlinkRecorder
from app.rendering.config import RenderConfig
from app.rendering.renderer import extract_audio, probe_media, render_original
from app.rendering.transcription import WhisperTranscriber, Word, usable_words
from app.video.boundaries import snap_to_speech_boundaries

LOGGER = logging.getLogger(__name__)


class LiveWatcherSession:
    """Manages active live stream recording and real-time spike clipping for Twitch or Kick."""

    def __init__(
        self,
        channel: str,
        output_dir: Path,
        platform: str = "twitch",
        min_messages: int = 5,
        spike_factor: float = 2.5,
        ai_mode: str = "groq",
        layout: str = "blurred_background",
        encoder: str = "auto",
        simulated_video: Path | None = None,
    ) -> None:
        raw_chan = channel.strip()
        self.platform = platform.lower().strip()
        if "kick.com" in raw_chan.lower():
            self.platform = "kick"

        if self.platform == "kick":
            clean_chan = re.sub(r"^https?://(www\.)?kick\.com/", "", raw_chan).strip("/").lstrip("#")
        else:
            clean_chan = re.sub(r"^https?://(www\.)?twitch\.tv/", "", raw_chan).strip("/").lstrip("#")

        self.channel = clean_chan.lower()
        self.output_dir = output_dir.resolve()
        self.min_messages = min_messages
        self.spike_factor = spike_factor
        self.ai_mode = ai_mode.lower().strip()
        self.layout = layout
        self.encoder = encoder
        self.simulated_video = simulated_video.resolve() if simulated_video else None

        prefix = "kick_" if self.platform == "kick" else ""
        self.session_id = hashlib.sha256(f"{self.platform}_{self.channel}_{time.time()}".encode()).hexdigest()[:16]
        self.run_dir = self.output_dir / f"{prefix}live_{self.channel}" / "runs" / self.session_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.shorts_dir = self.run_dir / "shorts"
        self.shorts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.run_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self._chat_client: TwitchIRCClient | KickChatClient | None = None
        self._recorder: StreamlinkRecorder | None = None
        self._spike_detector = ChatSpikeDetector(
            bucket_seconds=10.0,
            history_minutes=3.0,
            spike_factor=self.spike_factor,
            min_messages=self.min_messages,
            cooldown_seconds=35.0,
            lead_in_seconds=20.0,
            lookahead_seconds=15.0,
        )

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._transcriber: WhisperTranscriber | None = None
        self._encoder_selector = EncoderSelector(requested=self.encoder)
        self._generated_shorts: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._last_error: str | None = None

    @property
    def is_running(self) -> bool:
        return bool(self._worker_thread and self._worker_thread.is_alive() and not self._stop_event.is_set())

    def start(self) -> None:
        """Start chat listener, video recorder, and processing thread."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._started_at = time.time()

        # 1. Start Chat Client (Twitch IRC or Kick Centrifugo)
        if self.platform == "kick":
            self._chat_client = KickChatClient(
                channel=self.channel,
                on_message_callback=self._on_chat_message,
            )
        else:
            self._chat_client = TwitchIRCClient(
                channel=self.channel,
                on_message_callback=self._on_chat_message,
            )
        self._chat_client.start()

        # 2. Start Video Recorder (or use simulated video)
        if not self.simulated_video:
            stream_url = f"https://kick.com/{self.channel}" if self.platform == "kick" else f"https://twitch.tv/{self.channel}"
            self._recorder = StreamlinkRecorder(
                stream_url=stream_url,
                output_dir=self.run_dir / "recording",
                platform=self.platform,
            )
            try:
                self._recorder.start()
            except Exception as exc:
                LOGGER.warning("Could not start live streamlink recorder: %s", exc)

        # 3. Start supervisor thread
        self._worker_thread = threading.Thread(
            target=self._supervise_loop,
            name=f"watcher-{self.platform}-{self.channel}",
            daemon=True,
        )
        self._worker_thread.start()
        LOGGER.info("LiveWatcherSession active for %s:%s (session=%s)", self.platform, self.channel, self.session_id)

    def stop(self) -> None:
        """Stop all processes gracefully."""
        self._stop_event.set()
        if self._chat_client:
            self._chat_client.stop()
            self._chat_client = None

        if self._recorder:
            self._recorder.stop()
            self._recorder = None

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None

        LOGGER.info("LiveWatcherSession stopped for %s:%s", self.platform, self.channel)

    def get_status(self) -> dict[str, Any]:
        """Telemetry and metrics for web dashboard."""
        elapsed = round(time.time() - self._started_at, 1) if self._started_at else 0.0
        total_msgs = self._chat_client.total_messages_received if self._chat_client else 0
        chat_connected = self._chat_client.is_connected if self._chat_client else False
        rec_status = self._recorder.poll_status() if self._recorder else {"alive": bool(self.simulated_video)}

        with self._lock:
            recent_shorts = list(self._generated_shorts)
            spikes_count = len(self._spike_detector.detected_spikes)
            current_median = self._spike_detector.current_median
            bucket_count = self._spike_detector.current_bucket_count

        return {
            "is_running": self.is_running,
            "platform": self.platform,
            "channel": self.channel,
            "session_id": self.session_id,
            "elapsed_seconds": elapsed,
            "irc_connected": chat_connected,
            "chat_connected": chat_connected,
            "total_messages": total_msgs,
            "current_bucket_count": bucket_count,
            "current_median": round(current_median, 2),
            "spikes_detected": spikes_count,
            "shorts_count": len(recent_shorts),
            "generated_shorts": recent_shorts,
            "recorder": rec_status,
            "error": self._last_error or rec_status.get("error"),
        }

    def _on_chat_message(self, msg: ChatMessage) -> None:
        """Dispatched on every incoming Twitch message."""
        stream_elapsed = self._recorder.elapsed_seconds if self._recorder else (time.time() - (self._started_at or time.time()))
        spike = self._spike_detector.add_message(stream_elapsed)
        if spike:
            LOGGER.info(
                "CHAT SPIKE DETECTED at stream %.2fs! Messages=%d (%.2fx baseline median %.2f)",
                spike.timestamp,
                spike.message_count,
                spike.spike_ratio,
                spike.baseline_median,
            )
            # Dispatch clip generation in background worker
            threading.Thread(
                target=self._process_spike_moment,
                args=(spike,),
                daemon=True,
            ).start()

    def _supervise_loop(self) -> None:
        """Background health check and periodic flush."""
        while not self._stop_event.is_set():
            time.sleep(2.0)
            if self._recorder and not self._recorder.is_recording and not self._stop_event.is_set():
                status = self._recorder.poll_status()
                if status.get("error"):
                    self._last_error = status["error"]

    def _process_spike_moment(self, spike: ChatSpikeEvent) -> None:
        """Extract, snap speech boundaries, evaluate with AI, and render 9:16 Short."""
        try:
            # 1. Ensure lookahead video is recorded to disk
            wait_for_lookahead = max(0.0, (spike.context_end - (self._recorder.elapsed_seconds if self._recorder else 9999.0)) + 3.0)
            if wait_for_lookahead > 0 and wait_for_lookahead < 25.0:
                time.sleep(wait_for_lookahead)

            clip_idx = len(self._generated_shorts) + 1
            clip_id = f"spike_{clip_idx:02d}_{int(spike.timestamp)}s"
            work_dir = self.run_dir / "work" / clip_id
            work_dir.mkdir(parents=True, exist_ok=True)

            source_media = self.simulated_video or (self._recorder.output_file if self._recorder else None)
            if not source_media or not source_media.is_file():
                LOGGER.warning("No recorded video file available to extract clip %s", clip_id)
                return

            # 2. Extract raw candidate window
            raw_clip = work_dir / "raw.mp4"
            if self._recorder:
                self._recorder.extract_clip(spike.context_start, spike.context_end, raw_clip)
            else:
                # Simulated video seek
                self._extract_subclip_ffmpeg(source_media, spike.context_start, spike.context_end, raw_clip)

            if not raw_clip.is_file() or raw_clip.stat().st_size < 1000:
                LOGGER.warning("Extracted raw clip is empty: %s", raw_clip)
                return

            info = probe_media(raw_clip)

            # 3. Transcribe audio with Whisper
            audio_wav = work_dir / "audio.wav"
            extract_audio(raw_clip, audio_wav, 0.0, info.duration)

            render_cfg = RenderConfig(
                layout=self.layout,
                captions_enabled=True,
                crf=20,
                device="cuda" if self._has_cuda() else "cpu",
            )

            if self._transcriber is None:
                self._transcriber = WhisperTranscriber(render_cfg)

            transcript = self._transcriber.transcribe(audio_wav)
            segments = transcript.segments
            full_text = " ".join(s.text for s in segments).strip()
            raw_words, _ = usable_words(transcript, render_cfg, info.duration)

            # 4. Snap boundaries to avoid abrupt endings
            snapped_start = 0.0
            snapped_end = info.duration
            if segments:
                try:
                    s_s, s_e = snap_to_speech_boundaries(
                        raw_start=max(0.0, spike.lead_in_seconds - 15.0),
                        raw_end=info.duration,
                        segments=list(segments),
                        max_duration=info.duration,
                        reaction_padding=1.8,
                    )
                    snapped_start = s_s
                    snapped_end = s_e
                except Exception as exc:
                    LOGGER.debug("Boundary snapping fallback: %s", exc)

            final_duration = max(3.0, snapped_end - snapped_start)

            # Filter words for final window with relative offset
            valid_words = [
                Word(
                    start=max(0.0, w.start - snapped_start),
                    end=min(final_duration, w.end - snapped_start),
                    text=w.text,
                    probability=w.probability,
                )
                for w in raw_words
                if snapped_start <= w.start <= snapped_end
            ]

            # 5. AI Editorial Filter / Hook Generation (optional)
            hook_text = "WHAT JUST HAPPENED?!"
            editorial_rating = 4
            summary = f"Twitch chat surged {spike.spike_ratio:.1f}x baseline ({spike.message_count} messages in 10s)"

            if self.ai_mode in ("groq", "qwen"):
                ai_meta = self._run_ai_judgment(
                    summary_text=full_text,
                    spike=spike,
                    duration=final_duration,
                )
                if ai_meta:
                    hook_text = ai_meta.get("hook_suggestion", hook_text)
                    editorial_rating = ai_meta.get("editorial_rating", editorial_rating)
                    summary = ai_meta.get("summary", summary)

            # 6. Render final 9:16 vertical Short
            render_cfg = RenderConfig(
                layout=self.layout,
                captions_enabled=True,
                crf=20,
            )

            final_short_path = self.shorts_dir / f"{clip_id}.mp4"
            render_original(
                source=raw_clip,
                output=final_short_path,
                start=snapped_start,
                duration=final_duration,
                info=info,
                words=valid_words,
                config=render_cfg,
                work=work_dir,
                selector=self._encoder_selector,
            )

            # 7. Write metadata JSON
            meta = {
                "clip_id": clip_id,
                "title": f"#{self.channel.upper()} {summary[:50]}",
                "suggested_hook": hook_text,
                "description": summary,
                "start_seconds": round(spike.context_start + snapped_start, 2),
                "end_seconds": round(spike.context_start + snapped_end, 2),
                "duration": round(final_duration, 2),
                "spike_ratio": spike.spike_ratio,
                "message_count": spike.message_count,
                "editorial_rating": editorial_rating,
                "channel": self.channel,
                "path": str(final_short_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            (self.metadata_dir / f"{clip_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            with self._lock:
                self._generated_shorts.append(meta)

            LOGGER.info("RENDERED SHORT: %s (Duration: %.2fs, Hook: '%s')", final_short_path.name, final_duration, hook_text)

        except Exception as exc:
            LOGGER.error("Failed to process spike moment: %s", exc, exc_info=True)

    def _run_ai_judgment(self, summary_text: str, spike: ChatSpikeEvent, duration: float) -> dict[str, Any] | None:
        """Call Groq or Qwen for fast viral hook generation."""
        try:
            if self.ai_mode == "groq":
                from app.evaluation.providers.groq import GroqHighlightJudgeProvider
                provider = GroqHighlightJudgeProvider()
                if provider.has_valid_key():
                    prompt = (
                        f"Streamer #{self.channel} had a major Twitch chat reaction spike ({spike.message_count} msgs in 10s, {spike.spike_ratio:.1f}x baseline).\n"
                        f"Spoken dialogue: '{summary_text}'\n"
                        "Generate a viral YouTube Shorts on-screen caption hook (under 6 words, punchy, curiosity-inducing) and 1-sentence summary.\n"
                        'Return strict JSON: {"hook_suggestion": "...", "summary": "...", "editorial_rating": 4}'
                    )
                    raw, _ = provider._call_completion(
                        messages=[{"role": "user", "content": prompt}],
                        timeout_seconds=15.0,
                    )
                    return json.loads(raw)
        except Exception as exc:
            LOGGER.debug("Live AI judgment fallback: %s", exc)
        return None

    def trigger_manual_clip(self, clip_duration: float = 30.0) -> dict[str, Any]:
        """Trigger an immediate 30-second reel cut from the live stream recording."""
        if not self.is_running:
            raise RuntimeError("Live watcher is not actively running.")

        curr_time = 0.0
        if self._recorder and self._recorder.output_file.is_file():
            curr_time = self._recorder.get_duration()
        elif self.simulated_video and self.simulated_video.is_file():
            curr_time = probe_media(self.simulated_video).duration
        else:
            curr_time = round(time.time() - (self._started_at or time.time()), 2)

        if curr_time < 2.0:
            raise RuntimeError(f"Stream recording buffer is only {curr_time:.1f}s. Please wait a few seconds before testing.")

        start_time = max(0.0, curr_time - clip_duration)
        end_time = curr_time

        spike = ChatSpikeEvent(
            timestamp=curr_time,
            message_count=99,
            baseline_median=10.0,
            spike_ratio=9.9,
            lead_in_seconds=curr_time - start_time,
            lookahead_seconds=0.0,
            context_start=start_time,
            context_end=end_time,
        )

        # Trigger processing in background thread
        thread = threading.Thread(
            target=self._process_spike_moment,
            args=(spike,),
            name=f"manual-clip-{int(curr_time)}",
            daemon=True,
        )
        thread.start()

        return {
            "status": "TRIGGERED",
            "start_seconds": round(start_time, 2),
            "end_seconds": round(end_time, 2),
            "duration": round(end_time - start_time, 2),
            "channel": self.channel,
            "platform": self.platform,
        }

    def _extract_subclip_ffmpeg(self, source: Path, start: float, end: float, out_path: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(source),
            "-to", f"{end:.2f}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-c:a", "aac", str(out_path),
        ]
        import subprocess
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False


def create_test_short(
    output_dir: Path,
    layout: str = "blurred_background",
    encoder: str = "auto",
    duration_seconds: float = 30.0,
) -> dict[str, Any]:
    """Generate and render a 30s test vertical Short into completed output."""
    output_dir = Path(output_dir).resolve()
    session_id = f"test_{int(time.time())}"
    run_dir = output_dir / "test_live" / "runs" / session_id
    shorts_dir = run_dir / "shorts"
    meta_dir = run_dir / "metadata"
    work_dir = run_dir / "work"
    for d in (shorts_dir, meta_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Fast synthesis of 30s test footage with audio
    source_mp4 = work_dir / "test_source.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={duration_seconds:.1f}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration_seconds:.1f}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-t", f"{duration_seconds:.1f}",
        str(source_mp4),
    ]
    import subprocess
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)

    info = probe_media(source_mp4)
    selector = EncoderSelector(requested=encoder)
    render_cfg = RenderConfig(layout=layout, captions_enabled=True, crf=20)

    # 2. Add sample viral subtitles
    words = [
        Word(start=1.0, end=4.5, text="LIVE STREAMER MOMENT!", probability=1.0),
        Word(start=5.0, end=9.5, text="Chat is going crazy!", probability=1.0),
        Word(start=10.0, end=15.0, text="30-Second Reel Cut", probability=1.0),
        Word(start=15.5, end=20.0, text="Spike Detection Active", probability=1.0),
        Word(start=20.5, end=28.0, text="Ready for YouTube Shorts", probability=1.0),
    ]

    out_mp4 = shorts_dir / f"test_clip_{int(duration_seconds)}s.mp4"
    render_original(
        source=source_mp4,
        output=out_mp4,
        start=0.0,
        duration=duration_seconds,
        info=info,
        words=words,
        config=render_cfg,
        work=work_dir,
        selector=selector,
    )

    meta = {
        "clip_id": out_mp4.stem,
        "title": "⚡ Live Watcher 30s Test Reel",
        "suggested_hook": "WHAT A MOMENT! 😱",
        "description": "30-second test reel cut directly from Live Watcher pipeline with 9:16 layout and subtitles.",
        "start_seconds": 0.0,
        "end_seconds": duration_seconds,
        "duration": duration_seconds,
        "spike_ratio": 5.0,
        "message_count": 50,
        "editorial_rating": 5,
        "channel": "test_live",
        "path": str(out_mp4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (meta_dir / f"{out_mp4.stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Twitch & Kick Live Stream Chat Spike Watcher & 9:16 Shorts Generator")
    parser.add_argument("channel", help="Channel username or URL (e.g. kaicenat, https://kick.com/westcol)")
    parser.add_argument("--platform", choices=["twitch", "kick"], default="twitch", help="Streaming platform (default: twitch)")
    parser.add_argument("--kick", action="store_true", help="Watch Kick stream (shorthand for --platform kick)")
    parser.add_argument("--min", dest="min_messages", type=int, default=5, help="Minimum messages for spike (default: 5)")
    parser.add_argument("--spike-factor", type=float, default=2.5, help="Spike factor above 3-min median (default: 2.5)")
    parser.add_argument("--ai", choices=["groq", "qwen", "none"], default="groq", help="AI evaluation mode")
    parser.add_argument("--layout", choices=["blurred_background", "center_crop"], default="blurred_background")
    parser.add_argument("--encoder", choices=["auto", "h264_nvenc", "libx264"], default="auto")
    parser.add_argument("--simulated-video", type=Path, default=None, help="Simulate live stream using local video file")
    parser.add_argument("--out", type=Path, default=Path("data/completed"), help="Completed shorts output directory")
    args = parser.parse_args()

    platform = "kick" if (args.kick or "kick.com" in args.channel.lower()) else args.platform

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    session = LiveWatcherSession(
        channel=args.channel,
        output_dir=args.out,
        platform=platform,
        min_messages=args.min_messages,
        spike_factor=args.spike_factor,
        ai_mode=args.ai,
        layout=args.layout,
        encoder=args.encoder,
        simulated_video=args.simulated_video,
    )
    session.start()

    chat_desc = "Centrifugo WebSocket (Guest, no token)" if platform == "kick" else "irc.chat.twitch.tv:6667 (NICK justinfan, no token)"

    print(f"\n=======================================================")
    print(f" Live Stream Watcher Active: [{platform.upper()}] {session.channel}")
    print(f" Chat: {chat_desc}")
    print(f" Spike Trigger: >= {args.spike_factor}x 3-min median AND >= {args.min_messages} msgs/10s")
    print(f" Layout: {args.layout} | Subtitles: faster-whisper (GPU)")
    print(f" Output: {session.shorts_dir}")
    print(f" Press Ctrl+C to stop watching")
    print(f"=======================================================\n")

    try:
        while True:
            time.sleep(2.0)
            status = session.get_status()
            print(
                f"\r[{platform.upper()}:{session.channel}] Elapsed: {status['elapsed_seconds']}s | "
                f"Chat Msgs: {status['total_messages']} | "
                f"Current 10s Bucket: {status['current_bucket_count']} | "
                f"Median: {status['current_median']} | "
                f"Spikes: {status['spikes_detected']} | "
                f"Shorts: {status['shorts_count']}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopping Live Watcher Session...")
        session.stop()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
