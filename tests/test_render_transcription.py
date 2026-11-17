import os
from pathlib import Path

import pytest

from app.rendering.config import RenderConfig
from app.rendering.transcription import WhisperTranscriber, usable_words

SAMPLE = os.environ.get("STREAMCLIP_SPEECH_SAMPLE")


@pytest.mark.integration
@pytest.mark.skipif(not SAMPLE, reason="Set STREAMCLIP_SPEECH_SAMPLE to a short speech WAV to run real faster-whisper transcription")
def test_real_transcription_from_opt_in_speech_sample() -> None:
    import wave
    path = Path(SAMPLE)
    assert path.is_file(), f"STREAMCLIP_SPEECH_SAMPLE does not exist: {path}"
    with wave.open(str(path)) as audio:
        duration = audio.getnframes() / audio.getframerate()
    config = RenderConfig()
    result = WhisperTranscriber(config).transcribe(path)
    words, _ = usable_words(result, config, duration)
    assert words, "speech sample produced no usable word-timed speech"
    assert all(0 <= word.start < word.end <= duration for word in words)
