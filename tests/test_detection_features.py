from __future__ import annotations

import numpy as np

from app.detection.features import audio_features_by_bin, bin_layout, video_features_by_bin


def test_partial_last_bin_is_retained_and_has_features() -> None:
    layout = bin_layout(6.25)
    assert len(layout) == 7
    assert layout[-1] == (6.0, 0.25)

    samples = np.ones(16_000 * 7, dtype=np.float32) * 0.1
    audio = audio_features_by_bin(samples, layout, 16_000)
    assert len(audio) == 7
    assert audio[-1]["audio_rms_db"] > -30

    frames = np.zeros((25, 90, 160), dtype=np.uint8)
    frames[-1] = 255
    video = video_features_by_bin(frames, layout, 4.0)
    assert len(video) == 7
    assert video[-1]["motion"] > 0
