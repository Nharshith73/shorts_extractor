from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

import pytest

from app.clips.config import CandidateConfig
from app.clips.windows import Event, Window, availability, event_window, merge_windows, trim_gaps
from app.clips.store import CandidateStore
from app.clips.candidate_extractor import CandidateExtractor
from app.ingestion.rolling_buffer import RollingBuffer, SessionBoundary, to_utc_text
from app.ingestion.stream_reader import StreamReader, StreamConfig

ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)


def at(seconds: float) -> datetime:
    return ORIGIN + timedelta(seconds=seconds)


def event(peak: float, event_id: int = 1, session: str = "a") -> Event:
    return Event(event_id, "source", session, at(peak), at(peak), at(peak + 1), 10)


def segment(buffer: RollingBuffer, start: float, duration: float = 6, session: str = "a") -> int:
    path = buffer.segments_dir / f"segment_{session}_{start}.ts"
    path.write_bytes(b"fake media")
    buffer.manifest.add_writing(path, at(start), "source", session)
    buffer.manifest.complete(path, duration)
    return buffer.manifest.get_by_path(path).id


def add_event(store: CandidateStore, value: Event) -> None:
    with store.connect() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, source_id TEXT,session_id TEXT,peak_wall_utc TEXT,
            start_wall_utc TEXT,end_wall_utc TEXT,score REAL)""")
        connection.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (value.id, value.source_id,
            value.session_id, to_utc_text(value.peak), to_utc_text(value.start), to_utc_text(value.end), value.score))


def test_windows_clamp_and_merge_rules() -> None:
    config = CandidateConfig()
    normal = event_window(event(40), config)
    assert (normal.start, normal.end) == (at(25), at(51))
    long = event_window(replace(event(100), start=at(0), end=at(200)), config)
    assert 20 <= (long.end - long.start).total_seconds() <= 60
    assert long.start <= at(85) and long.start < at(100) < long.end
    short = event_window(event(40), replace(config, pre_roll_seconds=1, post_roll_seconds=1))
    assert (short.end - short.start).total_seconds() == 20
    assert len(merge_windows([normal, event_window(event(60, 2), config)], config)) == 1
    assert len(merge_windows([normal, event_window(event(78, 2), config)], config)) == 2


def test_discontinuity_trims_previous_end_and_rejects_short(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path, 900)
    left_id = segment(buffer, 0, 30)
    right_id = segment(buffer, 40, 30, "b")
    boundary = SessionBoundary(at(40), "source", "a", "b", left_id, right_id)
    window = Window((event(25),), at(10), at(50))
    trimmed = trim_gaps(window, [boundary], buffer.manifest.all_records())
    assert trimmed.end == at(30) and trimmed.has_gap
    after = Window((event(42, session="b"),), at(25), at(55))
    trimmed = trim_gaps(after, [boundary], buffer.manifest.all_records())
    assert trimmed.start == at(40)
    assert (trimmed.end - trimmed.start).total_seconds() < CandidateConfig().min_seconds


def test_wait_ready_expire_timeout(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path, 900)
    config = CandidateConfig()
    window = Window((event(20),), at(5), at(31))
    segment(buffer, 0, 24)
    records = buffer.manifest.all_records()
    assert availability(window, records, config, at(24), at(25)) == "waiting"
    assert availability(window, records, config, at(24), at(205)) == "timeout"
    segment(buffer, 24, 12)
    assert availability(window, buffer.manifest.all_records(), config, at(24), at(36)) == "ready"
    records[0].path.unlink()
    assert availability(window, buffer.manifest.all_records(), config, at(24), at(36)) == "expired"


def test_dedupe_restart_and_shared_pin_release(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path, 10)
    segment_id = segment(buffer, 0, 60)
    store = CandidateStore(buffer.manifest.database_path)
    window = event_window(event(20), CandidateConfig())
    records = buffer.manifest.all_records()
    first = store.create(window, records, at(30))
    second = store.create(event_window(event(22, 2), CandidateConfig()), records, at(30))
    restored = CandidateStore(buffer.manifest.database_path)
    assert restored.create(window, records, at(40)) == first
    assert len(restored.all()) == 2
    restored.update(first, "extracted")
    assert buffer.manifest.get_by_path(records[0].path).pinned
    assert buffer.cleanup_once(at(100)) == 0
    restored.update(second, "expired")
    assert not buffer.manifest.get_by_path(records[0].path).pinned
    assert buffer.cleanup_once(at(100)) == 1


def test_retention_race_ends_expired_and_restart_dedupes(tmp_path: Path) -> None:
    config = CandidateConfig(output_dir=tmp_path / "out")
    worker = CandidateExtractor(tmp_path / "segments", config)
    segment(worker.buffer, 0, 60)
    add_event(worker.store, event(20))
    worker.buffer.cleanup_once(at(1000))
    assert worker.run_once(now=at(1000))[0].status == "expired"
    restarted = CandidateExtractor(tmp_path / "segments", config)
    assert len(restarted.run_once(now=at(1001))) == 1


def test_worker_rejects_short_gap(tmp_path: Path) -> None:
    worker = CandidateExtractor(tmp_path / "segments", CandidateConfig(output_dir=tmp_path / "out"))
    segment(worker.buffer, 0, 30)
    segment(worker.buffer, 40, 30, "b")
    add_event(worker.store, event(42, session="b"))
    assert worker.run_once(now=at(80))[0].status == "rejected_gap"


def test_startup_recovers_playable_and_abandons_corrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reader = StreamReader(StreamConfig(source="source", segments_dir=tmp_path))
    good = tmp_path / "segment_old_good.ts"
    bad = tmp_path / "segment_old_bad.ts"
    orphan = tmp_path / "segment_orphan_good.ts"
    for path in (good, bad, orphan):
        path.write_bytes(b"media")
    reader.buffer.manifest.add_writing(good, at(0), reader.source_id, "old")
    reader.buffer.manifest.add_writing(bad, at(6), reader.source_id, "old")
    def probe(path: Path, binary: str) -> float:
        if path == bad:
            raise ValueError("corrupt")
        return 6.25
    monkeypatch.setattr("app.ingestion.stream_reader.probe_duration", probe)
    reader.recover_unfinished()
    rows = reader.buffer.manifest.all_records()
    assert reader.buffer.manifest.get_by_path(good).duration == 6.25
    assert reader.buffer.manifest.get_by_path(good).status == "complete"
    assert reader.buffer.manifest.get_by_path(orphan).session_id == "orphan"
    assert reader.buffer.manifest.get_by_path(bad).status == "abandoned"
    assert any(row.status == "discontinuity" for row in rows)


def test_later_events_merge_while_candidate_waits(tmp_path: Path) -> None:
    worker = CandidateExtractor(tmp_path / "segments", CandidateConfig(output_dir=tmp_path / "out"))
    segment(worker.buffer, 0, 24)
    add_event(worker.store, event(20))
    first = worker.run_once(now=at(25))[0]
    assert first.status == "waiting"
    add_event(worker.store, event(30, 2))
    candidates = worker.run_once(now=at(31))
    assert len(candidates) == 1
    assert candidates[0].id == first.id
    assert [e.id for e in candidates[0].window.events] == [1, 2]
    assert not worker.store.new_events()


def test_extracting_candidate_is_retried_after_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = CandidateConfig(output_dir=tmp_path / "out")
    worker = CandidateExtractor(tmp_path / "segments", config)
    segment(worker.buffer, 0, 60)
    candidate_id = worker.store.create(event_window(event(20), config), worker.buffer.manifest.all_records(), at(30))
    worker.store.update(candidate_id, "extracting")
    restarted = CandidateExtractor(tmp_path / "segments", config)
    output = config.output_dir / f"{candidate_id}.mp4"
    def export(identifier: str, window: Window) -> Path:
        output.write_bytes(b"exported")
        return output
    monkeypatch.setattr(restarted, "_export", export)
    candidates = restarted.run_once(now=at(61))
    assert candidates[0].status == "extracted"
    assert not any(record.pinned for record in restarted.buffer.manifest.all_records())


def test_manual_pin_added_during_reservation_survives_export(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path, 900)
    identifier = segment(buffer, 0, 60)
    store = CandidateStore(buffer.manifest.database_path)
    candidate = store.create(event_window(event(20), CandidateConfig()), buffer.manifest.all_records(), at(30))
    buffer.pin([identifier])
    store.update(candidate, "extracted")
    assert buffer.manifest.all_records()[0].pinned


def test_cleanup_rechecks_reservation_after_stale_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path, 10)
    segment(buffer, 0, 60)
    store = CandidateStore(buffer.manifest.database_path)
    original = buffer.manifest.records_with_status
    def race(statuses: tuple[str, ...]) -> list:
        stale = original(statuses)
        store.create(event_window(event(20), CandidateConfig()), stale, at(30))
        return stale
    monkeypatch.setattr(buffer.manifest, "records_with_status", race)
    assert buffer.cleanup_once(at(100)) == 0
    assert buffer.manifest.all_records()[0].path.exists()


def test_padding_and_timeout_release(tmp_path: Path) -> None:
    worker = CandidateExtractor(tmp_path / "segments", CandidateConfig(output_dir=tmp_path / "out"))
    for start in (0, 6, 12, 18, 24):
        segment(worker.buffer, start)
    add_event(worker.store, event(25))
    waiting = worker.run_once(now=at(30))[0]
    assert waiting.status == "waiting"
    # Window begins at 10: segment at 0 is the one segment of left padding.
    assert len(waiting.segment_ids) == 5
    timed_out = worker.run_once(now=at(211))[0]
    assert timed_out.status == "timeout"
    assert not any(record.pinned for record in worker.buffer.manifest.all_records())
