"""Production queue contracts independent of any model download."""
from types import SimpleNamespace

import pytest

from app.video.engine import Queue, select

def test_duplicate_claim_and_recovery(tmp_path):
    q = Queue(tmp_path / 'data')
    source = tmp_path / 'example.mp4'; source.write_bytes(b'example')
    identity = q.register(source)
    with pytest.raises(ValueError, match='duplicate'):
        q.register(source)
    assert q.claim(identity)['source_id'] == identity
    assert q.claim(identity) is None
    q.recover()
    assert q.claim(identity)['source_id'] == identity
    assert q.db.execute('SELECT count(*) FROM history').fetchone()[0] >= 4
    q.db.close()

def test_selection_rejects_weak_incomplete_and_overlapping():
    candidates = [dict(window_id=str(i), start_seconds=start, end_seconds=start + 10) for i, start in enumerate((0, 2, 30, 50))]
    records = [SimpleNamespace(candidate_id=str(i), editorial_rating=rating, confidence=0.9, context_complete=complete)
               for i, (rating, complete) in enumerate(((5, True), (4, True), (2, True), (5, False)))]
    chosen = select(candidates, records, 5, 4, 0.25, 60, True)
    assert [c['window_id'] for c in chosen] == ['0']
