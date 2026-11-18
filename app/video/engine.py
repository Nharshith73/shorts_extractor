"""Persistent single-worker orchestration over the existing VOD components."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import uuid

import yaml

from app.clips.candidate_extractor import worker_lock
from app.clips.render import EncoderSelector
from app.evaluation.judge import judge_candidates, load_candidate_context
from app.evaluation.inference import MockHighlightJudgeProvider
from app.evaluation.providers.qwen import QwenProviderConfig, QwenVisionLanguageProvider
from app.rendering.config import load_render_config
from app.rendering.renderer import probe_media
from app.rendering.transcription import WhisperTranscriber
from app.vod.pipeline import run
from app.vod.render import selected_clips, clip_transcript, render_clip
from .boundaries import resolve_boundaries, snap_to_speech_boundaries

EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm', '.ts', '.mpegts'}
ACTIVE = ('VALIDATING', 'ANALYZING', 'CONTEXT_EXTRACTION', 'AI_JUDGING', 'SHORTLIST_GENERATION', 'RENDERING')

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)

class Queue:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        for folder in ('inbox', 'processing', 'completed', 'failed'):
            (self.root / folder).mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / 'video_jobs.sqlite3', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS jobs (source_id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL,
          source_path TEXT NOT NULL, manifest TEXT NOT NULL, status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
          attempts INTEGER NOT NULL DEFAULT 0, error TEXT, updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, status TEXT NOT NULL,
          timestamp TEXT NOT NULL, detail TEXT);
        ''')
        self.db.commit()

    def transition(self, source_id, status, detail=None):
        with self.db:
            self.db.execute('UPDATE jobs SET status=?, error=?, updated=? WHERE source_id=?', (status, detail, now(), source_id))
            self.db.execute('INSERT INTO history(source_id,status,timestamp,detail) VALUES(?,?,?,?)', (source_id, status, now(), detail))

    def register(self, path, creator=None, source_url=None, authorization=None, priority=0):
        path = Path(path).resolve()
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            raise ValueError(f'unsupported or missing local media: {path}')
        before = path.stat(); fingerprint = digest(path); after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or not after.st_size:
            raise ValueError('source changed during fingerprinting or is empty')
        source_id = fingerprint[:24]
        manifest = dict(source_id=source_id, original_filename=path.name, source_path=str(path), fingerprint=fingerprint,
                        file_size=after.st_size, acquisition_method='creator-provided local file', creator_identity=creator,
                        source_url=source_url, authorization_reference=authorization, import_timestamp=now())
        try:
            with self.db:
                self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',
                    (source_id, fingerprint, str(path), json.dumps(manifest), 'DISCOVERED', priority, 0, None, now()))
                self.db.execute('INSERT INTO history(source_id,status,timestamp) VALUES(?,?,?)', (source_id, 'DISCOVERED', now()))
        except sqlite3.IntegrityError:
            raise ValueError(f'duplicate source: {source_id}') from None
        return source_id

    def claim(self, source_id=None, max_attempts=3):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            sql = "SELECT * FROM jobs WHERE status IN ('DISCOVERED','QUEUED','RETRY_WAIT') AND attempts < ?"
            args = [max_attempts]
            if source_id:
                sql += ' AND source_id=?'; args.append(source_id)
            row = self.db.execute(sql + ' ORDER BY priority DESC, updated LIMIT 1', args).fetchone()
            if row:
                self.db.execute('UPDATE jobs SET status=?, attempts=attempts+1, updated=? WHERE source_id=?', ('VALIDATING', now(), row['source_id']))
                self.db.execute('INSERT INTO history(source_id,status,timestamp) VALUES(?,?,?)', (row['source_id'], 'VALIDATING', now()))
            self.db.commit(); return row
        except BaseException:
            self.db.rollback(); raise

    def recover(self):
        for row in self.db.execute('SELECT source_id FROM jobs WHERE status IN (%s)' % ','.join('?' * len(ACTIVE)), ACTIVE).fetchall():
            self.transition(row['source_id'], 'RETRY_WAIT', 'Recovered interrupted worker')

def validate(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=60)
    data = json.loads(result.stdout)
    video = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video' and not s.get('disposition', {}).get('attached_pic')), None)
    duration = float(data.get('format', {}).get('duration', 0))
    if not video or duration <= 0 or not video.get('width') or not video.get('height'):
        raise ValueError('missing video stream or valid duration')
    # Full decode catches truncated files that a header-only probe accepts.
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(path), '-map', '0:v:0', '-map', '0:a:0?', '-f', 'null', '-'], check=True, capture_output=True)
    return dict(duration=duration, resolution=[video['width'], video['height']], frame_rate=video.get('avg_frame_rate'),
                video_codec=video.get('codec_name'), audio_codecs=[s.get('codec_name') for s in data['streams'] if s.get('codec_type') == 'audio'])

def make_provider(config, cpu=False, mock=False, provider_type=None):
    if mock: return MockHighlightJudgeProvider()
    raw = yaml.safe_load(Path(config).read_text(encoding='utf-8')).get('judge', {})
    provider_name = provider_type or raw.get('provider', 'local-qwen')
    if provider_name in ('groq', 'groq-cloud'):
        from app.evaluation.providers.groq import GroqHighlightJudgeProvider, GroqProviderConfig
        groq_cfg = GroqProviderConfig(
            api_key=raw.get('groq_api_key'),
            model_name=raw.get('groq_model', 'openai/gpt-oss-20b'),
            vision_model_name=raw.get('groq_vision_model', 'openai/gpt-oss-20b'),
            use_vision=raw.get('use_vision', False),
        )
        return GroqHighlightJudgeProvider(groq_cfg)
    keys = QwenProviderConfig.__dataclass_fields__
    values = {k: v for k, v in raw.items() if k in keys}
    for k in ('model_path', 'projector_path', 'server_binary'):
        if k in values: values[k] = Path(values[k])
    if cpu: values['n_gpu_layers'] = 0
    return QwenVisionLanguageProvider(QwenProviderConfig(**values))

def select(candidates, records, max_shorts, minimum, overlap, max_duration, require_context, min_shorts=1):
    by_id = {str(c['window_id']): c for c in candidates}
    sorted_records = sorted(records, key=lambda x: (x.editorial_rating, x.confidence), reverse=True)
    chosen = []

    # Primary pass: strict filter by minimum rating and context requirements
    for r in sorted_records:
        if r.candidate_id not in by_id:
            continue
        c = by_id[r.candidate_id]
        start, end = float(c['start_seconds']), float(c['end_seconds'])
        if r.editorial_rating < minimum or (require_context and not r.context_complete) or end - start > max_duration:
            continue
        if any(max(0, min(end, b) - max(start, a)) / min(end-start, b-a) > overlap for a, b, _ in chosen):
            continue
        chosen.append((start, end, c))
        if len(chosen) >= max_shorts:
            break

    # Secondary pass: satisfy minimum shorts guarantee if available candidates exist
    if len(chosen) < min_shorts:
        chosen_ids = {c['window_id'] for _, _, c in chosen}
        for r in sorted_records:
            if r.candidate_id not in by_id or r.candidate_id in chosen_ids:
                continue
            c = by_id[r.candidate_id]
            start, end = float(c['start_seconds']), float(c['end_seconds'])
            if end - start > max_duration:
                continue
            if any(max(0, min(end, b) - max(start, a)) / min(end-start, b-a) > overlap for a, b, _ in chosen):
                continue
            chosen.append((start, end, c))
            chosen_ids.add(c['window_id'])
            if len(chosen) >= min_shorts or len(chosen) >= max_shorts:
                break

    return [c for _, _, c in chosen]

def process_job(queue, row, provider, args):
    source_id = row['source_id']; source = Path(row['source_path'])
    work = queue.root / 'processing' / source_id; work.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output = work / 'staging' / run_id; output.mkdir(parents=True, exist_ok=False)
    final_run = queue.root / 'completed' / source_id / 'runs' / run_id
    manifest = json.loads(row['manifest']); started = time.perf_counter(); timings = {}
    model_files = {}
    if not args.mock:
        for key in ('model_path', 'projector_path'):
            path = getattr(provider.config, key, None)
            if path and Path(path).is_file(): model_files[key] = digest(path)
    model_fingerprint = hashlib.sha256(json.dumps({'model': provider.model_name, 'files': model_files}, sort_keys=True).encode()).hexdigest()
    manifest.update(run_id=run_id, created_at=now(), pipeline_config_fingerprint=digest(args.config),
                    judge_config_fingerprint=digest(args.judge_config), render_config_fingerprint=digest(args.render_config),
                    model=provider.model_name, model_fingerprint=model_fingerprint)
    if digest(source) != row['fingerprint']: raise ValueError('source fingerprint changed since import')
    manifest.update(validate(source)); manifest['processing_status'] = 'VALIDATING'; save(output / 'manifest.json', manifest)
    config_hash = hashlib.sha256((digest(args.config) + (digest(args.chat) if args.chat else '')).encode()).hexdigest()
    marker = work / 'analysis-cache.json'
    cache = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
    queue.transition(source_id, 'ANALYZING'); at = time.perf_counter()
    if cache.get('config_hash') != config_hash or not (work / 'clips.json').is_file() or not (work / 'features.csv').is_file():
        # Context and judgments depend on analysis and must not survive invalidation.
        import shutil
        for name in ('context', 'judgments'):
            shutil.rmtree(work / name, ignore_errors=True)
        marker.unlink(missing_ok=True)
        run(source, work, args.config, args.chat)
        save(marker, {'config_hash': config_hash, 'fingerprint': row['fingerprint']})
    timings['analysis_seconds'] = time.perf_counter() - at
    candidates = json.loads((work / 'clips.json').read_text(encoding='utf-8'))[:args.max_candidates]
    queue.transition(source_id, 'CONTEXT_EXTRACTION'); at = time.perf_counter()
    render_config = load_render_config(args.render_config)
    from dataclasses import replace
    render_config = replace(render_config, encoder=args.encoder or render_config.encoder, layout=args.layout or render_config.layout)
    render_config.validate(); transcriber = WhisperTranscriber(render_config)
    import shutil
    if args.chat: shutil.copy2(args.chat, work / 'chat.json')
    context_key = hashlib.sha256((digest(args.render_config) + digest(args.judge_config) + str(args.max_frames) + model_fingerprint).encode()).hexdigest()
    context_marker = work / 'context-cache.json'
    previous_context = json.loads(context_marker.read_text(encoding='utf-8')) if context_marker.exists() else {}
    if previous_context.get('key') != context_key:
        shutil.rmtree(work / 'context', ignore_errors=True)
        shutil.rmtree(work / 'judgments', ignore_errors=True)
        for path in (work / '.transcripts').glob('context-*.json'): path.unlink()
    # Existing transcript caches are clip-relative. Convert a copy to source time for context.
    import tempfile
    for clip in selected_clips(work, ranks={int(c['rank']) for c in candidates}) if candidates else []:
        info = probe_media(source)
        # Transcribe buffered audio (+14s lookforward) so Whisper transcribes past candidate end to capture punchlines
        buf_start = max(0.0, clip.start - 8.0)
        buf_end = min(info.duration, clip.end + 14.0)
        buf_clip = replace(clip, start=buf_start, end=buf_end)
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            transcript, _, _ = clip_transcript(buf_clip, work, render_config, info, Path(tmp), transcriber)
        segments = []
        for seg in transcript.segments:
            item = asdict(seg); item['start'] += buf_start; item['end'] += buf_start
            for word in item['words']: word['start'] += buf_start; word['end'] += buf_start; word['word'] = word.pop('text')
            segments.append(item)
        save(work / '.transcripts' / f'context-{clip.metadata["window_id"]}.json', {'language': transcript.language, 'segments': segments})
        load_candidate_context(clip.metadata['window_id'], work, clip.metadata, source, args.max_frames)
    save(context_marker, {'key': context_key})
    timings['context_and_transcription_seconds'] = time.perf_counter() - at
    queue.transition(source_id, 'AI_JUDGING'); at = time.perf_counter()
    records, _ = judge_candidates(work, provider, top_k=args.max_candidates, force=args.force, max_frames=args.max_frames)
    timings['judgment_seconds'] = time.perf_counter() - at
    queue.transition(source_id, 'SHORTLIST_GENERATION')
    min_shorts = getattr(args, 'min_shorts', 1) or 1
    chosen = select(candidates, records, args.max_shorts, args.min_score, args.max_overlap, args.max_duration, not args.allow_incomplete, min_shorts=min_shorts)
    save(output / 'shortlist.json', chosen)
    queue.transition(source_id, 'RENDERING'); at = time.perf_counter()
    selector = EncoderSelector(render_config.encoder)
    lookup = {r.candidate_id: r for r in records}
    for clip in selected_clips(work, ranks={int(c['rank']) for c in chosen}) if chosen else []:
        r = lookup[clip.metadata['window_id']]
        rec_start = r.recommended_start_seconds
        rec_end = r.recommended_end_seconds

        # Check speech segments to snap start/end to full sentence boundaries with generous reaction padding
        snapped_s = clip.start
        snapped_e = clip.end
        ctx_file = work / 'context' / clip.metadata['window_id'] / 'context.json'
        if ctx_file.is_file():
            try:
                ctx_data = json.loads(ctx_file.read_text(encoding='utf-8'))
                speech_segs = ctx_data.get('speech', {}).get('segments', [])
                if speech_segs:
                    snapped_s, snapped_e = snap_to_speech_boundaries(
                        clip.start, clip.end, clip.metadata.get('peak_seconds', clip.start), speech_segs,
                        min_duration=args.min_duration, max_duration=args.max_duration,
                        reaction_padding=1.8
                    )
            except Exception as exc:
                LOGGER.debug("Could not snap speech boundaries: %s", exc)

        final_rec_s = min(rec_start, snapped_s) if rec_start is not None else snapped_s
        final_rec_e = max(rec_end, snapped_e) if rec_end is not None else snapped_e

        ctx_start = max(0.0, min(clip.start, snapped_s, final_rec_s) - 1.0)
        ctx_end = min(manifest['duration'], max(clip.end, snapped_e, final_rec_e) + 2.0)
        boundary = resolve_boundaries(snapped_s, snapped_e, manifest['duration'], final_rec_s,
            final_rec_e, r.confidence, True, args.min_duration, args.max_duration,
            args.boundary_confidence, args.boundary_refinement, context_start=ctx_start, context_end=ctx_end)
        save(output / 'boundaries' / (clip.metadata['window_id'] + '.json'), boundary)
        clip = replace(clip, start=boundary['applied'][0], end=boundary['applied'][1])
        if not args.min_duration <= clip.end - clip.start <= args.max_duration:
            raise ValueError('fallback candidate duration is outside configured limits')
        result = render_clip(clip, work, render_config, transcriber, selector)
        import shutil
        target = output / 'shorts'; target.mkdir(exist_ok=True)
        for path in (work / 'final').glob(Path(result['clip_path']).stem + '.*'): shutil.copy2(path, target / path.name)
        validate(target / clip.name)
        save(output / 'metadata' / (Path(clip.name).stem + '.json'), dict(title=r.summary, suggested_hook=r.hook_suggestion,
            description=r.summary, keywords=['gaming', r.event_category], hashtags=['#gaming', '#shorts'], creator=manifest.get('creator_identity'),
            source_reference=manifest.get('source_url') or str(source), source_id=source_id, start_seconds=clip.start,
            end_seconds=clip.end, event_category=r.event_category, editorial_rating=r.editorial_rating,
            model=r.model_metadata, render_profile=asdict(render_config), boundaries=boundary, output=str(final_run / 'shorts' / clip.name)))
    timings['rendering_seconds'] = time.perf_counter() - at
    for folder, origin in [('analysis', 'run.json'), ('candidates', 'clips.json'), ('judgments', 'judgments'), ('contexts', 'context')]:
        import shutil
        dest = output / folder; dest.mkdir(exist_ok=True); src = work / origin
        if src.is_dir(): shutil.copytree(src, dest, dirs_exist_ok=True)
        elif src.exists(): shutil.copy2(src, dest / src.name)
    timings['total_seconds'] = time.perf_counter() - started
    save(output / 'reports' / 'production.json', {'timings': timings, 'source_duration': manifest['duration'],
         'processing_factor': manifest['duration'] / max(timings['total_seconds'], 1e-9), 'selected': len(chosen), 'candidates': len(candidates)})
    manifest['processing_status'] = 'COMPLETED'; manifest['completed_at'] = now()
    manifest['outputs'] = [{'path': str(p.relative_to(output)), 'sha256': digest(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    save(output / 'manifest.json', manifest)
    final_run.parent.mkdir(parents=True, exist_ok=True)
    output.rename(final_run)
    save(final_run.parent.parent / 'active.json', {'run_id': run_id, 'manifest': str(final_run / 'manifest.json')})
    queue.transition(source_id, 'COMPLETED')
