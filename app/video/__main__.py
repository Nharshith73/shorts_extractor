"""Operator CLI for authorized local video processing."""
import argparse
import logging
from pathlib import Path
import sys

from app.clips.candidate_extractor import worker_lock
from .engine import Queue, make_provider, process_job

def main():
    parser = argparse.ArgumentParser(description='Recorded video production')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('process')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path); group.add_argument('--all', action='store_true')
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--judge-config', type=Path, default=Path('judge.yaml'))
    p.add_argument('--render-config', type=Path, default=Path('render.yaml'))
    p.add_argument('--chat', type=Path); p.add_argument('--creator'); p.add_argument('--source-url'); p.add_argument('--authorization')
    p.add_argument('--priority', type=int, default=0); p.add_argument('--max-candidates', type=int, default=10)
    p.add_argument('--max-shorts', type=int, default=3); p.add_argument('--min-shorts', type=int, default=1); p.add_argument('--min-score', type=int, default=4)
    p.add_argument('--max-duration', type=float, default=60); p.add_argument('--max-overlap', type=float, default=0.25)
    p.add_argument('--min-duration', type=float, default=1); p.add_argument('--boundary-confidence', type=float, default=0.5)
    p.add_argument('--boundary-refinement', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-frames', type=int, default=8); p.add_argument('--max-attempts', type=int, default=3)
    p.add_argument('--allow-incomplete', action='store_true'); p.add_argument('--layout', choices=['blurred_background', 'stacked', 'center_crop'])
    p.add_argument('--encoder', choices=['auto', 'libx264', 'h264_nvenc'], default='libx264')
    p.add_argument('--provider', choices=['local-qwen', 'groq', 'mock'], default=None, help='AI Highlight Judge provider')
    p.add_argument('--cpu', action='store_true'); p.add_argument('--mock', action='store_true', help='Test provider; never a substitute for Qwen evaluation')
    p.add_argument('--force', action='store_true', default=False, help='Force re-analysis and re-evaluation')
    commands.add_parser('status'); commands.add_parser('results')
    for name in ('retry', 'rejudge'):
        sub = commands.add_parser(name); sub.add_argument('--source', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    q = Queue(args.data_dir)
    try:
        if args.command in ('status', 'results'):
            sql = 'SELECT source_id,status,priority,attempts,error,updated FROM jobs'
            if args.command == 'results': sql += " WHERE status='COMPLETED'"
            for row in q.db.execute(sql + ' ORDER BY updated DESC'): print(dict(row))
            return 0
        with worker_lock(q.root / '.video-worker.lock'):
            q.recover()
            if args.command in ('retry', 'rejudge'):
                row = q.db.execute('SELECT * FROM jobs WHERE source_id=?', (args.source,)).fetchone()
                if row is None: raise ValueError('unknown source ID')
                if args.command == 'rejudge':
                    import shutil
                    shutil.rmtree(q.root / 'processing' / args.source / 'judgments', ignore_errors=True)
                with q.db: q.db.execute('UPDATE jobs SET attempts=0 WHERE source_id=?', (args.source,))
                q.transition(args.source, 'QUEUED', args.command)
                print(f'{args.source} queued; run process --all'); return 0
            if args.max_candidates < 1 or args.max_shorts < 1 or args.min_shorts < 1 or args.min_shorts > args.max_shorts or args.max_attempts < 1 or not 0 <= args.min_score <= 5 or not 0 <= args.max_overlap <= 1 or not 0 < args.min_duration <= args.max_duration or not 0 <= args.boundary_confidence <= 1 or args.max_frames < 1:
                raise ValueError('invalid selection or retry settings')
            source_id = None
            if args.input:
                try:
                    source_id = q.register(args.input, args.creator, args.source_url, args.authorization, args.priority)
                except ValueError as exc:
                    if 'duplicate source' in str(exc):
                        from .engine import digest, now
                        source_id = digest(args.input)[:24]
                        with q.db:
                            q.db.execute("UPDATE jobs SET status='QUEUED', attempts=0, error=NULL, updated=? WHERE source_id=?", (now(), source_id))
                        q.transition(source_id, 'QUEUED', 'Re-queued for processing')
                        if args.force:
                            import shutil
                            work = q.root / 'processing' / source_id
                            (work / 'analysis-cache.json').unlink(missing_ok=True)
                            (work / 'context-cache.json').unlink(missing_ok=True)
                            shutil.rmtree(work / 'context', ignore_errors=True)
                            shutil.rmtree(work / 'judgments', ignore_errors=True)
                    else:
                        raise
            if args.all:
                for path in sorted((q.root / 'inbox').iterdir()):
                    if path.is_file() and path.suffix.lower() in ('.mp4', '.mkv', '.mov', '.webm', '.ts', '.mpegts'):
                        try: q.register(path)
                        except ValueError as exc: logging.info('%s', exc)
            provider = make_provider(args.judge_config, args.cpu, args.mock, provider_type=args.provider)
            failed = False
            try:
                # Each queued item is attempted once per invocation; retry waits persist.
                pending = [source_id] if source_id else [r['source_id'] for r in q.db.execute("SELECT source_id FROM jobs WHERE status IN ('DISCOVERED','QUEUED','RETRY_WAIT') ORDER BY priority DESC, updated")]
                for item in pending:
                    row = q.claim(item, args.max_attempts)
                    if not row:
                        current = q.db.execute('SELECT status,attempts FROM jobs WHERE source_id=?', (item,)).fetchone()
                        if current and current['status'] in ('DISCOVERED', 'QUEUED', 'RETRY_WAIT') and current['attempts'] >= args.max_attempts:
                            q.transition(item, 'FAILED', 'Retry limit exhausted')
                            failed = True
                        continue
                    try: process_job(q, row, provider, args)
                    except Exception as exc:
                        failed = True; logging.exception('Job %s failed', item)
                        attempts = q.db.execute('SELECT attempts FROM jobs WHERE source_id=?', (item,)).fetchone()[0]
                        q.transition(item, 'FAILED' if attempts >= args.max_attempts else 'RETRY_WAIT', str(exc))
            finally:
                if hasattr(provider, 'close'): provider.close()
            return 2 if failed else 0
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error('%s', exc); return 2
    finally: q.db.close()

if __name__ == '__main__':
    sys.exit(main())
