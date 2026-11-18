"""StreamClip AI — Minimalist Web Studio & Dashboard.

Provides an ultra-clean, white-background, zero-glow interface to:
1. Choose or download any video from inbox or YouTube.
2. Select AI Provider (Local Qwen2.5-VL on RTX 4070 or Groq Cloud 70B/Vision).
3. Monitor real-time progress across all pipeline stages.
4. Preview, play, and inspect all generated vertical 9:16 Shorts with transcripts & AI hooks.
5. Review all candidate moments and AI judgments.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from typing import Any
import urllib.parse

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.evaluation.providers.groq import resolve_groq_api_key

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
INBOX_DIR = DATA_DIR / "inbox"
PROCESSING_DIR = DATA_DIR / "processing"
COMPLETED_DIR = DATA_DIR / "completed"
DB_PATH = DATA_DIR / "video_jobs.sqlite3"

app = FastAPI(title="StreamClip Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active background jobs tracking
ACTIVE_TASKS: dict[str, dict[str, Any]] = {}


def sanitize_task(task_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    if not task_dict:
        return None
    return {
        "status": task_dict.get("status"),
        "logs": task_dict.get("logs", []),
        "started_at": task_dict.get("started_at"),
        "returncode": task_dict.get("returncode"),
        "error": task_dict.get("error"),
    }


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def list_inbox_videos() -> list[dict[str, Any]]:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    videos = []
    for ext in ("*.mp4", "*.mkv", "*.mov", "*.webm", "*.ts"):
        for p in INBOX_DIR.glob(ext):
            try:
                st = p.stat()
                videos.append({
                    "filename": p.name,
                    "path": str(p),
                    "size_mb": round(st.st_size / (1024 * 1024), 2),
                    "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                })
            except Exception:
                pass
    return sorted(videos, key=lambda v: v["mtime"], reverse=True)


@app.get("/favicon.ico", include_in_schema=False)
def api_favicon():
    return Response(content=b"", media_type="image/x-icon", status_code=204)


@app.get("/api/videos")
def api_videos():
    return {"videos": list_inbox_videos()}


@app.delete("/api/videos")
def api_delete_all_videos():
    deleted = []
    if INBOX_DIR.is_dir():
        for p in INBOX_DIR.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                    deleted.append(p.name)
                except Exception:
                    pass
    return {"status": "DELETED_ALL", "deleted": deleted}


@app.delete("/api/videos/{filename}")
def api_delete_video(filename: str):
    file_path = INBOX_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found in inbox")
    try:
        file_path.unlink()
        return {"status": "DELETED", "filename": filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {exc}")


@app.post("/api/upload")
async def api_upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename provided")
    safe_name = Path(file.filename).name
    target = INBOX_DIR / safe_name
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as buffer:
        while chunk := await file.read(1024 * 1024 * 4):
            buffer.write(chunk)
    return {
        "status": "UPLOADED",
        "filename": safe_name,
        "size_mb": round(target.stat().st_size / (1024 * 1024), 2),
    }


@app.delete("/api/jobs/{source_id}")
def api_delete_job(source_id: str):
    comp = COMPLETED_DIR / source_id
    if comp.is_dir():
        shutil.rmtree(comp, ignore_errors=True)
    proc = PROCESSING_DIR / source_id
    if proc.is_dir():
        shutil.rmtree(proc, ignore_errors=True)
    for k in list(ACTIVE_TASKS.keys()):
        if k == source_id or source_id in str(ACTIVE_TASKS.get(k, {})):
            ACTIVE_TASKS.pop(k, None)
    if DB_PATH.is_file():
        conn = get_db()
        try:
            conn.execute("DELETE FROM jobs WHERE source_id=?", (source_id,))
            conn.execute("DELETE FROM history WHERE source_id=?", (source_id,))
            conn.commit()
        finally:
            conn.close()
    return {"status": "DELETED", "source_id": source_id}


class CleanupRequest(BaseModel):
    clear_processing: bool = True
    clear_completed: bool = False
    clear_inbox: bool = False
    reset_db: bool = False


@app.post("/api/cleanup")
def api_cleanup(req: CleanupRequest):
    cleaned = []
    if req.clear_processing and PROCESSING_DIR.exists():
        for item in PROCESSING_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try: item.unlink()
                except Exception: pass
        cleaned.append("processing_temp_files")

    if req.clear_completed and COMPLETED_DIR.exists():
        for item in COMPLETED_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try: item.unlink()
                except Exception: pass
        cleaned.append("completed_shorts")

    if req.clear_inbox and INBOX_DIR.exists():
        for item in INBOX_DIR.glob("*"):
            if item.is_file():
                try: item.unlink()
                except Exception: pass
        cleaned.append("inbox_videos")

    if req.reset_db and DB_PATH.is_file():
        conn = get_db()
        try:
            conn.execute("DELETE FROM jobs")
            conn.execute("DELETE FROM history")
            conn.commit()
            cleaned.append("database_history")
        finally:
            conn.close()

    ACTIVE_TASKS.clear()
    return {"status": "CLEANED", "cleaned": cleaned}


@app.get("/api/shorts")
def api_list_shorts():
    """List all rendered 9:16 Shorts across completed VOD jobs and live Twitch/Kick watcher runs."""
    shorts = []
    if not COMPLETED_DIR.is_dir():
        return {"shorts": []}

    from app.evaluation.viral_metadata import ViralMetadataGenerator

    for mp4 in COMPLETED_DIR.rglob("shorts/*.mp4"):
        if not mp4.is_file():
            continue
        try:
            st = mp4.stat()
            meta_json = mp4.parent.parent / "metadata" / f"{mp4.stem}.json"
            alt_json = mp4.with_suffix(".json")
            meta: dict[str, Any] = {}
            if meta_json.is_file():
                try:
                    meta = json.loads(meta_json.read_text(encoding="utf-8"))
                except Exception:
                    pass
            elif alt_json.is_file():
                try:
                    meta = json.loads(alt_json.read_text(encoding="utf-8"))
                except Exception:
                    pass

            raw_title = meta.get("title", "")
            channel_name = meta.get("channel") or meta.get("creator") or ""
            # If title is missing or default stem or not attention grabbing, enhance it
            if not raw_title or raw_title == mp4.stem or raw_title.startswith(("spike_", "clip_", "rank_", "test_clip_")) or ("#Shorts" not in raw_title and "#shorts" not in raw_title):
                v = ViralMetadataGenerator.generate(
                    dialogue=meta.get("caption_text") or meta.get("summary") or "",
                    creator=channel_name,
                    platform=meta.get("platform") or "youtube",
                    category=meta.get("event_category", "funny"),
                    spike_ratio=float(meta.get("spike_ratio") or 0.0),
                    raw_summary=meta.get("description") or "",
                    use_ai=False,
                )
                title = v.title
                suggested_hook = v.suggested_hook
                tags = v.tags
                desc = v.description
            else:
                title = raw_title
                suggested_hook = meta.get("suggested_hook", "WHAT JUST HAPPENED?! 💀")
                tags = meta.get("tags") or ["Shorts", "viral", "gaming", "highlights"]
                desc = meta.get("description", "")

            shorts.append({
                "filename": mp4.name,
                "path": str(mp4),
                "url": f"/media/file?path={urllib.parse.quote(str(mp4))}",
                "size_mb": round(st.st_size / (1024 * 1024), 2),
                "mtime": st.st_mtime,
                "title": title,
                "suggested_hook": suggested_hook,
                "summary": desc,
                "tags": tags,
                "event_category": meta.get("event_category", "highlight"),
                "editorial_rating": meta.get("editorial_rating", 4),
                "start_seconds": meta.get("start_seconds"),
                "end_seconds": meta.get("end_seconds"),
                "channel": channel_name,
            })
        except Exception:
            pass

    shorts.sort(key=lambda s: s["mtime"], reverse=True)
    return {"shorts": shorts}


@app.delete("/api/shorts")
def api_delete_short(path: str = Query(...)):
    target = Path(path).resolve()
    if not str(target).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Short not found")
    try:
        target.unlink()
        return {"status": "DELETED", "filename": target.name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/tasks/dismiss")
def api_dismiss_tasks(task_id: str | None = None):
    if task_id and task_id in ACTIVE_TASKS:
        ACTIVE_TASKS.pop(task_id, None)
    else:
        ACTIVE_TASKS.clear()
    return {"status": "DISMISSED"}


@app.post("/api/tasks/stop")
def api_stop_tasks(task_id: str | None = None):
    targets = [task_id] if task_id and task_id in ACTIVE_TASKS else list(ACTIVE_TASKS.keys())
    stopped = []
    for tid in targets:
        task = ACTIVE_TASKS.get(tid)
        if task and "proc" in task:
            try:
                proc = task["proc"]
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        proc.kill()
                    stopped.append(tid)
            except Exception as e:
                LOGGER.warning("Failed to stop process %s: %s", tid, e)
            task["status"] = "STOPPED"
    return {"status": "STOPPED", "stopped": stopped}


@app.get("/api/jobs")
def api_jobs():
    if not DB_PATH.is_file():
        return {"jobs": []}
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT source_id, fingerprint, source_path, status, attempts, error, updated FROM jobs ORDER BY updated DESC"
        ).fetchall()
        jobs = []
        for r in rows:
            d = dict(r)
            d["filename"] = Path(d["source_path"]).name
            jobs.append(d)
        return {"jobs": jobs}
    finally:
        conn.close()


@app.get("/api/jobs/{source_id}")
def api_job_detail(source_id: str):
    conn = get_db()
    try:
        job = conn.execute("SELECT * FROM jobs WHERE source_id=?", (source_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_data = dict(job)
        job_data["filename"] = Path(job_data["source_path"]).name if job_data.get("source_path") else "video.mp4"
        job_data["manifest"] = json.loads(job_data["manifest"])

        history = conn.execute(
            "SELECT status, timestamp, detail FROM history WHERE source_id=? ORDER BY id ASC",
            (source_id,),
        ).fetchall()
        job_data["history"] = [dict(h) for h in history]

        # Check completed run output
        shorts = []
        reports = {}
        judgments = []
        candidates = []

        # Find run directory in completed
        run_dir = None
        runs_parent = COMPLETED_DIR / source_id / "runs"
        if runs_parent.is_dir():
            all_runs = sorted(runs_parent.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_runs:
                run_dir = all_runs[0]

        # Fallback to processing directory if not completed yet
        proc_dir = PROCESSING_DIR / source_id

        # Load reports
        if run_dir and (run_dir / "reports" / "production.json").is_file():
            try:
                reports = json.loads((run_dir / "reports" / "production.json").read_text(encoding="utf-8"))
            except Exception:
                pass

        # Load shorts
        search_dirs = []
        if run_dir and (run_dir / "shorts").is_dir():
            search_dirs.append(run_dir / "shorts")
        if (proc_dir / "final").is_dir():
            search_dirs.append(proc_dir / "final")

        seen_shorts = set()
        for sdir in search_dirs:
            for mp4 in sdir.glob("*.mp4"):
                if mp4.name in seen_shorts:
                    continue
                seen_shorts.add(mp4.name)

                # Look for matching metadata
                meta = {}
                meta_file = sdir.parent / "metadata" / f"{mp4.stem}.json"
                if not meta_file.is_file():
                    meta_file = sdir / f"{mp4.stem}.json"
                if meta_file.is_file():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                preview_file = sdir / f"{mp4.stem}.preview.jpg"
                if not preview_file.is_file() and (proc_dir / "final" / f"{mp4.stem}.preview.jpg").is_file():
                    preview_file = proc_dir / "final" / f"{mp4.stem}.preview.jpg"

                shorts.append({
                    "filename": mp4.name,
                    "path": str(mp4),
                    "url": f"/media/file?path={urllib_quote(str(mp4))}",
                    "preview_url": f"/media/file?path={urllib_quote(str(preview_file))}" if preview_file.is_file() else None,
                    "size_mb": round(mp4.stat().st_size / (1024 * 1024), 2),
                    "title": meta.get("title") or meta.get("suggested_hook") or mp4.stem,
                    "suggested_hook": meta.get("suggested_hook"),
                    "editorial_rating": meta.get("editorial_rating", 5),
                    "event_category": meta.get("event_category", "highlight"),
                    "start_seconds": meta.get("start_seconds"),
                    "end_seconds": meta.get("end_seconds"),
                    "summary": meta.get("description") or meta.get("title"),
                    "model": meta.get("model", {}).get("model"),
                })

        # Load judgments
        judg_dir = (run_dir / "judgments") if run_dir and (run_dir / "judgments").is_dir() else (proc_dir / "judgments")
        if judg_dir.is_dir():
            for jfile in judg_dir.glob("*.json"):
                try:
                    jdata = json.loads(jfile.read_text(encoding="utf-8"))
                    judgments.append(jdata)
                except Exception:
                    pass

        # Load candidates
        cand_file = (run_dir / "candidates" / "clips.json") if run_dir and (run_dir / "candidates" / "clips.json").is_file() else (proc_dir / "clips.json")
        if cand_file.is_file():
            try:
                candidates = json.loads(cand_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        job_data["shorts"] = shorts
        job_data["reports"] = reports
        job_data["judgments"] = judgments
        job_data["candidate_count"] = len(candidates)
        job_data["candidates"] = candidates[:15]

        # Active background status
        if source_id in ACTIVE_TASKS:
            job_data["active_progress"] = sanitize_task(ACTIVE_TASKS[source_id])
        else:
            stem = Path(job_data["source_path"]).stem
            if stem in ACTIVE_TASKS:
                job_data["active_progress"] = sanitize_task(ACTIVE_TASKS[stem])

        return job_data
    finally:
        conn.close()


def urllib_quote(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote(s)


@app.get("/media/file")
def stream_media_file(path: str, request: Request):
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")

    # Range request support for smooth HTML5 video seeking
    stat = file_path.stat()
    file_size = stat.st_size
    range_header = request.headers.get("Range")

    content_type = "video/mp4"
    if file_path.suffix.lower() in (".jpg", ".jpeg"):
        content_type = "image/jpeg"
    elif file_path.suffix.lower() == ".png":
        content_type = "image/png"
    elif file_path.suffix.lower() == ".json":
        content_type = "application/json"

    if range_header and content_type.startswith("video"):
        range_match = re.match(r"bytes=(\d+)-(\d+)?", range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            length = (end - start) + 1

            def iter_chunk():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    bytes_left = length
                    while bytes_left > 0:
                        chunk = f.read(min(bytes_left, 1024 * 64))
                        if not chunk:
                            break
                        bytes_left -= len(chunk)
                        yield chunk

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Type": content_type,
            }
            return StreamingResponse(iter_chunk(), status_code=206, headers=headers)

    return FileResponse(file_path, media_type=content_type)


class ProcessRequest(BaseModel):
    video_path: str
    creator: str | None = None
    source_url: str | None = None
    provider: str = "groq"  # 'groq' or 'local-qwen'
    max_candidates: int = 10
    max_shorts: int = 3
    min_shorts: int = 1
    min_score: int = 3
    encoder: str = "auto"
    layout: str = "blurred_background"


def run_pipeline_worker(source_path: Path, req: ProcessRequest, source_id_holder: list[str]):
    # Scale max_candidates to ensure enough moments to satisfy min_shorts
    effective_candidates = max(req.max_candidates, req.min_shorts * 3)
    cmd = [
        "python",
        "-m",
        "app.video",
        "process",
        "--input",
        str(source_path),
        "--provider",
        req.provider,
        "--max-candidates",
        str(effective_candidates),
        "--max-shorts",
        str(req.max_shorts),
        "--min-shorts",
        str(req.min_shorts),
        "--min-score",
        str(req.min_score),
        "--allow-incomplete",
        "--encoder",
        req.encoder,
        "--force",
    ]
    if req.layout:
        cmd.extend(["--layout", req.layout])
    if req.creator:
        cmd.extend(["--creator", req.creator])
    if req.source_url:
        cmd.extend(["--source-url", req.source_url])

    LOGGER.info("Starting production command: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_lines = []
    task_id = source_path.stem
    ACTIVE_TASKS[task_id] = {
        "status": "RUNNING",
        "command": cmd,
        "logs": log_lines,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "proc": proc,
    }

    try:
        for line in proc.stdout:
            clean = line.strip()
            if clean:
                log_lines.append(clean)
                if len(log_lines) > 200:
                    log_lines.pop(0)
        proc.wait()
        ACTIVE_TASKS[task_id]["status"] = "COMPLETED" if proc.returncode == 0 else "FAILED"
        ACTIVE_TASKS[task_id]["returncode"] = proc.returncode
    except Exception as exc:
        ACTIVE_TASKS[task_id]["status"] = "ERROR"
        ACTIVE_TASKS[task_id]["error"] = str(exc)


@app.post("/api/process")
def api_process_video(req: ProcessRequest, bg: BackgroundTasks):
    video_path = Path(req.video_path)
    if not video_path.is_file():
        # Check in inbox
        inbox_cand = INBOX_DIR / req.video_path
        if inbox_cand.is_file():
            video_path = inbox_cand
        else:
            raise HTTPException(status_code=400, detail=f"File not found: {req.video_path}")

    task_id = video_path.stem
    # Guard against duplicate concurrent worker collisions
    if task_id in ACTIVE_TASKS and ACTIVE_TASKS[task_id].get("status") == "RUNNING":
        return {
            "status": "RUNNING",
            "task_id": task_id,
            "video": str(video_path),
            "message": "Pipeline is already actively running for this video.",
        }

    # Automatically enforce max_shorts >= min_shorts
    req.max_shorts = max(req.max_shorts, req.min_shorts)

    source_id_holder = []
    thread = threading.Thread(
        target=run_pipeline_worker,
        args=(video_path, req, source_id_holder),
        daemon=True,
    )
    thread.start()

    return {
        "status": "QUEUED",
        "task_id": video_path.stem,
        "video": str(video_path),
        "provider": req.provider,
    }


class DownloadRequest(BaseModel):
    url: str
    output_name: str | None = None
    browser: str | None = None
    use_cookies: bool = True


def find_cookies_file() -> Path | None:
    for candidate in [DATA_DIR / "cookies.txt", ROOT_DIR / "cookies.txt"]:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def run_download_worker(url: str, output_name: str | None, browser: str | None = None, use_cookies: bool = True):
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(INBOX_DIR / (output_name or "%(id)s.%(ext)s"))
    cmd = [
        "python",
        "-m",
        "yt_dlp",
        "-f",
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o",
        out_tmpl,
        "--no-check-certificates",
    ]

    # 1. Enable Node.js JS runtime if node is on PATH (solves YouTube challenges)
    if shutil.which("node"):
        cmd.extend(["--js-runtimes", "node"])

    # 2. Check for cookies file or browser cookies (solves age-restriction & auth errors)
    cookies_path = find_cookies_file() if use_cookies else None
    if cookies_path:
        cmd.extend(["--cookies", str(cookies_path)])
        LOGGER.info("yt-dlp using cookies file: %s", cookies_path)
    elif browser and browser.lower() in ("chrome", "edge", "firefox", "brave", "opera"):
        cmd.extend(["--cookies-from-browser", browser.lower()])
        LOGGER.info("yt-dlp using cookies from browser: %s", browser)

    cmd.append(url)

    task_id = f"download_{int(time.time())}"
    ACTIVE_TASKS[task_id] = {
        "status": "DOWNLOADING",
        "url": url,
        "logs": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            clean = line.strip()
            if clean:
                ACTIVE_TASKS[task_id]["logs"].append(clean)
                if len(ACTIVE_TASKS[task_id]["logs"]) > 100:
                    ACTIVE_TASKS[task_id]["logs"].pop(0)
        proc.wait()
        if proc.returncode == 0:
            ACTIVE_TASKS[task_id]["status"] = "COMPLETED"
        else:
            log_str = " ".join(ACTIVE_TASKS[task_id]["logs"])
            if "Could not copy Chrome cookie database" in log_str:
                cf = find_cookies_file()
                if cf:
                    ACTIVE_TASKS[task_id]["logs"].append("⚠️ Chrome database locked by open browser. Retrying automatically with data/cookies.txt...")
                    fallback_cmd = [a for a in cmd if a not in ("--cookies-from-browser", (browser or "").lower())]
                    fallback_cmd.extend(["--cookies", str(cf)])
                    retry_proc = subprocess.Popen(
                        fallback_cmd,
                        cwd=str(ROOT_DIR),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in retry_proc.stdout:
                        clean = line.strip()
                        if clean:
                            ACTIVE_TASKS[task_id]["logs"].append(clean)
                    retry_proc.wait()
                    if retry_proc.returncode == 0:
                        ACTIVE_TASKS[task_id]["status"] = "COMPLETED"
                        return
                else:
                    ACTIVE_TASKS[task_id]["logs"].append("⚠️ Chrome database locked. Retrying download directly without browser cookies...")
                    fallback_cmd = [a for a in cmd if a not in ("--cookies-from-browser", (browser or "").lower())]
                    retry_proc = subprocess.Popen(
                        fallback_cmd,
                        cwd=str(ROOT_DIR),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in retry_proc.stdout:
                        clean = line.strip()
                        if clean:
                            ACTIVE_TASKS[task_id]["logs"].append(clean)
                    retry_proc.wait()
                    if retry_proc.returncode == 0:
                        ACTIVE_TASKS[task_id]["status"] = "COMPLETED"
                        return

            ACTIVE_TASKS[task_id]["status"] = "FAILED"
            if "Could not copy Chrome cookie database" in log_str:
                ACTIVE_TASKS[task_id]["logs"].append(
                    "💡 NOTE: Google Chrome is currently open on Windows, which locks its cookie database. "
                    "Either close Chrome completely while downloading, or export your cookies to cookies.txt (via 'Get cookies.txt LOCALLY' extension) and click 'Upload cookies.txt'."
                )
    except Exception as exc:
        ACTIVE_TASKS[task_id]["status"] = "ERROR"
        ACTIVE_TASKS[task_id]["error"] = str(exc)


@app.post("/api/download")
def api_download_video(req: DownloadRequest):
    if not req.url or not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    thread = threading.Thread(
        target=run_download_worker,
        args=(req.url, req.output_name, req.browser, req.use_cookies),
        daemon=True,
    )
    thread.start()
    return {"status": "DOWNLOADING_STARTED", "url": req.url}


@app.post("/api/upload-cookies")
async def api_upload_cookies(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    target = DATA_DIR / "cookies.txt"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        while chunk := await file.read(1024 * 64):
            f.write(chunk)
    return {"status": "UPLOADED", "filename": "cookies.txt", "size_bytes": target.stat().st_size}


@app.get("/api/cookies-status")
def api_cookies_status():
    f = find_cookies_file()
    return {
        "has_cookies_file": bool(f),
        "cookies_path": str(f) if f else None,
        "has_node_runtime": bool(shutil.which("node")),
    }


@app.get("/api/settings")
def api_get_settings():
    groq_key = resolve_groq_api_key()
    safe_tasks = {k: sanitize_task(v) for k, v in ACTIVE_TASKS.items()}
    return {
        "groq_configured": bool(groq_key and len(groq_key) > 10),
        "groq_key_masked": f"{groq_key[:6]}...{groq_key[-4:]}" if groq_key else None,
        "active_tasks": safe_tasks,
    }


@app.post("/api/settings/groq-key")
def api_save_groq_key(key: str = Query(...)):
    if not key or len(key) < 10:
        raise HTTPException(status_code=400, detail="Invalid Groq key")
    os.environ["GROQ_API_KEY"] = key.strip()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_SET_VALUE) as env:
            winreg.SetValueEx(env, "GROQ_API_KEY", 0, winreg.REG_SZ, key.strip())
    except Exception:
        pass
    return {"status": "SAVED", "configured": True}


ACTIVE_WATCHER_SESSION: Any = None


class WatcherStartRequest(BaseModel):
    channel: str
    platform: str = "twitch"
    min_messages: int = 5
    spike_factor: float = 2.5
    ai_mode: str = "groq"
    layout: str = "blurred_background"
    encoder: str = "auto"


@app.post("/api/watcher/start")
def api_watcher_start(req: WatcherStartRequest):
    global ACTIVE_WATCHER_SESSION
    if ACTIVE_WATCHER_SESSION and ACTIVE_WATCHER_SESSION.is_running:
        LOGGER.info("Stopping previous watcher session (%s on %s) to start new channel (%s)",
                    ACTIVE_WATCHER_SESSION.channel, ACTIVE_WATCHER_SESSION.platform, req.channel)
        try:
            ACTIVE_WATCHER_SESSION.stop()
        except Exception:
            pass
        ACTIVE_WATCHER_SESSION = None

    from app.live.watcher import LiveWatcherSession
    ACTIVE_WATCHER_SESSION = LiveWatcherSession(
        channel=req.channel,
        output_dir=COMPLETED_DIR,
        platform=req.platform,
        min_messages=req.min_messages,
        spike_factor=req.spike_factor,
        ai_mode=req.ai_mode,
        layout=req.layout,
        encoder=req.encoder,
    )
    ACTIVE_WATCHER_SESSION.start()
    return {"status": "STARTED", "channel": req.channel, "platform": ACTIVE_WATCHER_SESSION.platform, "session_id": ACTIVE_WATCHER_SESSION.session_id}


@app.post("/api/watcher/stop")
def api_watcher_stop():
    global ACTIVE_WATCHER_SESSION
    if ACTIVE_WATCHER_SESSION:
        ACTIVE_WATCHER_SESSION.stop()
        ACTIVE_WATCHER_SESSION = None
    return {"status": "STOPPED"}


@app.get("/api/watcher/status")
def api_watcher_status():
    global ACTIVE_WATCHER_SESSION
    if not ACTIVE_WATCHER_SESSION:
        return {"is_running": False}
    return ACTIVE_WATCHER_SESSION.get_status()


class WatcherTestClipRequest(BaseModel):
    duration: float = 30.0
    layout: str = "blurred_background"
    encoder: str = "auto"


@app.post("/api/watcher/test-clip")
def api_watcher_test_clip(req: WatcherTestClipRequest = WatcherTestClipRequest()):
    global ACTIVE_WATCHER_SESSION
    from app.live.watcher import create_test_short

    # 1. If active watcher session is running, trigger manual clip from live stream
    if ACTIVE_WATCHER_SESSION and ACTIVE_WATCHER_SESSION.is_running:
        try:
            res = ACTIVE_WATCHER_SESSION.trigger_manual_clip(clip_duration=req.duration)
            return {
                "status": "TRIGGERED",
                "mode": "live_stream",
                "message": f"Triggered {req.duration:.0f}s clip from live {ACTIVE_WATCHER_SESSION.platform} stream",
                "detail": res,
            }
        except RuntimeError as exc:
            LOGGER.warning("Manual live clip trigger failed: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))

    # 2. Standalone test short rendering (when no live stream is actively being recorded)
    meta = create_test_short(
        output_dir=COMPLETED_DIR,
        layout=req.layout,
        encoder=req.encoder,
        duration_seconds=req.duration,
    )
    return {
        "status": "COMPLETED",
        "mode": "standalone_test",
        "message": "Successfully rendered test reel into shorts directory",
        "short": meta,
    }


# ---------------------------------------------------------------------------
# YouTube Integration & Upload Queue Endpoints
# ---------------------------------------------------------------------------

class QueueAddRequest(BaseModel):
    items: list[dict[str, Any]]


class QueueUpdateRequest(BaseModel):
    item_id: str
    updates: dict[str, Any]


class QueueRemoveRequest(BaseModel):
    item_id: str


class UploadNowRequest(BaseModel):
    video_path: str
    title: str
    description: str = ""
    privacy: str = "public"
    tags: list[str] | None = None


@app.get("/api/youtube/status")
def api_youtube_status():
    """Return connected YouTube channel profile and statistics."""
    from app.youtube.client import get_channel_info
    return get_channel_info()


@app.get("/api/youtube/auth-url")
def api_youtube_auth_url(request: Request):
    """Generate Google OAuth authorization URL for YouTube upload permissions."""
    from app.youtube.client import get_auth_url
    redirect_uri = f"{request.base_url}api/youtube/oauth2callback".rstrip("/")
    if "127.0.0.1" in redirect_uri:
        redirect_uri = redirect_uri.replace("127.0.0.1", "localhost")
    try:
        auth_url, state = get_auth_url(redirect_uri)
        return {"auth_url": auth_url, "state": state, "redirect_uri": redirect_uri}
    except Exception as exc:
        LOGGER.error("Failed to generate YouTube auth URL: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/youtube/oauth2callback")
def api_youtube_oauth_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """OAuth redirect handler from Google consent screen."""
    if error:
        return HTMLResponse(f"""<!DOCTYPE html><html><body style="font-family:sans-serif;padding:30px;">
        <script>
            if (window.opener) {{
                window.opener.postMessage({{ type: 'YOUTUBE_AUTH_ERROR', error: '{error}' }}, '*');
                window.close();
            }} else {{
                window.location.href = '/?tab=youtube&auth_error={urllib.parse.quote(error)}';
            }}
        </script>
        <h3>Authentication Failed</h3>
        <p>Google returned error: {error}. You may close this window.</p>
        </body></html>""")

    from app.youtube.client import exchange_code
    redirect_uri = f"{request.base_url}api/youtube/oauth2callback".rstrip("/")
    if "127.0.0.1" in redirect_uri:
        redirect_uri = redirect_uri.replace("127.0.0.1", "localhost")

    try:
        info = exchange_code(code, redirect_uri, state=state)
        return HTMLResponse(f"""<!DOCTYPE html><html><body style="font-family:sans-serif;padding:30px;">
        <script>
            if (window.opener) {{
                window.opener.postMessage({{ type: 'YOUTUBE_AUTH_SUCCESS' }}, '*');
                window.close();
            }} else {{
                window.location.href = '/?tab=youtube&auth=success';
            }}
        </script>
        <h3>YouTube Connected!</h3>
        <p>Your channel has been linked successfully. Returning to dashboard...</p>
        </body></html>""")
    except Exception as exc:
        LOGGER.error("YouTube OAuth token exchange failed: %s", exc)
        return HTMLResponse(f"""<!DOCTYPE html><html><body style="font-family:sans-serif;padding:30px;">
        <script>
            if (window.opener) {{
                window.opener.postMessage({{ type: 'YOUTUBE_AUTH_ERROR', error: '{exc}' }}, '*');
                window.close();
            }} else {{
                window.location.href = '/?tab=youtube&auth_error={urllib.parse.quote(str(exc))}';
            }}
        </script>
        <h3>Authentication Error</h3>
        <p>{exc}. You may close this window.</p>
        </body></html>""")


@app.post("/api/youtube/disconnect")
def api_youtube_disconnect():
    """Disconnect active YouTube account by unlinking credentials."""
    from app.youtube.client import disconnect
    success = disconnect()
    return {"status": "DISCONNECTED", "success": success}


@app.get("/api/youtube/queue")
def api_youtube_queue():
    """Get current YouTube upload queue and upload worker status."""
    from app.youtube.queue_manager import GLOBAL_YOUTUBE_QUEUE
    return {
        "queue": GLOBAL_YOUTUBE_QUEUE.get_queue(),
        "is_uploading": GLOBAL_YOUTUBE_QUEUE.is_uploading,
    }


@app.post("/api/youtube/queue/add")
def api_youtube_queue_add(req: QueueAddRequest):
    """Add selected videos to YouTube upload queue."""
    from app.youtube.queue_manager import GLOBAL_YOUTUBE_QUEUE
    added = GLOBAL_YOUTUBE_QUEUE.add_to_queue(req.items)
    return {"added_count": len(added), "added": added}


@app.post("/api/youtube/queue/remove")
def api_youtube_queue_remove(req: QueueRemoveRequest):
    """Remove a video from YouTube upload queue."""
    from app.youtube.queue_manager import GLOBAL_YOUTUBE_QUEUE
    success = GLOBAL_YOUTUBE_QUEUE.remove_item(req.item_id)
    return {"success": success}


@app.post("/api/youtube/queue/update")
def api_youtube_queue_update(req: QueueUpdateRequest):
    """Update title, description, tags, or privacy for a queued item."""
    from app.youtube.queue_manager import GLOBAL_YOUTUBE_QUEUE
    updated = GLOBAL_YOUTUBE_QUEUE.update_item(req.item_id, req.updates)
    return {"updated": updated}


@app.post("/api/youtube/queue/clear")
def api_youtube_queue_clear():
    """Clear completed and failed items from upload queue."""
    from app.youtube.queue_manager import GLOBAL_YOUTUBE_QUEUE
    cleared = GLOBAL_YOUTUBE_QUEUE.clear_completed()
    return {"cleared_count": cleared}


@app.post("/api/youtube/queue/start")
def api_youtube_queue_start():
    """Start background sequential upload worker for pending items."""
    from app.youtube.queue_manager import GLOBAL_YOUTUBE_QUEUE
    res = GLOBAL_YOUTUBE_QUEUE.start_upload_worker()
    return res


@app.post("/api/youtube/upload-now")
def api_youtube_upload_now(req: UploadNowRequest):
    """Directly upload a single Short immediately and free storage upon success."""
    from app.youtube.uploader import upload_short
    from app.youtube.queue_manager import delete_uploaded_short_storage
    try:
        res = upload_short(
            video_path=req.video_path,
            title=req.title,
            description=req.description,
            tags=req.tags,
            privacy=req.privacy,
        )
        # Free storage space once uploaded
        cleanup = delete_uploaded_short_storage(req.video_path)
        res["freed_mb"] = cleanup.get("freed_mb", 0.0)
        return res
    except Exception as exc:
        LOGGER.error("Direct YouTube upload failed: %s", exc)
        err_msg = str(exc)
        if "uploadLimitExceeded" in err_msg:
            err_msg = "⚠️ Daily YouTube upload limit reached (~5-10 videos/day for new channels). Your quota resets in 24h. You can immediately increase this limit by verifying your channel in YouTube Studio (Settings > Channel > Feature Eligibility > Advanced Features)."
        raise HTTPException(status_code=400, detail=err_msg)


class QueueEnhanceRequest(BaseModel):
    item_id: str
    creator: str = ""


class MetadataGenerateRequest(BaseModel):
    dialogue: str = ""
    creator: str = ""
    platform: str = "youtube"
    category: str = "funny"
    spike_ratio: float = 0.0
    summary: str = ""


@app.post("/api/youtube/queue/enhance")
def api_youtube_queue_enhance(req: QueueEnhanceRequest):
    """Auto-enhance or re-roll viral title, description, and tags for a queued item."""
    from app.youtube.queue_manager import GLOBAL_YOUTUBE_QUEUE
    enhanced = GLOBAL_YOUTUBE_QUEUE.enhance_item(req.item_id, req.creator)
    if not enhanced:
        raise HTTPException(status_code=404, detail="Item not found in queue")
    return {"status": "ENHANCED", "item": enhanced}


@app.post("/api/metadata/generate")
def api_metadata_generate(req: MetadataGenerateRequest):
    """Generate attention-grabbing viral package on-demand."""
    from app.evaluation.viral_metadata import ViralMetadataGenerator
    viral = ViralMetadataGenerator.generate(
        dialogue=req.dialogue,
        creator=req.creator,
        platform=req.platform,
        category=req.category,
        spike_ratio=req.spike_ratio,
        raw_summary=req.summary,
        use_ai=True,
    )
    return viral.to_dict()


# Minimalist, White Background Dashboard HTML
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StreamClip Studio</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #ffffff;
      color: #111827;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      padding: 32px 24px;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid #e5e7eb;
      margin-bottom: 26px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand h1 {
      font-size: 19px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #111827;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 500;
      background: #f3f4f6;
      color: #374151;
      border: 1px solid #e5e7eb;
    }
    .badge-success { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
    .badge-active { background: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
    .badge-rating { background: #fef3c7; color: #92400e; border-color: #fde68a; font-weight: 600; }
    .badge-danger { background: #fef2f2; color: #991b1b; border-color: #fecaca; }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 26px;
    }
    @media (max-width: 840px) {
      .grid-2 { grid-template-columns: 1fr; }
    }

    .card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 20px;
    }
    .card-title {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #6b7280;
      margin-bottom: 16px;
    }

    .form-group {
      margin-bottom: 14px;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: #374151;
      margin-bottom: 5px;
    }
    input, select {
      width: 100%;
      padding: 8px 10px;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      font-size: 13px;
      background: #ffffff;
      color: #111827;
      outline: none;
      transition: border-color 0.15s;
    }
    input:focus, select:focus {
      border-color: #111827;
    }
    .field-hint {
      display: block;
      font-size: 11px;
      color: #6b7280;
      margin-top: 3px;
    }

    .row {
      display: flex;
      gap: 12px;
    }
    .row > * { flex: 1; }

    button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 14px;
      background: #111827;
      color: #ffffff;
      border: 1px solid #111827;
      border-radius: 4px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }
    button:hover {
      background: #27272a;
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    button.secondary {
      background: #ffffff;
      color: #111827;
      border: 1px solid #d1d5db;
    }
    button.secondary:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }
    button.btn-xs {
      padding: 4px 8px;
      font-size: 11px;
      border-radius: 4px;
    }
    button.btn-danger-subtle {
      background: #ffffff;
      color: #dc2626;
      border: 1px solid #fca5a5;
    }
    button.btn-danger-subtle:hover {
      background: #fef2f2;
      border-color: #ef4444;
    }
    button.btn-danger {
      background: #dc2626;
      color: #ffffff;
      border: 1px solid #dc2626;
    }
    button.btn-danger:hover {
      background: #b91c1c;
    }

    /* Dropzone */
    .dropzone {
      border: 2px dashed #d1d5db;
      border-radius: 6px;
      padding: 14px 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.15s;
      background: #fafafa;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: #111827;
      background: #f3f4f6;
    }

    /* Status Bar */
    .status-panel {
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 16px 20px;
      margin-bottom: 28px;
      background: #fafafa;
    }
    .status-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .progress-bar-bg {
      width: 100%;
      height: 4px;
      background: #e5e7eb;
      border-radius: 2px;
      overflow: hidden;
      margin-bottom: 12px;
    }
    .progress-bar-fill {
      height: 100%;
      background: #111827;
      width: 0%;
      transition: width 0.3s;
    }
    .log-box {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      color: #374151;
      padding: 10px 12px;
      max-height: 120px;
      overflow-y: auto;
      white-space: pre-wrap;
    }

    /* Shorts Grid */
    .shorts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }
    .short-card {
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      background: #ffffff;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .video-wrapper {
      position: relative;
      background: #000000;
      width: 100%;
      padding-top: 177.77%; /* 9:16 Aspect Ratio */
    }
    .video-wrapper video {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .short-body {
      padding: 16px;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .short-hook {
      font-weight: 700;
      font-size: 14px;
      line-height: 1.35;
      color: #111827;
    }
    .short-desc {
      font-size: 12px;
      color: #6b7280;
      flex: 1;
    }
    .short-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 8px;
      border-top: 1px solid #f3f4f6;
      font-size: 11px;
      color: #9ca3af;
    }
    .short-actions {
      display: flex;
      gap: 6px;
      margin-top: 6px;
    }
    .short-actions a {
      flex: 1;
      text-align: center;
      padding: 6px 10px;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 500;
      text-decoration: none;
      color: #111827;
      background: #ffffff;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .short-actions a:hover {
      background: #f9fafb;
    }

    /* Candidate Table */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 10px;
    }
    th {
      text-align: left;
      padding: 8px 12px;
      background: #f9fafb;
      color: #6b7280;
      font-weight: 600;
      border-bottom: 1px solid #e5e7eb;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid #f3f4f6;
      color: #374151;
    }
    tr:last-child td { border-bottom: none; }

    /* Modals */
    .modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(17, 24, 39, 0.4);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }
    .modal-card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      width: 100%;
      max-width: 520px;
      padding: 22px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .btn-close {
      background: transparent;
      border: none;
      font-size: 20px;
      line-height: 1;
      cursor: pointer;
      color: #9ca3af;
      padding: 2px 6px;
    }
    .btn-close:hover {
      color: #111827;
    }
    .checkbox-row {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      font-size: 12px;
      cursor: pointer;
      font-weight: normal;
      color: #374151;
    }
    .checkbox-row input[type="checkbox"] {
      width: auto;
      margin-top: 2px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1>StreamClip Studio</h1>
        <span class="badge" id="gpu-pill">RTX 4070 Ready</span>
        <span class="badge" id="groq-pill" onclick="openSettingsModal()" style="cursor: pointer;" title="Click to view/set Groq API Key">Groq Free Tier</span>
      </div>
      <div class="header-actions">
        <button class="secondary" onclick="refreshDashboard()" title="Refresh All Data">
          Refresh
        </button>
        <button class="secondary" onclick="openSettingsModal()" title="Groq API Key Settings">
          Groq Key
        </button>
        <button class="secondary btn-danger-subtle" onclick="openCleanupModal()" title="Clean Storage, Videos & DB">
          Storage & Reset
        </button>
      </div>
    </header>

    <!-- Dashboard Mode Switcher Tabs -->
    <div style="display: flex; gap: 8px; margin-bottom: 22px; border-bottom: 1px solid #e5e7eb; padding-bottom: 12px; flex-wrap: wrap;">
      <button id="tab-btn-vod" class="primary" onclick="setDashboardTab('vod')" style="padding: 7px 16px; font-size: 13px;">
        📁 Video Files & VOD Mode
      </button>
      <button id="tab-btn-live" class="secondary" onclick="setDashboardTab('live')" style="padding: 7px 16px; font-size: 13px;">
        🔴 Live Watcher (Twitch & Kick Spike Detection)
      </button>
      <button id="tab-btn-youtube" class="secondary" onclick="setDashboardTab('youtube')" style="padding: 7px 16px; font-size: 13px; display: inline-flex; align-items: center; gap: 6px;">
        <span>📺 YouTube & Upload Queue</span>
        <span id="nav-queue-badge" class="badge badge-active" style="display:none; font-size:10px; padding: 1px 6px;">0</span>
      </button>
    </div>

    <!-- Ingestion & Configuration Row -->
    <div id="panel-vod" class="grid-2">
      <!-- Select or Ingest Video -->
      <div class="card">
        <div class="card-title">1. Select Video & Ingestion</div>
        
        <div class="form-group">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <label for="video-select" style="margin-bottom: 0;">Inbox Video</label>
            <div style="display: flex; gap: 6px;">
              <button class="btn-xs btn-danger-subtle" onclick="deleteSelectedVideo()" title="Delete this video from inbox">Delete Video</button>
              <button class="btn-xs secondary" onclick="clearAllVideos()" title="Delete all videos from inbox">Clear All</button>
            </div>
          </div>
          <select id="video-select" onchange="onVideoSelected()">
            <option value="">Loading videos...</option>
          </select>
        </div>

        <!-- Direct Upload Dropzone -->
        <div class="form-group">
          <label>Upload Video from Computer</label>
          <div id="dropzone" class="dropzone" onclick="document.getElementById('file-uploader').click()">
            <input type="file" id="file-uploader" accept="video/mp4,video/x-matroska,video/quicktime,video/webm" style="display:none">
            <div style="font-size: 13px; font-weight: 500; color: #374151;">📁 Drop video file here or click to browse</div>
            <div style="font-size: 11px; color: #9ca3af; margin-top: 2px;">MP4, MKV, MOV, WebM (auto-saved to data/inbox)</div>
          </div>
          <div id="upload-progress-container" style="display: none; margin-top: 8px;">
            <div class="progress-bar-bg" style="margin-bottom: 4px;">
              <div class="progress-bar-fill" id="upload-progress-bar" style="width: 0%;"></div>
            </div>
            <div style="font-size: 11px; color: #4b5563;" id="upload-progress-text">Uploading...</div>
          </div>
        </div>

        <div class="form-group">
          <label for="yt-url">Or Download from YouTube</label>
          <div style="display: flex; gap: 8px;">
            <input type="text" id="yt-url" placeholder="https://www.youtube.com/watch?v=...">
            <button class="secondary" onclick="downloadYoutube()" style="white-space: nowrap;">Fetch</button>
          </div>
          <div style="display: flex; gap: 8px; align-items: center; margin-top: 6px; flex-wrap: wrap;">
            <span style="font-size: 11px; color: #6b7280; font-weight: 500;">Auth / Cookies:</span>
            <select id="yt-cookies-mode" style="padding: 2px 6px; font-size: 11px; width: auto;">
              <option value="auto" selected>Auto (cookies.txt if present)</option>
              <option value="chrome">Browser: Chrome</option>
              <option value="edge">Browser: Edge</option>
              <option value="firefox">Browser: Firefox</option>
              <option value="brave">Browser: Brave</option>
              <option value="none">None (Public Videos)</option>
            </select>
            <button class="btn-xs secondary" type="button" onclick="document.getElementById('cookies-file-input').click()" title="Upload cookies.txt for age-restricted/member videos">📁 Upload cookies.txt</button>
            <input type="file" id="cookies-file-input" accept=".txt" style="display: none;" onchange="uploadCookiesFile(this)">
            <span id="cookies-status-badge" class="badge" style="font-size: 10px;">Cookies: Checking...</span>
          </div>
          <span class="field-hint" style="margin-top: 4px;">Node.js runtime active (solves challenge scripts). For age-restricted videos, upload <code>cookies.txt</code> or select your browser.</span>
        </div>

        <div class="form-group" style="margin-bottom: 0;">
          <label for="creator-name">Creator / Streamer Identity</label>
          <input type="text" id="creator-name" placeholder="RawRawKnee" value="RawRawKnee">
        </div>
      </div>

      <!-- Pipeline Configuration -->
      <div class="card">
        <div class="card-title">2. AI Highlight & Render Settings</div>
        
        <div class="form-group">
          <label for="provider-select">Highlight Judge Provider</label>
          <select id="provider-select">
            <option value="groq" selected>Groq Cloud (Llama 3.3 70B & Vision, Free Tier)</option>
            <option value="local-qwen">Local Qwen2.5-VL-7B (RTX 4070 GPU, Free Offline)</option>
          </select>
        </div>

        <div class="row">
          <div class="form-group">
            <label for="layout-select">Vertical 9:16 Layout</label>
            <select id="layout-select">
              <option value="center_crop" selected>Full Screen 9:16 (Center Crop - Best for Mobile & Retention)</option>
              <option value="stacked">Stacked 9:16 (Top/Bottom Split)</option>
              <option value="blurred_background">Blurred Background (Cinematic Letterbox)</option>
            </select>
          </div>
          <div class="form-group">
            <label for="encoder-select">Video Encoder</label>
            <select id="encoder-select">
              <option value="auto" selected>Auto (Hardware NVENC if available)</option>
              <option value="h264_nvenc">NVIDIA NVENC (RTX 4070 Ultra Fast)</option>
              <option value="libx264">CPU libx264 (Software)</option>
            </select>
          </div>
        </div>

        <div class="row">
          <div class="form-group">
            <label for="min-shorts">Minimum Shorts (Guaranteed)</label>
            <input type="number" id="min-shorts" min="1" max="10" value="2">
            <span class="field-hint">Guarantees at least this many clips</span>
          </div>
          <div class="form-group">
            <label for="max-shorts">Maximum Shorts (Cap)</label>
            <input type="number" id="max-shorts" min="1" max="15" value="3">
            <span class="field-hint">Maximum clips to produce</span>
          </div>
          <div class="form-group">
            <label for="min-score">Target Quality</label>
            <input type="number" id="min-score" min="1" max="5" value="3">
            <span class="field-hint">Min star rating threshold (1-5)</span>
          </div>
        </div>

        <div style="display: flex; gap: 8px; margin-top: 10px;">
          <button style="flex: 2;" id="btn-extract" onclick="startProcessing()">Extract Real Shorts</button>
          <button class="secondary btn-danger-subtle" id="btn-stop" style="flex: 1; display: none;" onclick="stopActivePipeline()">Stop Pipeline</button>
        </div>
      </div>
    </div>

    <!-- Live Twitch & Kick Watcher Panel -->
    <div id="panel-live" class="grid-2" style="display: none;">
      <!-- Card 1: Channel & Rules -->
      <div class="card">
        <div class="card-title">1. Live Stream Source & Trigger Rules</div>
        
        <div class="row">
          <div class="form-group" style="flex: 1;">
            <label for="live-platform">Streaming Platform</label>
            <select id="live-platform" onchange="onPlatformChange()">
              <option value="twitch" selected>Twitch (irc.chat.twitch.tv)</option>
              <option value="kick">Kick (Centrifugo WebSocket)</option>
            </select>
          </div>
          <div class="form-group" style="flex: 2;">
            <label id="live-channel-label" for="live-channel">Twitch Channel Username</label>
            <input type="text" id="live-channel" placeholder="e.g. kaicenat, rawrawknee, tarik, shroud" value="rawrawknee">
          </div>
        </div>
        <span id="live-channel-hint" class="field-hint" style="margin-top: -6px; margin-bottom: 12px; display: block;">Connects anonymously to <code>irc.chat.twitch.tv:6667</code> (NICK justinfan, zero tokens required)</span>

        <div class="row">
          <div class="form-group">
            <label for="live-min-msgs">Min Messages Threshold</label>
            <input type="number" id="live-min-msgs" value="5" min="1" max="500">
            <span class="field-hint">Min messages in 10s</span>
          </div>
          <div class="form-group">
            <label for="live-spike-factor">Spike Sensitivity Factor</label>
            <input type="number" id="live-spike-factor" value="2.5" step="0.1" min="1.1" max="10.0">
            <span class="field-hint">x median of preceding 3 min</span>
          </div>
        </div>

        <div class="form-group">
          <label for="live-ai-mode">AI Highlight Judge & Viral Hook</label>
          <select id="live-ai-mode">
            <option value="groq" selected>Groq Cloud (Llama 3.3 70B & Vision, Free Tier)</option>
            <option value="qwen">Local Qwen2.5-VL-7B (NVIDIA RTX 4070)</option>
            <option value="none">Direct Fast Cut (No AI)</option>
          </select>
        </div>

        <div class="row">
          <div class="form-group">
            <label for="live-layout">Vertical 9:16 Layout</label>
            <select id="live-layout">
              <option value="center_crop" selected>Full Screen 9:16 (Center Crop - Best for Mobile & Retention)</option>
              <option value="blurred_background">Blurred Background (Cinematic Letterbox)</option>
            </select>
          </div>
          <div class="form-group">
            <label for="live-encoder">Video Encoder</label>
            <select id="live-encoder">
              <option value="auto" selected>Auto (Hardware NVENC)</option>
              <option value="h264_nvenc">NVIDIA NVENC</option>
              <option value="libx264">CPU libx264</option>
            </select>
          </div>
        </div>

        <div style="display: flex; gap: 8px; margin-top: 10px;">
          <button style="flex: 2;" id="btn-start-watcher" class="primary" onclick="startWatcher()">Start Watching Stream</button>
          <button class="secondary btn-danger-subtle" id="btn-stop-watcher" style="flex: 1; display: none;" onclick="stopWatcher()">Stop Watcher</button>
          <button style="flex: 1.2;" id="btn-test-clip" class="secondary" onclick="triggerTestClip()" title="Instantly cut a 30-second test reel">⚡ Test 30s Reel</button>
        </div>
      </div>

      <!-- Card 2: Real-time Telemetry & Metrics -->
      <div class="card">
        <div class="card-title">2. Live Telemetry & Audience Excitement Meter</div>

        <div style="display: flex; gap: 8px; margin-bottom: 16px;">
          <span class="badge" id="live-stream-badge">Stream: Inactive</span>
          <span class="badge" id="live-irc-badge">IRC: Disconnected</span>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
          <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px;">
            <div style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 600;">Current Velocity</div>
            <div id="live-metric-bucket" style="font-size: 24px; font-weight: 700; color: #111827; margin: 4px 0;">0</div>
            <div style="font-size: 11px; color: #9ca3af;">msgs in current 10s</div>
          </div>
          <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px;">
            <div style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 600;">3-Min Median Baseline</div>
            <div id="live-metric-median" style="font-size: 24px; font-weight: 700; color: #111827; margin: 4px 0;">0.0</div>
            <div style="font-size: 11px; color: #9ca3af;">msgs / 10s baseline</div>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
          <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px;">
            <div style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 600;">Spikes Detected</div>
            <div id="live-metric-spikes" style="font-size: 20px; font-weight: 700; color: #059669; margin: 4px 0;">0</div>
            <div style="font-size: 11px; color: #9ca3af;">chat spikes triggered</div>
          </div>
          <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px;">
            <div style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 600;">Shorts Produced</div>
            <div id="live-metric-shorts" style="font-size: 20px; font-weight: 700; color: #2563eb; margin: 4px 0;">0</div>
            <div style="font-size: 11px; color: #9ca3af;">ready in gallery below</div>
          </div>
        </div>

        <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; font-size: 11px; color: #4b5563;">
          <div><strong>Speech Boundary Snapping:</strong> Active (+1.8s punchline breathing room)</div>
          <div style="margin-top: 2px;"><strong>Subtitles:</strong> faster-whisper GPU burned-in karaoke captions</div>
        </div>
      </div>
    </div>

    <!-- YouTube Publisher & Upload Queue Panel -->
    <div id="panel-youtube" style="display: none; margin-bottom: 26px;">
      <!-- Channel Connection Banner Card -->
      <div class="card" style="margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div id="yt-channel-disconnected" style="display: flex; align-items: center; gap: 14px;">
          <div style="width: 44px; height: 44px; border-radius: 50%; background: #fee2e2; display: flex; align-items: center; justify-content: center; font-size: 20px;">
            📺
          </div>
          <div>
            <div style="font-weight: 700; font-size: 15px; color: #111827;">YouTube Account Not Connected</div>
            <div style="font-size: 12px; color: #6b7280;" id="yt-disconnected-subtitle">
              Connect your YouTube channel using Google OAuth to upload 9:16 Shorts with automated #Shorts tags.
            </div>
          </div>
        </div>
        <div id="yt-channel-connected" style="display: none; align-items: center; gap: 14px;">
          <img id="yt-channel-avatar" src="" alt="Avatar" style="width: 44px; height: 44px; border-radius: 50%; border: 1px solid #e5e7eb; object-fit: cover;">
          <div>
            <div style="display: flex; align-items: center; gap: 8px;">
              <span id="yt-channel-title" style="font-weight: 700; font-size: 15px; color: #111827;">Channel Name</span>
              <span class="badge badge-success">✓ YouTube Connected</span>
            </div>
            <div style="font-size: 12px; color: #6b7280; display: flex; gap: 12px; margin-top: 3px;">
              <span id="yt-channel-subs">0 subscribers</span>
              <span>•</span>
              <span id="yt-channel-videos">0 videos</span>
              <span>•</span>
              <a id="yt-channel-link" href="#" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 500;">View Channel ↗</a>
            </div>
          </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
          <button id="btn-yt-connect" class="primary" onclick="connectYouTube()" style="background: #ef4444; border-color: #ef4444; color: #ffffff;">
            🔴 Connect YouTube Channel
          </button>
          <button id="btn-yt-disconnect" class="secondary btn-danger-subtle" onclick="disconnectYouTube()" style="display: none;">
            Disconnect Channel
          </button>
        </div>
      </div>

      <!-- Viral Reach & Algorithm Strategy Card -->
      <div style="margin-bottom: 20px; padding: 14px 18px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; font-size: 11px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
        <div>
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 3px;">⚡ The 0.5s Hook Rule</div>
          <div style="color: #64748b; line-height: 1.4;">Algorithm halts impressions if swipe-away > 30%. Your shorts now start directly on action with bold burned-in hooks.</div>
        </div>
        <div>
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 3px;">🎯 Target Retention: >100%</div>
          <div style="color: #64748b; line-height: 1.4;">15–20s tight clips achieve 100%+ average percentage viewed on the first loop, signaling YouTube to blast it to thousands.</div>
        </div>
        <div>
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 3px;">📅 Optimal Posting Cadence</div>
          <div style="color: #64748b; line-height: 1.4;">Post 1 to 2 Shorts per day (peak hours: 4 PM – 8 PM). Never upload 5+ at once to avoid spam limits.</div>
        </div>
        <div>
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 3px;">💬 Pin a Conversation Comment</div>
          <div style="color: #64748b; line-height: 1.4;">Comments are weighted 3x heavier than likes. Pin a spicy question right after uploading to start debates.</div>
        </div>
      </div>

      <!-- Main Queue & Selection Grid -->
      <div class="grid-2" style="grid-template-columns: 1.15fr 0.85fr; align-items: start; gap: 20px;">
        <!-- Left: Rendered Shorts Selector Library -->
        <div class="card" style="padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
              <div class="card-title" style="margin-bottom: 2px;">1. Select Shorts For Upload Queue</div>
              <div style="font-size: 12px; color: #6b7280;" id="yt-library-subtitle">Choose produced Shorts to publish</div>
            </div>
            <div style="display: flex; gap: 6px;">
              <button class="btn-xs secondary" onclick="selectAllShortsForQueue(true)">Select All</button>
              <button class="btn-xs secondary" onclick="selectAllShortsForQueue(false)">Deselect</button>
              <button class="btn-xs secondary" onclick="loadYouTubeLibrary()">Refresh</button>
            </div>
          </div>

          <div style="margin-bottom: 14px;">
            <button id="btn-add-selected-to-queue" class="primary" onclick="addSelectedShortsToQueue()" style="width: 100%; justify-content: center;" disabled>
              ➕ Add Selected to Upload Queue (0)
            </button>
          </div>

          <div id="yt-shorts-selection-list" style="max-height: 560px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; padding-right: 4px;">
            <div style="text-align: center; padding: 40px 10px; color: #9ca3af; font-size: 12px;">
              Loading produced shorts...
            </div>
          </div>
        </div>

        <!-- Right: Upload Queue & Sequential Publisher -->
        <div class="card" style="padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
              <div class="card-title" style="margin-bottom: 2px;">2. YouTube Upload Queue</div>
              <div style="font-size: 12px; color: #6b7280;" id="yt-queue-summary">0 items in queue</div>
            </div>
            <div style="display: flex; gap: 6px;">
              <button class="btn-xs secondary" onclick="enhanceAllQueueItems()" title="Generate viral titles & tags for all pending queue items">✨ Enhance All</button>
              <button class="btn-xs btn-danger-subtle" onclick="clearCompletedQueue()">Clear Done</button>
              <button class="btn-xs secondary" onclick="fetchYouTubeQueue()">Refresh</button>
            </div>
          </div>

          <div style="margin-bottom: 14px;">
            <button id="btn-start-queue-upload" class="primary" onclick="startQueueUpload()" style="width: 100%; justify-content: center; background: #059669; border-color: #059669; color: #fff;">
              🚀 Upload All Pending Now
            </button>
          </div>

          <div id="yt-queue-list" style="max-height: 560px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; padding-right: 4px;">
            <div style="text-align: center; padding: 40px 10px; color: #9ca3af; font-size: 12px;">
              Your upload queue is empty.<br>Check videos on the left and click "Add Selected to Upload Queue".
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Live Status / Progress Panel -->
    <div class="status-panel" id="status-panel">
      <div class="status-header">
        <div style="display: flex; align-items: center; gap: 10px;">
          <span style="font-weight: 600; font-size: 13px;" id="status-title">Ready</span>
          <span class="badge" id="status-badge">Idle</span>
        </div>
        <div style="display: flex; gap: 6px;">
          <button class="btn-xs secondary" onclick="dismissStatus()" title="Dismiss this status from view">Dismiss Status</button>
          <button class="btn-xs btn-danger-subtle" onclick="deleteCurrentJob()" title="Delete job runs and reset database record">Delete Job Output</button>
        </div>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="progress-bar"></div>
      </div>
      <div class="log-box" id="log-box">Select a video and click "Extract Real Shorts" to begin production.</div>
    </div>

    <!-- Completed Shorts Gallery -->
    <div id="gallery-header" style="margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;">
      <div>
        <h2 style="font-size: 16px; font-weight: 700;">Generated Shorts (9:16 Vertical)</h2>
        <p style="font-size: 12px; color: #6b7280;" id="shorts-summary">0 clips generated</p>
      </div>
      <div>
        <button class="btn-xs secondary btn-danger-subtle" onclick="clearCurrentShorts()" id="btn-clear-shorts" style="display: none;">Clear All Shorts</button>
      </div>
    </div>
    <div class="shorts-grid" id="shorts-grid">
      <!-- Dynamic Shorts Cards inserted here -->
    </div>

    <!-- Candidate Moments Inspector -->
    <div class="card" id="candidates-card" style="margin-top: 24px;">
      <div class="card-title">Candidate Moments & AI Judgments</div>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Category</th>
              <th>AI Rating</th>
              <th>Editorial Summary & Hook</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="candidates-body">
            <tr>
              <td colspan="5" style="text-align: center; color: #9ca3af; padding: 20px;">No candidate data loaded yet.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Storage & Reset Modal -->
  <div id="cleanup-modal" class="modal-backdrop">
    <div class="modal-card">
      <div class="modal-header">
        <h3 style="font-size: 15px; font-weight: 700;">Storage & Clean Slate</h3>
        <button class="btn-close" onclick="closeCleanupModal()">&times;</button>
      </div>
      <div style="font-size: 13px; color: #4b5563; margin-bottom: 16px;">
        Select which storage areas to clean, or choose total reset to wipe everything back to a fresh state.
      </div>
      <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">
        <label class="checkbox-row">
          <input type="checkbox" id="clean-processing" checked>
          <span><strong>Processing Temp Files</strong> (<code>data/processing/</code> intermediate chunks & frames)</span>
        </label>
        <label class="checkbox-row">
          <input type="checkbox" id="clean-completed">
          <span><strong>Completed Shorts</strong> (<code>data/completed/</code> rendered vertical reels)</span>
        </label>
        <label class="checkbox-row">
          <input type="checkbox" id="clean-inbox">
          <span><strong>Inbox Source Videos</strong> (<code>data/inbox/</code> uploaded and downloaded VODs)</span>
        </label>
        <label class="checkbox-row">
          <input type="checkbox" id="clean-db">
          <span><strong>Reset Database History</strong> (Clear all entries in <code>video_jobs.sqlite3</code>)</span>
        </label>
      </div>
      <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #e5e7eb; padding-top: 16px;">
        <button class="secondary btn-danger-subtle" onclick="executeCleanup(true)">Total Wipe (Reset All)</button>
        <div style="display: flex; gap: 8px;">
          <button class="secondary" onclick="closeCleanupModal()">Cancel</button>
          <button onclick="executeCleanup(false)">Clean Selected</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Groq Settings Modal -->
  <div id="settings-modal" class="modal-backdrop">
    <div class="modal-card">
      <div class="modal-header">
        <h3 style="font-size: 15px; font-weight: 700;">Groq Cloud API Key</h3>
        <button class="btn-close" onclick="closeSettingsModal()">&times;</button>
      </div>
      <div style="font-size: 13px; color: #4b5563; margin-bottom: 14px;">
        Groq provides free high-speed Llama 3.3 70B & Vision Highlight Judging (8,000 TPM limit). The pipeline includes automatic backoff and retry handling.
      </div>
      <div class="form-group">
        <label>Current Status</label>
        <div id="groq-current-status" style="font-size: 12px; font-weight: 500; color: #374151; padding: 6px 10px; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 4px; margin-bottom: 12px;">Checking...</div>
      </div>
      <div class="form-group">
        <label for="groq-key-input">Groq API Key (<code>gsk_...</code>)</label>
        <input type="password" id="groq-key-input" placeholder="gsk_..." autocomplete="off">
        <span class="field-hint">Stored safely in your local environment.</span>
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 8px; border-top: 1px solid #e5e7eb; padding-top: 16px; margin-top: 16px;">
        <button class="secondary" onclick="closeSettingsModal()">Close</button>
        <button onclick="saveGroqKey()">Save Key</button>
      </div>
    </div>
  </div>

  <script>
    let currentJobId = null;
    let lastShortsCount = 0;

    async function init() {
      setupDragAndDrop();
      setupMinMaxSync();
      await loadVideos();
      await loadSettings();
      await loadAllShorts();
      await checkCookiesStatus();
      await pollStatus();
      await pollWatcherStatus();
      await fetchYouTubeStatus();
      await fetchYouTubeQueue();
      setInterval(pollStatus, 3000);
      setInterval(pollWatcherStatus, 2000);
      setInterval(fetchYouTubeQueue, 4000);

      const urlParams = new URLSearchParams(window.location.search);
      if (urlParams.get('tab') === 'youtube') {
        setDashboardTab('youtube');
      }
    }

    function setDashboardTab(tab) {
      const panelVod = document.getElementById('panel-vod');
      const panelLive = document.getElementById('panel-live');
      const panelYt = document.getElementById('panel-youtube');
      const tabVod = document.getElementById('tab-btn-vod');
      const tabLive = document.getElementById('tab-btn-live');
      const tabYt = document.getElementById('tab-btn-youtube');
      const statusPanel = document.getElementById('status-panel');
      const galleryHeader = document.getElementById('gallery-header');
      const shortsGrid = document.getElementById('shorts-grid');
      const candidatesCard = document.getElementById('candidates-card');

      if (tab === 'vod') {
        panelVod.style.display = 'grid';
        panelLive.style.display = 'none';
        panelYt.style.display = 'none';
        if (statusPanel) statusPanel.style.display = 'block';
        if (galleryHeader) galleryHeader.style.display = 'flex';
        if (shortsGrid) shortsGrid.style.display = 'grid';
        if (candidatesCard) candidatesCard.style.display = 'block';
        tabVod.className = 'primary';
        tabLive.className = 'secondary';
        tabYt.className = 'secondary';
      } else if (tab === 'live') {
        panelVod.style.display = 'none';
        panelLive.style.display = 'grid';
        panelYt.style.display = 'none';
        if (statusPanel) statusPanel.style.display = 'block';
        if (galleryHeader) galleryHeader.style.display = 'flex';
        if (shortsGrid) shortsGrid.style.display = 'grid';
        if (candidatesCard) candidatesCard.style.display = 'block';
        tabVod.className = 'secondary';
        tabLive.className = 'primary';
        tabYt.className = 'secondary';
      } else if (tab === 'youtube') {
        panelVod.style.display = 'none';
        panelLive.style.display = 'none';
        panelYt.style.display = 'block';
        if (statusPanel) statusPanel.style.display = 'none';
        if (galleryHeader) galleryHeader.style.display = 'none';
        if (shortsGrid) shortsGrid.style.display = 'none';
        if (candidatesCard) candidatesCard.style.display = 'none';
        tabVod.className = 'secondary';
        tabLive.className = 'secondary';
        tabYt.className = 'primary';
        refreshYouTubeTab();
      }
    }

    // ---------------------------------------------------------------------------
    // YouTube Integration & Upload Queue Logic
    // ---------------------------------------------------------------------------
    let ytAvailableShorts = [];
    let ytSelectedPaths = new Set();
    let ytQueuePollTimer = null;

    async function fetchYouTubeStatus() {
      try {
        const res = await fetch('/api/youtube/status');
        if (!res.ok) return;
        const data = await res.json();
        const disc = document.getElementById('yt-channel-disconnected');
        const conn = document.getElementById('yt-channel-connected');
        const btnConn = document.getElementById('btn-yt-connect');
        const btnDisc = document.getElementById('btn-yt-disconnect');

        if (data.connected && data.has_channel !== false) {
          disc.style.display = 'none';
          conn.style.display = 'flex';
          btnConn.style.display = 'none';
          btnDisc.style.display = 'inline-flex';

          document.getElementById('yt-channel-title').textContent = data.title || 'YouTube Channel';
          if (data.avatar_url) {
            document.getElementById('yt-channel-avatar').src = data.avatar_url;
            document.getElementById('yt-channel-avatar').style.display = 'block';
          } else {
            document.getElementById('yt-channel-avatar').style.display = 'none';
          }
          document.getElementById('yt-channel-subs').textContent = `${(data.subscriber_count || 0).toLocaleString()} subscribers`;
          document.getElementById('yt-channel-videos').textContent = `${(data.video_count || 0).toLocaleString()} videos`;
          const link = document.getElementById('yt-channel-link');
          if (data.custom_url) {
            link.href = `https://www.youtube.com/${data.custom_url.replace(/^@/, '')}`;
          } else if (data.channel_id) {
            link.href = `https://www.youtube.com/channel/${data.channel_id}`;
          }
        } else {
          disc.style.display = 'flex';
          conn.style.display = 'none';
          btnConn.style.display = 'inline-flex';
          btnDisc.style.display = 'none';
        }
      } catch (e) {
        console.error('Failed to fetch YouTube status:', e);
      }
    }

    async function connectYouTube() {
      try {
        const res = await fetch('/api/youtube/auth-url');
        if (!res.ok) {
          const err = await res.json();
          alert('Error preparing YouTube authorization: ' + (err.detail || 'Check client secret JSON file.'));
          return;
        }
        const data = await res.json();
        const width = 600, height = 750;
        const left = (window.innerWidth - width) / 2;
        const top = (window.innerHeight - height) / 2;
        const popup = window.open(data.auth_url, 'yt_oauth', `width=${width},height=${height},top=${top},left=${left}`);
        if (!popup || popup.closed || typeof popup.closed === 'undefined') {
          window.location.href = data.auth_url;
        }
      } catch (err) {
        alert('Could not start YouTube OAuth flow: ' + err);
      }
    }

    window.addEventListener('message', async (event) => {
      if (event.data && event.data.type === 'YOUTUBE_AUTH_SUCCESS') {
        await fetchYouTubeStatus();
        alert('YouTube channel connected successfully!');
      } else if (event.data && event.data.type === 'YOUTUBE_AUTH_ERROR') {
        alert('YouTube authorization error: ' + event.data.error);
      }
    });

    async function disconnectYouTube() {
      if (!confirm('Are you sure you want to disconnect this YouTube channel?')) return;
      try {
        const res = await fetch('/api/youtube/disconnect', { method: 'POST' });
        if (res.ok) {
          await fetchYouTubeStatus();
        }
      } catch (e) {
        alert('Error disconnecting YouTube: ' + e);
      }
    }

    async function loadYouTubeLibrary() {
      try {
        const res = await fetch('/api/shorts');
        if (!res.ok) return;
        const data = await res.json();
        ytAvailableShorts = data.shorts || [];
        renderYouTubeLibraryList();
      } catch (e) {
        console.error('Failed to load shorts library for YouTube tab:', e);
      }
    }

    function renderYouTubeLibraryList() {
      const container = document.getElementById('yt-shorts-selection-list');
      const subtitle = document.getElementById('yt-library-subtitle');
      if (!container) return;

      if (!ytAvailableShorts || ytAvailableShorts.length === 0) {
        container.innerHTML = `
          <div style="text-align: center; padding: 40px 10px; color: #9ca3af; font-size: 12px;">
            No rendered Shorts found yet.<br>Generate Shorts from VOD mode or Live Watcher first.
          </div>`;
        if (subtitle) subtitle.textContent = '0 produced Shorts available';
        return;
      }

      if (subtitle) subtitle.textContent = `${ytAvailableShorts.length} produced Shorts available`;

      container.innerHTML = ytAvailableShorts.map((s, idx) => {
        const isChecked = ytSelectedPaths.has(s.path);
        const titleText = escapeHtml(s.title || s.filename || 'Untitled Short');
        const hookText = s.suggested_hook ? escapeHtml(s.suggested_hook) : '';
        const durText = s.end_seconds && s.start_seconds ? `${Math.round(s.end_seconds - s.start_seconds)}s` : '';
        const sizeText = s.size_mb ? `${s.size_mb} MB` : '';
        const creatorTag = s.channel ? `@${escapeHtml(s.channel)}` : '';

        return `
          <div style="display: flex; align-items: center; gap: 10px; padding: 8px 10px; border: 1px solid ${isChecked ? '#2563eb' : '#e5e7eb'}; background: ${isChecked ? '#eff6ff' : '#ffffff'}; border-radius: 6px; transition: border 0.15s;">
            <input type="checkbox" class="yt-lib-check" data-path="${escapeHtml(s.path)}" data-idx="${idx}" ${isChecked ? 'checked' : ''} onchange="toggleShortSelection('${escapeHtml(s.path)}', this.checked)" style="width: 16px; height: 16px; cursor: pointer;">
            <div style="width: 36px; height: 60px; background: #000; border-radius: 4px; overflow: hidden; position: relative; flex-shrink: 0; cursor: pointer;" onclick="playShort('${escapeHtml(s.path)}', '${titleText}', '${hookText}', '')">
              <video src="${s.url}" style="width: 100%; height: 100%; object-fit: cover;" preload="metadata"></video>
              <div style="position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.25); color: #fff; font-size: 11px;">▶</div>
            </div>
            <div style="flex: 1; min-width: 0;">
              <div style="font-size: 13px; font-weight: 600; color: #111827; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${titleText}</div>
              ${hookText ? `<div style="font-size: 11px; color: #dc2626; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${hookText}</div>` : ''}
              <div style="font-size: 11px; color: #6b7280; display: flex; gap: 8px; margin-top: 2px;">
                ${durText ? `<span>${durText}</span>` : ''}
                ${sizeText ? `<span>${sizeText}</span>` : ''}
                ${creatorTag ? `<span style="color: #4b5563;">${creatorTag}</span>` : ''}
              </div>
            </div>
            <button class="btn-xs secondary" onclick="addSingleShortToQueue(${idx})" title="Add directly to upload queue" style="flex-shrink: 0;">
              + Queue
            </button>
          </div>
        `;
      }).join('');

      updateSelectionButton();
    }

    function toggleShortSelection(path, isChecked) {
      if (isChecked) {
        ytSelectedPaths.add(path);
      } else {
        ytSelectedPaths.delete(path);
      }
      renderYouTubeLibraryList();
    }

    function selectAllShortsForQueue(checked) {
      if (checked) {
        ytAvailableShorts.forEach(s => ytSelectedPaths.add(s.path));
      } else {
        ytSelectedPaths.clear();
      }
      renderYouTubeLibraryList();
    }

    function updateSelectionButton() {
      const btn = document.getElementById('btn-add-selected-to-queue');
      if (!btn) return;
      const count = ytSelectedPaths.size;
      btn.textContent = `➕ Add Selected to Upload Queue (${count})`;
      btn.disabled = (count === 0);
    }

    async function addSingleShortToQueue(idx) {
      const s = ytAvailableShorts[idx];
      if (!s) return;
      try {
        const payload = [{
          video_path: s.path,
          title: s.title || (s.suggested_hook ? `${s.suggested_hook} #Shorts` : s.filename),
          description: s.summary || `Exciting highlight Short from ${s.channel || 'stream'}! #Shorts`,
          tags: s.tags || [],
          suggested_hook: s.suggested_hook || '',
          channel: s.channel || '',
          privacy: 'public',
        }];
        const res = await fetch('/api/youtube/queue/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ items: payload }),
        });
        if (res.ok) {
          await fetchYouTubeQueue();
        }
      } catch (e) {
        alert('Error adding to queue: ' + e);
      }
    }

    async function addSelectedShortsToQueue() {
      if (ytSelectedPaths.size === 0) return;
      const selected = ytAvailableShorts.filter(s => ytSelectedPaths.has(s.path));
      const payload = selected.map(s => ({
        video_path: s.path,
        title: s.title || (s.suggested_hook ? `${s.suggested_hook} #Shorts` : s.filename),
        description: s.summary || `Exciting highlight Short from ${s.channel || 'stream'}! #Shorts`,
        tags: s.tags || [],
        suggested_hook: s.suggested_hook || '',
        channel: s.channel || '',
        privacy: 'public',
      }));

      try {
        const res = await fetch('/api/youtube/queue/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ items: payload }),
        });
        if (res.ok) {
          ytSelectedPaths.clear();
          renderYouTubeLibraryList();
          await fetchYouTubeQueue();
        }
      } catch (e) {
        alert('Error adding selected shorts to queue: ' + e);
      }
    }

    async function fetchYouTubeQueue() {
      try {
        const res = await fetch('/api/youtube/queue');
        if (!res.ok) return;
        const data = await res.json();
        const queue = data.queue || [];
        renderYouTubeQueueList(queue, data.is_uploading);

        const badge = document.getElementById('nav-queue-badge');
        if (badge) {
          if (queue.length > 0) {
            badge.style.display = 'inline-flex';
            badge.textContent = queue.length;
          } else {
            badge.style.display = 'none';
          }
        }

        if (data.is_uploading) {
          if (!ytQueuePollTimer) {
            ytQueuePollTimer = setInterval(fetchYouTubeQueue, 2000);
          }
        } else {
          if (ytQueuePollTimer) {
            clearInterval(ytQueuePollTimer);
            ytQueuePollTimer = null;
          }
        }
      } catch (e) {
        console.error('Failed to fetch YouTube queue:', e);
      }
    }

    function renderYouTubeQueueList(queue, isUploading) {
      const container = document.getElementById('yt-queue-list');
      const summary = document.getElementById('yt-queue-summary');
      const btnStart = document.getElementById('btn-start-queue-upload');
      if (!container) return;

      const pendingCount = queue.filter(q => q.status === 'pending').length;
      if (summary) {
        summary.textContent = `${queue.length} items (${pendingCount} pending)`;
      }

      if (btnStart) {
        if (isUploading) {
          btnStart.disabled = true;
          btnStart.textContent = '⏳ Upload In Progress...';
        } else {
          btnStart.disabled = (pendingCount === 0);
          btnStart.textContent = `🚀 Upload All Pending (${pendingCount})`;
        }
      }

      if (!queue || queue.length === 0) {
        container.innerHTML = `
          <div style="text-align: center; padding: 40px 10px; color: #9ca3af; font-size: 12px;">
            Your upload queue is empty.<br>Check videos on the left and click "Add Selected to Upload Queue".
          </div>`;
        return;
      }

      container.innerHTML = queue.map(item => {
        const titleVal = escapeHtml(item.title || '');
        const privacy = item.privacy || 'public';
        const tags = Array.isArray(item.tags) ? item.tags : [];
        const hookText = item.suggested_hook ? escapeHtml(item.suggested_hook) : '';

        let statusBadge = '';
        if (item.status === 'pending') {
          statusBadge = '<span class="badge">⏳ Pending</span>';
        } else if (item.status === 'uploading') {
          statusBadge = `<span class="badge badge-active">🚀 Uploading ${item.progress || 0}%</span>`;
        } else if (item.status === 'completed') {
          const freedLabel = item.freed_mb ? ` <span style="font-size: 10px; color: #059669; font-weight: 500; margin-left: 4px;">💾 Freed ${item.freed_mb}MB</span>` : (item.deleted_from_storage ? ` <span style="font-size: 10px; color: #059669; font-weight: 500; margin-left: 4px;">💾 Storage Freed</span>` : '');
          statusBadge = `<span class="badge badge-success">✓ Live on YouTube</span>${freedLabel}`;
        } else if (item.status === 'failed') {
          statusBadge = `<span class="badge badge-danger" title="${escapeHtml(item.error || 'Failed')}">✕ Failed</span>`;
        }

        return `
          <div style="border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; background: #ffffff;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
              <div style="display: flex; align-items: center; gap: 8px;">
                ${statusBadge}
                <span style="font-size: 11px; color: #6b7280;">${escapeHtml(item.file_name)}</span>
              </div>
              <button class="btn-xs secondary btn-danger-subtle" onclick="removeFromQueue('${item.id}')" title="Remove from queue">✕</button>
            </div>

            ${item.status === 'uploading' ? `
              <div class="progress-bar-bg" style="margin-bottom: 10px; height: 6px;">
                <div class="progress-bar-fill" style="width: ${item.progress || 0}%;"></div>
              </div>` : ''}

            ${item.shorts_url ? `
              <div style="margin-bottom: 10px; padding: 8px; background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12px; color: #065f46; font-weight: 500;">Short published successfully! ${item.deleted_from_storage ? '• 💾 Storage freed' : ''}</span>
                <a href="${item.shorts_url}" target="_blank" style="font-size: 12px; color: #2563eb; font-weight: 600; text-decoration: none;">Open Short ↗</a>
              </div>` : ''}

            ${item.error ? `
              <div style="margin-bottom: 8px; font-size: 11px; color: #dc2626; background: #fef2f2; padding: 6px; border-radius: 4px;">
                ${escapeHtml(item.error)}
              </div>` : ''}

            <div style="margin-bottom: 8px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                <label style="font-size: 11px; font-weight: 600; color: #374151;">Viral Short Title</label>
                ${item.status === 'pending' ? `
                  <button class="btn-xs secondary" onclick="enhanceQueueItem('${item.id}', this)" title="Re-roll with viral attention-grabbing formulas" style="font-size: 10px; padding: 1px 7px;">
                    ✨ Auto-Enhance
                  </button>` : ''}
              </div>
              <input type="text" value="${titleVal}" onchange="updateQueueItemMetadata('${item.id}', 'title', this.value)" style="padding: 5px 8px; font-size: 12px; font-weight: 500;" ${item.status !== 'pending' ? 'readonly' : ''}>
            </div>

            <div style="margin-bottom: 8px;">
              <details style="font-size: 11px;">
                <summary style="cursor: pointer; color: #4b5563; user-select: none; font-weight: 500;">
                  📝 Description & Tags (${tags.length} tags)
                </summary>
                <div style="margin-top: 6px;">
                  <label style="font-size: 10px; color: #6b7280; margin-bottom: 2px; display: block;">Structured Video Description:</label>
                  <textarea rows="4" onchange="updateQueueItemMetadata('${item.id}', 'description', this.value)" style="width: 100%; font-size: 11px; padding: 6px 8px; border: 1px solid #d1d5db; border-radius: 4px; font-family: inherit; resize: vertical;" ${item.status !== 'pending' ? 'readonly' : ''}>${escapeHtml(item.description || '')}</textarea>
                  
                  <div style="margin-top: 6px;">
                    <label style="font-size: 10px; color: #6b7280; margin-bottom: 2px; display: block;">Targeted Search Tags:</label>
                    <div style="display: flex; flex-wrap: wrap; gap: 4px; max-height: 48px; overflow-y: auto;">
                      ${tags.slice(0, 12).map(t => `<span style="display: inline-block; background: #f3f4f6; color: #374151; font-size: 10px; padding: 1px 6px; border-radius: 3px; border: 1px solid #e5e7eb;">#${escapeHtml(t)}</span>`).join('')}
                      ${tags.length > 12 ? `<span style="font-size: 10px; color: #9ca3af; align-self: center;">+${tags.length - 12} more</span>` : ''}
                    </div>
                  </div>
                </div>
              </details>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
              <div>
                <label style="font-size: 11px; margin-bottom: 2px;">Visibility</label>
                <select onchange="updateQueueItemMetadata('${item.id}', 'privacy', this.value)" style="padding: 5px 8px; font-size: 12px;" ${item.status !== 'pending' ? 'disabled' : ''}>
                  <option value="public" ${privacy === 'public' ? 'selected' : ''}>Public</option>
                  <option value="unlisted" ${privacy === 'unlisted' ? 'selected' : ''}>Unlisted</option>
                  <option value="private" ${privacy === 'private' ? 'selected' : ''}>Private</option>
                </select>
              </div>
              <div style="display: flex; align-items: flex-end;">
                ${item.status === 'pending' ? `
                  <button class="primary" onclick="uploadQueueItemDirect('${item.id}')" style="width: 100%; padding: 5px 8px; font-size: 11px; justify-content: center;">
                    Upload Now
                  </button>` : ''}
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    async function enhanceQueueItem(itemId, btnElem) {
      if (btnElem) {
        btnElem.disabled = true;
        btnElem.textContent = '⏳ Enhancing...';
      }
      try {
        const res = await fetch('/api/youtube/queue/enhance', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ item_id: itemId })
        });
        if (res.ok) {
          await fetchYouTubeQueue();
        } else {
          const err = await res.json();
          alert('Could not enhance item: ' + (err.detail || 'Error'));
        }
      } catch (e) {
        alert('Error enhancing item: ' + e);
      } finally {
        if (btnElem) {
          btnElem.disabled = false;
          btnElem.textContent = '✨ Auto-Enhance';
        }
      }
    }

    async function enhanceAllQueueItems() {
      const queueData = await (await fetch('/api/youtube/queue')).json();
      const pending = (queueData.queue || []).filter(q => q.status === 'pending');
      if (pending.length === 0) {
        alert('No pending queue items to enhance.');
        return;
      }
      for (const item of pending) {
        try {
          await fetch('/api/youtube/queue/enhance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_id: item.id })
          });
        } catch (e) {}
      }
      await fetchYouTubeQueue();
    }

    async function updateQueueItemMetadata(itemId, key, value) {
      try {
        const payload = { item_id: itemId, updates: { [key]: value } };
        await fetch('/api/youtube/queue/update', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } catch (e) {
        console.error('Failed to update queue item:', e);
      }
    }

    async function removeFromQueue(itemId) {
      try {
        const res = await fetch('/api/youtube/queue/remove', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ item_id: itemId }),
        });
        if (res.ok) {
          await fetchYouTubeQueue();
        }
      } catch (e) {
        alert('Error removing item: ' + e);
      }
    }

    async function clearCompletedQueue() {
      try {
        const res = await fetch('/api/youtube/queue/clear', { method: 'POST' });
        if (res.ok) {
          await fetchYouTubeQueue();
        }
      } catch (e) {
        alert('Error clearing queue: ' + e);
      }
    }

    async function startQueueUpload() {
      try {
        const res = await fetch('/api/youtube/queue/start', { method: 'POST' });
        if (res.ok) {
          await fetchYouTubeQueue();
        }
      } catch (e) {
        alert('Error starting upload: ' + e);
      }
    }

    async function uploadQueueItemDirect(itemId) {
      await startQueueUpload();
    }

    async function refreshYouTubeTab() {
      await fetchYouTubeStatus();
      await loadYouTubeLibrary();
      await fetchYouTubeQueue();
    }

    async function loadAllShorts() {
      try {
        const res = await fetch('/api/shorts');
        if (res.ok) {
          const data = await res.json();
          if (data.shorts && data.shorts.length > 0) {
            renderShorts(data.shorts);
          }
        }
      } catch (e) {}
    }

    function onPlatformChange() {
      const plat = document.getElementById('live-platform').value;
      const lbl = document.getElementById('live-channel-label');
      const input = document.getElementById('live-channel');
      const hint = document.getElementById('live-channel-hint');
      if (plat === 'kick') {
        lbl.textContent = 'Kick Channel Username or URL';
        input.placeholder = 'e.g. westcol, xqc, adinross, https://kick.com/westcol';
        hint.innerHTML = 'Connects anonymously via Kick Centrifugo WebSocket (Guest auth, zero tokens required)';
      } else {
        lbl.textContent = 'Twitch Channel Username';
        input.placeholder = 'e.g. kaicenat, rawrawknee, tarik, shroud';
        hint.innerHTML = 'Connects anonymously to <code>irc.chat.twitch.tv:6667</code> (NICK justinfan, zero tokens required)';
      }
    }

    async function startWatcher() {
      const platform = document.getElementById('live-platform').value;
      const channel = document.getElementById('live-channel').value.trim();
      if (!channel) {
        alert('Please enter a ' + (platform === 'kick' ? 'Kick' : 'Twitch') + ' channel username');
        return;
      }
      const minMsgs = parseInt(document.getElementById('live-min-msgs').value, 10) || 5;
      const spikeFactor = parseFloat(document.getElementById('live-spike-factor').value) || 2.5;
      const aiMode = document.getElementById('live-ai-mode').value;
      const layout = document.getElementById('live-layout').value;
      const encoder = document.getElementById('live-encoder').value;

      try {
        const res = await fetch('/api/watcher/start', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            channel: channel,
            platform: platform,
            min_messages: minMsgs,
            spike_factor: spikeFactor,
            ai_mode: aiMode,
            layout: layout,
            encoder: encoder
          })
        });
        const data = await res.json();
        if (data.status === 'STARTED' || data.status === 'ALREADY_RUNNING') {
          document.getElementById('btn-start-watcher').style.display = 'none';
          document.getElementById('btn-stop-watcher').style.display = 'inline-flex';
          pollWatcherStatus();
        } else {
          alert('Could not start watcher: ' + JSON.stringify(data));
        }
      } catch (err) {
        alert('Error starting watcher: ' + err);
      }
    }

    async function stopWatcher() {
      try {
        await fetch('/api/watcher/stop', { method: 'POST' });
        document.getElementById('btn-start-watcher').style.display = 'inline-flex';
        document.getElementById('btn-stop-watcher').style.display = 'none';
        pollWatcherStatus();
      } catch (err) {
        alert('Error stopping watcher: ' + err);
      }
    }

    async function triggerTestClip() {
      const btn = document.getElementById('btn-test-clip');
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '⏳ Cutting 30s Reel...';

      const layout = document.getElementById('live-layout').value;
      const encoder = document.getElementById('live-encoder').value;

      try {
        const res = await fetch('/api/watcher/test-clip', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            duration: 30.0,
            layout: layout,
            encoder: encoder
          })
        });
        const data = await res.json();
        btn.innerHTML = '✓ Reel Cut!';
        setTimeout(() => {
          btn.innerHTML = originalText;
          btn.disabled = false;
        }, 2000);

        // Refresh shorts gallery immediately
        await loadAllShorts();

        // If returned short has a path, play it automatically
        if (data.short && data.short.path) {
          playShort(data.short.path, data.short.title, data.short.suggested_hook, data.short.description);
        } else if (data.mode === 'live_stream') {
          alert('Live 30s reel extraction triggered! It will appear in Completed Shorts shortly.');
        }
      } catch (err) {
        alert('Error cutting test reel: ' + err);
        btn.innerHTML = originalText;
        btn.disabled = false;
      }
    }

    async function pollWatcherStatus() {
      try {
        const res = await fetch('/api/watcher/status');
        if (!res.ok) return;
        const data = await res.json();
        const btnStart = document.getElementById('btn-start-watcher');
        const btnStop = document.getElementById('btn-stop-watcher');
        const streamBadge = document.getElementById('live-stream-badge');
        const ircBadge = document.getElementById('live-irc-badge');

        if (data.is_running) {
          const platName = (data.platform || 'TWITCH').toUpperCase();
          if (btnStart) btnStart.style.display = 'none';
          if (btnStop) btnStop.style.display = 'inline-flex';

          const rec = data.recorder;
          const recError = (rec && rec.error) || data.error;
          const isAlive = rec ? rec.alive : true;

          if (streamBadge) {
            if (!isAlive || recError) {
              streamBadge.textContent = `⚠️ Offline / No Stream (${data.channel})`;
              streamBadge.className = 'badge badge-danger';
              streamBadge.title = recError || 'Streamer is offline or channel not found';
            } else {
              const mb = rec && rec.file_size_mb ? ` (${rec.file_size_mb} MB)` : '';
              streamBadge.textContent = `Stream: LIVE [${platName}] (${data.channel})${mb}`;
              streamBadge.className = 'badge badge-success';
            }
          }
          if (ircBadge) {
            const isConn = data.chat_connected ?? data.irc_connected;
            ircBadge.textContent = isConn ? `${platName} Chat: Connected` : `${platName} Chat: Connecting...`;
            ircBadge.className = isConn ? 'badge badge-success' : 'badge badge-active';
          }
          document.getElementById('live-metric-bucket').textContent = data.current_bucket_count || 0;
          document.getElementById('live-metric-median').textContent = data.current_median || '0.0';
          document.getElementById('live-metric-spikes').textContent = data.spikes_detected || 0;
          document.getElementById('live-metric-shorts').textContent = data.shorts_count || 0;

          if (data.shorts_count > lastShortsCount) {
            lastShortsCount = data.shorts_count;
            loadAllShorts();
          }
        } else {
          if (btnStart) btnStart.style.display = 'inline-flex';
          if (btnStop) btnStop.style.display = 'none';
          if (streamBadge) {
            streamBadge.textContent = 'Stream: Inactive';
            streamBadge.className = 'badge';
          }
          if (ircBadge) {
            ircBadge.textContent = 'Chat: Disconnected';
            ircBadge.className = 'badge';
          }
        }
      } catch (err) {}
    }

    function setupMinMaxSync() {
      const minEl = document.getElementById('min-shorts');
      const maxEl = document.getElementById('max-shorts');
      if (minEl && maxEl) {
        minEl.addEventListener('input', () => {
          const minV = parseInt(minEl.value, 10) || 1;
          const maxV = parseInt(maxEl.value, 10) || 1;
          if (maxV < minV) maxEl.value = minV;
        });
        maxEl.addEventListener('input', () => {
          const minV = parseInt(minEl.value, 10) || 1;
          const maxV = parseInt(maxEl.value, 10) || 1;
          if (maxV < minV) minEl.value = maxV;
        });
      }
    }

    function setupDragAndDrop() {
      const dropzone = document.getElementById('dropzone');
      const fileInput = document.getElementById('file-uploader');

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });

      dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          uploadFile(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
          uploadFile(fileInput.files[0]);
        }
      });
    }

    function uploadFile(file) {
      const pcontainer = document.getElementById('upload-progress-container');
      const pbar = document.getElementById('upload-progress-bar');
      const ptext = document.getElementById('upload-progress-text');
      pcontainer.style.display = 'block';
      pbar.style.width = '0%';
      ptext.textContent = `Uploading ${file.name}... (0%)`;

      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/upload');

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          pbar.style.width = pct + '%';
          const loadedMB = (e.loaded / (1024 * 1024)).toFixed(1);
          const totalMB = (e.total / (1024 * 1024)).toFixed(1);
          ptext.textContent = `Uploading ${file.name}: ${pct}% (${loadedMB}MB / ${totalMB}MB)`;
        }
      };

      xhr.onload = async () => {
        if (xhr.status === 200) {
          ptext.textContent = `✓ Uploaded ${file.name} successfully!`;
          setTimeout(() => { pcontainer.style.display = 'none'; }, 2500);
          await loadVideos(file.name);
        } else {
          ptext.textContent = `Upload failed: ${xhr.responseText}`;
        }
      };

      xhr.onerror = () => {
        ptext.textContent = 'Upload network error';
      };

      xhr.send(formData);
    }

    async function loadVideos(selectFilename = null) {
      try {
        const res = await fetch('/api/videos');
        const data = await res.json();
        const sel = document.getElementById('video-select');
        sel.innerHTML = '';
        if (data.videos.length === 0) {
          sel.innerHTML = '<option value="">No videos found in data/inbox</option>';
          renderShorts([]);
          renderCandidates([], []);
          return;
        }
        data.videos.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.filename;
          opt.textContent = `${v.filename} (${v.size_mb} MB)`;
          if (selectFilename && v.filename === selectFilename) {
            opt.selected = true;
          }
          sel.appendChild(opt);
        });
        await onVideoSelected();
      } catch (err) {
        console.error('Failed to load videos:', err);
      }
    }

    async function deleteSelectedVideo() {
      const sel = document.getElementById('video-select').value;
      if (!sel) return alert('No video selected');
      if (!confirm(`Are you sure you want to permanently delete "${sel}" from inbox?`)) return;

      try {
        const res = await fetch(`/api/videos/${encodeURIComponent(sel)}`, { method: 'DELETE' });
        if (res.ok) {
          await loadVideos();
        } else {
          const err = await res.json();
          alert(`Failed to delete video: ${err.detail || res.statusText}`);
        }
      } catch (err) {
        alert(`Error: ${err}`);
      }
    }

    async function clearAllVideos() {
      if (!confirm('Are you sure you want to delete ALL videos from the inbox? This cannot be undone.')) return;
      try {
        const res = await fetch('/api/videos', { method: 'DELETE' });
        if (res.ok) {
          await loadVideos();
        } else {
          alert('Failed to clear inbox videos.');
        }
      } catch (err) {
        alert(`Error: ${err}`);
      }
    }

    async function loadSettings() {
      try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        const groqPill = document.getElementById('groq-pill');
        const statusEl = document.getElementById('groq-current-status');
        if (data.groq_configured) {
          groqPill.textContent = 'Groq Cloud Active';
          groqPill.className = 'badge badge-success';
          if (statusEl) statusEl.textContent = `Configured (${data.groq_key_masked || 'Active'})`;
        } else {
          groqPill.textContent = 'Groq Key Unset';
          groqPill.className = 'badge badge-rating';
          if (statusEl) statusEl.textContent = 'Not configured. Please enter a valid API key.';
        }
      } catch (err) {}
    }

    async function onVideoSelected() {
      const sel = document.getElementById('video-select').value;
      if (!sel) {
        renderShorts([]);
        renderCandidates([], []);
        return;
      }
      try {
        const res = await fetch('/api/jobs');
        const data = await res.json();
        const match = data.jobs.find(j => j.filename === sel);
        if (match) {
          currentJobId = match.source_id;
          await loadJobDetail(match.source_id);
        } else {
          currentJobId = null;
          renderShorts([]);
          renderCandidates([], []);
        }
      } catch (err) {}
    }

    async function loadJobDetail(sourceId) {
      try {
        const res = await fetch(`/api/jobs/${sourceId}`);
        if (!res.ok) return;
        const job = await res.json();
        renderShorts(job.shorts || []);
        renderCandidates(job.candidates || [], job.judgments || []);

        const statusBadge = document.getElementById('status-badge');
        const statusTitle = document.getElementById('status-title');
        const pbar = document.getElementById('progress-bar');
        const logBox = document.getElementById('log-box');

        if (job.status === 'COMPLETED') {
          statusTitle.textContent = `${job.filename} (COMPLETED)`;
          statusBadge.textContent = 'COMPLETED';
          statusBadge.className = 'badge badge-success';
          pbar.style.width = '100%';
          logBox.textContent = `Job completed. ${job.shorts ? job.shorts.length : 0} vertical Shorts produced.`;
        }
      } catch (err) {}
    }

    function renderShorts(shorts) {
      const grid = document.getElementById('shorts-grid');
      const summary = document.getElementById('shorts-summary');
      const clearBtn = document.getElementById('btn-clear-shorts');
      summary.textContent = `${shorts.length} vertical Shorts produced`;
      grid.innerHTML = '';

      if (shorts.length === 0) {
        grid.innerHTML = '<div style="grid-column: 1/-1; padding: 32px; text-align: center; color: #9ca3af; border: 1px dashed #e5e7eb; border-radius: 6px;">No rendered Shorts yet. Click "Extract Real Shorts" to produce clips.</div>';
        if (clearBtn) clearBtn.style.display = 'none';
        return;
      }

      if (clearBtn) clearBtn.style.display = 'inline-flex';

      shorts.forEach(s => {
        const card = document.createElement('div');
        card.className = 'short-card';
        const hookText = s.suggested_hook || s.title || 'Gaming Highlight';
        const rating = s.editorial_rating ? `★ ${s.editorial_rating}/5` : '★ 4/5';
        const encodedPath = encodeURIComponent(s.path || '');
        const encodedHook = encodeURIComponent(hookText);

        card.innerHTML = `
          <div class="video-wrapper">
            <video controls playsinline preload="metadata">
              <source src="${s.url}" type="video/mp4">
            </video>
          </div>
          <div class="short-body">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <span class="badge badge-rating">${rating}</span>
              <span class="badge">${s.event_category || 'highlight'}</span>
            </div>
            <div class="short-hook">${escapeHtml(hookText)}</div>
            <div class="short-desc">${escapeHtml(s.summary || '')}</div>
            <div class="short-meta">
              <span>Time: ${s.start_seconds ? formatTime(s.start_seconds) : ''} - ${s.end_seconds ? formatTime(s.end_seconds) : ''}</span>
              <span>${s.size_mb} MB</span>
            </div>
            <div class="short-actions">
              <button class="secondary btn-xs" onclick="copyHook(decodeURIComponent('${encodedHook}'), this)" title="Copy Hook to clipboard">📋 Copy Hook</button>
              <a href="${s.url}" download="${s.filename}" class="btn-xs" style="text-align: center; text-decoration: none;">⬇️ Download</a>
              <button class="btn-xs btn-danger-subtle" onclick="deleteShort(decodeURIComponent('${encodedPath}'))" title="Delete this short">🗑️ Delete</button>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    function renderCandidates(candidates, judgments) {
      const tbody = document.getElementById('candidates-body');
      tbody.innerHTML = '';
      if (!candidates || candidates.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #9ca3af; padding: 20px;">No candidate moments evaluated yet.</td></tr>';
        return;
      }
      const judgMap = {};
      (judgments || []).forEach(j => { judgMap[j.candidate_id] = j; });

      candidates.forEach(c => {
        const j = judgMap[c.window_id] || {};
        const tr = document.createElement('tr');
        const start = formatTime(c.start_seconds);
        const end = formatTime(c.end_seconds);
        const rating = j.editorial_rating ? `★ ${j.editorial_rating}/5` : '-';
        const isSelected = (j.editorial_rating && j.editorial_rating >= 4);

        tr.innerHTML = `
          <td><strong>${start} – ${end}</strong></td>
          <td><span class="badge">${j.event_category || c.rule || 'gaming'}</span></td>
          <td><strong>${rating}</strong></td>
          <td style="max-width: 400px;">${escapeHtml(j.summary || 'Awaiting evaluation')}</td>
          <td>${isSelected ? '<span class="badge badge-success">Approved</span>' : '<span class="badge">Reviewed</span>'}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function copyHook(text, btnElement) {
      navigator.clipboard.writeText(text).then(() => {
        const original = btnElement.textContent;
        btnElement.textContent = '✓ Copied!';
        btnElement.style.color = '#059669';
        btnElement.style.borderColor = '#059669';
        setTimeout(() => {
          btnElement.textContent = original;
          btnElement.style.color = '';
          btnElement.style.borderColor = '';
        }, 2000);
      }).catch(err => {
        alert('Failed to copy hook: ' + err);
      });
    }

    async function deleteShort(path) {
      if (!path) return alert('No path specified.');
      if (!confirm('Are you sure you want to permanently delete this generated short?')) return;
      try {
        const res = await fetch(`/api/shorts?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
        if (res.ok) {
          if (currentJobId) {
            await loadJobDetail(currentJobId);
          }
        } else {
          alert('Failed to delete short.');
        }
      } catch (err) {
        alert(`Error: ${err}`);
      }
    }

    async function clearCurrentShorts() {
      if (!currentJobId) return alert('No active job loaded.');
      if (!confirm('Are you sure you want to delete all generated shorts for this video?')) return;
      try {
        const res = await fetch(`/api/jobs/${encodeURIComponent(currentJobId)}`, { method: 'DELETE' });
        if (res.ok) {
          currentJobId = null;
          renderShorts([]);
          renderCandidates([], []);
          dismissStatus();
          await loadVideos();
        } else {
          alert('Failed to clear shorts.');
        }
      } catch (err) {
        alert(`Error: ${err}`);
      }
    }

    async function dismissStatus() {
      try {
        await fetch('/api/tasks/dismiss', { method: 'POST' });
        document.getElementById('status-title').textContent = 'Ready';
        document.getElementById('status-badge').textContent = 'IDLE';
        document.getElementById('status-badge').className = 'badge';
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('log-box').textContent = 'Status cleared. Ready for next pipeline run.';
      } catch (err) {}
    }

    async function deleteCurrentJob() {
      if (!currentJobId) return alert('No job loaded to delete.');
      if (!confirm('Delete this job output, completed shorts, and reset database history?')) return;
      try {
        const res = await fetch(`/api/jobs/${encodeURIComponent(currentJobId)}`, { method: 'DELETE' });
        if (res.ok) {
          currentJobId = null;
          dismissStatus();
          renderShorts([]);
          renderCandidates([], []);
          refreshDashboard();
        } else {
          alert('Failed to delete job.');
        }
      } catch (err) {
        alert(`Error: ${err}`);
      }
    }

    async function stopActivePipeline() {
      if (!confirm('Stop the currently running pipeline?')) return;
      try {
        await fetch('/api/tasks/stop', { method: 'POST' });
        document.getElementById('status-title').textContent = 'Pipeline Stopped';
        document.getElementById('status-badge').textContent = 'STOPPED';
        document.getElementById('status-badge').className = 'badge badge-rating';
      } catch (err) {
        alert(`Error stopping pipeline: ${err}`);
      }
    }

    function openCleanupModal() {
      document.getElementById('cleanup-modal').style.display = 'flex';
    }

    function closeCleanupModal() {
      document.getElementById('cleanup-modal').style.display = 'none';
    }

    async function executeCleanup(resetAll = false) {
      let body = {};
      if (resetAll) {
        if (!confirm('WARNING: TOTAL WIPE. This will permanently delete ALL inbox videos, processing temp files, generated shorts, and reset the SQLite database. Are you sure?')) {
          return;
        }
        body = {
          clear_processing: true,
          clear_completed: true,
          clear_inbox: true,
          reset_db: true
        };
      } else {
        body = {
          clear_processing: document.getElementById('clean-processing').checked,
          clear_completed: document.getElementById('clean-completed').checked,
          clear_inbox: document.getElementById('clean-inbox').checked,
          reset_db: document.getElementById('clean-db').checked
        };
        if (!body.clear_processing && !body.clear_completed && !body.clear_inbox && !body.reset_db) {
          alert('Please select at least one item to clean.');
          return;
        }
      }

      try {
        const res = await fetch('/api/cleanup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        if (res.ok) {
          const data = await res.json();
          alert(`Cleanup completed: ${data.cleaned.join(', ') || 'Done'}`);
          closeCleanupModal();
          currentJobId = null;
          dismissStatus();
          renderShorts([]);
          renderCandidates([], []);
          await loadVideos();
        } else {
          alert('Cleanup failed.');
        }
      } catch (err) {
        alert(`Error during cleanup: ${err}`);
      }
    }

    function openSettingsModal() {
      document.getElementById('settings-modal').style.display = 'flex';
      loadSettings();
    }

    function closeSettingsModal() {
      document.getElementById('settings-modal').style.display = 'none';
    }

    async function saveGroqKey() {
      const key = document.getElementById('groq-key-input').value.trim();
      if (!key || key.length < 10) {
        alert('Please enter a valid Groq API key (starts with gsk_...)');
        return;
      }
      try {
        const res = await fetch(`/api/settings/groq-key?key=${encodeURIComponent(key)}`, { method: 'POST' });
        if (res.ok) {
          alert('Groq API Key saved successfully!');
          closeSettingsModal();
          await loadSettings();
        } else {
          alert('Failed to save Groq API key.');
        }
      } catch (err) {
        alert(`Error: ${err}`);
      }
    }

    function formatTime(secs) {
      if (!secs && secs !== 0) return '00:00';
      const m = Math.floor(secs / 60);
      const s = Math.floor(secs % 60);
      return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function escapeJsString(str) {
      if (!str) return '';
      return JSON.stringify(str).slice(1, -1);
    }

    async function startProcessing() {
      const video = document.getElementById('video-select').value;
      if (!video) {
        alert('Please select a video first');
        return;
      }
      const provider = document.getElementById('provider-select').value;
      const creator = document.getElementById('creator-name').value;
      const layout = document.getElementById('layout-select').value;
      const encoder = document.getElementById('encoder-select').value;
      const minShorts = parseInt(document.getElementById('min-shorts').value, 10) || 1;
      let maxShorts = parseInt(document.getElementById('max-shorts').value, 10) || 3;
      if (maxShorts < minShorts) {
        maxShorts = minShorts;
        document.getElementById('max-shorts').value = maxShorts;
      }
      const minScore = parseInt(document.getElementById('min-score').value, 10) || 3;
      const maxCandidates = Math.max(10, minShorts * 3);

      const btn = document.getElementById('btn-extract');
      const btnStop = document.getElementById('btn-stop');
      if (btn) {
        btn.disabled = true;
        btn.textContent = 'Processing Pipeline Active...';
      }
      if (btnStop) btnStop.style.display = 'inline-flex';

      document.getElementById('status-title').textContent = 'Starting pipeline...';
      document.getElementById('status-badge').textContent = 'QUEUED';
      document.getElementById('status-badge').className = 'badge badge-active';
      document.getElementById('progress-bar').style.width = '15%';

      try {
        const res = await fetch('/api/process', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            video_path: video,
            provider: provider,
            creator: creator,
            min_shorts: minShorts,
            max_shorts: maxShorts,
            max_candidates: maxCandidates,
            min_score: minScore,
            layout: layout,
            encoder: encoder
          })
        });
        const data = await res.json();
        document.getElementById('status-title').textContent = `Processing ${video}...`;
        document.getElementById('status-badge').textContent = 'RUNNING';
        pollStatus();
      } catch (err) {
        alert('Failed to launch pipeline: ' + err);
        if (btn) {
          btn.disabled = false;
          btn.textContent = 'Extract Real Shorts';
        }
        if (btnStop) btnStop.style.display = 'none';
      }
    }

    async function downloadYoutube() {
      const url = document.getElementById('yt-url').value.trim();
      if (!url) return alert('Enter a valid YouTube URL');
      const cookiesMode = document.getElementById('yt-cookies-mode').value;
      const useCookies = cookiesMode !== 'none';
      const browser = (cookiesMode !== 'auto' && cookiesMode !== 'none') ? cookiesMode : null;

      document.getElementById('status-title').textContent = 'Downloading video from YouTube...';
      document.getElementById('status-badge').textContent = 'DOWNLOADING';
      document.getElementById('status-badge').className = 'badge badge-active';
      try {
        await fetch('/api/download', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            url: url,
            browser: browser,
            use_cookies: useCookies
          })
        });
        pollStatus();
      } catch (err) {
        alert('Download request failed: ' + err);
      }
    }

    async function checkCookiesStatus() {
      try {
        const res = await fetch('/api/cookies-status');
        if (!res.ok) return;
        const data = await res.json();
        const badge = document.getElementById('cookies-status-badge');
        if (!badge) return;
        if (data.has_cookies_file) {
          badge.textContent = 'cookies.txt: Active';
          badge.className = 'badge badge-success';
        } else {
          badge.textContent = 'cookies.txt: None';
          badge.className = 'badge';
        }
      } catch (e) {}
    }

    async function uploadCookiesFile(input) {
      if (!input.files || input.files.length === 0) return;
      const file = input.files[0];
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/api/upload-cookies', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.status === 'UPLOADED') {
          alert('cookies.txt uploaded successfully! yt-dlp will now use your cookies for age-restricted & authenticated videos.');
          checkCookiesStatus();
        } else {
          alert('Failed to upload cookies: ' + JSON.stringify(data));
        }
      } catch (err) {
        alert('Error uploading cookies: ' + err);
      }
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        const tasks = data.active_tasks || {};
        const logBox = document.getElementById('log-box');
        const pbar = document.getElementById('progress-bar');
        const statusBadge = document.getElementById('status-badge');
        const statusTitle = document.getElementById('status-title');
        const video = document.getElementById('video-select').value;
        const btnExtract = document.getElementById('btn-extract');
        const btnStop = document.getElementById('btn-stop');

        // Check if there is a running task
        const activeKeys = Object.keys(tasks);
        let runningTask = null;
        let failedTask = null;

        for (const k of activeKeys) {
          const t = tasks[k];
          if (t.status === 'RUNNING' || t.status === 'DOWNLOADING') {
            runningTask = { key: k, ...t };
            break;
          }
          if (t.status === 'FAILED' || t.status === 'ERROR') {
            failedTask = { key: k, ...t };
          }
        }

        if (runningTask) {
          statusTitle.textContent = `${runningTask.key} (${runningTask.status})`;
          statusBadge.textContent = runningTask.status;
          statusBadge.className = 'badge badge-active';
          pbar.style.width = '65%';
          if (btnExtract) {
            btnExtract.disabled = true;
            btnExtract.textContent = 'Processing Pipeline Active...';
          }
          if (btnStop) btnStop.style.display = 'inline-flex';
          if (runningTask.logs && runningTask.logs.length > 0) {
            logBox.textContent = runningTask.logs.slice(-20).join('\\n');
            logBox.scrollTop = logBox.scrollHeight;
          }
          return;
        }

        // Pipeline is not actively running
        if (btnStop) btnStop.style.display = 'none';
        if (btnExtract) {
          btnExtract.disabled = false;
          btnExtract.textContent = 'Extract Real Shorts';
        }

        // Check DB status for selected video: database completed truth takes priority
        const jobsRes = await fetch('/api/jobs');
        if (jobsRes.ok) {
          const jobsData = await jobsRes.json();
          const dbJob = (jobsData.jobs || []).find(j => j.filename === video);
          if (dbJob && dbJob.status === 'COMPLETED') {
            statusTitle.textContent = `${video} (COMPLETED)`;
            statusBadge.textContent = 'COMPLETED';
            statusBadge.className = 'badge badge-success';
            pbar.style.width = '100%';
            if (currentJobId !== dbJob.source_id) {
              currentJobId = dbJob.source_id;
              await loadJobDetail(dbJob.source_id);
            }
            return;
          }
        }

        if (failedTask) {
          statusTitle.textContent = `${failedTask.key} (${failedTask.status})`;
          statusBadge.textContent = failedTask.status;
          statusBadge.className = 'badge badge-danger';
          pbar.style.width = '100%';
          if (failedTask.logs && failedTask.logs.length > 0) {
            logBox.textContent = failedTask.logs.slice(-20).join('\\n');
            logBox.scrollTop = logBox.scrollHeight;
          }
        }
      } catch (err) {}
    }

    function refreshDashboard() {
      loadVideos();
      loadSettings();
      pollStatus();
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.web.server:app", host="127.0.0.1", port=8000, reload=False)
