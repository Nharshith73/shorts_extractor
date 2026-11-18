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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
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
    """List all rendered 9:16 Shorts across completed VOD jobs and live Twitch watcher runs."""
    shorts = []
    if not COMPLETED_DIR.is_dir():
        return {"shorts": []}

    for mp4 in COMPLETED_DIR.rglob("shorts/*.mp4"):
        if not mp4.is_file():
            continue
        try:
            st = mp4.stat()
            meta_json = mp4.parent.parent / "metadata" / f"{mp4.stem}.json"
            meta: dict[str, Any] = {}
            if meta_json.is_file():
                try:
                    meta = json.loads(meta_json.read_text(encoding="utf-8"))
                except Exception:
                    pass

            shorts.append({
                "filename": mp4.name,
                "path": str(mp4),
                "url": f"/media/file?path={urllib.parse.quote(str(mp4))}",
                "size_mb": round(st.st_size / (1024 * 1024), 2),
                "mtime": st.st_mtime,
                "title": meta.get("title", mp4.stem),
                "suggested_hook": meta.get("suggested_hook"),
                "summary": meta.get("description", ""),
                "event_category": meta.get("event_category", "highlight"),
                "editorial_rating": meta.get("editorial_rating", 4),
                "start_seconds": meta.get("start_seconds"),
                "end_seconds": meta.get("end_seconds"),
                "channel": meta.get("channel"),
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
            ACTIVE_TASKS[task_id]["status"] = "FAILED"
            log_str = " ".join(ACTIVE_TASKS[task_id]["logs"])
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
        return {"status": "ALREADY_RUNNING", "channel": ACTIVE_WATCHER_SESSION.channel, "platform": ACTIVE_WATCHER_SESSION.platform}

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
    <div style="display: flex; gap: 8px; margin-bottom: 22px; border-bottom: 1px solid #e5e7eb; padding-bottom: 12px;">
      <button id="tab-btn-vod" class="primary" onclick="setDashboardTab('vod')" style="padding: 7px 16px; font-size: 13px;">
        📁 Video Files & VOD Mode
      </button>
      <button id="tab-btn-live" class="secondary" onclick="setDashboardTab('live')" style="padding: 7px 16px; font-size: 13px;">
        🔴 Live Watcher (Twitch & Kick Spike Detection)
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
              <option value="blurred_background" selected>Blurred Background (Reels Standard)</option>
              <option value="stacked">Stacked 9:16 (Top/Bottom Split)</option>
              <option value="center_crop">Center Crop 9:16</option>
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
              <option value="blurred_background" selected>Blurred Background (Reels Standard)</option>
              <option value="center_crop">Center Crop 9:16</option>
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
    <div style="margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;">
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
    <div class="card" style="margin-top: 24px;">
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
      setInterval(pollStatus, 3000);
      setInterval(pollWatcherStatus, 2000);
    }

    function setDashboardTab(tab) {
      const panelVod = document.getElementById('panel-vod');
      const panelLive = document.getElementById('panel-live');
      const tabVod = document.getElementById('tab-btn-vod');
      const tabLive = document.getElementById('tab-btn-live');

      if (tab === 'vod') {
        panelVod.style.display = 'grid';
        panelLive.style.display = 'none';
        tabVod.className = 'primary';
        tabLive.className = 'secondary';
      } else {
        panelVod.style.display = 'none';
        panelLive.style.display = 'grid';
        tabVod.className = 'secondary';
        tabLive.className = 'primary';
      }
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
          if (streamBadge) {
            streamBadge.textContent = `Stream: LIVE [${platName}] (${data.channel})`;
            streamBadge.className = 'badge badge-success';
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
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
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

    window.onload = init;
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
