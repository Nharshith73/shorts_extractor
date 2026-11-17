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

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
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
            job_data["active_progress"] = ACTIVE_TASKS[source_id]

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
    provider: str = "local-qwen"  # 'local-qwen' or 'groq'
    max_candidates: int = 10
    max_shorts: int = 3
    min_shorts: int = 1
    min_score: int = 3
    encoder: str = "auto"


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


def run_download_worker(url: str, output_name: str | None):
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(INBOX_DIR / (output_name or "%(id)s.%(ext)s"))
    cmd = [
        "python",
        "-m",
        "yt_dlp",
        "-f",
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "-o",
        out_tmpl,
        url,
    ]
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
        ACTIVE_TASKS[task_id]["status"] = "COMPLETED" if proc.returncode == 0 else "FAILED"
    except Exception as exc:
        ACTIVE_TASKS[task_id]["status"] = "ERROR"
        ACTIVE_TASKS[task_id]["error"] = str(exc)


@app.post("/api/download")
def api_download_video(req: DownloadRequest):
    if not req.url or not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    thread = threading.Thread(
        target=run_download_worker,
        args=(req.url, req.output_name),
        daemon=True,
    )
    thread.start()
    return {"status": "DOWNLOADING_STARTED", "url": req.url}


@app.get("/api/settings")
def api_get_settings():
    groq_key = resolve_groq_api_key()
    return {
        "groq_configured": bool(groq_key and len(groq_key) > 10),
        "groq_key_masked": f"{groq_key[:6]}...{groq_key[-4:]}" if groq_key else None,
        "active_tasks": ACTIVE_TASKS,
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
      padding-bottom: 24px;
      border-bottom: 1px solid #e5e7eb;
      margin-bottom: 28px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand h1 {
      font-size: 18px;
      font-weight: 600;
      letter-spacing: -0.01em;
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

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 28px;
    }
    @media (max-width: 800px) {
      .grid-2 { grid-template-columns: 1fr; }
    }

    .card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 20px;
    }
    .card-title {
      font-size: 13px;
      font-weight: 600;
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
      font-weight: 500;
      color: #374151;
      margin-bottom: 6px;
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
    }
    input:focus, select:focus {
      border-color: #111827;
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
      padding: 9px 16px;
      background: #111827;
      color: #ffffff;
      border: 1px solid #111827;
      border-radius: 4px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s;
    }
    button:hover {
      background: #27272a;
    }
    button.secondary {
      background: #ffffff;
      color: #111827;
      border: 1px solid #d1d5db;
    }
    button.secondary:hover {
      background: #f9fafb;
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
      font-family: monospace;
      font-size: 11px;
      color: #4b5563;
      padding: 8px 12px;
      max-height: 110px;
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
      font-weight: 600;
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
      gap: 8px;
      margin-top: 8px;
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
      font-weight: 500;
      border-bottom: 1px solid #e5e7eb;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid #f3f4f6;
      color: #374151;
    }
    tr:last-child td { border-bottom: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1>StreamClip Studio</h1>
        <span class="badge" id="gpu-pill">RTX 4070 Ready</span>
        <span class="badge" id="groq-pill">Groq Free Tier</span>
      </div>
      <div>
        <button class="secondary" onclick="refreshDashboard()" style="padding: 6px 12px; font-size: 12px;">Refresh</button>
      </div>
    </header>

    <!-- Ingestion & Configuration Row -->
    <div class="grid-2">
      <!-- Select or Ingest Video -->
      <div class="card">
        <div class="card-title">1. Select Video</div>
        <div class="form-group">
          <label for="video-select">Inbox Video</label>
          <select id="video-select" onchange="onVideoSelected()">
            <option value="">Loading videos...</option>
          </select>
        </div>
        <div class="form-group">
          <label for="yt-url">Or Download from YouTube</label>
          <div style="display: flex; gap: 8px;">
            <input type="text" id="yt-url" placeholder="https://www.youtube.com/watch?v=...">
            <button class="secondary" onclick="downloadYoutube()" style="white-space: nowrap;">Fetch</button>
          </div>
        </div>
        <div class="form-group" style="margin-top: 18px;">
          <label for="creator-name">Creator / Streamer Identity</label>
          <input type="text" id="creator-name" placeholder="RawRawKnee" value="RawRawKnee">
        </div>
      </div>

      <!-- Pipeline Configuration -->
      <div class="card">
        <div class="card-title">2. AI Highlight Settings</div>
        <div class="form-group">
          <label for="provider-select">Highlight Judge Provider</label>
          <select id="provider-select">
            <option value="local-qwen">Local Qwen2.5-VL-7B (RTX 4070 GPU, Free Offline)</option>
            <option value="groq">Groq Cloud (Llama 3.3 70B & Vision, Free Tier)</option>
          </select>
        </div>
        <div class="row">
          <div class="form-group">
            <label for="min-shorts">Minimum Shorts (Guaranteed)</label>
            <input type="number" id="min-shorts" min="1" max="10" value="2">
            <span style="display: block; font-size: 11px; color: #6b7280; margin-top: 3px;">Guarantees at least this many clips</span>
          </div>
          <div class="form-group">
            <label for="max-shorts">Maximum Shorts (Cap)</label>
            <input type="number" id="max-shorts" min="1" max="15" value="3">
            <span style="display: block; font-size: 11px; color: #6b7280; margin-top: 3px;">Maximum clips to produce</span>
          </div>
          <div class="form-group">
            <label for="min-score">Target Quality</label>
            <input type="number" id="min-score" min="1" max="5" value="3">
            <span style="display: block; font-size: 11px; color: #6b7280; margin-top: 3px;">Min star rating threshold</span>
          </div>
        </div>
        <div class="form-group" style="margin-top: 8px;">
          <button style="width: 100%;" onclick="startProcessing()">Extract Real Shorts</button>
        </div>
      </div>
    </div>

    <!-- Live Status / Progress Panel -->
    <div class="status-panel" id="status-panel">
      <div class="status-header">
        <span style="font-weight: 600; font-size: 13px;" id="status-title">Ready</span>
        <span class="badge badge-active" id="status-badge">Idle</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="progress-bar"></div>
      </div>
      <div class="log-box" id="log-box">Select a video and click "Extract Real Shorts" to begin production.</div>
    </div>

    <!-- Completed Shorts Gallery -->
    <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: flex-end;">
      <div>
        <h2 style="font-size: 16px; font-weight: 600;">Generated Shorts (9:16 Vertical)</h2>
        <p style="font-size: 12px; color: #6b7280;" id="shorts-summary">0 clips generated</p>
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
              <th>Editorial Summary</th>
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

  <script>
    let activePoll = null;
    let currentJobId = null;

    async function init() {
      await loadVideos();
      await loadSettings();
      await pollStatus();
      setInterval(pollStatus, 3000);
    }

    async function loadVideos() {
      try {
        const res = await fetch('/api/videos');
        const data = await res.json();
        const sel = document.getElementById('video-select');
        sel.innerHTML = '';
        if (data.videos.length === 0) {
          sel.innerHTML = '<option value="">No videos found in data/inbox</option>';
          return;
        }
        data.videos.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.filename;
          opt.textContent = `${v.filename} (${v.size_mb} MB)`;
          sel.appendChild(opt);
        });
        onVideoSelected();
      } catch (err) {
        console.error('Failed to load videos:', err);
      }
    }

    async function loadSettings() {
      try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        const groqPill = document.getElementById('groq-pill');
        if (data.groq_configured) {
          groqPill.textContent = 'Groq Cloud Active';
          groqPill.className = 'badge badge-success';
        } else {
          groqPill.textContent = 'Groq Key Unset';
          groqPill.className = 'badge';
        }
      } catch (err) {}
    }

    async function onVideoSelected() {
      const sel = document.getElementById('video-select').value;
      if (!sel) return;
      // Load completed job data for this video if exists
      const res = await fetch('/api/jobs');
      const data = await res.json();
      const match = data.jobs.find(j => j.filename === sel);
      if (match) {
        currentJobId = match.source_id;
        loadJobDetail(match.source_id);
      }
    }

    async function loadJobDetail(sourceId) {
      try {
        const res = await fetch(`/api/jobs/${sourceId}`);
        if (!res.ok) return;
        const job = await res.json();
        renderShorts(job.shorts || []);
        renderCandidates(job.candidates || [], job.judgments || []);

        const statusBadge = document.getElementById('status-badge');
        statusBadge.textContent = job.status;
        if (job.status === 'COMPLETED') {
          statusBadge.className = 'badge badge-success';
          document.getElementById('progress-bar').style.width = '100%';
        } else if (job.status === 'RUNNING' || job.status === 'ANALYZING' || job.status === 'AI_JUDGING' || job.status === 'RENDERING') {
          statusBadge.className = 'badge badge-active';
        }
      } catch (err) {}
    }

    function renderShorts(shorts) {
      const grid = document.getElementById('shorts-grid');
      const summary = document.getElementById('shorts-summary');
      summary.textContent = `${shorts.length} vertical Shorts produced`;
      grid.innerHTML = '';

      if (shorts.length === 0) {
        grid.innerHTML = '<div style="grid-column: 1/-1; padding: 32px; text-align: center; color: #9ca3af; border: 1px dashed #e5e7eb; border-radius: 6px;">No rendered Shorts yet. Click "Extract Real Shorts" to produce clips.</div>';
        return;
      }

      shorts.forEach(s => {
        const card = document.createElement('div');
        card.className = 'short-card';
        card.innerHTML = `
          <div class="video-wrapper">
            <video controls playsinline preload="metadata" poster="${s.preview_url || ''}">
              <source src="${s.url}" type="video/mp4">
            </video>
          </div>
          <div class="short-body">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <span class="badge badge-rating">★ ${s.editorial_rating}/5</span>
              <span class="badge">${s.event_category}</span>
            </div>
            <div class="short-hook">${s.suggested_hook || s.title}</div>
            <div class="short-desc">${s.summary || ''}</div>
            <div class="short-meta">
              <span>Time: ${s.start_seconds ? formatTime(s.start_seconds) : ''} - ${s.end_seconds ? formatTime(s.end_seconds) : ''}</span>
              <span>${s.size_mb} MB</span>
            </div>
            <div class="short-actions">
              <a href="${s.url}" download="${s.filename}">Download Short</a>
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
          <td style="max-width: 400px;">${j.summary || 'Awaiting evaluation'}</td>
          <td>${isSelected ? '<span class="badge badge-success">Approved</span>' : '<span class="badge">Reviewed</span>'}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function formatTime(secs) {
      if (!secs && secs !== 0) return '00:00';
      const m = Math.floor(secs / 60);
      const s = Math.floor(secs % 60);
      return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    async function startProcessing() {
      const video = document.getElementById('video-select').value;
      if (!video) {
        alert('Please select a video first');
        return;
      }
      const provider = document.getElementById('provider-select').value;
      const creator = document.getElementById('creator-name').value;
      const minShorts = parseInt(document.getElementById('min-shorts').value, 10) || 1;
      const maxShorts = parseInt(document.getElementById('max-shorts').value, 10) || 3;
      const minScore = parseInt(document.getElementById('min-score').value, 10) || 3;
      const maxCandidates = Math.max(10, minShorts * 3);

      document.getElementById('status-title').textContent = 'Starting pipeline...';
      document.getElementById('status-badge').textContent = 'QUEUED';
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
            min_score: minScore
          })
        });
        const data = await res.json();
        document.getElementById('status-title').textContent = `Processing ${video}...`;
        document.getElementById('status-badge').textContent = 'RUNNING';
        pollStatus();
      } catch (err) {
        alert('Failed to launch pipeline: ' + err);
      }
    }

    async function downloadYoutube() {
      const url = document.getElementById('yt-url').value;
      if (!url) return alert('Enter a valid YouTube URL');
      document.getElementById('status-title').textContent = 'Downloading video from YouTube...';
      document.getElementById('status-badge').textContent = 'DOWNLOADING';
      try {
        await fetch('/api/download', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({url: url})
        });
        pollStatus();
      } catch (err) {
        alert('Download request failed: ' + err);
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

        const activeKeys = Object.keys(tasks);
        if (activeKeys.length > 0) {
          const latestKey = activeKeys[activeKeys.length - 1];
          const t = tasks[latestKey];
          statusTitle.textContent = `${latestKey} (${t.status})`;
          statusBadge.textContent = t.status;
          if (t.logs && t.logs.length > 0) {
            logBox.textContent = t.logs.slice(-15).join('\\n');
            logBox.scrollTop = logBox.scrollHeight;
          }
          if (t.status === 'RUNNING') {
            pbar.style.width = '65%';
          } else if (t.status === 'COMPLETED') {
            pbar.style.width = '100%';
            statusBadge.className = 'badge badge-success';
            onVideoSelected();
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
