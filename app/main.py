"""FastAPI app: serves the drag-and-drop UI and streams the pipeline over SSE."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline
from .config import OUTPUT_DIR, UPLOAD_DIR, WEB_DIR

app = FastAPI(title="Climbing Cam")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def _sse(events) -> "StreamingResponse":
    def gen():
        for ev in events:
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text()


@app.post("/api/upload")
async def upload(file: UploadFile) -> dict:
    job = pipeline.create_job(file.filename or "video.mp4")
    dest = Path(job.source)
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024 * 4):  # 4 MB chunks, streamed
            out.write(chunk)
    job.save_meta()
    return {"job_id": job.id, "filename": job.filename}


@app.get("/api/analyze/{job_id}")
def analyze(job_id: str):
    job = pipeline.Job.load(job_id)
    return _sse(pipeline.analyze_stream(job))


@app.get("/api/recut/{job_id}")
def recut(job_id: str):
    """Re-cut from cached detections (e.g. after tuning thresholds)."""
    job = pipeline.Job.load(job_id)
    return _sse(pipeline.recut_stream(job))


@app.get("/api/frame/{job_id}")
def frame(job_id: str):
    path = OUTPUT_DIR / job_id / "_confirm.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/clip/{job_id}/{name}")
def clip_file(job_id: str, name: str):
    path = OUTPUT_DIR / job_id / Path(name).name
    if not path.exists() or path.suffix != ".mp4":
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4", filename=name)


@app.get("/api/clips/{job_id}")
def clips(job_id: str) -> dict:
    return {"clips": pipeline.list_clips(job_id)}


@app.post("/api/reveal/{job_id}")
def reveal(job_id: str) -> dict:
    out_dir = OUTPUT_DIR / job_id
    if not out_dir.exists():
        raise HTTPException(404)
    subprocess.run(["open", str(out_dir)], check=False)  # macOS Finder
    return {"ok": True}
