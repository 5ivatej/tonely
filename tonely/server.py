"""FastAPI web app: upload → background analysis job → poll progress → result."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, models
from .config import Config
from .pipeline import analyze, cache_key

log = logging.getLogger("tonely")
WEB = Path(__file__).parent / "web"
CACHE = Path(os.environ.get("TONELY_CACHE", Path.home() / ".cache" / "tonely"))
MAX_UPLOAD = 500 * 1024 * 1024


class Jobs:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pool = ThreadPoolExecutor(max_workers=1)  # models are big; one job at a time
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        (CACHE / "results").mkdir(parents=True, exist_ok=True)

    def submit(self, path: Path, name: str) -> str:
        jid = uuid.uuid4().hex[:12]
        key = cache_key(path, self.cfg)
        cached = CACHE / "results" / f"{key}.json"
        job = {"id": jid, "name": name, "status": "queued", "progress": 0.0, "stage": None,
               "label": "Waiting in queue", "error": None, "result_path": None, "cached": False}
        with self.lock:
            self.jobs[jid] = job
        if cached.exists():
            job.update(status="done", progress=1.0, label="Loaded from cache", result_path=str(cached), cached=True)
            path.unlink(missing_ok=True)
            return jid
        self.pool.submit(self._run, job, path, name, cached)
        return jid

    def _run(self, job, path: Path, name: str, out: Path):
        def progress(stage, frac, label):
            job.update(status="running", stage=stage, progress=round(frac, 4), label=label)

        try:
            job.update(status="running", label="Starting")
            result = analyze(path, self.cfg, progress, name=name)
            out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            job.update(status="done", progress=1.0, label="Done", result_path=str(out))
        except Exception as exc:  # noqa: BLE001
            log.exception("analysis failed")
            job.update(status="error", error=str(exc), label="Failed")
        finally:
            path.unlink(missing_ok=True)

    def get(self, jid: str) -> dict:
        job = self.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config()
    app = FastAPI(title="tonely", version=__version__)
    jobs = Jobs(cfg)

    @app.get("/api/info")
    def info():
        return {
            "version": __version__,
            "device": models.torch_device(cfg.device),
            "analyzers": {
                "acoustic": True, "asr": cfg.use_asr, "text": cfg.use_asr and cfg.use_text,
                "ser": cfg.use_ser, "ser_dim": cfg.use_ser_dim, "clap": cfg.use_clap,
            },
            "whisper": cfg.whisper_model,
        }

    @app.post("/api/jobs")
    async def create_job(file: UploadFile = File(...)):
        suffix = Path(file.filename or "audio").suffix[:10] or ".bin"
        tmp = Path(tempfile.mkstemp(prefix="tonely-", suffix=suffix)[1])
        size = 0
        with open(tmp, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    f.close()
                    tmp.unlink(missing_ok=True)
                    raise HTTPException(413, "file too large (500 MB max)")
                f.write(chunk)
        return {"id": jobs.submit(tmp, file.filename or "audio")}

    @app.get("/api/jobs/{jid}")
    def job_status(jid: str):
        job = jobs.get(jid)
        return {k: v for k, v in job.items() if k != "result_path"}

    @app.get("/api/jobs/{jid}/result")
    def job_result(jid: str):
        job = jobs.get(jid)
        if job["status"] != "done":
            raise HTTPException(409, f"job is {job['status']}")
        return FileResponse(job["result_path"], media_type="application/json")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.exception_handler(ValueError)
    async def bad_input(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return app


def run(host: str, port: int, cfg: Config, open_browser: bool = True):
    import uvicorn

    if shutil.which("ffmpeg") is None:
        log.debug("system ffmpeg not found — fine, PyAV bundles its own decoder")
    url = f"http://{host}:{port}"
    log.info("tonely %s → %s", __version__, url)
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
