
import sys
import uuid
import asyncio
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, HTMLResponse
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from worker.worker import (
    analyze_video,
    MODEL_OPTIONS,
    OUTPUT_DIR,
)

app = FastAPI(
    title="WasteLens Predict API",
    description="AI-powered waste monitoring and video analysis",
    version="2.0.0",
)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html_path = PROJECT_DIR / "frontend" / "index.html"

    if not html_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dashboard file not found.",
        )

    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8")
    )

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE = 500 * 1024 * 1024

jobs = {}


@app.get("/")
def home():
    return {
        "project": "WasteLens Predict",
        "status": "running",
        "message": "WasteLens API is ready",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/model-info")
def model_info():
    return {
        "models": {
            name: {
                "available": path.is_file(),
                "path": str(path),
                "description": (
                    "Experimental overflowing-bin detector; "
                    "outdoor accuracy still needs testing."
                    if name == "candidate"
                    else "Original four-class waste detector."
                ),
            }
            for name, path in MODEL_OPTIONS.items()
        },
        "default_model": "roboflow",
        "inference": "Roboflow RF-DETR cloud inference",
        "cloud_gpu_connected": True,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    original_name = Path(file.filename or "video").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Use MP4, AVI, MOV, or MKV video files.",
        )

    job_id = str(uuid.uuid4())
    destination = UPLOAD_DIR / f"{job_id}{extension}"
    total_size = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 500 MB limit.",
                    )

                output.write(chunk)

    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    job = {
        "job_id": job_id,
        "filename": original_name,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "status": "uploaded",
        "message": "Video uploaded and ready for analysis.",
        "model_choice": None,
        "results": None,
    }

    jobs[job_id] = job
    return job


@app.get("/api/jobs")
def list_jobs():
    return {"total": len(jobs), "jobs": list(jobs.values())}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found. Jobs reset after a server restart.",
        )
    return job


@app.post("/api/jobs/{job_id}/analyze")
async def analyze_uploaded_video(
    job_id: str,
    model_choice: str = Query(
        default="original",
        pattern="^(original|candidate|roboflow)$",
    ),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0),
    sample_every_n_frames: int = Query(default=5, ge=1, le=300),
):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="This video is already being analyzed.",
        )

    if job["status"] == "completed":
        if job["model_choice"] == model_choice:
            return {
                "job_id": job_id,
                "status": "completed",
                "results": job["results"],
            }

        raise HTTPException(
            status_code=409,
            detail=(
                "This job already has results from another model. "
                "Upload the video again to test a different model."
            ),
        )

    if model_choice != "roboflow":
        if not MODEL_OPTIONS[model_choice].is_file():
            raise HTTPException(
                status_code=503,
                detail=f"The {model_choice} model file is missing.",
        )

    extension = Path(job["filename"]).suffix.lower()
    video_path = UPLOAD_DIR / f"{job_id}{extension}"

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Uploaded video file not found.",
        )

    job["status"] = "processing"
    job["model_choice"] = model_choice
    job["message"] = f"Analyzing video with the {model_choice} model."

    try:
        result = await asyncio.to_thread(
            analyze_video,
            str(video_path),
            sample_every_n_frames,
            model_choice,
            confidence,
        )

        result["annotated_video_url"] = (
            f"/api/jobs/{job_id}/video"
        )
        result["results_url"] = f"/api/jobs/{job_id}/results"

        job["status"] = "completed"
        job["results"] = result
        job["message"] = "Video analysis completed."

        return {
            "job_id": job_id,
            "status": "completed",
            "results": result,
        }

    except Exception as exc:
        job["status"] = "failed"
        job["message"] = "Analysis failed. Check the backend terminal."
        print(f"Analysis error for job {job_id}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=job["message"],
        )


@app.get("/api/jobs/{job_id}/video")
def get_annotated_video(job_id: str):
    job = jobs.get(job_id)
    if job is None or not job.get("results"):
        raise HTTPException(
            status_code=404,
            detail="Analysis results are not available.",
        )

    filename = job["results"].get("annotated_video")
    if not filename:
        raise HTTPException(
            status_code=404,
            detail="Annotated video is not available.",
        )

    output_path = (OUTPUT_DIR / filename).resolve()
    if output_path.parent != OUTPUT_DIR.resolve() or not output_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Annotated video file not found.",
        )

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=filename,
        content_disposition_type="inline",
    )


@app.get("/api/jobs/{job_id}/results")
def get_analysis_results(job_id: str):
    job = jobs.get(job_id)
    if job is None or not job.get("results"):
        raise HTTPException(
            status_code=404,
            detail="Analysis results are not available.",
        )

    return job["results"]
