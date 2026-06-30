import io
import time
import os
import urllib.parse
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from loguru import logger

from app.core.config import MODELS_ONNX_DIR
from app.pipelines.onnx_pipeline import ONNXPipeline

try:
    from app.pipelines.sam2_pipeline import MattingPipeline as SAM2Pipeline
except ImportError as e:
    logger.warning(f"SAM2 pipeline dependencies not found ({e}). 'ben2+sam2matting' mode will be unavailable.")
    SAM2Pipeline = None

# Global pipeline dictionary
pipelines = {}

# Locks to ensure thread-safe inference
pipeline_locks = {
    "birefnet": asyncio.Lock(),
    "sam2": asyncio.Lock()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize pipelines on startup
    logger.info("Initializing models...")
    pipelines["birefnet"] = ONNXPipeline(
        model_path=MODELS_ONNX_DIR / "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
    )

    if SAM2Pipeline:
        pipelines["sam2"] = SAM2Pipeline()
    logger.info("Models ready.")
    
    yield
    
    # Clean up on shutdown
    pipelines.clear()
    logger.info("Models unloaded.")

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Background Remover", version="0.1.0", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.post("/api/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    mode: str = Form("birefnet"),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Format {file.content_type} not supported.",
        )

    data = await file.read()

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB.",
        )

    allowed_modes = ["birefnet", "ben2+sam2matting"]
    if mode not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{mode}'. Valid: {allowed_modes}",
        )

    if mode == "ben2+sam2matting" and "sam2" not in pipelines:
        raise HTTPException(status_code=500, detail="SAM2 pipeline not initialized.")

    try:
        pil_image = Image.open(io.BytesIO(data))
        t0 = time.time()
        
        # Run CPU/GPU-heavy inference in a separate thread to avoid blocking the asyncio event loop
        # We use a Lock to prevent concurrent model inference which could cause OOM or state corruption
        if mode == "birefnet":
            async with pipeline_locks["birefnet"]:
                result_rgba = await asyncio.to_thread(pipelines["birefnet"].process_image, pil_image)
        else:
            async with pipeline_locks["sam2"]:
                result_rgba = await asyncio.to_thread(pipelines["sam2"].process_image, pil_image)
            
        elapsed = time.time() - t0
        logger.info(f"{mode}: {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    buf = io.BytesIO()
    result_rgba.save(buf, format="PNG")
    buf.seek(0)

    base_name = os.path.splitext(file.filename or "image")[0]
    out_filename = f"{base_name}_removed_bg.png"
    encoded_filename = urllib.parse.quote(out_filename)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{encoded_filename}",
            "X-Processing-Time": f"{elapsed:.2f}",
        },
    )
