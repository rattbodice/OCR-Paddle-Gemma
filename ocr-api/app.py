# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from celery.result import AsyncResult
import os, shutil, uuid
import cv2
import numpy as np
from celery_app import celery
from tasks import run_ocr, run_paddleocr

app = FastAPI()

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── API Key Auth ──────────────────────────────────────────────────
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    expected = os.getenv("OCR_API_KEY", "")
    if not expected:
        return
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


def has_poppler():
    return shutil.which("pdftoppm") is not None


@app.get("/health")
def health():
    return {
        "ok": True,
        "poppler": has_poppler(),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1"),
        "model": os.getenv("OLLAMA_MODEL", "gemma4:8b-instruct-q4_K_M"),
        "backend": "Ollama",
        "queue": "Celery + Redis",
    }


@app.post("/ocr")
async def ocr_api(
    file: UploadFile = File(...),
    target_name: str = Form(None),
    skip_preprocess: bool = Form(False),
    _: None = Security(verify_api_key),
):
    if file.filename.lower().endswith(".pdf") and not has_poppler():
        raise HTTPException(
            status_code=500,
            detail="Poppler (pdftoppm) is required for PDF. Please install poppler-utils.",
        )

    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    tmp_path = f"{UPLOAD_DIR}/{file_id}{ext}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    task = run_ocr.delay(tmp_path, skip_preprocess=skip_preprocess, target_name=target_name)
    return {"task_id": task.id, "status": "queued"}


@app.get("/preview/{filename}")
def get_preview_image(
    filename: str,
    _: None = Security(verify_api_key),
):
    """ดูภาพที่ผ่าน preprocessing แล้ว (_pre)"""
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path) or "_pre" not in filename:
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(filename)[1].lower()
    media = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media)


def _detect_text_regions(img: np.ndarray) -> list[dict]:
    """หา bounding box ของ text line โดยใช้ morphological operations"""
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 4,
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    kernel_w = max(20, iw // 40)
    dilated = cv2.dilate(binary,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3)),
                         iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = iw * ih * 0.0003
    pad = 6
    regions = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area or w < 15 or h < 6:
            continue
        regions.append({
            "id": i,
            "x": max(0, x - pad),
            "y": max(0, y - pad),
            "w": min(iw - max(0, x - pad), w + pad * 2),
            "h": min(ih - max(0, y - pad), h + pad * 2),
        })

    regions.sort(key=lambda r: (r["y"], r["x"]))
    for i, r in enumerate(regions):
        r["id"] = i
    return regions


@app.post("/detect-regions")
async def detect_regions(
    file: UploadFile = File(...),
    _: None = Security(verify_api_key),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    ih, iw = img.shape[:2]
    regions = _detect_text_regions(img)
    return {"width": iw, "height": ih, "regions": regions}


@app.post("/ocr-region")
async def ocr_region(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
    _: None = Security(verify_api_key),
):
    """Crop ภาพตามพิกัดที่ระบุ → PaddleOCR เฉพาะส่วนนั้น → คืน text"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    ih, iw = img.shape[:2]
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = max(1, min(w, iw - x))
    h = max(1, min(h, ih - y))

    crop = img[y:y + h, x:x + w]

    # CLAHE บน crop
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    crop = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b_ch]), cv2.COLOR_LAB2BGR)

    # Scale ให้สูงอย่างน้อย 64px เพื่อให้ PaddleOCR อ่านได้
    ch, cw = crop.shape[:2]
    if ch < 64:
        scale = 64 / ch
        crop = cv2.resize(crop, (int(cw * scale), 64), interpolation=cv2.INTER_CUBIC)

    text = run_paddleocr(crop)
    return {"text": text, "crop": {"x": x, "y": y, "w": w, "h": h}}


@app.get("/task/{task_id}")
def get_task(
    task_id: str,
    _: None = Security(verify_api_key),
):
    result = AsyncResult(task_id, app=celery)

    if result.state == "PENDING":
        return {"task_id": task_id, "status": "pending"}
    elif result.state == "STARTED":
        return {"task_id": task_id, "status": "processing"}
    elif result.state == "SUCCESS":
        return {"task_id": task_id, "status": "success", **result.result}
    elif result.state == "FAILURE":
        return {"task_id": task_id, "status": "failure", "error": str(result.result)}
    else:
        return {"task_id": task_id, "status": result.state.lower()}
