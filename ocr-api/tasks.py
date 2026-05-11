import os
import re
import json
import time
import cv2
import numpy as np
from celery_app import celery
from paddleocr import PaddleOCR
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────
TARGET_WIDTH = 1280

_paddle = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    use_gpu=True,
    show_log=False,
)


# ── Robust Preprocessing ─────────────────────────────────────────
def correct_perspective_robust(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > (w * h * 0.15):
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            return warped
    return img

def enhance_for_ocr(img):
    h, w = img.shape[:2]
    ratio = TARGET_WIDTH / float(w)
    new_h = int(h * ratio)
    img = cv2.resize(img, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    enhanced = cv2.medianBlur(enhanced, 3)
    return enhanced


# ── Title Extraction & Fuzzy Matching ────────────────────────────
_TITLE_GENDER = {
    "นาย": "M", "mr": "M",
    "นาง": "F", "mrs": "F",
    "นางสาว": "F", "miss": "F", "ms": "F", "น.ส.": "F",
    "ดช": "M", "ดญ": "F",
}

def _extract_title(s: str):
    s = s.strip().lower().replace(".", "")
    for title in sorted(_TITLE_GENDER, key=len, reverse=True):
        if s.startswith(title):
            return title
    return None

def _titles_compatible(a: str, b: str) -> bool:
    ta = _extract_title(a)
    tb = _extract_title(b)
    if ta and tb:
        return _TITLE_GENDER.get(ta) == _TITLE_GENDER.get(tb)
    return True

_TITLE_STRIP_RE = None

def _strip_title(s: str) -> str:
    global _TITLE_STRIP_RE
    if _TITLE_STRIP_RE is None:
        titles = ["นางสาว", "น.ส.", "นาง", "นาย", "ดช.", "ดญ.", "ดช", "ดญ",
                  "miss", "mrs.", "mr.", "ms.", "mrs", "mr", "ms", "คุณ"]
        titles.sort(key=len, reverse=True)
        _TITLE_STRIP_RE = re.compile(
            r"^(" + "|".join(re.escape(t) for t in titles) + r")\s*",
            re.IGNORECASE,
        )
    return _TITLE_STRIP_RE.sub("", s.strip()).strip()

def _fuzzy_score(a: str, b: str) -> float:
    from rapidfuzz import fuzz
    a = _strip_title(a).lower().replace(".", "")
    b = _strip_title(b).lower().replace(".", "")
    return fuzz.token_set_ratio(a, b)


# ── PaddleOCR Text Extraction ─────────────────────────────────────
def run_paddleocr(image_input) -> str:
    """Run PaddleOCR on file path or numpy array; returns joined text lines."""
    result = _paddle.ocr(image_input, cls=True)
    if not result or not result[0]:
        return ""
    lines = []
    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        if conf > 0.5:
            lines.append(text)
    return "\n".join(lines)


# ── Ollama Text-based Extraction ──────────────────────────────────
BANK_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "account_number": {"type": ["string", "null"]},
        "account_name":   {"type": ["string", "null"]},
        "bank_name":      {"type": ["string", "null"]},
    },
    "required": ["account_number", "account_name", "bank_name"],
    "additionalProperties": False,
}

EXTRACT_PROMPT = (
    "จากข้อความต่อไปนี้ที่อ่านได้จากภาพบัญชีธนาคาร ให้ดึงข้อมูล:\n"
    "- account_number: เลขที่บัญชี (ทุก format)\n"
    "- account_name: ชื่อ-นามสกุลเจ้าของบัญชี\n"
    "- bank_name: ชื่อธนาคาร\n"
    "ถ้าหาค่าไหนไม่เจอให้ใส่ null\n"
    "ตอบเป็น JSON เท่านั้น ห้ามมีคำอธิบาย\n\n"
    "ข้อความ:\n{ocr_text}"
)

def extract_bank_info_from_text(ocr_text: str, target_name: str, base_url: str, api_key: str, model: str) -> dict:
    client = OpenAI(base_url=base_url, api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": EXTRACT_PROMPT.format(ocr_text=ocr_text),
        }],
        max_tokens=150,
        temperature=0.0,
        extra_body={"format": BANK_INFO_SCHEMA},
    )

    content = response.choices[0].message.content.strip()
    print(f"[extract] raw response: {content}")

    if content.startswith("```"):
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m_num  = re.search(r'"account_number"\s*:\s*"([^"]*)"', content)
        m_name = re.search(r'"account_name"\s*:\s*"([^"]*)"', content)
        m_bank = re.search(r'"bank_name"\s*:\s*"([^"]*)"', content)
        data = {
            "account_number": m_num.group(1)  if m_num  else None,
            "account_name":   m_name.group(1) if m_name else None,
            "bank_name":      m_bank.group(1) if m_bank else None,
        }
        print(f"[extract] JSON parse failed, recovered: {data}")

    for k in ("account_number", "account_name", "bank_name"):
        if data.get(k) in (None, "", "null", "NULL"):
            data[k] = None
        elif data[k]:
            data[k] = data[k].strip()

    if not target_name:
        data["match_status"] = "mismatch"
        data["fuzzy_score"] = None
    else:
        account_name_str = data.get("account_name") or ""
        title_ok = _titles_compatible(account_name_str, target_name)
        score = _fuzzy_score(account_name_str, target_name)
        if not title_ok or score < 75:
            data["match_status"] = "mismatch"
        elif score < 90:
            data["match_status"] = "review"
        else:
            data["match_status"] = "match"
        data["fuzzy_score"] = score
        print(f"[extract] title_ok={title_ok} fuzzy={score:.1f} match_status={data['match_status']}: '{account_name_str}' vs '{target_name}'")

    return data


# ── Celery Task ──────────────────────────────────────────────────
@celery.task(bind=True, max_retries=3, soft_time_limit=120, time_limit=150)
def run_ocr(self, file_path: str, skip_preprocess: bool = False, target_name: str = None):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
    api_key  = "ollama"
    model    = os.getenv("OLLAMA_MODEL",    "gemma4:8b-instruct-q4_K_M")

    print(f"[run_ocr] start task for {file_path} (skip_preprocess={skip_preprocess})")
    print(f"[run_ocr] target_name='{target_name}'")

    blur_threshold = float(os.getenv("BLUR_THRESHOLD", "80"))

    pre_path = None
    try:
        if skip_preprocess:
            ocr_input = file_path
            print("[run_ocr] skip_preprocess=True: ข้าม preprocess")
        else:
            img = cv2.imread(file_path)
            if img is None: raise ValueError("Image not found")
            img = correct_perspective_robust(img)
            img = enhance_for_ocr(img)
            pre_path = file_path.replace(os.path.splitext(file_path)[1], "_pre.jpg")
            cv2.imwrite(pre_path, img)
            ocr_input = pre_path

        # ประเมินความชัดของภาพก่อน OCR
        check_img = cv2.imread(ocr_input)
        gray = cv2.cvtColor(check_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"[run_ocr] blur_score={blur_score:.1f} (threshold={blur_threshold})")
        if blur_score < blur_threshold:
            return {
                "ocr_text": None,
                "bank_info": {"account_number": None, "account_name": None, "bank_name": None, "match_status": "blurry"},
                "blur_score": blur_score,
                "target_name_checked": target_name,
                "pre_file": os.path.basename(pre_path) if pre_path else None,
            }

        # Step 1: PaddleOCR (CPU)
        ocr_text = run_paddleocr(ocr_input)
        print(f"[run_ocr] PaddleOCR extracted {len(ocr_text)} chars")
        print(ocr_text)

        if not ocr_text.strip():
            print("[run_ocr] no text from PaddleOCR - skipping extraction")
            bank_info = {"account_number": None, "account_name": None, "bank_name": None, "match_status": "no_text"}
        else:
            # Step 2: Gemma 4 E4B via Ollama (text-only, 1 call)
            bank_info = extract_bank_info_from_text(ocr_text, target_name, base_url, api_key, model)

        return {
            "ocr_text": ocr_text,
            "bank_info": bank_info,
            "blur_score": blur_score,
            "target_name_checked": target_name,
            "pre_file": os.path.basename(pre_path) if pre_path else None,
        }
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
