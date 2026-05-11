# ocr-api/Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 1) System deps: Python 3.10 + OpenCV + poppler for PDF
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenblas-dev \
 && ln -sf python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

# 2) PaddlePaddle GPU - CUDA 11.8 build (stable, widely tested)
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.2.post118 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 3) PaddleOCR + remaining dependencies (numpy<2 in same command so resolver respects it)
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    paddleocr==2.7.3 \
    fastapi \
    uvicorn \
    python-multipart \
    celery \
    redis \
    Pillow \
    openai \
    rapidfuzz \
    opencv-python-headless

WORKDIR /app
COPY app.py celery_app.py tasks.py ./

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--log-level","info"]
