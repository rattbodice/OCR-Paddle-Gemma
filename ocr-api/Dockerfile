# ocr-api/Dockerfile
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# 1) System deps: Python 3.10 + OpenCV + poppler for PDF
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    libcudnn8 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenblas-dev \
 && ln -sf python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

# 2) PaddlePaddle GPU - install from PaddlePaddle index (CUDA 12.0 build, compatible with 12.2)
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/

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
