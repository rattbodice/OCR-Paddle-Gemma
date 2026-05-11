# celery_app.py
import os
from celery import Celery

celery = Celery(
    "ocr_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
    include=["tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,  # ผลลัพธ์หมดอายุใน 1 ชั่วโมง
)
