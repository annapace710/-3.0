#!/usr/bin/env python3
"""
Скачивает модель PoseLandmarker и фиксирует её SHA-256 в pin-файл.

Заменяет скачивание при импорте main.py (side effect на import, race condition
при нескольких воркерах). Запуск:

    python scripts/download_pose_model.py

Скрипт идемпотентен: если модель уже есть и хеш совпадает — не перекачивает.
При нескольких параллельных запусках скачивание атомарно (tmp-файл + os.replace).
"""

import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем пути из main (без выполнения анализа, только константы)
from main import MODEL_PATH, MODEL_SHA256_PATH, _file_sha256, verify_pose_model

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def main() -> int:
    if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_SHA256_PATH):
        with open(MODEL_SHA256_PATH, "r", encoding="utf-8") as f:
            pinned = f.read().strip()
        if _file_sha256(MODEL_PATH) == pinned:
            print(f"Модель уже на месте и хеш совпадает: {MODEL_PATH}")
            return 0
        print("Хеш существующей модели не совпадает с pin-файлом — перекачиваю...")

    print(f"Скачивание модели: {MODEL_URL}")
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(MODEL_PATH), suffix=".task.tmp"
    )
    os.close(tmp_fd)
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        sha256 = _file_sha256(tmp_path)
        # Атомарная установка файла и pin-хеша (безопасно при параллельных воркерах)
        os.replace(tmp_path, MODEL_PATH)
        with open(MODEL_SHA256_PATH, "w", encoding="utf-8") as f:
            f.write(sha256)
        print(f"SHA-256: {sha256}")
        print(f"Pin-файл записан: {MODEL_SHA256_PATH}")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if not verify_pose_model():
        print("Ошибка: проверка после скачивания не прошла.", file=sys.stderr)
        return 1
    print("Модель PoseLandmarker готова.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
