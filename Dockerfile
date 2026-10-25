# --- Этап 1: сборка зависимостей -------------------------------------------
# Отдельный этап, чтобы в рантайм-образ не попадали компиляторы и лишние пакеты.
FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Этап 2: рантайм ---------------------------------------------------------
FROM python:3.12-slim

# Системные библиотеки OpenCV и MediaPipe (тот же набор, что в .sourcecraft/ci.yaml)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libx11-6 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости из этапа сборки
COPY --from=builder /install /usr/local

# Код приложения (кроме данных пользователей и временных файлов)
COPY main.py storage.py club_tracker.py report_generator.py ./
COPY agent/ ./agent/
COPY scripts/ ./scripts/
COPY index.html live.html script.js style.css ./

# Модель PoseLandmarker: скачивание + верификация по pin-хешу на этапе сборки.
# Контейнер становится самодостаточным и стартует без обращения к сети.
RUN python scripts/download_pose_model.py

# Непривилегированный пользователь; каталоги данных — volume
RUN useradd --create-home appuser \
    && mkdir -p /app/uploads/temp \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]