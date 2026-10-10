import os
import time
import threading
import uuid
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, HTMLResponse
import cv2  # OpenCV
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from club_tracker import ClubTracker, analyze_club_flight, assign_clubs_to_athletes
import storage
import report_generator
from agent.llm_client import (
    LLMNotConfiguredError,
    LLMRequestError,
    get_llm_client,
)
from agent.prompts import build_report_context

app = FastAPI()

# Размер группы для группового формата (число гимнасток)
GROUP_SIZE = 5

# --- Авторизация и rate limiting на /upload (пункт 6 ревью) ---
# Список валидных API-ключей задается переменной окружения API_KEYS (ключи через запятую).
# Если переменная не задана или пуста — открытый доступ (режим разработки/MVP).
# Для продакшена: задайте API_KEYS в окружении, клиенты передают ключ в заголовке X-API-Key.
API_KEYS = [
    k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()
]
API_KEY_HEADER = "X-API-Key"

# Rate limit на /upload: сколько загрузок в минуту разрешено на один ключ/IP.
# Счетчики хранятся в памяти процесса (сбрасываются при рестарте) — для MVP достаточно.
UPLOAD_RATE_LIMIT_PER_MINUTE = int(os.environ.get("UPLOAD_RATE_LIMIT_PER_MINUTE", "5"))

_rate_limit_lock = threading.Lock()
_rate_limit_buckets: dict = {}  # client_id -> list[timestamp] последних запросов


def _client_id(request: Request, api_key: str) -> str:
    """Идентификатор клиента для rate limit: API-ключ или IP-адрес."""
    return api_key or (request.client.host if request.client else "unknown")


def _check_rate_limit(client_id: str) -> bool:
    """
    Скользящее окно 60 секунд: возвращает True, если запрос разрешен
    (не превышен UPLOAD_RATE_LIMIT_PER_MINUTE).
    """
    now = time.monotonic()
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(client_id, [])
        # Убираем записи старше 60 секунд
        bucket[:] = [ts for ts in bucket if now - ts < 60.0]
        if len(bucket) >= UPLOAD_RATE_LIMIT_PER_MINUTE:
            return False
        bucket.append(now)
        return True


def _require_api_key(request: Request) -> str:
    """
    Проверяет API-ключ (если авторизация настроена через API_KEYS).
    Возвращает ключ клиента (или ""), если авторизация выключена.
    Бросает HTTPException 401/429 при ошибке.
    """
    api_key = request.headers.get(API_KEY_HEADER, "")
    if API_KEYS:
        if api_key not in API_KEYS:
            raise HTTPException(
                status_code=401,
                detail="Недействительный API-ключ. Передайте ключ в заголовке X-API-Key.",
            )
    if not _check_rate_limit(_client_id(request, api_key)):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Превышен лимит загрузок: {UPLOAD_RATE_LIMIT_PER_MINUTE} в минуту. "
                "Повторите попытку позже."
            ),
        )
    return api_key


# --- Ленивый singleton PoseLandmarker (пункт 5 ревью) ---
# Инстанс MediaPipe дорогой: создается один раз и переиспользуется между запросами.
# threading.Lock защищает от гонки при одновременных первых запросах (threadpool).
_pose_lock = threading.Lock()
_pose_landmarker = None

# Каждое новое видео анализируется со своим базовым таймстампом:
# detect_for_video требует монотонно растущие таймстампы между вызовами
# одного инстанса, поэтому к таймстампам кадра добавляем смещение per-video.
_VIDEO_TS_OFFSET_MS = 100_000  # запас > максимальной длительности одного видео
_pose_ts_offset = [0]  # общий счетчик смещений (под защищенным доступом via _pose_lock)


def get_pose_landmarker():
    """Возвращает переиспользуемый инстанс PoseLandmarker (создается лениво)."""
    global _pose_landmarker
    if _pose_landmarker is None:
        with _pose_lock:
            if _pose_landmarker is None:  # double-checked locking
                base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
                options = mp_vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


# CORS: разрешаем обращения с фронтенда (SourceCraft Sites), локальной разработки
# и туннелей Google Colab (Cloudflare/ngrok)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://annapace.sourcecraft.site",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://(.*\.trycloudflare\.com|.*\.ngrok-free\.app|.*\.ngrok\.io)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Директория для временного хранения загруженных видео
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads", "temp")
ALLOWED_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".m4v"}
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Гарантируем существование директории для загрузок
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- MediaPipe Tasks API (новый API, совместим с Python 3.13) ---
# Модель pose_landmarker_lite.task должна лежать в корне проекта.
# Скачивание при импорте убрано (side effect на import + race condition воркеров):
# модель ставится скриптом scripts/download_pose_model.py, который записывает
# pin-файл pose_landmarker_lite.sha256; при старте приложения целостность
# файла проверяется по этому pin-хешу.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
MODEL_SHA256_PATH = MODEL_PATH + ".sha256"


def _file_sha256(path: str) -> str:
    """Вычисляет SHA-256 файла (потоково, для больших файлов)."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_pose_model() -> bool:
    """
    Проверяет наличие модели и её целостность по pin-хешу (если pin-файл есть).

    Возвращает True, только если модель существует И (pin-файл отсутствует —
    тогда в /health будет degraded до первой проверки скриптом; или хеш совпал).
    """
    if not os.path.exists(MODEL_PATH):
        print(
            "Модель PoseLandmarker не найдена. Запустите: "
            "python scripts/download_pose_model.py"
        )
        return False
    if not os.path.exists(MODEL_SHA256_PATH):
        print(
            "ПРЕДУПРЕЖДЕНИЕ: pin-файл SHA-256 модели отсутствует "
            f"({MODEL_SHA256_PATH}). Запустите python scripts/download_pose_model.py, "
            "чтобы зафиксировать доверенный хеш."
        )
        return False
    with open(MODEL_SHA256_PATH, "r", encoding="utf-8") as f:
        pinned = f.read().strip()
    if _file_sha256(MODEL_PATH) != pinned:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: SHA-256 модели PoseLandmarker не совпадает с pin-хешем. "
            "Запустите python scripts/download_pose_model.py, чтобы скачать доверенную версию."
        )
        return False
    return True


POSE_MODEL_OK = verify_pose_model()

# Кастомная ONNX-модель детекции булав (обучается через train_club_detector.py).
# Если файл отсутствует — используется HSV-сегментация (club_tracker.py).
CLUB_MODEL_PATH = os.path.join(os.path.dirname(__file__), "club_detector.onnx")

# Индексы ключевых точек MediaPipe Pose (33 точки)
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24
LANDMARK_LEFT_KNEE = 25
LANDMARK_RIGHT_KNEE = 26
LANDMARK_LEFT_ANKLE = 27
LANDMARK_RIGHT_ANKLE = 28


def validate_video_file(filename: str, content_length: int) -> None:
    """Проверяет формат и размер загружаемого видеофайла."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Недопустимый формат файла: {ext}. "
            f"Поддерживаются: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if content_length > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Файл слишком большой. Максимальный размер — {MAX_FILE_SIZE_MB} МБ."
        )


def _point(landmarks, index: int) -> np.ndarray:
    """Возвращает координаты (x, y) ключевой точки."""
    lm = landmarks[index]
    return np.array([lm.x, lm.y])


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Угол между двумя векторами в градусах."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_torso_tilt(landmarks) -> float:
    """
    Угол наклона корпуса от вертикали (в градусах).
    Используется середина плеч и середина таза.
    """
    shoulder_mid = (_point(landmarks, LANDMARK_LEFT_SHOULDER) + _point(landmarks, LANDMARK_RIGHT_SHOULDER)) / 2
    hip_mid = (_point(landmarks, LANDMARK_LEFT_HIP) + _point(landmarks, LANDMARK_RIGHT_HIP)) / 2

    torso_vec = shoulder_mid - hip_mid  # от таза к плечам
    vertical = np.array([0.0, -1.0])    # направление «вверх»

    return _angle_between(torso_vec, vertical)


def compute_wrist_height(landmarks) -> float:
    """
    Нормализованная высота запястий (среднее y левого и правого).
    Значение y растёт вниз: чем МЕНЬШЕ y, тем ВЫШЕ руки.
    """
    left_y = landmarks[LANDMARK_LEFT_WRIST].y
    right_y = landmarks[LANDMARK_RIGHT_WRIST].y
    return float((left_y + right_y) / 2)


def compute_center_of_mass(landmarks) -> np.ndarray:
    """
    Приблизительный центр масс тела: среднее всех видимых точек.
    Используется для отслеживания траектории движения.
    """
    visible = [
        np.array([lm.x, lm.y])
        for lm in landmarks
        if getattr(lm, "visibility", 1.0) > 0.5
    ]
    if not visible:
        return np.array([0.5, 0.5])
    return np.mean(visible, axis=0)


def compute_knee_angle(landmarks, side: str) -> float:
    """
    Угол в коленном суставе (0–180°).
    side: "left" или "right". 180° — полностью выпрямленная нога.
    """
    if side == "left":
        hip = _point(landmarks, LANDMARK_LEFT_HIP)
        knee = _point(landmarks, LANDMARK_LEFT_KNEE)
        ankle = _point(landmarks, LANDMARK_LEFT_ANKLE)
    else:
        hip = _point(landmarks, LANDMARK_RIGHT_HIP)
        knee = _point(landmarks, LANDMARK_RIGHT_KNEE)
        ankle = _point(landmarks, LANDMARK_RIGHT_ANKLE)

    v1 = hip - knee
    v2 = ankle - knee
    return _angle_between(v1, v2)


def compute_leg_opening_angle(landmarks) -> float:
    """
    Угол раскрытия ног (шпагат) при прыжке: угол между векторами
    «таз → левая стопа» и «таз → правая стопа» (0–180°).
    Чем ближе к 180°, тем шире раскрыты ноги.
    """
    hip_mid = (_point(landmarks, LANDMARK_LEFT_HIP) + _point(landmarks, LANDMARK_RIGHT_HIP)) / 2
    left_vec = _point(landmarks, LANDMARK_LEFT_ANKLE) - hip_mid
    right_vec = _point(landmarks, LANDMARK_RIGHT_ANKLE) - hip_mid
    return _angle_between(left_vec, right_vec)


def compute_hip_height(landmarks) -> float:
    """
    Нормализованная высота таза (среднее y левого и правого бедра).
    Значение y растёт вниз: чем МЕНЬШЕ y, тем ВЫШЕ таз.
    """
    left_y = landmarks[LANDMARK_LEFT_HIP].y
    right_y = landmarks[LANDMARK_RIGHT_HIP].y
    return float((left_y + right_y) / 2)


def compute_ankle_y(landmarks, side: str) -> float:
    """
    Нормализованная высота лодыжки (y) для заданной стороны.
    Значение y растёт вниз: чем МЕНЬШЕ y, тем ВЫШЕ стопа.
    Используется для определения опорной ноги в равновесии.
    """
    idx = LANDMARK_LEFT_ANKLE if side == "left" else LANDMARK_RIGHT_ANKLE
    return float(landmarks[idx].y)


def compute_ankle_height(landmarks) -> float:
    """
    Средняя нормализованная высота стоп (среднее y левой и правой лодыжки).
    Чем МЕНЬШЕ значение, тем ВЫШЕ подняты стопы (прыжок).
    """
    left_y = landmarks[LANDMARK_LEFT_ANKLE].y
    right_y = landmarks[LANDMARK_RIGHT_ANKLE].y
    return float((left_y + right_y) / 2)


def compute_arm_symmetry(landmarks) -> float:
    """
    Асимметрия рук: модуль разницы высот левого и правого запястья (0..1).
    Чем ближе к 0, тем симметричнее руки (важно для пируэта).
    """
    left_y = landmarks[LANDMARK_LEFT_WRIST].y
    right_y = landmarks[LANDMARK_RIGHT_WRIST].y
    return float(abs(left_y - right_y))


def compute_rotation_angle_world(world_landmarks) -> float:
    """
    Угол ориентации плечевой линии в горизонтальной плоскости (градусы).
    Использует 3D world landmarks (x — вправо, y — вверх, z — от камеры).
    Применяется для подсчёта числа витков в пируэте.
    """
    left = world_landmarks[LANDMARK_LEFT_SHOULDER]
    right = world_landmarks[LANDMARK_RIGHT_SHOULDER]
    dx = right.x - left.x
    dz = right.z - left.z
    return float(np.degrees(np.arctan2(dz, dx)))


def compute_torso_tilt_signed(landmarks) -> float:
    """
    Угол наклона корпуса от вертикали в градусах со знаком.
    Положительный — наклон влево, отрицательный — вправо (в плоскости кадра).
    Используется для анализа симметрии оси вращения в пируэте.
    """
    shoulder_mid = (_point(landmarks, LANDMARK_LEFT_SHOULDER) + _point(landmarks, LANDMARK_RIGHT_SHOULDER)) / 2
    hip_mid = (_point(landmarks, LANDMARK_LEFT_HIP) + _point(landmarks, LANDMARK_RIGHT_HIP)) / 2

    torso_vec = shoulder_mid - hip_mid  # от таза к плечам
    vertical = np.array([0.0, -1.0])    # направление «вверх»

    # Угол со знаком: положительный при наклоне влево от вертикали
    # (если плечи смещены влево относительно таза, torso_vec[0] < 0 -> угол положительный)
    signed_deg = float(np.degrees(np.arctan2(-torso_vec[0], -torso_vec[1])))
    # Ограничиваем диапазон до [-90, 90]
    return float(np.clip(signed_deg, -90.0, 90.0))


def select_target_athletes(poses, format_type: str, group_size: int = GROUP_SIZE):
    """
    Выбирает целевых гимнасток из списка всех обнаруженных поз.

    poses: список списков landmark-точек (каждый элемент — скелет одной гимнастки).
    format_type: "individual" — один самый крупный скелет; "group" — group_size самых крупных.

    Возвращает список выбранных скелетов (для individual — не более 1).
    """
    if not poses:
        return []

    # Оценка «размера» скелета по bounding box ключевых точек
    def _size(landmarks):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    ranked = sorted(poses, key=_size, reverse=True)

    if format_type == "group":
        return ranked[:group_size]
    # individual — один самый крупный
    return ranked[:1]


# Максимальное нормализованное расстояние для сопоставления скелета с треком гимнастки
MAX_ATHLETE_MATCH_DIST = 0.30


def _skeleton_center(landmarks) -> tuple:
    """Центр скелета: среднее всех ключевых точек."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _skeleton_size(landmarks) -> float:
    """Площадь bounding box скелета (для ранжирования «кто на переднем плане»)."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


class AthleteTrack:
    """
    Трек одной гимнастки: накапливает временные ряды признаков по кадрам.
    Используется для последующего вычисления агрегированных метрик.
    """

    def __init__(self, index: int):
        self.index = index
        self.torso_angles = []
        self.wrist_heights = []
        self.wrist_y_left = []   # высота левого запястья по кадрам (для асимметрии подброса)
        self.wrist_y_right = []  # высота правого запястья по кадрам
        self.com_positions = []
        self.knee_angles_left = []
        self.knee_angles_right = []
        self.leg_opening_angles = []
        self.hip_heights = []
        self.signed_torso_angles = []
        self.ankle_heights = []
        self.ankle_y_left = []
        self.ankle_y_right = []
        self.arm_symmetries = []
        self.rotation_angles_world = []
        self.wrist_positions = []   # средние позиции запястий (для связывания булав)
        self.last_center = None
        self.size_sum = 0.0
        self.frames_seen = 0

    def add_frame(self, landmarks, world_landmarks=None) -> None:
        """Добавляет кадр: вычисляет все признаки и сохраняет в ряды."""
        self.torso_angles.append(compute_torso_tilt(landmarks))
        self.wrist_heights.append(compute_wrist_height(landmarks))
        self.wrist_y_left.append(float(landmarks[LANDMARK_LEFT_WRIST].y))
        self.wrist_y_right.append(float(landmarks[LANDMARK_RIGHT_WRIST].y))
        self.com_positions.append(compute_center_of_mass(landmarks))
        self.knee_angles_left.append(compute_knee_angle(landmarks, "left"))
        self.knee_angles_right.append(compute_knee_angle(landmarks, "right"))
        self.leg_opening_angles.append(compute_leg_opening_angle(landmarks))
        self.hip_heights.append(compute_hip_height(landmarks))
        self.signed_torso_angles.append(compute_torso_tilt_signed(landmarks))
        self.ankle_heights.append(compute_ankle_height(landmarks))
        self.ankle_y_left.append(compute_ankle_y(landmarks, "left"))
        self.ankle_y_right.append(compute_ankle_y(landmarks, "right"))
        self.arm_symmetries.append(compute_arm_symmetry(landmarks))

        left_w = _point(landmarks, LANDMARK_LEFT_WRIST)
        right_w = _point(landmarks, LANDMARK_RIGHT_WRIST)
        self.wrist_positions.append(((left_w[0] + right_w[0]) / 2, (left_w[1] + right_w[1]) / 2))

        if world_landmarks is not None:
            try:
                self.rotation_angles_world.append(compute_rotation_angle_world(world_landmarks))
            except Exception:
                pass

        self.last_center = _skeleton_center(landmarks)
        self.size_sum += _skeleton_size(landmarks)
        self.frames_seen += 1

    @property
    def avg_size(self) -> float:
        """Средний размер скелета за всё время наблюдения."""
        return self.size_sum / max(self.frames_seen, 1)


def track_athletes(
    frames_with_poses,
    fps: float,
) -> list:
    """
    Строит треки гимнасток по последовательности кадров.

    frames_with_poses: список кортежей (frame_bgr, pose_landmarks, world_landmarks).
    Возвращает список AthleteTrack.
    """
    tracks: list = []

    for frame, pose_landmarks_list, world_landmarks_list in frames_with_poses:
        if not pose_landmarks_list:
            continue

        used_tracks = set()
        for i, landmarks in enumerate(pose_landmarks_list):
            center = _skeleton_center(landmarks)
            world = None
            if world_landmarks_list and i < len(world_landmarks_list):
                world = world_landmarks_list[i]

            # Ищем ближайший трек, у которого есть last_center
            best_track = None
            best_dist = MAX_ATHLETE_MATCH_DIST  # порог
            for t in tracks:
                if id(t) in used_tracks or t.last_center is None:
                    continue
                d = (center[0] - t.last_center[0]) ** 2 + (center[1] - t.last_center[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_track = t

            if best_track is None:
                best_track = AthleteTrack(index=len(tracks))
                tracks.append(best_track)

            best_track.add_frame(landmarks, world)
            used_tracks.add(id(best_track))

    return tracks


def select_target_tracks(tracks: list, format_type: str, group_size: int = GROUP_SIZE) -> list:
    """
    Выбирает целевые треки: individual — самый крупный, group — group_size самых крупных.
    """
    ranked = sorted(tracks, key=lambda t: t.avg_size, reverse=True)
    if format_type == "group":
        return ranked[:group_size]
    return ranked[:1]


def compute_deep_metrics(
    torso_angles: list,
    wrist_heights: list,
    com_positions: list,
    fps: float,
    knee_angles_left: list = None,
    knee_angles_right: list = None,
    leg_opening_angles: list = None,
    hip_heights: list = None,
    signed_torso_angles: list = None,
    ankle_heights: list = None,
    ankle_y_left: list = None,
    ankle_y_right: list = None,
    arm_symmetries: list = None,
    rotation_angles_world: list = None,
    wrist_y_left: list = None,
    wrist_y_right: list = None,
) -> dict:
    """Агрегирует метрики углублённого анализа движений."""
    result = {}

    # 1. Угол наклона корпуса при кувырке
    if torso_angles:
        arr = np.array(torso_angles)
        result["torso_tilt"] = {
            "avg_deg": round(float(np.mean(arr)), 1),
            "max_deg": round(float(np.max(arr)), 1),
            "min_deg": round(float(np.min(arr)), 1),
        }
    else:
        result["torso_tilt"] = None

    # 2. Высота рук при подбросе булав
    if wrist_heights:
        arr = np.array(wrist_heights)
        highest_y = float(np.min(arr))          # самая верхняя точка запястий
        wrist_metrics = {
            "avg_y": round(float(np.mean(arr)), 3),
            "highest_y": round(highest_y, 3),
            # Доля высоты кадра, на которую поднимались руки (0..1)
            "reach_ratio": round(max(0.0, 1.0 - highest_y), 3),
        }
        # Высоты каждой руки отдельно: усреднение левого/правого запястья
        # «размывает» асимметрию, поэтому досягаемость проверяется и по каждой руке
        for side, side_arr in (("left", wrist_y_left), ("right", wrist_y_right)):
            if side_arr:
                s = np.array(side_arr)
                side_highest_y = float(np.min(s))
                wrist_metrics[f"{side}_highest_y"] = round(side_highest_y, 3)
                wrist_metrics[f"{side}_reach_ratio"] = round(max(0.0, 1.0 - side_highest_y), 3)
        # Самая низкая точка запястий за всё видео: признак «руки опущены на пол»
        # (переворот вперёд — постановка рук на опору перед проходом)
        wrist_metrics["lowest_y"] = round(float(np.max(arr)), 3)
        result["wrist_height"] = wrist_metrics
    else:
        result["wrist_height"] = None

    # 3. Траектория движения (центр масс)
    if len(com_positions) > 1:
        com = np.array(com_positions)
        diffs = np.diff(com, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        result["trajectory"] = {
            "total_distance": round(float(np.sum(distances)), 4),
            "x_range": round(float(np.ptp(com[:, 0])), 4),
            "y_range": round(float(np.ptp(com[:, 1])), 4),
        }
    else:
        result["trajectory"] = None

    # 4. Скорость и плавность движения
    if len(com_positions) > 2:
        com = np.array(com_positions)
        diffs = np.diff(com, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        speeds = distances * fps  # нормализованные единицы в секунду

        accelerations = np.diff(speeds)
        # Количество «рывков» — смен знака ускорения
        jerk_count = int(np.sum(np.abs(np.diff(np.sign(accelerations))) > 0))
        # Плавность: 1 при нулевом разбросе ускорений, стремится к 0 при рывках
        smoothness = round(1.0 / (1.0 + float(np.std(accelerations))), 4)

        result["motion"] = {
            "avg_speed": round(float(np.mean(speeds)), 4),
            "max_speed": round(float(np.max(speeds)), 4),
            "smoothness": smoothness,
            "jerk_count": jerk_count,
        }
    else:
        result["motion"] = None

    # 5. Признаки, специфичные для упражнений (собираются только при наличии данных)
    # Равновесие: угол в колене опорной ноги (чем ближе к 180°, тем выпрямленнее нога)
    if knee_angles_left or knee_angles_right:
        knee = {}
        if knee_angles_left:
            arr_l = np.array(knee_angles_left)
            knee["left_avg_deg"] = round(float(np.mean(arr_l)), 1)
            knee["left_min_deg"] = round(float(np.min(arr_l)), 1)
        if knee_angles_right:
            arr_r = np.array(knee_angles_right)
            knee["right_avg_deg"] = round(float(np.mean(arr_r)), 1)
            knee["right_min_deg"] = round(float(np.min(arr_r)), 1)
        result["knee_angle"] = knee

    # Прыжок: угол раскрытия ног (шпагат) — чем ближе к 180°, тем шире
    if leg_opening_angles:
        arr = np.array(leg_opening_angles)
        result["leg_opening"] = {
            "max_deg": round(float(np.max(arr)), 1),
            "avg_deg": round(float(np.mean(arr)), 1),
        }

    # Прыжок/кувырок: высота таза (для оценки вертикального отрыва и группировки)
    if hip_heights:
        arr = np.array(hip_heights)
        result["hip_height"] = {
            "min_y": round(float(np.min(arr)), 3),  # самая высокая точка таза (прыжок)
            "max_y": round(float(np.max(arr)), 3),  # самая низкая точка таза (кувырок)
            "avg_y": round(float(np.mean(arr)), 3),
        }

    # Пируэт: наклон корпуса со знаком (симметрия оси вращения)
    if signed_torso_angles:
        arr = np.array(signed_torso_angles)
        result["torso_tilt_signed"] = {
            "avg_deg": round(float(np.mean(arr)), 1),
            "abs_max_deg": round(float(np.max(np.abs(arr))), 1),
        }

    # Прыжок: высота стоп при отрыве
    if ankle_heights:
        arr = np.array(ankle_heights)
        result["ankle_height"] = {
            "min_y": round(float(np.min(arr)), 3),  # самая высокая точка стоп
            "avg_y": round(float(np.mean(arr)), 3),
        }

    # Равновесие: высоты лодыжек (для определения опорной ноги)
    if ankle_y_left and ankle_y_right:
        result["ankle_y"] = {
            "left_avg_y": round(float(np.mean(ankle_y_left)), 3),
            "right_avg_y": round(float(np.mean(ankle_y_right)), 3),
            # Средняя разница высот: > 0 => левая выше (правая опорная), < 0 => наоборот
            "avg_diff": round(float(np.mean(np.array(ankle_y_left) - np.array(ankle_y_right))), 3),
        }
    elif ankle_y_left:
        result["ankle_y"] = {
            "left_avg_y": round(float(np.mean(ankle_y_left)), 3),
            "right_avg_y": None,
            "avg_diff": None,
        }
    elif ankle_y_right:
        result["ankle_y"] = {
            "left_avg_y": None,
            "right_avg_y": round(float(np.mean(ankle_y_right)), 3),
            "avg_diff": None,
        }

    # Пируэт: асимметрия рук
    if arm_symmetries:
        arr = np.array(arm_symmetries)
        result["arm_symmetry"] = {
            "avg_diff": round(float(np.mean(arr)), 3),
            "max_diff": round(float(np.max(arr)), 3),
        }

    # Пируэт: число витков по углу поворота плечевой линии
    if rotation_angles_world:
        angles = np.unwrap(np.radians(np.array(rotation_angles_world)))
        total_rotation_deg = float(np.degrees(angles[-1] - angles[0]))
        result["rotation_turns"] = {
            "total_deg": round(total_rotation_deg, 1),
            "turns": round(abs(total_rotation_deg) / 360.0, 2),
        }

    return result


# --- Детекция ошибок и генерация рекомендаций ---

# Эвристические пороги для оценки техники упражнений.
# Значения подобраны на основе типовых ошибок начинающих гимнасток
# и будут уточняться по мере накопления данных.
THRESHOLDS = {
    "risk-buhavy": {
        "min_tuck_tilt_deg": 60.0,   # минимальный макс. угол наклона при кувырке
        "min_reach_ratio": 0.30,     # минимальная высота подброса булав (0..1)
        "min_hip_low_y": 0.55,       # минимальный нижний уровень таза при кувырке (0..1, чем больше — тем ниже)
        "max_club_height_diff": 0.08,  # допустимая разница высот броска двух булав (0..1)
        "max_peak_frame_diff": 4,      # допустимая разница кадров достижения пика (одновременность)
        "max_catch_x_deviation": 0.20, # допустимое отклонение точки ловли от центра (по x)
        "max_jerk_count": 15,        # допустимое число рывков при ловле
    },
    "jump": {
        "min_jump_y_range": 0.15,    # минимальный вертикальный размах центра масс
        "min_backbend_tilt_deg": 35.0,  # минимальный макс. наклон корпуса при прогибе
        "min_leg_opening_deg": 90.0, # минимальный угол раскрытия ног (шпагат)
        "max_hip_lift_y": 0.45,      # максимальная верхняя точка таза при прыжке (0..1, чем меньше — тем выше)
        "max_ankle_lift_y": 0.80,    # максимальная высота стоп при отрыве (0..1, чем меньше — тем выше)
        "max_jerk_count": 15,        # допустимое число рывков при приземлении
    },
    "balance": {
        "max_com_x_range": 0.10,     # допустимый горизонтальный дрейф центра масс
        "max_avg_tilt_deg": 10.0,    # допустимый средний наклон корпуса
        "min_knee_extension_deg": 150.0,  # минимальный угол в колене опорной ноги (чем ближе к 180°, тем прямее)
        "min_ankle_height_diff": 0.08,  # минимальная разница высот стоп для определения опорной ноги
    },
    "rotation": {
        "max_avg_tilt_deg": 12.0,    # допустимый средний наклон оси вращения
        "max_com_x_range": 0.12,     # допустимое смещение оси вращения
        "max_signed_tilt_deg": 10.0, # допустимый систематический завал оси в одну сторону
        "max_arm_symmetry": 0.10,    # допустимая разница высот запястий (0..1)
        "min_turns": 1.0,            # ожидаемое минимальное число витков
        "max_jerk_count": 20,        # допустимое число рывков при вращении
    },
    "walkover-front": {
        "min_pass_tilt_deg": 150.0,    # полный проход через стойку на руках (макс. наклон корпуса)
        "min_split_deg": 150.0,        # раскрытие ног (шпагат) при проходе над головой
        "min_leg_straight_deg": 150.0, # прямые ноги при проходе (мин. угол в колене)
        "min_hands_down_y": 0.70,      # руки должны опуститься до пола (0..1, чем больше — тем ниже)
        "max_jerk_count": 15,          # допустимое число рывков при выходе из переворота
    },
}

# Штрафы за ошибки (используются для расчёта общей оценки)
SEVERITY_PENALTY = {
    "critical": 25,
    "warning": 10,
}

# Корректирующие упражнения: подсказка по ключевой фразе ошибки -> предлагаемое упражнение.
# Тренер может дополнить/уточнить формулировки — они выводятся в PDF-карточке.
SUGGESTED_DRILLS = [
    ("кувырок", "Перекаты в группировке вперёд-назад, 3×8 повторений"),
    ("подброс", "Подброс булав от плеча у стены с контролем высоты, 3×10"),
    ("прыжок слишком низкий", "Прыжки из полуприседа с выпрыгиванием вверх, 3×10"),
    ("прогиб", "Мостик из положения лёжа и «корзиночка», 2×10"),
    ("раскрыт", "Шпагаты и складки у опоры, ежедневная растяжка 10 мин"),
    ("вертикальный отрыв", "Прыжки на скакалке с высокой голенью, 3×30 сек"),
    ("стопы", "Подъёмы на носки и прыжки «стрелочка» с вытягиванием носков, 3×12"),
    ("раскачивание", "Стойка на одной ноге с закрытыми глазами у опоры, 3×30 сек"),
    ("опорн", "Подъёмы на полупальцы одной ноги у опоры, 3×12 на каждую"),
    ("ось вращения", "Повороты на 180°/360° с фиксацией позы, 3×8"),
    ("витков", "Пируэты с партнёром-страховкой, постепенно наращивая число витков"),
    ("асимметричн", "Отработка позиций рук перед зеркалом, 2×10"),
    ("одновременно", "Синхронный подброс двух булав под счёт, 3×10"),
    ("разной высоте", "Подброс одной булавы каждой рукой поочерёдно на одинаковую высоту, 3×10"),
    ("смещена от центра", "Ловля булав с фиксацией рук на оси тела перед зеркалом, 2×10"),
    ("жёсткая", "Жонглирование/ловля с мягким «доигрыванием» кистью, 3×5 мин"),
    ("стойку на руках", "Стойка на руках у стены с выходом в вертикаль, 4×15 сек"),
    ("не опускаются до пола", "Наклоны вперёд с постановкой рук на пол и переносом веса на руки, 3×8"),
    ("согнуты в коленях", "Махи ногами с выпрямленными коленями и растяжка под шпагат, 3×10"),
    ("резкий", "Плавные приземления из стойки на руках с амортизацией в колене, 3×8"),
]


def _suggested_drill(message: str):
    """Подбирает корректирующее упражнение по тексту ошибки."""
    msg = (message or "").lower()
    for key, drill in SUGGESTED_DRILLS:
        if key in msg:
            return drill
    return None


def _fmt(value, suffix=""):
    """Форматирует число для текста рекомендации."""
    if value is None:
        return "—"
    return f"{value}{suffix}"


def detect_errors_and_recommendations(exercise: str, movement: dict, clubs: dict = None) -> dict:
    """
    Детектирует ошибки техники и формирует рекомендации для конкретного
    упражнения на основе агрегированных метрик движения.

    clubs: данные трекинга булав (результат analyze_club_flight) или None.

    Возвращает:
    {
        "overall_score": 0..100,
        "errors": [{"severity": "critical"|"warning", "message": "..."}],
        "recommendations": ["..."],
        "summary": "..."
    }
    """
    errors = []
    recommendations = []

    if exercise not in THRESHOLDS:
        return {
            "overall_score": None,
            "errors": [],
            "recommendations": [
                "Упражнение не распознано. Выберите упражнение из списка и повторите анализ."
            ],
            "summary": "Не удалось подобрать правила проверки для выбранного упражнения.",
        }

    thresholds = THRESHOLDS[exercise]

    torso = movement.get("torso_tilt") or {}
    wrist = movement.get("wrist_height") or {}
    trajectory = movement.get("trajectory") or {}
    motion = movement.get("motion") or {}
    knee = movement.get("knee_angle") or {}
    leg_opening = movement.get("leg_opening") or {}
    hip_height = movement.get("hip_height") or {}
    torso_signed = movement.get("torso_tilt_signed") or {}
    ankle_height = movement.get("ankle_height") or {}
    ankle_y = movement.get("ankle_y") or {}
    arm_symmetry = movement.get("arm_symmetry") or {}
    rotation_turns = movement.get("rotation_turns") or {}

    # Если поза почти не была обнаружена — данных для проверки недостаточно
    if not torso and not wrist and not trajectory:
        return {
            "overall_score": None,
            "errors": [
                {"severity": "critical", "message": "Недостаточно данных о позе: человек не был стабильно обнаружен в кадре."}
            ],
            "recommendations": [
                "Снимите видео при хорошем освещении, в полный рост, без перекрытий.",
                "Убедитесь, что тело целиком попадает в кадр на протяжении всего упражнения.",
            ],
            "summary": "Анализ невозможен из-за отсутствия данных о позе.",
        }

    # --- Риск: ловля двух булав после кувырка ---
    if exercise == "risk-buhavy":
        max_tilt = torso.get("max_deg")
        if max_tilt is not None and max_tilt < thresholds["min_tuck_tilt_deg"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Кувырок выполнен недостаточно глубоко: максимальный наклон корпуса "
                    f"{_fmt(max_tilt, '°')} (нужно не менее {thresholds['min_tuck_tilt_deg']:.0f}°). "
                    "Булавы не успеют перелететь над кувырком."
                ),
            })
            recommendations.append(
                "Сильнее группируйтесь и активнее перекатывайтесь через плечо, чтобы корпус "
                "проходил под траекторией полёта булав."
            )

        reach_ratio = wrist.get("reach_ratio")
        if reach_ratio is not None and reach_ratio < thresholds["min_reach_ratio"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Подброс булав недостаточно высокий: руки поднялись лишь на "
                    f"{_fmt(round(reach_ratio * 100), '%')} высоты кадра "
                    f"(нужно не менее {thresholds['min_reach_ratio'] * 100:.0f}%)."
                ),
            })
            recommendations.append(
                "Подбрасывайте булавы энергичным движением от плеча, направляя их вверх, "
                "и добавляйте небольшой подъём на носки для увеличения высоты полёта."
            )

        # Асимметричный подброс: усреднение запястий может маскировать низкий
        # подброс одной руки, поэтому проверяем досягаемость каждой руки отдельно
        side_reaches = [
            (label, wrist.get(f"{side}_reach_ratio"))
            for side, label in (("left", "левой"), ("right", "правой"))
        ]
        for side_label, side_reach in side_reaches:
            if side_reach is not None and side_reach < thresholds["min_reach_ratio"]:
                errors.append({
                    "severity": "critical",
                    "message": (
                        f"Подброс недостаточно высокий для {side_label} руки: досягаемость "
                        f"{_fmt(round(side_reach * 100), '%')} высоты кадра "
                        f"(нужно не менее {thresholds['min_reach_ratio'] * 100:.0f}%). "
                        "Булава со стороны этой руки летит ниже."
                    ),
                })
                recommendations.append(
                    f"Поднимайте {side_label} руку так же высоко, как противоположную, "
                    "чтобы бросок был симметричным и булавы летели на одной высоте."
                )

        # Кувырок: низкая группировка — таз должен опускаться достаточно низко
        hip_max_y = hip_height.get("max_y")
        if hip_max_y is not None and hip_max_y < thresholds["min_hip_low_y"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Недостаточная группировка при кувырке: таз опускается только до "
                    f"{_fmt(hip_max_y)} высоты кадра (нужно ниже {thresholds['min_hip_low_y']}). "
                    "Корпус проходит слишком высоко над полом."
                ),
            })
            recommendations.append(
                "Сильнее группируйтесь: подтягивайте колени к груди и ниже опускайте таз, "
                "чтобы корпус проходил под булавами."
            )

        # --- Трекинг булав: параметры полёта предметов ---
        clubs_info = clubs or {}
        throw_height = clubs_info.get("throw_height") or {}
        symmetry = clubs_info.get("symmetry") or {}
        simultaneity = clubs_info.get("simultaneity") or {}
        catch_position = clubs_info.get("catch_position") or {}

        if clubs_info.get("clubs_detected", 0) >= 2:
            # Высота броска (по пикам траекторий)
            max_height_y = throw_height.get("max")
            if max_height_y is not None and max_height_y > thresholds["min_reach_ratio"]:
                errors.append({
                    "severity": "warning",
                    "message": (
                        f"Бросок булав недостаточно высокий: максимальная точка полёта "
                        f"{_fmt(max_height_y)} высоты кадра (нужно выше "
                        f"{thresholds['min_reach_ratio']})."
                    ),
                })
                recommendations.append(
                    "Увеличивайте высоту подброса: энергичнее выталкивайте булавы вверх "
                    "от плеча и добавляйте импульс кистью."
                )

            # Симметрия высоты броска
            height_diff = symmetry.get("height_diff")
            if height_diff is not None and height_diff > thresholds["max_club_height_diff"]:
                errors.append({
                    "severity": "warning",
                    "message": (
                        f"Булавы летят на разной высоте: разница пиков "
                        f"{_fmt(height_diff)} (допустимо не более "
                        f"{thresholds['max_club_height_diff']})."
                    ),
                })
                recommendations.append(
                    "Подбрасывайте обе булавы с одинаковым усилием и на одну высоту, "
                    "контролируя симметрию движения рук."
                )

            # Одновременность
            peak_frame_diff = simultaneity.get("peak_frame_diff")
            if peak_frame_diff is not None and peak_frame_diff > thresholds["max_peak_frame_diff"]:
                errors.append({
                    "severity": "warning",
                    "message": (
                        f"Булавы подброшены не одновременно: разница достижения верхней точки "
                        f"{_fmt(peak_frame_diff, ' кадр.')} (допустимо не более "
                        f"{thresholds['max_peak_frame_diff']})."
                    ),
                })
                recommendations.append(
                    "Работайте руками синхронно: подбрасывайте булавы одним движением, "
                    "не растягивая фазу броска."
                )

            # Позиция ловли: проверяем отклонение от центра по x
            for side_key, label in [("left", "левой"), ("right", "правой")]:
                pos = catch_position.get(side_key) or {}
                catch_x = pos.get("x")
                if catch_x is not None and abs(catch_x - 0.5) > thresholds["max_catch_x_deviation"]:
                    errors.append({
                        "severity": "warning",
                        "message": (
                            f"Ловля {label} булавы смещена от центра: позиция по x "
                            f"{_fmt(round(catch_x, 2))} (центр 0.50, допустимое отклонение "
                            f"{thresholds['max_catch_x_deviation']})."
                        ),
                    })
                    recommendations.append(
                        f"Ловите {label} булаву ближе к оси тела, не уводя руки в сторону."
                    )
        elif clubs_info.get("clubs_detected") == 1:
            errors.append({
                "severity": "warning",
                "message": "Обнаружена только одна булава — оценить симметрию и одновременность броска невозможно.",
            })
            recommendations.append(
                "Убедитесь, что обе булавы хорошо видны в кадре и имеют яркую, контрастную окраску."
            )
        elif clubs is not None and clubs_info.get("clubs_detected") == 0:
            errors.append({
                "severity": "warning",
                "message": "Булавы не обнаружены в кадре — параметры полёта предметов не оценены.",
            })
            recommendations.append(
                "Снимайте при хорошем освещении; булавы должны быть яркими и не сливаться с фоном."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ловля булав жёсткая: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Встречайте булавы мягким движением кистей и амортизируйте ловлю "
                "сгибанием рук в локтях."
            )

    # --- Прыжок с прогибом назад ---
    elif exercise == "jump":
        y_range = trajectory.get("y_range")
        if y_range is not None and y_range < thresholds["min_jump_y_range"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Прыжок слишком низкий: вертикальный размах центра масс "
                    f"{_fmt(y_range)} (нужно не менее {thresholds['min_jump_y_range']}). "
                    "Прогиб не успевает раскрыться в полёте."
                ),
            })
            recommendations.append(
                "Отталкивайтесь активнее обеими ногами и вытягивайтесь вверх перед прогибом, "
                "чтобы увеличить время полёта."
            )

        max_tilt = torso.get("max_deg")
        if max_tilt is not None and max_tilt < thresholds["min_backbend_tilt_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Недостаточный прогиб назад: максимальный наклон корпуса "
                    f"{_fmt(max_tilt, '°')} (нужно не менее {thresholds['min_backbend_tilt_deg']:.0f}°)."
                ),
            })
            recommendations.append(
                "В верхней точке прыжка активнее прогибайтесь в грудном отделе и тяните "
                "руки назад, раскрывая плечи."
            )

        # Прыжок: раскрытие ног (шпагат) — угол должен быть широким
        opening_max = leg_opening.get("max_deg")
        if opening_max is not None and opening_max < thresholds["min_leg_opening_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ноги раскрыты недостаточно широко: максимальный угол раскрытия "
                    f"{_fmt(opening_max, '°')} (нужно не менее {thresholds['min_leg_opening_deg']:.0f}°)."
                ),
            })
            recommendations.append(
                "В полёте активнее разводите ноги в стороны, тяните носки, "
                "чтобы раскрыть шпагат и увеличить амплитуду прыжка."
            )

        # Прыжок: вертикальный отрыв — таз должен подниматься достаточно высоко
        hip_min_y = hip_height.get("min_y")
        if hip_min_y is not None and hip_min_y > thresholds["max_hip_lift_y"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Недостаточный вертикальный отрыв: таз поднимается только до "
                    f"{_fmt(hip_min_y)} высоты кадра (нужно выше {thresholds['max_hip_lift_y']}). "
                    "Прыжок получается «плоским»."
                ),
            })
            recommendations.append(
                "Добавьте вертикальный импульс: активнее выталкивайтесь стопами "
                "и тянитесь макушкой вверх в момент отрыва."
            )

        # Прыжок: высота отрыва стоп — стопы должны подниматься от пола
        ankle_min_y = ankle_height.get("min_y")
        if ankle_min_y is not None and ankle_min_y > thresholds["max_ankle_lift_y"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Стопы почти не отрываются от пола: максимальная высота стоп "
                    f"{_fmt(ankle_min_y)} (нужно выше {thresholds['max_ankle_lift_y']}). "
                    "Отрыв недостаточно высокий."
                ),
            })
            recommendations.append(
                "Выталкивайтесь активнее носками, поднимайте стопы от пола "
                "и вытягивайте носки в полёте."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Приземление жёсткое: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Приземляйтесь мягко: с носка на полную стопу, слегка сгибая колени "
                "и сохраняя баланс."
            )

    # --- Равновесие на одной ноге ---
    elif exercise == "balance":
        x_range = trajectory.get("x_range")
        if x_range is not None and x_range > thresholds["max_com_x_range"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Раскачивание корпуса: центр масс смещается по горизонтали на "
                    f"{_fmt(x_range)} (допустимо не более {thresholds['max_com_x_range']}). "
                    "Равновесие теряется."
                ),
            })
            recommendations.append(
                "Зафиксируйте взгляд в одной точке, напрягите мышцы пресса и удерживайте "
                "опорную ногу неподвижной."
            )

        avg_tilt = torso.get("avg_deg")
        if avg_tilt is not None and avg_tilt > thresholds["max_avg_tilt_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Корпус наклонён: средний наклон {_fmt(avg_tilt, '°')} "
                    f"(допустимо не более {thresholds['max_avg_tilt_deg']:.0f}°). "
                    "Поза выглядит неустойчивой."
                ),
            })
            recommendations.append(
                "Держите спину вертикально, подтяните живот и не опускайте плечи."
            )

        # Равновесие: выпрямленность опорной ноги — угол в колене должен быть близок к 180°
        left_avg = knee.get("left_avg_deg")
        right_avg = knee.get("right_avg_deg")
        # Определяем опорную ногу по высоте лодыжек: опорная стопа обычно ниже (стоит на полу),
        # свободная — поднята. Используем среднюю разницу высот (left - right).
        # y растёт вниз: если left_y - right_y > 0, левая лодыжка НИЖЕ -> опорная левая.
        ankle_diff = ankle_y.get("avg_diff")

        support_side = None
        if ankle_diff is not None:
            if ankle_diff > thresholds["min_ankle_height_diff"]:
                support_side = "left"
            elif ankle_diff < -thresholds["min_ankle_height_diff"]:
                support_side = "right"
        # Запасной вариант: если разница высот мала, используем более выпрямленное колено
        if support_side is None:
            if left_avg is not None and right_avg is not None:
                support_side = "left" if left_avg >= right_avg else "right"
            elif left_avg is not None:
                support_side = "left"
            elif right_avg is not None:
                support_side = "right"

        support_avg = None
        support_label = None
        if support_side == "left":
            support_avg = left_avg
            support_label = "левая"
        elif support_side == "right":
            support_avg = right_avg
            support_label = "правая"

        if support_avg is not None and support_avg < thresholds["min_knee_extension_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Опорная нога ({support_label}) согнута: угол в колене "
                    f"{_fmt(support_avg, '°')} (нужно не менее {thresholds['min_knee_extension_deg']:.0f}°). "
                    "Согнутая нога делает равновесие менее устойчивым."
                ),
            })
            recommendations.append(
                f"Выпрямляйте опорную ({support_label}) ногу в колене, тяните коленную "
                "чашечку вверх, не заваливайтесь на пятку."
            )

    # --- Поворот (пируэт) ---
    elif exercise == "rotation":
        avg_tilt = torso.get("avg_deg")
        if avg_tilt is not None and avg_tilt > thresholds["max_avg_tilt_deg"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Ось вращения завалена: средний наклон корпуса {_fmt(avg_tilt, '°')} "
                    f"(допустимо не более {thresholds['max_avg_tilt_deg']:.0f}°). "
                    "Вращение получается неустойчивым."
                ),
            })
            recommendations.append(
                "Держите ось вращения строго вертикальной: тянитесь макушкой вверх "
                "и держите корпус собранным."
            )

        x_range = trajectory.get("x_range")
        if x_range is not None and x_range > thresholds["max_com_x_range"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Центр масс смещается по горизонтали на {_fmt(x_range)} "
                    f"(допустимо не более {thresholds['max_com_x_range']}). "
                    "Теряется ось вращения."
                ),
            })
            recommendations.append(
                "Перед началом поворота перенесите вес точно на опорную ногу "
                "и сохраняйте проекцию центра масс над стопой."
            )

        # Пируэт: систематический завал оси в одну сторону
        signed_avg = torso_signed.get("avg_deg")
        if signed_avg is not None and abs(signed_avg) > thresholds["max_signed_tilt_deg"]:
            direction = "влево" if signed_avg > 0 else "вправо"
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ось вращения систематически завалена {direction}: средний наклон "
                    f"{_fmt(signed_avg, '°')} (допустимо не более "
                    f"{thresholds['max_signed_tilt_deg']:.0f}° в любую сторону). "
                    "Вращение уходит в одну сторону."
                ),
            })
            recommendations.append(
                "Следите за симметрией: держите плечи ровно, не наклоняйте корпус "
                f"{direction}, тянитесь строго вверх."
            )

        # Пируэт: число витков (по углу поворота плечевой линии)
        turns = rotation_turns.get("turns")
        total_deg = rotation_turns.get("total_deg")
        if turns is not None and turns < thresholds["min_turns"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Мало витков в пируэте: выполнен поворот на {_fmt(total_deg, '°')} "
                    f"(примерно {_fmt(turns, ' об.')}, нужно не менее {thresholds['min_turns']:.0f})."
                ),
            })
            recommendations.append(
                "Увеличивайте темп вращения: активнее работайте руками и не замедляйтесь "
                "между витками."
            )

        # Пируэт: симметрия рук — запястья должны быть на примерно одинаковой высоте
        arm_diff = arm_symmetry.get("avg_diff")
        if arm_diff is not None and arm_diff > thresholds["max_arm_symmetry"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Руки асимметричны: разница высот запястий {_fmt(arm_diff)} "
                    f"(допустимо не более {thresholds['max_arm_symmetry']}). "
                    "Это нарушает баланс и ось вращения."
                ),
            })
            recommendations.append(
                "Держите руки на одной высоте, симметрично относительно оси вращения, "
                "не опускайте одну из рук."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Вращение неравномерное: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Работайте руками равномерно, не «замирайте» между витками — "
                "вращение должно быть непрерывным."
            )

    # --- Переворот вперёд ---
    elif exercise == "walkover-front":
        # Проход через стойку на руках: корпус должен пройти через вертикаль
        max_tilt = torso.get("max_deg")
        if max_tilt is not None and max_tilt < thresholds["min_pass_tilt_deg"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Проход через стойку на руках недостаточно вертикальный: максимальный "
                    f"наклон корпуса {_fmt(max_tilt, '°')} (нужно не менее "
                    f"{thresholds['min_pass_tilt_deg']:.0f}°). Переворот выполнен «низко»."
                ),
            })
            recommendations.append(
                "Отрабатывайте стойку на руках у стены и заход в переворот махом, "
                "чтобы корпус проходил через вертикаль."
            )

        # Руки должны опуститься до пола: опора перед проходом
        lowest_y = wrist.get("lowest_y")
        if lowest_y is not None and lowest_y < thresholds["min_hands_down_y"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Руки не опускаются до пола: самая низкая точка запястий "
                    f"{_fmt(lowest_y)} высоты кадра (нужно не менее "
                    f"{thresholds['min_hands_down_y']}). Опора под руки недостаточно глубокая."
                ),
            })
            recommendations.append(
                "Наклоняйтесь глубже вперёд и ставьте руки на пол ближе к линии ног, "
                "чтобы перевернуться через руки, а не «провисать» в наклоне."
            )

        # Раскрытие ног (шпагат) при проходе над головой
        opening_max = leg_opening.get("max_deg")
        if opening_max is not None and opening_max < thresholds["min_split_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ноги раскрыты недостаточно широко: максимальный угол раскрытия "
                    f"{_fmt(opening_max, '°')} (нужно не менее "
                    f"{thresholds['min_split_deg']:.0f}°). Проход над головой «согнутый»."
                ),
            })
            recommendations.append(
                "В перевороте тяните ноги в шпагат: активнее разводите их в стороны "
                "и тяните носки, пока корпус идёт через стойку."
            )

        # Прямые ноги: минимальный угол в колене за всё видео
        knee_mins = [
            v for v in (knee.get("left_min_deg"), knee.get("right_min_deg"))
            if v is not None
        ]
        min_knee = min(knee_mins) if knee_mins else None
        if min_knee is not None and min_knee < thresholds["min_leg_straight_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ноги согнуты в коленях при проходе: минимальный угол в колене "
                    f"{_fmt(min_knee, '°')} (нужно не менее "
                    f"{thresholds['min_leg_straight_deg']:.0f}°)."
                ),
            })
            recommendations.append(
                "Держите ноги прямыми на протяжении всего переворота: тяните колени "
                "и носки, не сгибайте колени при махе."
            )

        # Плавность выхода из переворота
        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Выход из переворота резкий: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Плавно завершайте переворот: приземляйтесь на маховую ногу "
                "с амортизацией в колене и сразу поднимайтесь в стойку."
            )

        # Подводящие упражнения для переворота вперёд: выводятся всегда,
        # независимо от найденных ошибок (и как развитие элемента в идеальном случае)
        recommendations.extend([
            "Колесо через одну руку с постепенным уменьшением опоры — база для захода "
            "в переворот вперёд.",
            "Перекат вперёд через стойку у стены с мягким приземлением — отработка фазы прохода.",
            "Мостик и растяжка плечевых суставов — подвижность плеч для глубокой постановки рук.",
        ])

    # --- Общая оценка и итоговая рекомендация ---
    if not errors:
        overall_score = 100
        summary = "Отличная техника! Ошибок не обнаружено, продолжайте в том же духе."
        recommendations.insert(
            0,
            "Отличная техника! Для закрепления результата повторите упражнение "
            "в медленном темпе и под контролем тренера.",
        )
    else:
        overall_score = max(
            0,
            100 - sum(SEVERITY_PENALTY[e["severity"]] for e in errors),
        )
        if overall_score >= 80:
            summary = "Техника близка к идеалу: есть небольшие недочёты, которые легко исправить."
        elif overall_score >= 50:
            summary = "Упражнение выполняется, но есть заметные ошибки — обратите внимание на рекомендации."
        else:
            summary = "Техника требует серьёзной доработки. Разберите упражнение по элементам с тренером."

    # Подбираем корректирующее упражнение для каждой ошибки (используется в PDF-отчёте)
    for e in errors:
        e["suggested_drill"] = _suggested_drill(e.get("message"))

    return {
        "overall_score": overall_score,
        "errors": errors,
        "recommendations": recommendations,
        "summary": summary,
    }


def _capture_error_frames(
    video_path: str,
    fps: float,
    frames_with_poses: list,
    feedback: dict,
    frames_dir: str,
) -> dict:
    """
    Снимает скриншоты моментов ошибок из видео.

    Для каждой ошибки feedback выбирается кадр из числа анализированных:
    для ошибок про наклон/кувырок — кадр с максимальным наклоном корпуса,
    для остальных — равномерно распределённые кадры. На PNG добавляется
    обводка скелета (координаты landmark'ов сохраняются в результат).

    Возвращает {index: {"path", "timestamp", "landmarks"}}.
    """
    result = {}
    errors = (feedback or {}).get("errors") or []
    if not errors or not frames_with_poses:
        return result

    os.makedirs(frames_dir, exist_ok=True)

    # Кадры с одной (целевой) позой — самая крупная гимнастка
    target_frames = []
    for idx, (frame, pose_list, _) in enumerate(frames_with_poses):
        if not pose_list:
            continue
        best = max(pose_list, key=_skeleton_size)
        target_frames.append((idx, frame, best))

    if not target_frames:
        return result

    # Кадр с максимальным наклоном корпуса (момент кувырка/прогиба)
    peak_idx = max(
        range(len(target_frames)),
        key=lambda i: compute_torso_tilt(target_frames[i][2]),
        default=None,
    )

    total = len(errors)
    for i, _err in enumerate(errors):
        # Первая (самая серьёзная) ошибка — на пиковом кадре,
        # остальные — равномерно по видео
        if peak_idx is not None and i == 0:
            pick = peak_idx
        else:
            pick = int(len(target_frames) * (i + 1) / (total + 1)) % len(target_frames)
        idx, frame, landmarks = target_frames[pick]

        timestamp = round(idx * (1.0 / max(fps, 1e-6)), 2)
        frame_png = os.path.join(frames_dir, f"error_{i}.png")
        cv2.imwrite(frame_png, frame)

        result[i] = {
            "path": frame_png,
            "timestamp": timestamp,
            "landmarks": {
                j: {"x": float(lm.x), "y": float(lm.y)}
                for j, lm in enumerate(landmarks)
                if getattr(lm, "visibility", 1.0) > 0.5
            },
        }
    return result


def _build_heatmap(frames_with_poses, heatmap_path: str, width_cells: int = 20, height_cells: int = 12):
    """
    Строит тепловую карту позиций гимнасток по площадке (групповой формат).

    Собирает центры скелетов всех гимнасток по кадрам, агрегирует в сетку
    width_cells × height_cells и сохраняет PNG. Возвращает stats или None.
    """
    if not frames_with_poses:
        return None

    centers = []
    for _, pose_list, _ in frames_with_poses:
        for landmarks in pose_list:
            centers.append(_skeleton_center(landmarks))

    if not centers:
        return None

    grid = np.zeros((height_cells, width_cells), dtype=float)
    for cx, cy in centers:
        col = min(width_cells - 1, max(0, int(cx * width_cells)))
        row = min(height_cells - 1, max(0, int(cy * height_cells)))
        grid[row, col] += 1

    total = float(grid.sum())
    if total <= 0:
        return None

    # Доля попаданий в «плотные» ячейки, где одновременно сидят >= 4 гимнастки
    dense_cells = grid >= 4
    overlap_ratio = float(grid[dense_cells].sum() / total) if dense_cells.any() else 0.0
    stats = {
        "peak_density": int(grid.max()),
        "overlap_ratio": round(overlap_ratio, 3),
    }

    # Рисуем тепловую карту: JET-палитра, пустые ячейки — тёмный фон.
    # Маска накладывается ДО resize, иначе размеры массива не совпадут.
    normalized = (grid / grid.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored[normalized == 0] = (30, 30, 30)
    colored = cv2.resize(colored, (480, 270), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(heatmap_path, colored)
    return stats


# --- Командная обратная связь (групповой формат) ---

# Порог скорости центра масс (нормализованных ед./кадр), ниже которого гимнастка
# считается неподвижной («пауза»)
PAUSE_SPEED_THRESHOLD = 0.01
# Минимальная длительность паузы в кадрах анализа
PAUSE_MIN_FRAMES = 2
# Допустимый рассинхрон старта движения внутри команды (сек)
TEAM_START_SYNC_TOLERANCE_S = 0.3
# Порог «заметного движения» для определения старта (нормализованных ед./кадр)
MOVEMENT_START_SPEED = 0.02


def _pause_frames(com_positions: list) -> int:
    """Число подряд идущих кадров, где гимнастка неподвижна (пауза)."""
    if len(com_positions) < 2:
        return 0
    com = np.array(com_positions)
    speeds = np.linalg.norm(np.diff(com, axis=0), axis=1)
    current = max_run = 0
    for s in speeds:
        if s < PAUSE_SPEED_THRESHOLD:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _movement_start_frame(com_positions: list) -> int:
    """Кадр (индекс в ряду анализа), где скорость впервые превысила порог движения."""
    if len(com_positions) < 2:
        return 0
    com = np.array(com_positions)
    speeds = np.linalg.norm(np.diff(com, axis=0), axis=1)
    for i, s in enumerate(speeds):
        if s >= MOVEMENT_START_SPEED:
            return i + 1
    return 0


def detect_team_feedback(target_tracks: list, fps: float, clubs_per_athlete: list) -> dict:
    """
    Командная обратная связь для группового формата: сопоставляет события
    разных гимнасток по кадрам и формирует общие для команды замечания.

    clubs_per_athlete: список результатов analyze_club_flight по гимнасткам
    (каждый может содержать поле "dropped": {"left": bool, "right": bool}).

    Возвращает:
    {
        "messages": [{"severity": ..., "message": ...}],
        "start_spread_sec": ...,
    }
    """
    messages = []

    # --- Правило 1: рассинхрон старта движения ---
    start_frames = [
        _movement_start_frame(t.com_positions) if t.com_positions else None
        for t in target_tracks
    ]
    valid = [f for f in start_frames if f is not None]
    start_spread_sec = None
    if len(valid) >= 2:
        spread_frames = max(valid) - min(valid)
        start_spread_sec = round(spread_frames / max(fps, 1e-6), 2)
        if start_spread_sec > TEAM_START_SYNC_TOLERANCE_S:
            # Кто начинает раньше всех / позже всех (нумерация с 1 для тренера)
            early_idx = start_frames.index(min(valid))
            late_idx = start_frames.index(max(valid))
            messages.append({
                "severity": "warning",
                "message": (
                    f"Рассинхрон старта движения {start_spread_sec} с: гимнастка №{early_idx + 1} "
                    f"начинает раньше остальных, гимнастка №{late_idx + 1} — позже. "
                    "Синхронизируйте ритм разбега под общий счёт."
                ),
            })

    # --- Правило 2: потеря ритма после падения булавы ---
    # Гимнастка, у которой упала булава, и кадр падения (последний кадр траектории)
    drop_events = []  # (athlete_idx, drop_frame)
    for a_idx, clubs_info in enumerate(clubs_per_athlete or []):
        dropped = (clubs_info or {}).get("dropped") or {}
        # Траектории не сохраняются в результате analyze_club_flight,
        # поэтому кадр падения определяем приближённо: момент ловли неизвестен,
        # используем кадр пика как верхнюю границу.
        if any(v for v in dropped.values() if v is not None):
            drop_events.append(a_idx)

    for a_idx in drop_events:
        # Пауза у каждой гимнастки в целом по видео — прокси для «замерли»
        paused = [
            i for i, t in enumerate(target_tracks)
            if i != a_idx and _pause_frames(t.com_positions) >= PAUSE_MIN_FRAMES
        ]
        if len(paused) >= len(target_tracks) - 1:
            messages.append({
                "severity": "warning",
                "message": (
                    f"Команда теряет ритм после падения булавы у гимнастки №{a_idx + 1}: "
                    f"{len(paused)} из {len(target_tracks) - 1} остальных остановились. "
                    "Продолжайте комбинацию и подхватывайте элемент без общей паузы."
                ),
            })

    return {
        "messages": messages,
        "start_spread_sec": start_spread_sec,
    }


def _latency_stats(frame_latencies_ms: list) -> dict:
    """
    Статистика латентности обработки кадра (мс): средняя, p95, максимум.
    Используется для контроля требований realtime («задержка кадра < 100 мс»).
    """
    if not frame_latencies_ms:
        return None
    arr = np.array(frame_latencies_ms)
    return {
        "avg_ms": round(float(np.mean(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "samples": len(frame_latencies_ms),
    }


def _open_video_with_rotation(video_path: str):
    """
    Открывает VideoCapture и учитывает rotation metadata вертикальных видео
    (смартфонная съемка). OpenCV игнорирует метаданные поворота: кадр приходит
    «лежащим» на 90/270° и детекция позы ломается. Возвращает (cap, rotate_code),
    где rotate_code — cv2.ROTATE_90_CLOCKWISE / ROTATE_180 / ROTATE_90_COUNTERCLOCKWISE
    или None, если поворот не нужен.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    try:
        orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    except (cv2.error, TypeError, ValueError):
        orientation = 0

    rotate_code = None
    if orientation in (90, -90):
        # -90 и 90 в OpenCV трактуются одинаково как 90° по часовой;
        # -180/180 не встречаются в CAP_PROP_ORIENTATION_META (только 0/90/180/270)
        rotate_code = cv2.ROTATE_90_CLOCKWISE if orientation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
    elif orientation == 180:
        rotate_code = cv2.ROTATE_180
    return cap, rotate_code


def analyze_video(video_path: str, exercise: str = "unknown", format_type: str = "individual") -> dict:
    """
    Анализирует видео с помощью MediaPipe PoseLandmarker (Tasks API):
    определяет наличие человека/группы, собирает статистику, вычисляет
    метрики углублённого анализа движений и формирует
    рекомендации по технике для выбранного упражнения.

    format_type: "individual" — одна гимнастка (самая крупная в кадре),
                 "group" — группа из GROUP_SIZE гимнасток.
    """
    cap, rotate_code = _open_video_with_rotation(video_path)
    if cap is None:
        return {"error": "Не удалось открыть видеофайл"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Анализируем примерно 2 кадра в секунду для экономии ресурсов
    frame_interval = max(1, int(fps / 2))

    frames_analyzed = 0
    pose_detections = 0
    # Замер латентности обработки кадра (мс): для контроля требований realtime
    frame_latencies_ms = []

    # Кадры с позами всех обнаруженных людей (для построения треков гимнасток)
    frames_with_poses = []
    # Кадры для трекинга булав (анализируем те же кадры, что и позу)
    frames_for_clubs = []
    # Трекер булав: использует кастомную ONNX-модель, если она есть, иначе HSV
    club_tracker = ClubTracker(model_path=CLUB_MODEL_PATH)

    landmarker = get_pose_landmarker()
    # Смещение таймстампов: инстанс PoseLandmarker singleton, и detect_for_video
    # требует монотонно растущие таймстампы между видео; каждому видео — свой диапазон
    with _pose_lock:
        ts_offset = _pose_ts_offset[0]
        _pose_ts_offset[0] += _VIDEO_TS_OFFSET_MS

    frame_idx_holder = [0]  # мутируемый счетчик кадров (int в замыкании не мутируется)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx_holder[0] % frame_interval == 0:
            frame_start = time.perf_counter()
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_idx = frame_idx_holder[0]
            timestamp_ms = (
                ts_offset + int(frame_idx * 1000 / fps) if fps else ts_offset
            )
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Сохраняем кадр для трекинга булав (если выбрано упражнение с булавами)
            if exercise == "risk-buhavy":
                frames_for_clubs.append(frame)

            pose_list = list(results.pose_landmarks) if results.pose_landmarks else []
            world_list = list(results.world_landmarks) if results.world_landmarks else []
            if pose_list:
                pose_detections += 1
                frames_with_poses.append((frame, pose_list, world_list))

            # Латентность обработки кадра (детекция позы + булавы), мс
            frame_latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)
            frames_analyzed += 1

        frame_idx_holder[0] += 1

    cap.release()

    # Строим треки всех гимнасток и выбираем целевых
    all_tracks = track_athletes(frames_with_poses, fps)
    target_tracks = select_target_tracks(all_tracks, format_type)

    if not target_tracks:
        return {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "frames_analyzed": frames_analyzed,
            "pose_detections": pose_detections,
            "pose_presence": round(pose_detections / max(frames_analyzed, 1), 3),
            "latency": _latency_stats(frame_latencies_ms),
            "movement": None,
            "athletes": [],
            "clubs": None,
            "feedback": {
                "overall_score": None,
                "errors": [
                    {"severity": "critical", "message": "Не удалось обнаружить гимнастку в кадре."}
                ],
                "recommendations": [
                    "Убедитесь, что спортсменка полностью в кадре и хорошо освещена."
                ],
                "summary": "Анализ невозможен: человек не обнаружен.",
            },
            "format": format_type,
        }

    # Для каждой целевой гимнастки вычисляем агрегированные метрики
    athlete_results = []
    for i, track in enumerate(target_tracks):
        movement = compute_deep_metrics(
            track.torso_angles,
            track.wrist_heights,
            track.com_positions,
            fps,
            knee_angles_left=track.knee_angles_left,
            knee_angles_right=track.knee_angles_right,
            leg_opening_angles=track.leg_opening_angles,
            hip_heights=track.hip_heights,
            signed_torso_angles=track.signed_torso_angles,
            ankle_heights=track.ankle_heights,
            ankle_y_left=track.ankle_y_left,
            ankle_y_right=track.ankle_y_right,
            arm_symmetries=track.arm_symmetries,
            rotation_angles_world=track.rotation_angles_world,
            wrist_y_left=track.wrist_y_left,
            wrist_y_right=track.wrist_y_right,
        )
        athlete_results.append({
            "index": i,
            "movement": movement,
            "frames_seen": track.frames_seen,
        })

    # Трекинг булав (только для упражнения с булавами)
    clubs = None
    if exercise == "risk-buhavy" and frames_for_clubs:
        trajectories = club_tracker.track(frames_for_clubs)

        # Для индивидуального формата связываем булавы с единственной гимнасткой
        if format_type == "individual" and target_tracks:
            wrist_positions = [target_tracks[0].wrist_positions]
            assigned = assign_clubs_to_athletes(trajectories, wrist_positions)
            if assigned:
                trajectories = assigned[0]
        # Для группового формата связываем булавы с каждой гимнасткой
        elif format_type == "group" and target_tracks:
            wrist_positions = [t.wrist_positions for t in target_tracks]
            assigned_groups = assign_clubs_to_athletes(trajectories, wrist_positions)
            # Сохраняем траектории по гимнасткам для дальнейшего анализа
            clubs_per_athlete = [analyze_club_flight(group, fps) for group in assigned_groups]
            clubs = {
                "format": "group",
                "athletes": clubs_per_athlete,
                "clubs_detected_total": len(trajectories),
            }
        else:
            clubs = analyze_club_flight(trajectories, fps)
    else:
        clubs = None

    # Командная обратная связь (только групповой формат): сопоставление событий
    # гимнасток по кадрам — рассинхрон старта, потеря ритма после падения булавы
    team_feedback = None
    if format_type == "group" and len(target_tracks) >= 2:
        clubs_per_athlete_for_team = None
        if isinstance(clubs, dict) and clubs.get("format") == "group":
            clubs_per_athlete_for_team = clubs.get("athletes")
        team_feedback = detect_team_feedback(target_tracks, fps, clubs_per_athlete_for_team)

    # Детекция ошибок для первой гимнастки (для обратной совместимости ответа)
    first_movement = athlete_results[0]["movement"] if athlete_results else {}
    if isinstance(clubs, dict) and clubs.get("format") == "group":
        feedback = detect_errors_and_recommendations(exercise, first_movement, None)
    else:
        feedback = detect_errors_and_recommendations(exercise, first_movement, clubs)

    # Для группового формата — отдельный feedback для каждой гимнастки
    athletes_output = []
    for res in athlete_results:
        athletes_output.append({
            **res,
            "feedback": detect_errors_and_recommendations(exercise, res["movement"], None),
        })

    return {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "frames_analyzed": frames_analyzed,
        "pose_detections": pose_detections,
        "pose_presence": round(pose_detections / max(frames_analyzed, 1), 3),
        "latency": _latency_stats(frame_latencies_ms),
        "movement": first_movement if first_movement else None,
        "athletes": athletes_output,
        "clubs": clubs,
        "feedback": feedback,
        "team_feedback": team_feedback,
        "format": format_type,
        # Кадры с позами (numpy-массивы): используются для создания отчёта
        # (скриншоты ошибок, тепловая карта) и удаляются до сериализации ответа
        "_frames_with_poses": frames_with_poses,
    }


def create_report(
    analysis: dict,
    video_path: str,
    frames_with_poses: list,
    fps: float,
    reports_dir: str = storage.REPORTS_DIR,
) -> dict:
    """
    Сохраняет отчёт о тренировке: видео, метаданные, кадры ошибок
    и тепловую карту (для группового формата). Возвращает документ отчёта.

    Синхронная функция: выполняется в threadpool FastAPI вместе с analyze_video,
    чтобы не блокировать event loop (см. upload_video).
    """
    report_id = storage.create_report_id()
    report_dir = storage.get_report_dir(report_id, reports_dir)
    frames_dir = os.path.join(report_dir, "frames")

    analysis["report_id"] = report_id

    # Кадры моментов ошибок (для карточки PDF)
    error_frames = _capture_error_frames(
        video_path, fps, frames_with_poses,
        analysis.get("feedback") or {}, frames_dir,
    )
    analysis["error_frames"] = {
        str(i): {
            "timestamp": f.get("timestamp"),
            "landmarks": f.get("landmarks"),
        }
        for i, f in error_frames.items()
    }

    # Тепловая карта (групповой формат): где гимнастки сбиваются в кучу
    heatmap_stats = None
    if analysis.get("format") == "group":
        heatmap_path = os.path.join(frames_dir, "heatmap.png")
        heatmap_stats = _build_heatmap(frames_with_poses, heatmap_path)
        if heatmap_stats:
            analysis["heatmap"] = heatmap_path
            analysis["heatmap_stats"] = heatmap_stats

    doc = storage.save_report(report_id, analysis, video_path, reports_dir)
    if heatmap_stats:
        doc["analysis"]["heatmap"] = os.path.join(
            storage.get_report_dir(report_id, reports_dir), "frames", "heatmap.png"
        )
    doc["error_frames_full"] = error_frames
    storage.cleanup_old_reports(reports_dir)
    return doc


@app.post("/upload")
def upload_video(
    request: Request,
    video_file: UploadFile = File(...),
    exercise: str = Form("unknown"),
    format_type: str = Form("individual"),
):
    """
    Загрузка видео и синхронный анализ.

    Обычный def (не async): FastAPI выполняет эндпоинт в threadpool, поэтому
    CPU-интенсивный analyze_video не блокирует event loop и конкурентные запросы.
    """
    # 0. Авторизация и rate limiting (если настроены API_KEYS)
    _require_api_key(request)

    filename = video_file.filename or "video.mp4"

    # 1. Валидация файла (fast-path по Content-Length). Ошибка клиента ->
    # корректный HTTP-статус вместо 200 с {"error": ...} в теле.
    try:
        validate_video_file(filename, video_file.size or 0)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Сохранение файла во временную директорию (с жестким лимитом размера:
    #    video_file.size может быть неизвестен при стриминговой загрузке)
    unique_name = f"{uuid.uuid4().hex}_{os.path.basename(filename)}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        written = 0
        with open(save_path, "wb") as buffer:
            while chunk := video_file.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_FILE_SIZE_BYTES:
                    break
                buffer.write(chunk)

        if written > MAX_FILE_SIZE_BYTES:
            os.remove(save_path)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Файл слишком большой. Максимальный размер — "
                    f"{MAX_FILE_SIZE_MB} МБ."
                ),
            )

        # 3. Анализ видео
        analysis = analyze_video(save_path, exercise, format_type)

        # Логирование латентности: контроль бюджета обработки кадра
        latency = analysis.get("latency")
        if latency:
            print(
                f"[latency] file={unique_name} exercise={exercise} format={format_type} "
                f"avg={latency['avg_ms']}ms p95={latency['p95_ms']}ms "
                f"max={latency['max_ms']}ms samples={latency['samples']}"
            )

        # 4. Отчёт: видео сохраняется, кадры ошибок и тепловая карта строятся,
        #    если анализ был успешным; иначе видео просто удаляется
        report_id = None
        if not analysis.get("error") and analysis.get("feedback"):
            # Кадры ошибок снимаются с тех же кадров, что использовались в анализе
            frames_with_poses = analysis.pop("_frames_with_poses", [])
            fps = analysis.get("fps") or 30.0
            doc = create_report(analysis, save_path, frames_with_poses, fps)
            report_id = doc.get("report_id")
            analysis = doc["analysis"]
            # Абсолютная ссылка на плеер — используется в PDF для таймкодов
            analysis["video_url"] = str(request.base_url).rstrip("/") + f"/video-player/{report_id}"
            storage.update_report_analysis(report_id, analysis)
        else:
            if os.path.exists(save_path):
                os.remove(save_path)

        return {
            "message": "Видео получено",
            "file": unique_name,
            "exercise": exercise,
            "format": format_type,
            "report_id": report_id,
            "analysis": analysis,
        }
    finally:
        # Временный файл мог быть перемещён в отчёт; удаляем только если остался
        if os.path.exists(save_path):
            os.remove(save_path)


def _get_report_or_none(report_id: str):
    """Загружает отчёт или возвращает None (для FastAPI 404)."""
    return storage.get_report(report_id)


@app.get("/stats/{exercise}")
async def get_stats(exercise: str, format_type: str = "individual", k: int = 3):
    """
    Тренд критических ошибок по данным детектора (pre/post-контроль).

    Сравнивает среднее число critical-ошибок в первых k и последних k сессиях
    по этому упражнению и формату. Возвращает None-тренд, если сессий < 2*k.
    """
    trend = storage.error_trend(exercise, format_type, k=k)
    return {
        "exercise": exercise,
        "format": format_type,
        "trend": trend,
        "note": (
            "Тренд отражает снижение срабатываний детектора ошибок. "
            "Валидация правил против экспертизы тренера не проводилась."
        ),
    }


@app.get("/reports/{report_id}/pdf")
async def get_report_pdf(report_id: str):
    """Возвращает PDF-карточку тренировки («Сохранить отчёт»)."""
    doc = _get_report_or_none(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Отчёт не найден")

    analysis = doc.get("analysis") or {}
    report_dir = storage.get_report_dir(report_id)
    frames_dir = os.path.join(report_dir, "frames")

    # Миниатюра видео: первый кадр, сохранённый во время анализа
    thumb_path = os.path.join(frames_dir, "thumb.png")
    if not os.path.exists(thumb_path):
        video_full_path = os.path.join(report_dir, doc.get("video_file") or "")
        _save_video_thumbnail(video_full_path, thumb_path)

    # Кадры ошибок (PNG + координаты скелета для обводки)
    saved_frames = analysis.get("error_frames") or {}
    error_frames = {}
    for key, f in saved_frames.items():
        png_path = os.path.join(frames_dir, f"error_{key}.png")
        if os.path.exists(png_path):
            error_frames[int(key)] = {
                "path": png_path,
                "timestamp": f.get("timestamp"),
                "landmarks": f.get("landmarks"),
            }

    # Предыдущий отчёт по тому же упражнению и формату — для блока динамики
    previous = storage.find_previous_report(
        doc.get("exercise"), doc.get("format"), doc.get("created_at") or ""
    )

    pdf_path = os.path.join(report_dir, "report.pdf")
    report_generator.build_report_pdf(
        doc,
        video_thumb_path=thumb_path if os.path.exists(thumb_path) else None,
        error_frames=error_frames,
        previous_meta=previous,
        output_path=pdf_path,
    )

    if not os.path.exists(pdf_path):
        return {"error": "Не удалось сформировать PDF"}

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"training-report-{report_id[:8]}.pdf",
    )


def _save_video_thumbnail(video_path: str, thumb_path: str) -> None:
    """Сохраняет первый кадр видео как PNG-миниатюру (кэшируется на диске)."""
    if os.path.exists(thumb_path) or not video_path or not os.path.exists(video_path):
        return
    cap = cv2.VideoCapture(video_path)
    try:
        ret, frame = cap.read()
        if ret:
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            cv2.imwrite(thumb_path, frame)
    finally:
        cap.release()


@app.get("/reports/{report_id}/video")
async def get_report_video(report_id: str, request: Request, t: float = None):
    """
    Раздаёт видео отчёта. Параметр t (секунды) используется фронтенд-плеером
    для перескока к моменту ошибки по кликабельным таймкодам из PDF.
    """
    doc = _get_report_or_none(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Отчёт не найден")
    video_path = os.path.join(storage.get_report_dir(report_id), doc.get("video_file") or "")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Видео отчёта не найдено")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path),
    )


@app.get("/reports/{report_id}/frames/{frame_name}")
async def get_report_frame(report_id: str, frame_name: str):
    """Раздаёт PNG-кадры отчёта (скриншоты ошибок, тепловая карта)."""
    if not storage.is_valid_report_id(report_id):
        return {"error": "Некорректный report_id"}
    # Имя файла строго ограничено (защита от path traversal)
    allowed = {"error_0.png", "error_1.png", "error_2.png", "error_3.png", "error_4.png", "heatmap.png", "thumb.png"}
    if frame_name not in allowed:
        return {"error": "Кадр не найден"}
    frame_path = os.path.join(storage.get_report_dir(report_id), "frames", frame_name)
    if not os.path.exists(frame_path):
        return {"error": "Кадр не найден"}
    return FileResponse(frame_path, media_type="image/png")


@app.get("/video-player/{report_id}")
async def video_player(report_id: str, t: float = 0.0):
    """HTML-плеер видео с автопереходом к таймкоду ошибки (?t=<секунды>)."""
    doc = _get_report_or_none(report_id)
    if doc is None:
        return HTMLResponse("<h1>Отчёт не найден</h1>", status_code=404)
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>Видео тренировки</title></head>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center">
  <h2 style="padding:12px">Видео тренировки — переход к моменту ошибки</h2>
  <video id="player" width="960" controls autoplay>
    <source src="/reports/{report_id}/video" type="video/mp4">
  </video>
  <p><a href="/reports/{report_id}/pdf" style="color:#7ab7ff">Скачать PDF-карточку</a></p>
  <script>
    const player = document.getElementById("player");
    player.currentTime = {float(t)};
  </script>
  </body>
</html>"""
    return HTMLResponse(html)


@app.get("/health")
async def health():
    """
    Health-check для деплой-проверок и мониторинга: статус сервиса
    и готовность модели PoseLandmarker (наличие + совпадение pin-хеша).
    """
    return {
        "status": "ok" if POSE_MODEL_OK else "degraded",
        "pose_model_ok": POSE_MODEL_OK,
    }


# --- LLM-агент-ассистент тренера (MVP: GigaChat, бесплатный тариф) ---

# Rate limit на агент: сообщения в минуту на клиента (жёстче, чем /upload,
# потому что бесплатный тариф GigaChat имеет лимиты запросов).
AGENT_RATE_LIMIT_PER_MINUTE = int(os.environ.get("AGENT_RATE_LIMIT_PER_MINUTE", "10"))
# Максимум сообщений истории диалога, отправляемых в LLM (ограничение стоимости)
AGENT_MAX_HISTORY_MESSAGES = 12


def _check_agent_rate_limit(client_id: str) -> bool:
    """Скользящее окно 60 с для эндпоинта агента (AGENT_RATE_LIMIT_PER_MINUTE)."""
    now = time.monotonic()
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(f"agent:{client_id}", [])
        bucket[:] = [ts for ts in bucket if now - ts < 60.0]
        if len(bucket) >= AGENT_RATE_LIMIT_PER_MINUTE:
            return False
        bucket.append(now)
        return True


@app.post("/agent/chat/{report_id}")
async def agent_chat(
    report_id: str,
    request: Request,
):
    """
    Чат с LLM-агентом по отчёту о тренировке.

    Тело запроса (JSON):
      {"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}

    Ответ: {"reply": "..."}.
    Ошибки: 401/429 — auth/лимит; 404 — отчёт не найден; 400 — пустое сообщение;
    503 — LLM не настроен; 502 — ошибка провайдера LLM.
    """
    # 0. Auth + отдельный rate limit для агента
    api_key = _require_api_key(request)
    if not _check_agent_rate_limit(_client_id(request, api_key)):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Превышен лимит сообщений: {AGENT_RATE_LIMIT_PER_MINUTE} в минуту. "
                "Повторите попытку позже."
            ),
        )

    # 1. Тело запроса
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    message = (body or {}).get("message") or ""
    history = (body or {}).get("history") or []
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=400, detail="Поле message обязательно и не может быть пустым")
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="Поле history должно быть списком")

    # 2. LLM должен быть настроен до обращения к хранилищу (fail fast)
    try:
        llm = get_llm_client()
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 3. Контекст отчёта (детерминированные данные; отчёт не найден -> 404)
    try:
        context = build_report_context(report_id, reports_dir=storage.REPORTS_DIR)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 4. Диалог: системный промпт + контекст + ограниченная история + сообщение
    from agent.prompts import SYSTEM_PROMPT, format_context_as_json

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Данные отчёта (JSON):\n"
                + format_context_as_json(context)
                + "\n\nВопрос тренера: "
                + message.strip()
            ),
        },
    ]
    # История: только последние AGENT_MAX_HISTORY_MESSAGES сообщений
    for h in history[-AGENT_MAX_HISTORY_MESSAGES:]:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})

    # 5. Запрос к LLM (цикл function calling внутри ask(): модель может
    #    запросить инструменты динамики — error_trend / previous_report)
    try:
        reply = llm.ask(messages)
    except LLMRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"reply": reply, "report_id": report_id}


@app.get("/agent/health")
async def agent_health():
    """Доступность LLM-агента: настроен ли провайдер и установлен ли SDK."""
    from agent.llm_client import is_llm_configured

    return {
        "llm_configured": is_llm_configured(),
        "provider": os.environ.get("LLM_PROVIDER", "gigachat").lower(),
    }