import os
import sys
import unittest

import cv2
import numpy as np

import storage

# Добавляем корень проекта в путь, чтобы импортировать main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    validate_video_file,
    MAX_FILE_SIZE_BYTES,
    detect_errors_and_recommendations,
    compute_knee_angle,
    compute_leg_opening_angle,
    compute_hip_height,
    compute_torso_tilt_signed,
    compute_ankle_y,
    compute_ankle_height,
    compute_arm_symmetry,
    compute_rotation_angle_world,
    select_target_tracks,
    AthleteTrack,
    track_athletes,
    compute_deep_metrics,
    _suggested_drill,
    _build_heatmap,
    _latency_stats,
    _pause_frames,
    _movement_start_frame,
    detect_team_feedback,
)
from club_tracker import ClubTracker, analyze_club_flight, ClubTrajectory, assign_clubs_to_athletes


class TestValidateVideoFile(unittest.TestCase):
    """Тесты для проверки валидации загружаемого видеофайла."""

    def test_valid_mp4(self):
        """Валидный MP4-файл не должен вызывать ошибку."""
        validate_video_file("video.mp4", 1024)
        # Если ошибка не возникла — тест пройден

    def test_valid_webm(self):
        """Валидный WebM-файл не должен вызывать ошибку."""
        validate_video_file("video.webm", 1024)

    def test_valid_m4v(self):
        """Валидный M4V-файл не должен вызывать ошибку."""
        validate_video_file("video.m4v", 1024)

    def test_valid_hevc(self):
        """Валидный HEVC-файл (.hevc) не должен вызывать ошибку."""
        validate_video_file("video.hevc", 1024)

    def test_valid_h264(self):
        """Валидный H.264-файл (.h264) не должен вызывать ошибку."""
        validate_video_file("video.h264", 1024)

    def test_valid_h265(self):
        """Валидный H.265-файл (.h265) не должен вызывать ошибку."""
        validate_video_file("video.h265", 1024)

    def test_valid_uppercase_extension(self):
        """Расширение в верхнем регистре (.MP4) должно быть принято."""
        validate_video_file("video.MP4", 1024)

    def test_invalid_extension(self):
        """Файл с недопустимым расширением должен вызывать ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_video_file("file.txt", 1024)
        self.assertIn("Недопустимый формат файла", str(ctx.exception))

    def test_invalid_extension_no_dot(self):
        """Файл без расширения должен вызывать ValueError."""
        with self.assertRaises(ValueError):
            validate_video_file("file", 1024)

    def test_file_too_large(self):
        """Файл больше максимального размера должен вызывать ValueError."""
        oversized = MAX_FILE_SIZE_BYTES + 1
        with self.assertRaises(ValueError) as ctx:
            validate_video_file("video.mp4", oversized)
        self.assertIn("Файл слишком большой", str(ctx.exception))

    def test_exact_max_size(self):
        """Файл точно максимального размера должен проходить валидацию."""
        validate_video_file("video.mp4", MAX_FILE_SIZE_BYTES)


class TestDetectErrorsAndRecommendations(unittest.TestCase):
    """Тесты для детекции ошибок и генерации рекомендаций."""

    def _movement(self, **overrides):
        """Базовый набор метрик без ошибок (идеальная техника)."""
        movement = {
            "torso_tilt": {"avg_deg": 5.0, "max_deg": 85.0, "min_deg": 2.0},
            "wrist_height": {"avg_y": 0.3, "highest_y": 0.1, "reach_ratio": 0.9},
            "trajectory": {"total_distance": 2.0, "x_range": 0.05, "y_range": 0.3},
            "motion": {"avg_speed": 0.1, "max_speed": 0.3, "smoothness": 0.9, "jerk_count": 5},
        }
        movement.update(overrides)
        return movement

    def test_perfect_risk_buhavy(self):
        """Идеальная техника риска с булавами не должна содержать ошибок."""
        result = detect_errors_and_recommendations("risk-buhavy", self._movement())
        self.assertEqual(result["overall_score"], 100)
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["recommendations"])

    def test_shallow_tuck_risk_buhavy(self):
        """Мелкий кувырок должен определяться как критическая ошибка."""
        movement = self._movement(torso_tilt={"avg_deg": 10.0, "max_deg": 30.0, "min_deg": 5.0})
        result = detect_errors_and_recommendations("risk-buhavy", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertLess(result["overall_score"], 100)

    def test_low_reach_risk_buhavy(self):
        """Низкий подброс булав должен определяться как критическая ошибка."""
        movement = self._movement(wrist_height={"avg_y": 0.6, "highest_y": 0.8, "reach_ratio": 0.1})
        result = detect_errors_and_recommendations("risk-buhavy", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertTrue(any("Подброс" in e["message"] for e in result["errors"]))

    def test_asymmetric_reach_risk_buhavy(self):
        """Низкий подброс одной руки должен определяться, даже если среднее выглядит нормальным."""
        # Средний reach_ratio высокий (0.9), но правая рука поднялась слабо (0.2)
        movement = self._movement(
            wrist_height={
                "avg_y": 0.15, "highest_y": 0.1, "reach_ratio": 0.9,
                "left_reach_ratio": 0.9, "right_reach_ratio": 0.2,
            }
        )
        result = detect_errors_and_recommendations("risk-buhavy", movement)
        self.assertTrue(any("правой руки" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(any(e["severity"] == "critical" for e in result["errors"]))

    def test_symmetric_reach_risk_buhavy(self):
        """Симметричный высокий подброс обеих рук не должен давать ошибок о подбросе."""
        movement = self._movement(
            wrist_height={
                "avg_y": 0.15, "highest_y": 0.1, "reach_ratio": 0.9,
                "left_reach_ratio": 0.9, "right_reach_ratio": 0.9,
            }
        )
        result = detect_errors_and_recommendations("risk-buhavy", movement)
        self.assertFalse(any("Подброс" in e["message"] or "руки:" in e["message"] for e in result["errors"]))

    def test_low_jump(self):
        """Низкий прыжок должен определяться как критическая ошибка."""
        movement = self._movement(trajectory={"total_distance": 1.0, "x_range": 0.05, "y_range": 0.05})
        result = detect_errors_and_recommendations("jump", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertTrue(any("Прыжок слишком низкий" in e["message"] for e in result["errors"]))

    def test_no_backbend_jump(self):
        """Недостаточный прогиб в прыжке должен определяться как ошибка."""
        movement = self._movement(torso_tilt={"avg_deg": 5.0, "max_deg": 15.0, "min_deg": 2.0})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertTrue(any("прогиб" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_swaying_balance(self):
        """Раскачивание в равновесии должно определяться как критическая ошибка."""
        movement = self._movement(trajectory={"total_distance": 3.0, "x_range": 0.35, "y_range": 0.1})
        result = detect_errors_and_recommendations("balance", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertTrue(any("Раскачивание" in e["message"] for e in result["errors"]))

    def test_tilted_rotation(self):
        """Заваленная ось вращения должна определяться как критическая ошибка."""
        movement = self._movement(torso_tilt={"avg_deg": 25.0, "max_deg": 40.0, "min_deg": 10.0})
        result = detect_errors_and_recommendations("rotation", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertTrue(any("Ось вращения" in e["message"] for e in result["errors"]))

    def test_unknown_exercise(self):
        """Неизвестное упражнение не должно давать ошибок, но должно дать рекомендацию."""
        result = detect_errors_and_recommendations("unknown-exercise", self._movement())
        self.assertIsNone(result["overall_score"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["recommendations"])

    def test_no_pose_data(self):
        """Отсутствие данных о позе должно давать критическую ошибку."""
        result = detect_errors_and_recommendations("balance", {})
        self.assertIsNone(result["overall_score"])
        self.assertTrue(result["errors"])
        self.assertEqual(result["errors"][0]["severity"], "critical")

    def test_poor_grouping_risk_buhavy(self):
        """Недостаточная группировка (таз недостаточно низко) должна определяться как ошибка."""
        movement = self._movement(hip_height={"min_y": 0.2, "max_y": 0.4, "avg_y": 0.3})
        result = detect_errors_and_recommendations("risk-buhavy", movement)
        self.assertTrue(any("группировк" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_clubs_asymmetric_throw_risk_buhavy(self):
        """Асимметричный бросок булав должен определяться как ошибка."""
        movement = self._movement()
        clubs = {
            "clubs_detected": 2,
            "throw_height": {"left": 0.2, "right": 0.2, "max": 0.2, "min": 0.2},
            "symmetry": {"height_diff": 0.25, "peak_frame_diff": 1},
            "simultaneity": {"peak_frame_diff": 1, "simultaneous": True},
            "catch_position": {"left": {"x": 0.4, "y": 0.7}, "right": {"x": 0.6, "y": 0.7}},
        }
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("разной высоте" in e["message"].lower() for e in result["errors"]))

    def test_clubs_not_simultaneous_risk_buhavy(self):
        """Неодновременный подброс булав должен определяться как ошибка."""
        movement = self._movement()
        clubs = {
            "clubs_detected": 2,
            "throw_height": {"left": 0.2, "right": 0.2, "max": 0.2, "min": 0.2},
            "symmetry": {"height_diff": 0.02, "peak_frame_diff": 8},
            "simultaneity": {"peak_frame_diff": 8, "simultaneous": False},
            "catch_position": {"left": {"x": 0.4, "y": 0.7}, "right": {"x": 0.6, "y": 0.7}},
        }
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("не одновременно" in e["message"].lower() for e in result["errors"]))

    def test_clubs_catch_offset_risk_buhavy(self):
        """Ловля булавы со смещением от центра должна определяться как ошибка."""
        movement = self._movement()
        clubs = {
            "clubs_detected": 2,
            "throw_height": {"left": 0.2, "right": 0.2, "max": 0.2, "min": 0.2},
            "symmetry": {"height_diff": 0.02, "peak_frame_diff": 1},
            "simultaneity": {"peak_frame_diff": 1, "simultaneous": True},
            "catch_position": {"left": {"x": 0.1, "y": 0.7}, "right": {"x": 0.9, "y": 0.7}},
        }
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("смещена от центра" in e["message"].lower() for e in result["errors"]))

    def test_clubs_one_detected_risk_buhavy(self):
        """Одна обнаруженная булава — предупреждение о невозможности оценки симметрии."""
        movement = self._movement()
        clubs = {"clubs_detected": 1, "throw_height": None, "symmetry": None,
                 "simultaneity": None, "catch_position": None}
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("только одна" in e["message"].lower() for e in result["errors"]))

    def test_clubs_none_detected_risk_buhavy(self):
        """Булавы не обнаружены — предупреждение."""
        movement = self._movement()
        clubs = {"clubs_detected": 0, "throw_height": None, "symmetry": None,
                 "simultaneity": None, "catch_position": None}
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("не обнаружены" in e["message"].lower() for e in result["errors"]))

    def test_clubs_none_param_risk_buhavy(self):
        """Если clubs=None (трекинг не выполнялся) — ошибок про булавы быть не должно."""
        movement = self._movement()
        result = detect_errors_and_recommendations("risk-buhavy", movement, None)
        self.assertFalse(any("булав" in e["message"].lower() for e in result["errors"]))

    def test_narrow_legs_jump(self):
        """Недостаточное раскрытие ног в прыжке должно определяться как ошибка."""
        movement = self._movement(leg_opening={"max_deg": 60.0, "avg_deg": 40.0})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertTrue(any("раскрыт" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_low_hip_lift_jump(self):
        """Недостаточный вертикальный отрыв (таз низко) должен определяться как ошибка."""
        movement = self._movement(hip_height={"min_y": 0.6, "max_y": 0.9, "avg_y": 0.75})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertTrue(any("отрыв" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_bent_support_leg_balance(self):
        """Согнутая опорная нога в равновесии должна определяться как ошибка."""
        # left_y(0.95) - right_y(0.35) = 0.6 > 0 => левая лодыжка ниже => опорная левая (120° < 150°)
        movement = self._movement(
            ankle_y={"left_avg_y": 0.95, "right_avg_y": 0.35, "avg_diff": 0.6},
            knee_angle={"left_avg_deg": 120.0, "right_avg_deg": 170.0},
        )
        result = detect_errors_and_recommendations("balance", movement)
        self.assertTrue(any("опорн" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_straight_support_leg_balance(self):
        """Выпрямленная опорная нога в равновесии не должна давать ошибку."""
        movement = self._movement(knee_angle={"left_avg_deg": 175.0, "right_avg_deg": 172.0})
        result = detect_errors_and_recommendations("balance", movement)
        self.assertFalse(any("опорн" in e["message"].lower() for e in result["errors"]))

    def test_systematic_tilt_rotation(self):
        """Систематический завал оси вращения должен определяться как ошибка."""
        movement = self._movement(torso_tilt_signed={"avg_deg": 18.0, "abs_max_deg": 25.0})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertTrue(any("завалена" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_symmetric_rotation(self):
        """Симметричное вращение не должно давать ошибку систематического завала."""
        movement = self._movement(torso_tilt_signed={"avg_deg": 3.0, "abs_max_deg": 8.0})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertFalse(any("систематически" in e["message"].lower() for e in result["errors"]))

    def test_few_turns_rotation(self):
        """Малое число витков в пируэте должно определяться как ошибка."""
        movement = self._movement(rotation_turns={"total_deg": 120.0, "turns": 0.33})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertTrue(any("витк" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_enough_turns_rotation(self):
        """Достаточное число витков не должно давать ошибку."""
        movement = self._movement(rotation_turns={"total_deg": 720.0, "turns": 2.0})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertFalse(any("витк" in e["message"].lower() for e in result["errors"]))

    def test_asymmetric_arms_rotation(self):
        """Асимметрия рук должна определяться как ошибка."""
        movement = self._movement(arm_symmetry={"avg_diff": 0.25, "max_diff": 0.4})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertTrue(any("асимметричн" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_symmetric_arms_rotation(self):
        """Симметричные руки не должны давать ошибку."""
        movement = self._movement(arm_symmetry={"avg_diff": 0.03, "max_diff": 0.08})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertFalse(any("асимметричн" in e["message"].lower() for e in result["errors"]))

    def test_low_ankle_lift_jump(self):
        """Низкий отрыв стоп в прыжке должен определяться как ошибка."""
        movement = self._movement(ankle_height={"min_y": 0.85, "avg_y": 0.9})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertTrue(any("стоп" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_high_ankle_lift_jump(self):
        """Высокий отрыв стоп в прыжке не должен давать ошибку."""
        movement = self._movement(ankle_height={"min_y": 0.5, "avg_y": 0.7})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertFalse(any("стоп" in e["message"].lower() for e in result["errors"]))

    def test_bent_support_leg_by_ankle_balance(self):
        """Опорная нога определяется по высоте лодыжек; согнутая — ошибка."""
        movement = self._movement(
            ankle_y={"left_avg_y": 0.3, "right_avg_y": 0.9, "avg_diff": -0.6},
            knee_angle={"left_avg_deg": 120.0, "right_avg_deg": 175.0},
        )
        result = detect_errors_and_recommendations("balance", movement)
        # Левая лодыжка выше -> опорная правая (175°), ошибки быть не должно
        self.assertFalse(any("опорн" in e["message"].lower() for e in result["errors"]))

    def test_bent_support_leg_left_balance(self):
        """Если опорная левая нога согнута — ошибка с указанием ноги."""
        # left_y(0.95) - right_y(0.35) = 0.6 > 0 => левая лодыжка ниже => опорная левая
        movement = self._movement(
            ankle_y={"left_avg_y": 0.95, "right_avg_y": 0.35, "avg_diff": 0.6},
            knee_angle={"left_avg_deg": 120.0, "right_avg_deg": 175.0},
        )
        result = detect_errors_and_recommendations("balance", movement)
        self.assertTrue(any("опорн" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(any("левая" in e["message"].lower() for e in result["errors"]))

    def test_straight_support_leg_right_balance(self):
        """Если опорная правая нога выпрямлена — ошибки нет."""
        # left_y(0.35) - right_y(0.95) = -0.6 < 0 => правая лодыжка ниже => опорная правая
        movement = self._movement(
            ankle_y={"left_avg_y": 0.35, "right_avg_y": 0.95, "avg_diff": -0.6},
            knee_angle={"left_avg_deg": 175.0, "right_avg_deg": 175.0},
        )
        result = detect_errors_and_recommendations("balance", movement)
        self.assertFalse(any("опорн" in e["message"].lower() for e in result["errors"]))

    # --- Переворот вперёд (walkover-front) ---

    def _walkover_movement(self, **overrides):
        """Базовые метрики идеального переворота вперёд (без ошибок)."""
        movement = {
            "torso_tilt": {"avg_deg": 100.0, "max_deg": 170.0, "min_deg": 10.0},
            "wrist_height": {
                "avg_y": 0.45, "highest_y": 0.1, "reach_ratio": 0.9,
                "lowest_y": 0.9,
            },
            "trajectory": {"total_distance": 2.5, "x_range": 0.4, "y_range": 0.6},
            "motion": {"avg_speed": 0.15, "max_speed": 0.4, "smoothness": 0.85, "jerk_count": 5},
            "knee_angle": {"left_avg_deg": 175.0, "right_avg_deg": 174.0,
                           "left_min_deg": 170.0, "right_min_deg": 168.0},
            "leg_opening": {"max_deg": 170.0, "avg_deg": 120.0},
        }
        movement.update(overrides)
        return movement

    def test_perfect_walkover_front(self):
        """Идеальный переворот вперёд: 100 баллов, ошибок нет."""
        result = detect_errors_and_recommendations("walkover-front", self._walkover_movement())
        self.assertEqual(result["overall_score"], 100)
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["recommendations"])

    def test_shallow_pass_walkover_front(self):
        """Проход без стойки на руках (наклон < 150°) — критическая ошибка."""
        movement = self._walkover_movement(
            torso_tilt={"avg_deg": 80.0, "max_deg": 120.0, "min_deg": 10.0}
        )
        result = detect_errors_and_recommendations("walkover-front", movement)
        self.assertTrue(any("стойку на руках" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(any(e["severity"] == "critical" for e in result["errors"]))
        self.assertLess(result["overall_score"], 100)

    def test_hands_not_down_walkover_front(self):
        """Руки не опускаются до пола (lowest_y < 0.70) — критическая ошибка."""
        movement = self._walkover_movement(
            wrist_height={"avg_y": 0.4, "highest_y": 0.1, "reach_ratio": 0.9, "lowest_y": 0.55}
        )
        result = detect_errors_and_recommendations("walkover-front", movement)
        self.assertTrue(any("не опускаются до пола" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(any(e["severity"] == "critical" for e in result["errors"]))

    def test_narrow_split_walkover_front(self):
        """Недостаточный шпагат при проходе (раскрытие < 150°) — замечание."""
        movement = self._walkover_movement(
            leg_opening={"max_deg": 110.0, "avg_deg": 70.0}
        )
        result = detect_errors_and_recommendations("walkover-front", movement)
        self.assertTrue(any("раскрыт" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(any(e["severity"] == "warning" for e in result["errors"]))

    def test_bent_legs_walkover_front(self):
        """Согнутые ноги при проходе (мин. угол в колене < 150°) — замечание."""
        movement = self._walkover_movement(
            knee_angle={"left_avg_deg": 165.0, "right_avg_deg": 160.0,
                        "left_min_deg": 130.0, "right_min_deg": 155.0}
        )
        result = detect_errors_and_recommendations("walkover-front", movement)
        self.assertTrue(any("согнуты в коленях" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(any(e["severity"] == "warning" for e in result["errors"]))

    def test_jerky_exit_walkover_front(self):
        """Рывки при выходе из переворота (jerk_count > 15) — замечание."""
        movement = self._walkover_movement(
            motion={"avg_speed": 0.2, "max_speed": 0.5, "smoothness": 0.4, "jerk_count": 25}
        )
        result = detect_errors_and_recommendations("walkover-front", movement)
        self.assertTrue(any("резкий" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(any(e["severity"] == "warning" for e in result["errors"]))

    def test_walkover_front_no_leg_data(self):
        """Если данных о ногах нет — переворот оценивается только по корпусу и рукам."""
        movement = self._walkover_movement()
        movement.pop("leg_opening")
        movement.pop("knee_angle")
        result = detect_errors_and_recommendations("walkover-front", movement)
        self.assertEqual(result["overall_score"], 100)

    def test_walkover_front_drills_always_present(self):
        """Подводящие упражнения для переворота присутствуют в рекомендациях всегда."""
        result = detect_errors_and_recommendations("walkover-front", self._walkover_movement())
        joined = " ".join(result["recommendations"]).lower()
        for key in ("колесо через одну руку", "перекат вперёд через стойку", "мостик"):
            self.assertIn(key, joined)

    def test_walkover_front_drills_do_not_affect_score(self):
        """Подводящие упражнения не искажают оценку: без ошибок остаётся 100 баллов."""
        result = detect_errors_and_recommendations("walkover-front", self._walkover_movement())
        self.assertEqual(result["overall_score"], 100)
        self.assertEqual(result["errors"], [])
        self.assertGreaterEqual(len(result["recommendations"]), 4)


class TestComputeDeepMetricsWristLowestY(unittest.TestCase):
    """Тесты новой агрегированной метрики wrist_height.lowest_y."""

    def test_lowest_y_present(self):
        """lowest_y — максимальное значение y запястий (самая нижняя точка)."""
        wrist_heights = [0.3, 0.2, 0.5, 0.4]
        result = compute_deep_metrics(
            [10.0, 20.0], wrist_heights, [(0.5, 0.5), (0.5, 0.5)], 30.0,
        )
        wrist = result["wrist_height"]
        self.assertEqual(wrist["lowest_y"], 0.5)

    def test_lowest_y_with_per_hand_fields(self):
        """По-рукавные поля (left_highest_y и т.д.) не ломаются добавлением lowest_y."""
        result = compute_deep_metrics(
            [10.0, 20.0], [0.3, 0.4], [(0.5, 0.5), (0.5, 0.5)], 30.0,
            wrist_y_left=[0.25, 0.45],
            wrist_y_right=[0.35, 0.35],
        )
        wrist = result["wrist_height"]
        self.assertEqual(wrist["lowest_y"], 0.4)
        self.assertEqual(wrist["left_highest_y"], 0.25)
        self.assertEqual(wrist["right_highest_y"], 0.35)


class _Landmark:
    """Простой объект-заглушка с координатами x, y (и visibility)."""

    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def _pose(landmarks_dict):
    """Создаёт список из 33 ключевых точек, недостающие заполняет нулями."""
    pose = [_Landmark(0.0, 0.0) for _ in range(33)]
    for idx, (x, y) in landmarks_dict.items():
        pose[idx] = _Landmark(x, y)
    return pose


class TestExerciseSpecificFeatures(unittest.TestCase):
    """Тесты для exercise-специфичных признаков."""

    def test_knee_angle_straight(self):
        """Полностью выпрямленная нога должна давать угол ~180°."""
        # hip(0.5, 0.4), knee(0.5, 0.7), ankle(0.5, 1.0)
        pose = _pose({23: (0.5, 0.4), 25: (0.5, 0.7), 27: (0.5, 1.0)})
        angle = compute_knee_angle(pose, "left")
        self.assertAlmostEqual(angle, 180.0, delta=1.0)

    def test_knee_angle_bent(self):
        """Согнутая нога должна давать угол заметно меньше 180°."""
        # hip(0.5, 0.4), knee(0.5, 0.7), ankle(0.3, 0.9)
        pose = _pose({23: (0.5, 0.4), 25: (0.5, 0.7), 27: (0.3, 0.9)})
        angle = compute_knee_angle(pose, "left")
        self.assertLess(angle, 160.0)

    def test_leg_opening_narrow(self):
        """Ноги вместе должны давать малый угол раскрытия."""
        # hip_mid(0.5, 0.5), обе стопы в одной точке под тазом
        pose = _pose({23: (0.45, 0.5), 24: (0.55, 0.5), 27: (0.5, 1.0), 28: (0.5, 1.0)})
        angle = compute_leg_opening_angle(pose)
        self.assertLess(angle, 30.0)

    def test_leg_opening_wide(self):
        """Широко раскрытые ноги (шпагат) должны давать угол близкий к 180°."""
        # Горизонтальный шпагат: таз по центру, стопы на том же уровне по горизонтали
        pose = _pose({23: (0.45, 0.5), 24: (0.55, 0.5), 27: (0.0, 0.5), 28: (1.0, 0.5)})
        angle = compute_leg_opening_angle(pose)
        self.assertGreater(angle, 150.0)

    def test_hip_height(self):
        """Высота таза — среднее y левого и правого бедра."""
        pose = _pose({23: (0.4, 0.3), 24: (0.6, 0.5)})
        self.assertAlmostEqual(compute_hip_height(pose), 0.4, delta=0.001)

    def test_torso_tilt_signed_zero(self):
        """Вертикальный корпус должен давать нулевой наклон."""
        pose = _pose({11: (0.4, 0.2), 12: (0.6, 0.2), 23: (0.4, 0.6), 24: (0.6, 0.6)})
        self.assertAlmostEqual(compute_torso_tilt_signed(pose), 0.0, delta=1.0)

    def test_torso_tilt_signed_positive(self):
        """Наклон корпуса влево должен давать положительный угол."""
        # плечи смещены влево относительно таза
        pose = _pose({11: (0.35, 0.2), 12: (0.45, 0.2), 23: (0.4, 0.6), 24: (0.6, 0.6)})
        self.assertGreater(compute_torso_tilt_signed(pose), 0.0)

    def test_ankle_y(self):
        """Высота лодыжки — просто y ключевой точки."""
        pose = _pose({27: (0.5, 0.95), 28: (0.5, 0.3)})
        self.assertAlmostEqual(compute_ankle_y(pose, "left"), 0.95, delta=0.001)
        self.assertAlmostEqual(compute_ankle_y(pose, "right"), 0.3, delta=0.001)

    def test_ankle_height(self):
        """Средняя высота стоп — среднее y лодыжек."""
        pose = _pose({27: (0.5, 0.4), 28: (0.5, 0.8)})
        self.assertAlmostEqual(compute_ankle_height(pose), 0.6, delta=0.001)

    def test_arm_symmetry(self):
        """Асимметрия рук — модуль разницы высот запястий."""
        pose = _pose({15: (0.3, 0.25), 16: (0.7, 0.65)})
        self.assertAlmostEqual(compute_arm_symmetry(pose), 0.4, delta=0.001)

    def test_arm_symmetry_zero(self):
        """Симметричные руки дают нулевую асимметрию."""
        pose = _pose({15: (0.3, 0.4), 16: (0.7, 0.4)})
        self.assertAlmostEqual(compute_arm_symmetry(pose), 0.0, delta=0.001)

    def test_rotation_angle_world(self):
        """Угол ориентации плечевой линии по world-координатам."""
        class WL:
            def __init__(self, x, z):
                self.x = x
                self.z = z

        # Плечи направлены вдоль X (dz = 0) -> угол 0°
        world = {11: WL(0.0, 0.0), 12: WL(1.0, 0.0)}
        self.assertAlmostEqual(compute_rotation_angle_world(world), 0.0, delta=1.0)

        # Плечи направлены вдоль Z (dx = 0) -> угол 90°
        world = {11: WL(0.0, 0.0), 12: WL(0.0, 1.0)}
        self.assertAlmostEqual(compute_rotation_angle_world(world), 90.0, delta=1.0)

    def test_rotation_angle_world_negative(self):
        """Поворот в другую сторону даёт отрицательный угол."""
        class WL:
            def __init__(self, x, z):
                self.x = x
                self.z = z

        world = {11: WL(0.0, 0.0), 12: WL(0.0, -1.0)}
        self.assertAlmostEqual(compute_rotation_angle_world(world), -90.0, delta=1.0)


class TestClubTracker(unittest.TestCase):
    """Тесты для трекинга булав и анализа траекторий."""

    def test_analyze_club_flight_empty(self):
        """Без траекторий метрики полёта пустые."""
        result = analyze_club_flight([], 30.0)
        self.assertEqual(result["clubs_detected"], 0)
        self.assertIsNone(result["throw_height"])
        self.assertIsNone(result["symmetry"])

    def test_analyze_club_flight_two_clubs(self):
        """Две траектории: высота, симметрия, одновременность, ловля."""
        left = ClubTrajectory()
        left.positions = [(0.3, 0.8), (0.3, 0.5), (0.3, 0.2)]
        left.peak_y = 0.2
        left.peak_frame = 2
        left.catch_x, left.catch_y = 0.3, 0.7

        right = ClubTrajectory()
        right.positions = [(0.7, 0.8), (0.7, 0.4), (0.7, 0.2)]
        right.peak_y = 0.2
        right.peak_frame = 2
        right.catch_x, right.catch_y = 0.7, 0.7

        result = analyze_club_flight([left, right], 30.0)
        self.assertEqual(result["clubs_detected"], 2)
        self.assertEqual(result["throw_height"]["max"], 0.2)
        self.assertEqual(result["symmetry"]["height_diff"], 0.0)
        self.assertTrue(result["simultaneity"]["simultaneous"])
        self.assertEqual(result["catch_position"]["left"]["x"], 0.3)

    def test_analyze_club_flight_asymmetric(self):
        """Асимметричный бросок: разная высота и неодновременность."""
        left = ClubTrajectory()
        left.positions = [(0.3, 0.8), (0.3, 0.1), (0.3, 0.6)]
        left.peak_y = 0.1
        left.peak_frame = 1
        left.catch_x, left.catch_y = 0.3, 0.6

        right = ClubTrajectory()
        right.positions = [(0.7, 0.8), (0.7, 0.5), (0.7, 0.7)]
        right.peak_y = 0.5
        right.peak_frame = 6
        right.catch_x, right.catch_y = 0.7, 0.7

        result = analyze_club_flight([left, right], 30.0)
        self.assertEqual(result["throw_height"]["max"], 0.1)
        self.assertEqual(result["symmetry"]["height_diff"], 0.4)
        self.assertEqual(result["simultaneity"]["peak_frame_diff"], 5)
        self.assertFalse(result["simultaneity"]["simultaneous"])

    def test_detect_hsv_yellow_club(self):
        """HSV-детекция: жёлтый объект на тёмном фоне обнаруживается."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Рисуем жёлтый круг (BGR: (0, 255, 255))
        cv2.circle(frame, (50, 50), 12, (0, 255, 255), -1)
        tracker = ClubTracker()
        detections = tracker.detect(frame)
        self.assertGreaterEqual(len(detections), 1)

    def test_detect_hsv_no_clubs(self):
        """Без ярких объектов детекций нет."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Тёмный фон — ничего не должно детектироваться
        tracker = ClubTracker()
        detections = tracker.detect(frame)
        self.assertEqual(len(detections), 0)

    def test_track_returns_trajectories(self):
        """Трекинг последовательности кадров возвращает траектории с пиками."""
        frames = []
        for i in range(10):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            y = 180 - i * 10  # булава летит вверх
            cv2.circle(frame, (50, y), 10, (0, 255, 255), -1)  # жёлтая
            frames.append(frame)

        tracker = ClubTracker()
        trajectories = tracker.track(frames)
        self.assertGreaterEqual(len(trajectories), 1)
        self.assertIsNotNone(trajectories[0].peak_y)


class TestAthleteTracking(unittest.TestCase):
    """Тесты для трекинга гимнасток и связывания булав."""

    def _pose_with(self, center_x, center_y, scale=0.1):
        """Создаёт скелет (33 точки), сосредоточенный вокруг центра."""
        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.visibility = 1.0

        pts = []
        for i in range(33):
            pts.append(LM(center_x + (i % 5 - 2) * scale / 2,
                          center_y + (i % 7 - 3) * scale / 2))
        return pts

    def test_select_target_tracks_individual(self):
        """individual выбирает самый крупный трек."""
        t1 = AthleteTrack(index=0)
        t1.size_sum = 10.0
        t1.frames_seen = 1
        t2 = AthleteTrack(index=1)
        t2.size_sum = 30.0
        t2.frames_seen = 1
        selected = select_target_tracks([t1, t2], "individual")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].index, 1)

    def test_select_target_tracks_group(self):
        """group выбирает 5 самых крупных треков."""
        tracks = []
        for i in range(7):
            t = AthleteTrack(index=i)
            t.size_sum = float(i)
            t.frames_seen = 1
            tracks.append(t)
        selected = select_target_tracks(tracks, "group")
        self.assertEqual(len(selected), 5)
        # Самые крупные — индексы 6,5,4,3,2
        idxs = [t.index for t in selected]
        self.assertEqual(idxs, [6, 5, 4, 3, 2])

    def test_assign_clubs_to_athletes(self):
        """Булавы привязываются к ближайшей гимнастке по запястьям."""
        left = ClubTrajectory()
        left.positions = [(0.2, 0.3), (0.2, 0.2)]
        right = ClubTrajectory()
        right.positions = [(0.8, 0.3), (0.8, 0.2)]

        # Гимнастка 0 — слева (запястья ~0.2), гимнастка 1 — справа (~0.8)
        wrist_positions = [
            [(0.2, 0.4), (0.2, 0.35)],
            [(0.8, 0.4), (0.8, 0.35)],
        ]
        result = assign_clubs_to_athletes([left, right], wrist_positions)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)
        # Левая булава — гимнастке 0, правая — гимнастке 1
        self.assertAlmostEqual(result[0][0].positions[0][0], 0.2, delta=0.001)
        self.assertAlmostEqual(result[1][0].positions[0][0], 0.8, delta=0.001)

    def test_athlete_track_add_frame(self):
        """AthleteTrack накапливает ряды признаков по кадрам."""
        # Заглушка landmarks: объект с x, y, visibility
        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.visibility = 1.0

        pts = []
        for i in range(33):
            pts.append(LM(0.1 + i * 0.02, 0.1 + (i % 3) * 0.02))
        track = AthleteTrack(index=0)
        track.add_frame(pts)
        self.assertEqual(track.frames_seen, 1)
        self.assertEqual(len(track.torso_angles), 1)
        self.assertEqual(len(track.wrist_heights), 1)
        self.assertIsNotNone(track.last_center)

    def test_track_athletes_two_people(self):
        """Два человека в кадре → два трека."""
        frames = [
            (None, [self._pose_with(0.3, 0.4), self._pose_with(0.7, 0.4)], None),
            (None, [self._pose_with(0.3, 0.41), self._pose_with(0.7, 0.42)], None),
        ]
        tracks = track_athletes(frames, 30.0)
        self.assertEqual(len(tracks), 2)


class TestSuggestedDrills(unittest.TestCase):
    """Тесты подбора корректирующего упражнения по тексту ошибки."""

    def test_drill_for_bent_support_leg(self):
        """Ошибка про опорную ногу → упражнение для стопы/полупальцев."""
        drill = _suggested_drill("Опорная нога (левая) согнута: угол в колене 120.0°")
        self.assertIsNotNone(drill)
        self.assertIn("опоры", drill.lower())

    def test_drill_for_low_throw(self):
        """Ошибка про низкий подброс → упражнение на подброс булав."""
        drill = _suggested_drill("Подброс булав недостаточно высокий: руки поднялись лишь на 15%")
        self.assertIsNotNone(drill)
        self.assertIn("подброс", drill.lower())

    def test_drill_unknown_error(self):
        """Для нераспознанного текста упражнения не подбирается (None)."""
        self.assertIsNone(_suggested_drill("Совершенно непонятная ошибка без ключевых слов"))


class TestHeatmap(unittest.TestCase):
    """Тесты тепловой карты перемещений (групповой формат)."""

    def _pose_with(self, center_x, center_y, scale=0.05):
        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.visibility = 1.0

        return [LM(center_x, center_y) for _ in range(33)]

    def test_heatmap_counts_positions(self):
        """Тепловая карта собирает статистику по центрам скелетов."""
        frames = [
            (None, [self._pose_with(0.2, 0.3)] * 5, None),
            (None, [self._pose_with(0.2, 0.3)] * 5, None),
        ]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "heatmap.png")
            stats = _build_heatmap(frames, path)
            self.assertIsNotNone(stats)
            # 5 гимнасток в одной точке два кадра подряд → пик плотности 10
            self.assertEqual(stats["peak_density"], 10)
            # Все попадания в «плотную» ячейку (>= 4) → overlap 1.0
            self.assertEqual(stats["overlap_ratio"], 1.0)
            self.assertTrue(os.path.exists(path))

    def test_heatmap_empty(self):
        """Без кадров статистика не строится."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "heatmap.png")
            stats = _build_heatmap([], path)
            self.assertIsNone(stats)


class TestTeamFeedback(unittest.TestCase):
    """Тесты командной обратной связи (групповой формат)."""

    def _track_with_com(self, com_positions):
        """Трек-заглушка с заданной траекторией центра масс."""
        t = AthleteTrack(index=0)
        t.com_positions = com_positions
        return t

    def test_pause_frames_detects_stillness(self):
        """Пауза: подряд идущие кадры с почти нулевой скоростью."""
        com = [(0.5, 0.5)] * 4  # неподвижность
        self.assertGreaterEqual(_pause_frames(com), 2)

    def test_pause_frames_no_pause(self):
        """Непрерывное движение — паузы нет."""
        com = [(0.1 + i * 0.05, 0.5) for i in range(6)]
        self.assertEqual(_pause_frames(com), 0)

    def test_movement_start_frame(self):
        """Старт движения — первый кадр, где скорость превысила порог."""
        com = [(0.5, 0.5), (0.5, 0.5), (0.6, 0.5), (0.7, 0.5)]
        self.assertEqual(_movement_start_frame(com), 2)

    def test_team_start_desync_detected(self):
        """Рассинхрон старта > допуска — замечание для команды."""
        # Гимнастка 1 стартует на кадре 0, гимнастка 2 — на кадре 12 (fps=30 -> 0.4 c)
        t1 = self._track_with_com([(0.5 + i * 0.05, 0.5) for i in range(10)])
        t2 = self._track_with_com(
            [(0.5, 0.5)] * 12 + [(0.5 + i * 0.05, 0.5) for i in range(5)]
        )
        result = detect_team_feedback([t1, t2], 30.0, None)
        self.assertTrue(any("рассинхрон" in m["message"].lower() for m in result["messages"]))
        self.assertEqual(result["start_spread_sec"], 0.4)

    def test_team_start_sync_no_message(self):
        """Синхронный старт — замечаний о рассинхроне нет."""
        t1 = self._track_with_com([(0.5 + i * 0.05, 0.5) for i in range(10)])
        t2 = self._track_with_com([(0.3 + i * 0.05, 0.5) for i in range(10)])
        result = detect_team_feedback([t1, t2], 30.0, None)
        self.assertFalse(any("рассинхрон" in m["message"].lower() for m in result["messages"]))

    def test_team_rhythm_loss_after_drop(self):
        """Падение булавы у одной гимнастки + паузы у остальных — командное замечание."""
        t_moving = self._track_with_com([(0.5 + i * 0.05, 0.5) for i in range(10)])
        t_paused = self._track_with_com([(0.5, 0.5)] * 10)
        clubs = {"dropped": {"left": True, "right": False}}
        result = detect_team_feedback([t_moving, t_paused, t_paused], 30.0, [clubs, None, None])
        self.assertTrue(
            any("теряет ритм" in m["message"].lower() for m in result["messages"])
        )

    def test_team_no_drop_no_rhythm_message(self):
        """Без падения булавы командного замечания о ритме нет."""
        t_moving = self._track_with_com([(0.5 + i * 0.05, 0.5) for i in range(10)])
        clubs = {"dropped": {"left": False, "right": False}}
        result = detect_team_feedback([t_moving], 30.0, [clubs])
        self.assertFalse(
            any("теряет ритм" in m["message"].lower() for m in result["messages"])
        )


class TestLatencyStats(unittest.TestCase):
    """Тесты статистики латентности обработки кадра."""

    def test_latency_stats_values(self):
        """avg, p95 и max считаются корректно."""
        stats = _latency_stats([10.0, 20.0, 30.0, 40.0, 100.0])
        self.assertEqual(stats["samples"], 5)
        self.assertEqual(stats["avg_ms"], 40.0)
        self.assertEqual(stats["max_ms"], 100.0)
        self.assertLessEqual(stats["p95_ms"], 100.0)
        self.assertGreaterEqual(stats["p95_ms"], 40.0)

    def test_latency_stats_empty(self):
        """Без замеров статистика пустая."""
        self.assertIsNone(_latency_stats([]))


class TestStorage(unittest.TestCase):
    """Тесты хранения отчётов."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.reports_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_get_report(self):
        """Отчёт сохраняется и загружается с видео и метаданными."""
        import shutil
        video_path = os.path.join(self._tmp.name, "video.mp4")
        with open(video_path, "wb") as f:
            f.write(b"fake video bytes")

        rid = storage.create_report_id()
        analysis = {"exercise": "balance", "format": "individual", "feedback": {}}
        storage.save_report(rid, analysis, video_path, self.reports_dir)

        doc = storage.get_report(rid, self.reports_dir)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["exercise"], "balance")
        # Видео перемещено в директорию отчёта
        self.assertTrue(
            os.path.exists(os.path.join(storage.get_report_dir(rid, self.reports_dir), doc["video_file"]))
        )

    def test_find_previous_report(self):
        """Динамика: находится ближайший предыдущий отчёт по упражнению и формату."""
        rid1 = storage.create_report_id()
        storage.save_report(
            rid1,
            {"exercise": "balance", "format": "individual", "feedback": {},
             "movement": {"knee_angle": {"left_avg_deg": 150.0, "right_avg_deg": 150.0}}},
            None,
            self.reports_dir,
        )
        rid2 = storage.create_report_id()
        storage.save_report(
            rid2,
            {"exercise": "balance", "format": "individual", "feedback": {},
             "movement": {"knee_angle": {"left_avg_deg": 155.0, "right_avg_deg": 155.0}}},
            None,
            self.reports_dir,
        )
        doc2 = storage.get_report(rid2, self.reports_dir)
        prev = storage.find_previous_report("balance", "individual", doc2["created_at"], self.reports_dir)
        self.assertIsNotNone(prev)
        self.assertEqual(prev["report_id"], rid1)

    def test_invalid_report_id(self):
        """Подделка report_id (path traversal) отклоняется."""
        self.assertFalse(storage.is_valid_report_id("../../etc"))
        self.assertIsNone(storage.get_report("../../etc", self.reports_dir))

    def test_cleanup_old_reports(self):
        """Отчёты старше срока хранения удаляются."""
        rid = storage.create_report_id()
        storage.save_report(rid, {"exercise": "jump", "format": "individual", "feedback": {}}, None, self.reports_dir)
        # Уменьшаем срок хранения до 0 дней — отчёт должен удалиться
        removed = storage.cleanup_old_reports(self.reports_dir, retention_days=0)
        self.assertEqual(removed, 1)
        self.assertIsNone(storage.get_report(rid, self.reports_dir))


class TestErrorTrend(unittest.TestCase):
    """Тесты тренда критических ошибок (pre/post-контроль по данным детектора)."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.reports_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _save_session(self, exercise: str, critical: int, warning: int = 0):
        rid = storage.create_report_id()
        errors = (
            [{"severity": "critical", "message": "e"}] * critical
            + [{"severity": "warning", "message": "w"}] * warning
        )
        storage.save_report(
            rid,
            {"exercise": exercise, "format": "individual",
             "feedback": {"errors": errors, "overall_score": 100 - critical * 25}},
            None,
            self.reports_dir,
        )

    def test_error_trend_improvement(self):
        """Ранние сессии с большим числом critical, поздние с меньшим → тренд вниз."""
        for _ in range(3):
            self._save_session("balance", critical=3)
        for _ in range(3):
            self._save_session("balance", critical=1)
        trend = storage.error_trend("balance", "individual", k=3, reports_dir=self.reports_dir)
        self.assertIsNotNone(trend)
        self.assertEqual(trend["sessions"], 6)
        self.assertEqual(trend["early_avg_critical"], 3.0)
        self.assertEqual(trend["late_avg_critical"], 1.0)
        self.assertLess(trend["delta_pct"], 0)

    def test_error_trend_insufficient_sessions(self):
        """Меньше 2*k сессий → тренд не считается (None)."""
        self._save_session("balance", critical=2)
        self.assertIsNone(storage.error_trend("balance", "individual", k=3, reports_dir=self.reports_dir))

    def test_error_trend_filters_by_exercise_and_format(self):
        """Отчёты по другим упражнениям/форматам не учитываются."""
        # 6 сессий по balance — достаточно для порога 2*k при k=3;
        # сессии по jump не должны попадать в тренд по balance.
        for _ in range(6):
            self._save_session("balance", critical=2)
        for _ in range(3):
            self._save_session("jump", critical=5)
        trend = storage.error_trend("balance", "individual", k=3, reports_dir=self.reports_dir)
        self.assertIsNotNone(trend)
        self.assertEqual(trend["early_avg_critical"], 2.0)
        self.assertEqual(trend["late_avg_critical"], 2.0)

    def test_error_trend_zero_early(self):
        """Если в ранних сессиях ноль critical — delta_pct равен None (деление на ноль)."""
        for _ in range(3):
            self._save_session("balance", critical=0)
        for _ in range(3):
            self._save_session("balance", critical=2)
        trend = storage.error_trend("balance", "individual", k=3, reports_dir=self.reports_dir)
        self.assertIsNotNone(trend)
        self.assertIsNone(trend["delta_pct"])


if __name__ == "__main__":
    unittest.main()