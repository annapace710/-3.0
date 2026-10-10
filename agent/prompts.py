"""Системный промпт и сборка контекста отчёта для LLM-агента."""

import json

import storage

SYSTEM_PROMPT = """Ты — методист по художественной гимнастике, ассистент тренера.

Правила:
1. Ты отвечаешь ТОЛЬКО на основе данных отчёта, переданных в сообщении пользователя.
2. НЕ выдумывай метрики, ошибки и рекомендации: если данных нет в отчёте —
   скажи, что данных недостаточно.
3. Ошибки техники уже определены детерминированным детектором (правила с порогами);
   твоя задача — объяснить их понятным языком и помочь составить план отработки.
4. Отвечай кратко, по-русски, структурированно: сначала суть, потом конкретные шаги.
5. Ты не ставишь диагнозы здоровья и не даёшь медицинских советов.
6. Если вопрос не связан с техникой упражнений, отчётом или планом тренировки —
   вежливо верни разговор к этим темам.
7. Для вопросов о прогрессе и сравнении тренировок используй инструменты
   (get_error_trend, get_previous_report), а не догадки.

Данные отчёта (JSON) приложены в сообщении пользователя."""


def build_report_context(report_id: str, reports_dir: str = storage.REPORTS_DIR) -> dict:
    """
    Собирает контекст для агента из отчёта и истории тренировок.

    Включает: упражнение, формат, оценку, ошибки, рекомендации, тренд
    критических ошибок (динамика) — всё, что уже вычислено детерминированно.

    Бросает ValueError, если отчёт не найден.
    """
    doc = storage.get_report(report_id, reports_dir)
    if doc is None:
        raise ValueError(f"Отчёт {report_id} не найден")

    analysis = doc.get("analysis") or {}
    exercise = doc.get("exercise") or ""
    format_type = doc.get("format") or ""

    # Динамика: тренд критических ошибок по этому упражнению и формату
    trend = storage.error_trend(exercise, format_type, k=3, reports_dir=reports_dir)

    return {
        "report_id": report_id,
        "created_at": doc.get("created_at"),
        "exercise": exercise,
        "format": format_type,
        "movement": analysis.get("movement"),
        "feedback": analysis.get("feedback"),
        "athletes": analysis.get("athletes"),
        "clubs": analysis.get("clubs"),
        "team_feedback": analysis.get("team_feedback"),
        "error_trend": trend,
    }


def format_context_as_json(context: dict) -> str:
    """Сериализует контекст в компактный JSON для вставки в промпт."""
    return json.dumps(context, ensure_ascii=False, default=str)
