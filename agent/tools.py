"""
Инструменты (function calling) LLM-агента: динамика тренировок.

Агент сам решает, когда вызвать инструмент, на основе схем из TOOL_SPECS.
Все инструменты детерминированные (поверх storage) — LLM не генерирует данные,
а получает факты из хранилища отчётов.
"""

import json

import storage

# Схемы инструментов в формате OpenAI/gigachat function calling.
TOOL_SPECS = [
    {
        "name": "get_error_trend",
        "description": (
            "Тренд числа критических ошибок по упражнению и формату: среднее "
            "число critical-ошибок в первых k и последних k сессиях. Используй "
            "для вопросов о прогрессе/динамике ('есть ли прогресс?', 'как в динамике?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "exercise": {
                    "type": "string",
                    "description": "Код упражнения: risk-buhavy, jump, balance, rotation, walkover-front",
                },
                "format": {
                    "type": "string",
                    "description": "Формат: individual или group",
                },
                "k": {
                    "type": "integer",
                    "description": "Размер окна сессий (по умолчанию 3)",
                },
            },
            "required": ["exercise", "format"],
        },
    },
    {
        "name": "get_previous_report",
        "description": (
            "Метаданные и метрики предыдущего отчёта по тому же упражнению и "
            "формату (для сравнения с текущей тренировкой)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "exercise": {
                    "type": "string",
                    "description": "Код упражнения",
                },
                "format": {
                    "type": "string",
                    "description": "Формат: individual или group",
                },
                "created_at": {
                    "type": "string",
                    "description": "ISO-дата текущего отчёта (created_at из контекста)",
                },
            },
            "required": ["exercise", "format", "created_at"],
        },
    },
]


def execute_tool(
    name: str,
    arguments: dict,
    reports_dir: str = storage.REPORTS_DIR,
) -> str:
    """
    Выполняет инструмент по имени и возвращает результат JSON-строкой.
    Неизвестный инструмент -> сообщение об ошибке (LLM увидит и исправится).
    """
    try:
        if name == "get_error_trend":
            trend = storage.error_trend(
                arguments.get("exercise", ""),
                arguments.get("format", "individual"),
                k=int(arguments.get("k", 3)),
                reports_dir=reports_dir,
            )
            return json.dumps(
                {"trend": trend}, ensure_ascii=False, default=str
            )

        if name == "get_previous_report":
            doc = storage.find_previous_report(
                arguments.get("exercise", ""),
                arguments.get("format", "individual"),
                arguments.get("created_at", ""),
                reports_dir=reports_dir,
            )
            if doc is None:
                return json.dumps(
                    {"previous_report": None,
                     "note": "Предыдущий отчёт не найден — сравнение невозможно."},
                    ensure_ascii=False,
                )
            analysis = doc.get("analysis") or {}
            return json.dumps(
                {
                    "previous_report": {
                        "report_id": doc.get("report_id"),
                        "created_at": doc.get("created_at"),
                        "movement": analysis.get("movement"),
                        "feedback": analysis.get("feedback"),
                    }
                },
                ensure_ascii=False,
                default=str,
            )

        return json.dumps(
            {"error": f"Неизвестный инструмент: {name}"}, ensure_ascii=False
        )
    except Exception as e:  # ошибка инструмента не должна ломать диалог
        return json.dumps({"error": f"Ошибка инструмента {name}: {e}"}, ensure_ascii=False)
