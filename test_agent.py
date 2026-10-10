"""
Тесты LLM-агента (MVP: GigaChat). Внешние вызовы не выполняются: LLM мокается.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient

import main
import storage
from agent.llm_client import (
    BaseLLMClient,
    LLMNotConfiguredError,
    is_llm_configured,
    get_llm_client,
)
from agent.prompts import SYSTEM_PROMPT, build_report_context, format_context_as_json


class _StubLLM(BaseLLMClient):
    """Мок-клиент: возвращает фиксированный ответ, запоминает messages."""

    def __init__(self):
        self.calls = []

    def ask(self, messages):
        self.calls.append(messages)
        return "ОК: кувырок недостаточно глубокий, нужна отработка перекатов."


class AgentTestCase(unittest.TestCase):
    """Базовый класс: приложение + мок LLM + временная директория отчётов."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.reports_dir = self._tmp.name
        self._orig_reports = storage.REPORTS_DIR
        storage.REPORTS_DIR = self.reports_dir

        self.stub = _StubLLM()
        self._orig_client = main.get_llm_client
        # Патчим фабрику клиента и флаг конфигурации в agent.llm_client
        import agent.llm_client as llm_mod

        self._orig_llm_get = llm_mod.get_llm_client
        self._orig_llm_flag = llm_mod.is_llm_configured
        llm_mod.get_llm_client = lambda: self.stub
        llm_mod.is_llm_configured = lambda: True
        main.get_llm_client = lambda: self.stub

        self.client = TestClient(main.app)

    def tearDown(self):
        storage.REPORTS_DIR = self._orig_reports
        main.get_llm_client = self._orig_client
        import agent.llm_client as llm_mod

        llm_mod.get_llm_client = self._orig_llm_get
        llm_mod.is_llm_configured = self._orig_llm_flag
        self._tmp.cleanup()

    def _save_report(self, exercise="balance", critical=2):
        rid = storage.create_report_id()
        errors = [{"severity": "critical", "message": "e"} for _ in range(critical)]
        storage.save_report(
            rid,
            {
                "exercise": exercise,
                "format": "individual",
                "feedback": {"overall_score": 50, "errors": errors},
                "movement": {"torso_tilt": {"avg_deg": 15.0}},
            },
            None,
            self.reports_dir,
        )
        return rid


class TestAgentChat(AgentTestCase):
    def test_chat_success(self):
        rid = self._save_report()
        resp = self.client.post(
            f"/agent/chat/{rid}", json={"message": "Почему оценка 50?"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["report_id"], rid)
        self.assertIn("кувырок", data["reply"])
        # LLM получил системный промпт и контекст
        messages = self.stub.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(SYSTEM_PROMPT[:40], messages[0]["content"])
        # В контексте есть данные отчёта
        user_msg = messages[1]["content"]
        self.assertIn('"exercise": "balance"', user_msg)

    def test_chat_report_not_found(self):
        fake = storage.create_report_id()  # валидный формат, но не существует
        resp = self.client.post(f"/agent/chat/{fake}", json={"message": "привет"})
        self.assertEqual(resp.status_code, 404)

    def test_chat_empty_message(self):
        rid = self._save_report()
        resp = self.client.post(f"/agent/chat/{rid}", json={"message": "  "})
        self.assertEqual(resp.status_code, 400)

    def test_chat_invalid_body(self):
        rid = self._save_report()
        resp = self.client.post(f"/agent/chat/{rid}", content=b"not json")
        self.assertEqual(resp.status_code, 400)

    def test_history_is_passed_and_limited(self):
        rid = self._save_report()
        history = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        self.client.post(
            f"/agent/chat/{rid}", json={"message": "что дальше?", "history": history}
        )
        messages = self.stub.calls[0]
        # system + контекст-сообщение + не более AGENT_MAX_HISTORY_MESSAGES истории
        limit = main.AGENT_MAX_HISTORY_MESSAGES
        self.assertLessEqual(len(messages), 2 + limit)
        self.assertEqual(messages[-1]["content"], "msg 29")

    def test_rate_limit(self):
        rid = self._save_report()
        main.AGENT_RATE_LIMIT_PER_MINUTE = 2
        try:
            self.client.post(f"/agent/chat/{rid}", json={"message": "1"})
            self.client.post(f"/agent/chat/{rid}", json={"message": "2"})
            resp = self.client.post(f"/agent/chat/{rid}", json={"message": "3"})
            self.assertEqual(resp.status_code, 429)
        finally:
            main.AGENT_RATE_LIMIT_PER_MINUTE = 10

    def test_llm_not_configured_raises(self):
        # Оригинальная фабрика без ключа в окружении должна бросать LLMNotConfiguredError
        os.environ.pop("LLM_API_KEY", None)
        with self.assertRaises(LLMNotConfiguredError):
            self._orig_llm_get()


class TestAgentHealth(AgentTestCase):
    def test_agent_health(self):
        resp = self.client.get("/agent/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("llm_configured", data)
        self.assertIn("provider", data)


class TestContextBuilder(AgentTestCase):
    def test_build_report_context_includes_trend(self):
        # 6 сессий, чтобы тренд считался (2*k при k=3)
        for _ in range(6):
            self._save_report(critical=2)
        rid = self._save_report(critical=3)
        ctx = build_report_context(rid, reports_dir=self.reports_dir)
        self.assertEqual(ctx["exercise"], "balance")
        self.assertIn("feedback", ctx)
        self.assertIsNotNone(ctx["error_trend"])
        # JSON-сериализация работает
        self.assertIn("balance", format_context_as_json(ctx))

    def test_build_report_context_not_found(self):
        fake = storage.create_report_id()
        with self.assertRaises(ValueError):
            build_report_context(fake, reports_dir=self.reports_dir)


class TestAgentTools(unittest.TestCase):
    """Тесты function calling: схемы инструментов и execute_tool (без LLM)."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.reports_dir = self._tmp.name
        self._orig_reports = storage.REPORTS_DIR
        storage.REPORTS_DIR = self.reports_dir

    def tearDown(self):
        storage.REPORTS_DIR = self._orig_reports
        self._tmp.cleanup()

    def _save_report(self, exercise="balance", critical=2, format_type="individual"):
        rid = storage.create_report_id()
        errors = [{"severity": "critical", "message": "e"} for _ in range(critical)]
        doc = storage.save_report(
            rid,
            {
                "exercise": exercise,
                "format": format_type,
                "feedback": {"overall_score": 50, "errors": errors},
                "movement": {"torso_tilt": {"avg_deg": 15.0}},
            },
            None,
            self.reports_dir,
        )
        return doc

    def test_tool_specs_valid(self):
        from agent.tools import TOOL_SPECS

        names = [spec["name"] for spec in TOOL_SPECS]
        self.assertIn("get_error_trend", names)
        self.assertIn("get_previous_report", names)
        for spec in TOOL_SPECS:
            self.assertIn("description", spec)
            self.assertEqual(spec["parameters"]["type"], "object")

    def test_execute_tool_get_error_trend(self):
        from agent.tools import execute_tool

        # 6 сессий, чтобы тренд считался (2*k при k=3)
        for _ in range(6):
            self._save_report(critical=2)
        result = json.loads(
            execute_tool(
                "get_error_trend",
                {"exercise": "balance", "format": "individual"},
                reports_dir=self.reports_dir,
            )
        )
        self.assertIn("trend", result)
        self.assertEqual(result["trend"]["sessions"], 6)

    def test_execute_tool_get_previous_report(self):
        from agent.tools import execute_tool

        prev = self._save_report(critical=1)
        result = json.loads(
            execute_tool(
                "get_previous_report",
                {
                    "exercise": "balance",
                    "format": "individual",
                    "created_at": "2999-01-01T00:00:00+00:00",
                },
                reports_dir=self.reports_dir,
            )
        )
        self.assertEqual(
            result["previous_report"]["report_id"], prev["report_id"]
        )

    def test_execute_tool_previous_report_not_found(self):
        from agent.tools import execute_tool

        result = json.loads(
            execute_tool(
                "get_previous_report",
                {
                    "exercise": "rotation",
                    "format": "group",
                    "created_at": "2999-01-01T00:00:00+00:00",
                },
                reports_dir=self.reports_dir,
            )
        )
        self.assertIsNone(result["previous_report"])

    def test_execute_tool_unknown_and_error(self):
        from agent.tools import execute_tool

        unknown = json.loads(
            execute_tool("no_such_tool", {}, reports_dir=self.reports_dir)
        )
        self.assertIn("error", unknown)

        # Невалидные аргументы не должны ронять вызов — возвращается error-JSON
        broken = json.loads(
            execute_tool(
                "get_error_trend",
                {"k": "не число"},
                reports_dir=self.reports_dir,
            )
        )
        self.assertIn("error", broken)


if __name__ == "__main__":
    unittest.main()
