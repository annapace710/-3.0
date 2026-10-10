"""
Клиент LLM с абстракцией провайдера.

MVP: GigaChat (Сбер, бесплатный тариф) через SDK `gigachat`.
Смена провайдера = смена переменных окружения LLM_PROVIDER/LLM_API_KEY/LLM_MODEL;
SDK импортируется лениво, поэтому приложение работает и без него
(эндпоинт агента тогда отвечает 503).

Function calling: клиент поддерживает цикл tool-calls — модель может запросить
инструмент из TOOL_SPECS, клиент выполняет его через agent.tools.execute_tool
(детерминированные данные из storage) и отправляет результат модели обратно.
"""

import json
import os
import threading

from agent.tools import TOOL_SPECS, execute_tool


class LLMNotConfiguredError(RuntimeError):
    """LLM-провайдер не настроен (нет LLM_API_KEY или SDK не установлен)."""


class LLMRequestError(RuntimeError):
    """Ошибка запроса к LLM-провайдеру (сеть, лимиты, отказ сервиса)."""


class BaseLLMClient:
    """Единый интерфейс: ask(messages) -> str. Реализуется для каждого провайдера."""

    def ask(self, messages: list) -> str:
        raise NotImplementedError


class GigaChatClient(BaseLLMClient):
    """
    Клиент GigaChat. SDK `gigachat` импортируется лениво — при первом запросе.

    Поддерживает цикл function calling: если модель возвращает tool_calls,
    инструменты выполняются (максимум MAX_TOOL_ROUNDS раундов), а результаты
    отправляются модели как сообщения role="function".
    """

    MAX_TOOL_ROUNDS = 3  # защита от зацикливания tool-calls

    def __init__(self, api_key: str, model: str = "GigaChat"):
        self._api_key = api_key
        self._model = model
        self._client = None
        self._lock = threading.Lock()

    def _ensure_client(self):
        """Ленивая инициализация SDK: импорт при первом запросе, не при импорте модуля."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        from gigachat import GigaChat
                    except ImportError as e:
                        raise LLMNotConfiguredError(
                            "SDK gigachat не установлен. Выполните: pip install gigachat"
                        ) from e
                    self._client = GigaChat(
                        credentials=self._api_key,
                        model=self._model,
                        verify_ssl_certs=False,  # самоподписанный сертификат Сбера
                    )
        return self._client

    def _ask_once(self, messages: list, functions=None):
        """Один вызов chat(). Возвращает сырой ответ SDK."""
        client = self._ensure_client()
        kwargs = {"messages": messages}
        if functions:
            kwargs["functions"] = functions
        try:
            return client.chat(**kwargs)
        except Exception as e:
            raise LLMRequestError(f"Ошибка запроса к GigaChat: {e}") from e

    @staticmethod
    def _message_text(response) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMRequestError("Пустой ответ GigaChat")
        message = choices[0].message
        return getattr(message, "content", None) or ""

    @staticmethod
    def _message_tool_calls(response) -> list:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return []
        message = choices[0].message
        # У gigachat SDK tool_calls может быть атрибутом или None
        return list(getattr(message, "tool_calls", None) or [])

    def ask(self, messages: list) -> str:
        """
        Отправляет диалог и возвращает финальный текст ответа.

        Если модель просит вызвать инструмент — выполняет его (execute_tool),
        добавляет результат как сообщение role="function" и запрашивает снова,
        до MAX_TOOL_ROUNDS раундов.
        """
        conversation = list(messages)
        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self._ask_once(conversation, functions=TOOL_SPECS)

            tool_calls = self._message_tool_calls(response)
            if not tool_calls:
                return self._message_text(response)

            # Фиксируем сообщение ассистента с tool_calls (требование формата диалога)
            assistant_message = response.choices[0].message
            conversation.append(assistant_message)

            for call in tool_calls:
                call_id = getattr(call, "id", "") or ""
                func = getattr(call, "function", None)
                name = getattr(func, "name", "") or ""
                args_raw = getattr(func, "arguments", None) or "{}"
                try:
                    arguments = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                result = execute_tool(name, arguments)
                conversation.append({
                    "role": "function",
                    "name": name,
                    "content": result,
                })

        # Превышен лимит раундов: возвращаем текст последнего ответа как есть
        return self._message_text(response)

    # --- Небольшие хелперы для совместимости с тестами ---
    @property
    def model(self) -> str:
        return self._model


# --- Фабрика провайдера (по переменным окружения) ---

def is_llm_configured() -> bool:
    """True, если настроен хотя бы один провайдер (сейчас — gigachat)."""
    provider = os.environ.get("LLM_PROVIDER", "gigachat").strip().lower()
    if provider != "gigachat":
        return False
    return bool(os.environ.get("LLM_API_KEY", "").strip())


def get_llm_client() -> BaseLLMClient:
    """
    Возвращает клиент LLM по переменным окружения.

    LLM_PROVIDER=gigachat (по умолчанию), LLM_API_KEY обязателен,
    LLM_MODEL — модель GigaChat (по умолчанию GigaChat).

    Raises:
        LLMNotConfiguredError: если провайдер неизвестен или нет ключа.
    """
    provider = os.environ.get("LLM_PROVIDER", "gigachat").strip().lower()
    api_key = os.environ.get("LLM_API_KEY", "").strip()

    if provider != "gigachat":
        raise LLMNotConfiguredError(
            f"Неизвестный провайдер LLM: {provider}. Поддерживается: gigachat"
        )
    if not api_key:
        raise LLMNotConfiguredError(
            "LLM_API_KEY не задан. Укажите ключ GigaChat в переменных окружения."
        )
    model = os.environ.get("LLM_MODEL", "GigaChat").strip() or "GigaChat"
    return GigaChatClient(api_key=api_key, model=model)
