"""
Клиент LLM с абстракцией провайдера.

MVP: GigaChat (Сбер, бесплатный тариф) через SDK `gigachat`.
Смена провайдера = смена переменных окружения LLM_PROVIDER/LLM_API_KEY/LLM_MODEL;
SDK импортируется лениво, поэтому приложение работает и без него
(эндпоинт агента тогда отвечает 503).
"""

import os
import threading


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

    GigaChatClient создаётся один раз (singleton в get_llm_client), ключ
    читается из окружения при создании. Авторизация через credentials-токен.
    """

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
                        verify_ssl_certs=False,  # самоподписанный сертификат Сбер
                    )
        return self._client

    def ask(self, messages: list) -> str:
        """
        Отправляет список сообщений [{"role": ..., "content": ...}] и возвращает текст ответа.
        """
        client = self._ensure_client()
        try:
            response = client.chat(messages)
        except Exception as e:
            raise LLMRequestError(f"Ошибка запроса к GigaChat: {e}") from e
        # Формат ответа SDK: {"choices": [{"message": {"content": ...}}]}
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMRequestError("Пустой ответ GigaChat")
        return choices[0].message.content


# --- Фабрика клиента (singleton) ---

_llm_client = None
_llm_lock = threading.Lock()


def is_llm_configured() -> bool:
    """True, если ключ LLM задан (провайдер giga — единственный в MVP)."""
    provider = os.environ.get("LLM_PROVIDER", "gigachat").lower()
    return bool(os.environ.get("LLM_API_KEY", "")) and provider == "gigachat"


def get_llm_client() -> BaseLLMClient:
    """
    Возвращает singleton-клиент LLM по переменным окружения:
      LLM_PROVIDER = gigachat (по умолчанию)
      LLM_API_KEY  = ключ из консоли Сбера
      LLM_MODEL    = GigaChat (по умолчанию)
    Бросает LLMNotConfiguredError, если ключ не задан.
    """
    global _llm_client
    if _llm_client is None:
        with _llm_lock:
            if _llm_client is None:  # double-checked locking
                provider = os.environ.get("LLM_PROVIDER", "gigachat").lower()
                api_key = os.environ.get("LLM_API_KEY", "")
                model = os.environ.get("LLM_MODEL", "GigaChat")
                if not api_key:
                    raise LLMNotConfiguredError(
                        "Ключ LLM не задан. Установите переменную окружения LLM_API_KEY."
                    )
                if provider != "gigachat":
                    raise LLMNotConfiguredError(
                        f"Провайдер '{provider}' не поддерживается в MVP. "
                        "Доступен: gigachat."
                    )
                _llm_client = GigaChatClient(api_key, model)
    return _llm_client
