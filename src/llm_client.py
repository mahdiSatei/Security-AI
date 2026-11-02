"""OpenAI-compatible chat client.

Primary path uses the official `openai` SDK; if it is not installed the module
falls back to a stdlib HTTP implementation of /chat/completions, so the same code
runs in Colab, a bare venv or a locked-down machine.

Settings come from the environment, and `.env` in the project root is loaded when
python-dotenv is available:

    LLM_API_KEY      (fallback: OPENAI_API_KEY)
    LLM_BASE_URL     (fallback: OPENAI_BASE_URL, default https://api.openai.com/v1)
    LLM_MODEL        (default gpt-4o-mini)
    LLM_TEMPERATURE  (default 0)
    LLM_TIMEOUT      (default 60 seconds)
    LLM_MAX_TOKENS   (default 800, empty/unset disables the field)
    LLM_JSON_MODE    (default true)

The key is never printed or written anywhere.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:  # python-dotenv is optional; plain environment variables still work
    pass

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 800
DEFAULT_MAX_RETRIES = 2


class LLMError(RuntimeError):
    """Raised when the provider cannot produce an answer."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _settings(api_key=None, base_url=None, model=None) -> Dict[str, Any]:
    key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError(
            "No API key found. Put LLM_API_KEY in .env (or set the environment variable)."
        )
    return {
        "api_key": key,
        "base_url": (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/"),
        "model": model or os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        "temperature": _env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        "timeout": _env_float("LLM_TIMEOUT", DEFAULT_TIMEOUT),
        "max_tokens": _env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        "json_mode": _env_bool("LLM_JSON_MODE", True),
    }


class OpenAIClient:
    """Chat client built on the official openai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        from openai import OpenAI  # imported here so the module works without the SDK

        settings = _settings(api_key, base_url, model)
        self.model = settings["model"]
        self.temperature = settings["temperature"]
        self.max_tokens = settings["max_tokens"]
        self.json_mode = settings["json_mode"]
        self._client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=settings["timeout"],
            max_retries=max_retries,
        )

    def complete(
        self, messages: Sequence[Dict[str, str]], json_mode: Optional[bool] = None
    ) -> str:
        use_json = self.json_mode if json_mode is None else json_mode
        try:
            return self._call(messages, use_json, self.max_tokens)
        except Exception as exc:
            # Some OpenAI-compatible providers reject response_format or max_tokens.
            if _is_bad_request(exc):
                try:
                    return self._call(messages, False, None)
                except Exception as retry_exc:
                    raise LLMError(f"Provider rejected the request: {retry_exc}") from retry_exc
            raise LLMError(f"LLM request failed: {exc}") from exc

    def _call(
        self,
        messages: Sequence[Dict[str, str]],
        json_mode: bool,
        max_tokens: Optional[int],
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content if response.choices else None
        if not content or not str(content).strip():
            raise LLMError("Provider returned an empty message.")
        return str(content)


class HTTPChatClient:
    """Stdlib fallback used when the openai SDK is unavailable."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        settings = _settings(api_key, base_url, model)
        self.api_key = settings["api_key"]
        self.base_url = settings["base_url"]
        self.model = settings["model"]
        self.temperature = settings["temperature"]
        self.timeout = settings["timeout"]
        self.max_tokens = settings["max_tokens"]
        self.json_mode = settings["json_mode"]
        self.max_retries = max_retries

    def complete(
        self, messages: Sequence[Dict[str, str]], json_mode: Optional[bool] = None
    ) -> str:
        use_json = self.json_mode if json_mode is None else json_mode
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._post(messages, use_json)
            except _BadRequest as exc:
                if use_json:
                    use_json = False
                    continue
                raise LLMError(f"Provider rejected the request: {exc}") from exc
            except _Retryable as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)

        raise LLMError(f"LLM request failed after {self.max_retries} retries: {last_error}")

    def _post(self, messages: Sequence[Dict[str, str]], json_mode: bool) -> str:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (400, 404, 422):
                raise _BadRequest(f"HTTP {exc.code}: {detail}") from exc
            if exc.code == 429 or exc.code >= 500:
                raise _Retryable(f"HTTP {exc.code}: {detail}") from exc
            raise LLMError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise _Retryable(f"network error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise _Retryable(f"invalid JSON from provider: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected provider response shape: {payload}") from exc

        if not content or not str(content).strip():
            raise _Retryable("provider returned an empty message")
        return str(content)


class _BadRequest(Exception):
    """The provider rejected the request shape (often response_format)."""


class _Retryable(Exception):
    """Rate limit, server error or network failure."""


def _is_bad_request(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in (400, 404, 422)


def OpenAICompatibleClient(**kwargs) -> Any:
    """Return the SDK client when `openai` is installed, else the stdlib client."""
    try:
        import openai  # noqa: F401
    except ImportError:
        return HTTPChatClient(**kwargs)
    return OpenAIClient(**kwargs)


class FakeClient:
    """Offline stand-in used by tests and dry runs."""

    def __init__(self, replies):
        if isinstance(replies, str):
            replies = [replies]
        self.replies: List[str] = list(replies)
        self.calls = 0

    def complete(self, messages: Sequence[Dict[str, str]], json_mode: Optional[bool] = None) -> str:
        self.calls += 1
        if not self.replies:
            raise LLMError("FakeClient has no canned reply left.")
        return self.replies.pop(0)
