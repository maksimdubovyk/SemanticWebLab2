"""Спільний клієнт Gemini: кеш на диску, паузи між викликами, повторні спроби.

Кеш важливий на безкоштовному рівні: повторний запуск не витрачає квоту, а після
вичерпання денного ліміту пайплайн продовжує з місця зупинки.

Перевірка ключа і моделі:  python llm.py
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

LLM_MODEL = os.getenv("LLM_MODEL") or "gemini-3.5-flash-lite"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "gemini-embedding-001"
EMBED_DIM = 768
MIN_INTERVAL = float(os.getenv("MIN_INTERVAL") or 7)
MAX_ATTEMPTS = 6
RETRYABLE = {429, 500, 502, 503, 504}
CACHE_DIR = ROOT / ".cache"

T = TypeVar("T", bound=BaseModel)
_client: Optional[genai.Client] = None
_last_call = 0.0


class QuotaExhausted(RuntimeError):
    """Денний ліміт вичерпано: повтори в межах запуску нічого не дадуть."""


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Немає GEMINI_API_KEY: скопіюйте .env.example у .env і вставте ключ.")
        _client = genai.Client(api_key=key)
    return _client


def _pace() -> None:
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _with_retry(call):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        _pace()
        try:
            return call()
        except errors.APIError as error:
            message = str(error)
            if error.code == 429 and "PerDay" in message:
                raise QuotaExhausted(
                    f"Денну квоту вичерпано. Кеш збережено: запустіть скрипт пізніше.\n{message}"
                ) from error
            if error.code not in RETRYABLE or attempt == MAX_ATTEMPTS:
                raise
            hinted = re.search(r"retry(?:Delay)?\D{1,12}(\d+)", message, re.IGNORECASE)
            delay = int(hinted.group(1)) + 1 if hinted else 10 * 2 ** (attempt - 1)
            print(f"  API {error.code}: спроба {attempt}/{MAX_ATTEMPTS}, пауза {delay} с")
            time.sleep(delay)


def _cached(kind: str, key_parts: list, produce):
    digest = hashlib.sha256(json.dumps(key_parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / kind / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def generate(prompt: str, temperature: float = 0.2) -> str:
    """Вільна текстова відповідь моделі."""

    def produce() -> str:
        response = _with_retry(lambda: client().models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        ))
        if not response.text:
            raise RuntimeError(f"Порожня відповідь моделі: {response.candidates}")
        return response.text

    return _cached("generate", [LLM_MODEL, prompt, temperature], produce)


def generate_structured(prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
    """Structured output: відповідь гарантовано відповідає pydantic-схемі.

    У кеш потрапляє лише відповідь, що пройшла валідацію.
    """

    def produce() -> dict:
        response = _with_retry(lambda: client().models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        ))
        return schema.model_validate_json(response.text).model_dump(mode="json")

    key = [LLM_MODEL, prompt, temperature, schema.model_json_schema()]
    return schema.model_validate(_cached("structured", key, produce))


def embed(texts: list[str], task_type: str, batch_size: int = 50) -> list[list[float]]:
    """Ембединги з кешем на кожен текст; запитуються лише відсутні в кеші."""

    def path_for(text: str) -> Path:
        key = json.dumps([EMBED_MODEL, EMBED_DIM, task_type, text], ensure_ascii=False)
        return CACHE_DIR / "embed" / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    missing = list(dict.fromkeys(t for t in texts if not path_for(t).exists()))
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        print(f"  ембединги {start + len(batch)}/{len(missing)}")
        response = _with_retry(lambda: client().models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
        ))
        for text, embedding in zip(batch, response.embeddings, strict=True):
            path = path_for(text)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(embedding.values), encoding="utf-8")
    return [json.loads(path_for(t).read_text(encoding="utf-8")) for t in texts]


if __name__ == "__main__":
    print(f"LLM_MODEL={LLM_MODEL}  EMBED_MODEL={EMBED_MODEL}")
    names = [m.name.removeprefix("models/") for m in client().models.list()]
    print("Flash і embedding-моделі, доступні цьому ключу:")
    for name in sorted(n for n in names if "flash" in n or "embedding" in n):
        print(" ", name)
    probe = client().models.generate_content(model=LLM_MODEL, contents="Reply with the single word: ok")
    print(f"{LLM_MODEL}: {probe.text.strip()!r}")
    vector = client().models.embed_content(
        model=EMBED_MODEL, contents="test",
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    ).embeddings[0].values
    print(f"{EMBED_MODEL}: розмірність {len(vector)}")
