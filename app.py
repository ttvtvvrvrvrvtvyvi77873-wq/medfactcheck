"""Flask-бэкенд МедФактЧек.

Проверяет медицинские утверждения: сначала ищет в локальной базе фактов,
затем идёт в GigaChat. Принимает текст и изображения (OCR — Tesseract).
Держит очередь запросов, кэш ответов и фоновую автомодерацию фактов.

Содержимое базы наружу не отдаётся: /api/facts возвращает только счётчик.
Текст пользовательских запросов в кэш и pending не пишется — только хеши.
Является ли запрос медицинским — решает GigaChat по промпту.
Ссылки от GigaChat проверяются HTTP-запросом, мёртвые не отдаются.
"""

import concurrent.futures
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import requests
import urllib3
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix


load_dotenv()

AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
CLIENT_ID = os.getenv("GIGACHAT_CLIENT_ID")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
MODERATION_ENABLED = os.getenv("MODERATION_ENABLED", "True").lower() == "true"
MODERATION_MIN_CONFIDENCE = int(os.getenv("MODERATION_MIN_CONFIDENCE", "70"))
GIGACHAT_CONCURRENCY = max(1, int(os.getenv("GIGACHAT_CONCURRENCY", "1")))
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "feedback@medfactcheck.ru")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

if not AUTH_KEY:
    raise RuntimeError("Не задан GIGACHAT_AUTH_KEY. Проверь .env рядом с app.py.")

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

# У Сбера самоподписанный корневой сертификат, иначе requests ругается на каждый вызов
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = Path(__file__).parent

RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "10"))
RATE_LIMIT_WINDOW = 60
CACHE_TTL = 60 * 60 * 24

CACHE_FILE = BASE_DIR / "cache.json"
PENDING_FACTS_FILE = BASE_DIR / "pending_facts.json"
FACTS_FILE = BASE_DIR / "facts.json"
MODERATION_LOG_FILE = BASE_DIR / "moderation_log.json"
DEFERRED_FACTS_FILE = BASE_DIR / "deferred_facts.json"
GAME_QUESTIONS_FILE = BASE_DIR / "game_questions.json"
SYSTEM_PROMPT_FILE = BASE_DIR / "system_prompt.txt"

DEFAULT_SYSTEM_PROMPT = "Ты — медицинский фактчекер. Отвечай строго в формате JSON."

# Проверка живости ссылок от GigaChat. Range не используем — часть серверов
# на него отвечает 416 или рвёт соединение. Timeout и сетевые ошибки считаем
# «неизвестно» и оставляем ссылку, чтобы не резать медленные PubMed и Cochrane.
URL_CHECK_TIMEOUT = 6
URL_CHECK_UA = "Mozilla/5.0 (compatible; MedFactCheck/1.0)"


def read_json(path: Path, default):
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as exc:
        print(f"[WARN] не удалось прочитать {path.name}: {exc}")
        return default


def write_json(path: Path, data) -> None:
    # Пишем через временный файл и os.replace — иначе при сбое оставим обрезанный JSON
    try:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
    except Exception as exc:
        print(f"[ERROR] не удалось записать {path.name}: {exc}")


_facts_lock = threading.RLock()


def load_facts_db():
    return read_json(
        FACTS_FILE,
        {"facts": [], "trusted_domains": [], "sources": [], "contact_email": CONTACT_EMAIL},
    )


FACTS_DB = load_facts_db()


def get_facts_db():
    with _facts_lock:
        return FACTS_DB


def set_facts_db(new_db):
    global FACTS_DB
    with _facts_lock:
        FACTS_DB = new_db


try:
    SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def anonymize_text(text: str, max_len: int = 20) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= max_len else text[:max_len] + "..."


def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def clean_json_response(ai_text: str) -> str:
    ai_text = ai_text.strip()
    if ai_text.startswith("```"):
        ai_text = ai_text.split("```")[1]
        if ai_text.startswith("json"):
            ai_text = ai_text[4:]
    return ai_text.strip()


def parse_json_tolerant(text: str) -> dict:
    """GigaChat иногда вставляет сырые control-символы (\n, \t, \r) внутрь
    строковых значений JSON. По RFC 8259 их там быть не должно, и json.loads
    падает с Invalid control character. Заменяем их на пробел и пробуем
    ещё раз. Текст explanation схлопнется в одну строку — для HTML это
    визуально то же самое."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)
        return json.loads(cleaned)


def is_garbage_claim(claim: str) -> bool:
    """Отсеивает OCR-мусор: слишком короткие строки, наборы коротких
    заглавных латинских токенов, строки без кириллицы. Это не про
    медицину — про то, что Tesseract может вернуть белиберду, и её
    незачем гнать в GigaChat и в базу."""
    stripped = claim.strip()
    if len(stripped) < 15:
        return True

    tokens = stripped.split()
    if len(tokens) < 3:
        return True

    upper_junk = sum(
        1 for t in tokens
        if t.isupper() and t.isascii() and 2 <= len(t) <= 6
    )
    if upper_junk / len(tokens) > 0.20:
        return True

    cyrillic = sum(1 for c in stripped if 'а' <= c.lower() <= 'я' or c.lower() == 'ё')
    if len(stripped) > 60 and cyrillic / len(stripped) < 0.45:
        return True

    return False


# Пул параллельных запросов к GigaChat. GIGACHAT_CONCURRENCY = 1 на
# бесплатном тарифе PERS; на платном можно поднять. Семафор, а не Lock —
# чтобы при увеличении лимита всё работало без правок.
_gigachat_semaphore = threading.Semaphore(GIGACHAT_CONCURRENCY)


def gigachat_post(headers, body, timeout=30):
    with _gigachat_semaphore:
        response = requests.post(CHAT_URL, headers=headers, json=body, verify=False, timeout=timeout)
        response.raise_for_status()
        return response


_token_cache = {"token": None, "expires_at": 0}
_token_lock = threading.Lock()


def get_access_token() -> str:
    with _token_lock:
        if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
            return _token_cache["token"]

        response = requests.post(
            OAUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {AUTH_KEY}",
            },
            data={"scope": "GIGACHAT_API_PERS"},
            verify=False,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        _token_cache["token"] = payload["access_token"]
        _token_cache["expires_at"] = payload["expires_at"] / 1000
        expires = time.strftime("%H:%M:%S", time.localtime(_token_cache["expires_at"]))
        print(f"[AUTH] токен обновлён до {expires}")
        return payload["access_token"]


def ensure_protocol(url: str) -> str:
    """GigaChat иногда отдаёт домен без схемы. Добавляем https://."""
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return url
    return "https://" + url.lstrip("/")


def filter_trusted_urls(urls):
    """Оставляем только ссылки с доверенных доменов. Никаких фолбэков:
    если GigaChat не дал валидных ссылок, пользователь увидит «Источники
    не указаны» — это честнее, чем вести его на пустой поиск."""
    trusted = get_facts_db().get("trusted_domains", [])
    normalized = [ensure_protocol(u) for u in (urls or []) if isinstance(u, str)]
    return [u for u in normalized if any(d in u for d in trusted)]


def url_is_alive(url: str) -> bool:
    """True, если ссылка не отдаёт явный 4xx.

    Timeout и сетевые ошибки считаем «неизвестно» и ссылку оставляем —
    иначе режем живые, но медленные PubMed, Cochrane и ВОЗ. Range-запрос
    не используем: некоторые серверы на него отвечают 416 или рвут
    соединение, и валидная страница улетает в мусор."""
    try:
        response = requests.head(
            url,
            timeout=URL_CHECK_TIMEOUT,
            allow_redirects=True,
            headers={"User-Agent": URL_CHECK_UA},
        )
        # HEAD не всегда поддерживается — тогда повторяем через GET
        if response.status_code in (403, 405, 501):
            response = requests.get(
                url,
                timeout=URL_CHECK_TIMEOUT,
                allow_redirects=True,
                stream=True,
                headers={"User-Agent": URL_CHECK_UA},
            )
            response.close()

        return response.status_code < 400
    except requests.exceptions.Timeout:
        return True
    except requests.exceptions.RequestException:
        return True


def filter_alive_urls(urls):
    """Параллельно проверяет ссылки. GigaChat иногда склеивает правдоподобный
    URL из шаблона, и он ведёт в 404. Живость проверяем, но медленные и
    недоступные по сети страницы не выбрасываем."""
    if not urls:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as pool:
        checks = list(pool.map(url_is_alive, urls))
    return [u for u, ok in zip(urls, checks) if ok]


def call_gigachat(text: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }
    body = {
        "model": "GigaChat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Проверь утверждение: {text}"},
        ],
        "temperature": 0.2,
        "max_tokens": 2500,
        "tools": [{"type": "web_search"}],
    }

    last_error = None
    for attempt in range(2):
        try:
            response = gigachat_post(headers, body, timeout=30)
            raw = clean_json_response(response.json()["choices"][0]["message"]["content"])
            parsed = parse_json_tolerant(raw)
            if isinstance(parsed.get("sources"), list):
                raw_sources = parsed["sources"]
                trusted = filter_trusted_urls(raw_sources)
                alive = filter_alive_urls(trusted)
                print(f"[SOURCES] от GigaChat: {len(raw_sources)} | "
                      f"после whitelist: {len(trusted)} | "
                      f"после проверки: {len(alive)}")
                if raw_sources and not alive:
                    print(f"[SOURCES] было: {raw_sources}")
                parsed["sources"] = alive
            return parsed
        except Exception as exc:
            last_error = exc
            print(f"[GIGACHAT] попытка {attempt + 1} не удалась: {exc}")
            if attempt < 1:
                time.sleep(4)

    raise last_error


_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()


def check_rate_limit(ip: str):
    now = time.time()
    with _rate_limit_lock:
        for stale_ip in [k for k, v in _rate_limit_store.items() if not v or now - v[-1] > RATE_LIMIT_WINDOW]:
            _rate_limit_store.pop(stale_ip, None)

        history = [ts for ts in _rate_limit_store.get(ip, []) if now - ts < RATE_LIMIT_WINDOW]
        if len(history) >= RATE_LIMIT_MAX:
            wait = int(RATE_LIMIT_WINDOW - (now - min(history))) + 1
            return False, wait

        history.append(now)
        _rate_limit_store[ip] = history
        return True, 0


_queue_semaphore = threading.Semaphore(1)
_queue_waiting = 0
_queue_total = 0
_queue_lock = threading.Lock()


def queue_enter():
    global _queue_waiting
    with _queue_lock:
        _queue_waiting += 1

    if not _queue_semaphore.acquire(timeout=120):
        with _queue_lock:
            _queue_waiting = max(0, _queue_waiting - 1)
        raise TimeoutError("слишком долгое ожидание в очереди")


def queue_leave() -> None:
    global _queue_waiting, _queue_total
    with _queue_lock:
        _queue_waiting = max(0, _queue_waiting - 1)
        _queue_total += 1
    _queue_semaphore.release()


def queue_snapshot() -> dict:
    with _queue_lock:
        position = max(0, _queue_waiting - 1)
        return {
            "position": position,
            "waitSeconds": position * 5,
            "totalProcessed": _queue_total,
        }


_cache_lock = threading.Lock()


def cache_get(text: str):
    key = hash_text(text)
    with _cache_lock:
        cache = read_json(CACHE_FILE, {})
        entry = cache.get(key)
        if not entry:
            return None
        if time.time() - entry.get("timestamp", 0) > CACHE_TTL:
            cache.pop(key, None)
            write_json(CACHE_FILE, cache)
            return None
        return entry.get("result")


def cache_set(text: str, result: dict) -> None:
    # Текст запроса не сохраняем — только хеш и результат.
    key = hash_text(text)
    with _cache_lock:
        cache = read_json(CACHE_FILE, {})
        cache[key] = {"result": result, "timestamp": time.time()}
        cutoff = time.time() - CACHE_TTL
        cache = {k: v for k, v in cache.items() if v.get("timestamp", 0) > cutoff}
        write_json(CACHE_FILE, cache)


def how_to_verify_for(url: str) -> str:
    if not url:
        return "Проверьте это утверждение в нескольких авторитетных источниках или обратитесь к врачу."

    hints = {
        "who.int": "Зайдите на who.int, введите в поиск ключевые слова, откройте актуальный факт-лист ВОЗ и сравните данные.",
        "nccih.nih.gov": "Откройте nccih.nih.gov, найдите раздел по теме, посмотрите «What the Science Says».",
        "ods.od.nih.gov": "Откройте ods.od.nih.gov, найдите факт-лист по веществу, посмотрите раздел о доказательствах.",
        "pubmed.ncbi.nlm.nih.gov": "Откройте pubmed.ncbi.nlm.nih.gov, введите ключевые слова, посмотрите систематические обзоры и мета-анализы.",
        "msdmanuals.com": "Откройте msdmanuals.com/ru, найдите раздел по теме, посмотрите «Диагностика и лечение».",
        "nhs.uk": "Откройте nhs.uk, введите в поиск ключевые слова, прочитайте «Overview» и «Treatment».",
        "cdc.gov": "Откройте cdc.gov, введите в поиск ключевые слова, посмотрите «Basics» и «Prevention».",
        "nhlbi.nih.gov": "Откройте nhlbi.nih.gov, найдите раздел по теме, посмотрите «Health Topics».",
        "cancer.gov": "Откройте cancer.gov, введите в поиск ключевые слова, посмотрите «Patient Version».",
        "rlsnet.ru": "Откройте rlsnet.ru, найдите препарат или вещество, посмотрите «Показания» и «Противопоказания».",
        "vidal.ru": "Откройте vidal.ru, найдите препарат или вещество, посмотрите «Клинические исследования».",
        "minzdrav.gov.ru": "Откройте minzdrav.gov.ru, введите в поиск ключевые слова, посмотрите официальные разъяснения.",
        "cr.minzdrav.gov.ru": "Откройте cr.minzdrav.gov.ru, найдите клиническую рекомендацию по теме, посмотрите «Диагностика».",
        "nice.org.uk": "Откройте nice.org.uk, найдите guideline по теме, посмотрите «Recommendations».",
        "cochranelibrary.com": "Откройте cochranelibrary.com, введите ключевые слова, посмотрите систематические обзоры Cochrane.",
        "medlineplus.gov": "Откройте medlineplus.gov, введите в поиск ключевые слова, прочитайте «Summary» и «Learn More».",
        "almazovcentre.ru": "Откройте almazovcentre.ru, посмотрите раздел с публикациями Центра Алмазова.",
    }
    for domain, hint in hints.items():
        if domain in url:
            return hint
    return "Перейдите по ссылке в источнике, найдите раздел по теме и сравните данные с утверждением."


def search_in_facts(text: str):
    """Ищет наиболее близкий факт. Порог совпадений зависит от длины запроса.
    Записи без url пропускаем — это старые заглушки, они портят ответ."""
    words = [w for w in normalize(text).split() if len(w) > 3] or \
            [w for w in normalize(text).split() if len(w) > 2]
    if not words:
        return None

    best, best_score = None, 0
    for fact in get_facts_db().get("facts", []):
        claim_norm = normalize(fact.get("claim", ""))
        if not claim_norm:
            continue
        score = sum(1 for w in words if w in claim_norm.split())
        if score > best_score:
            best, best_score = fact, score

    if not best:
        return None

    n = len(words)
    if n <= 2:
        min_score = n
    elif n <= 4:
        min_score = max(2, n - 1)
    else:
        min_score = max(2, int(n * 0.5))

    if best_score < min_score:
        return None

    # Записи без url — это старые заглушки, им не место в ответе.
    # Лучше отправить запрос в GigaChat, чем показывать «источник неизвестен».
    if not best.get("url"):
        return None

    url = best["url"]
    if best["verdict"] == "confirmed":
        explanation = (
            f"Утверждение «{best['claim']}» подтверждается авторитетными источниками. "
            f"Согласно {best['source']}, это соответствует принципам доказательной медицины."
        )
    elif best["verdict"] == "refuted":
        explanation = (
            f"Утверждение «{best['claim']}» противоречит авторитетным источникам. "
            f"По данным {best['source']}, это утверждение классифицируется как недостоверное."
        )
    else:
        explanation = (
            f"По утверждению «{best['claim']}» недостаточно данных для однозначного вывода. "
            f"Источник {best['source']} указывает, что требуется дополнительная проверка."
        )

    return {
        "verdict": best["verdict"],
        "confidence": best.get("confidence", 90),
        "bottom_line": f"Найдено в базе знаний: {best['source']}.",
        "explanation": explanation,
        "sources": [url],
        "how_to_verify": how_to_verify_for(url),
        "_from": "facts",
    }


_moderation_lock = threading.Lock()


def heuristic_quality_check(claim, verdict, confidence, sources, explanation, bottom_line):
    if not explanation or len(explanation.strip()) < 80:
        return False, "Уровень 4: слишком короткое объяснение (< 80 символов)"

    lowered = explanation.lower()
    for phrase in ("не знаю", "не уверен", "возможно, это", "может быть, это",
                   "я не могу", "затрудняюсь", "не имею информации"):
        if phrase in lowered:
            return False, f"Уровень 4: неуверенная формулировка «{phrase}»"

    if not bottom_line or len(bottom_line.strip()) < 10:
        return False, "Уровень 4: пустой или слишком короткий краткий вывод"

    if not any(isinstance(u, str) and u.startswith(("http://", "https://")) for u in sources or []):
        return False, "Уровень 4: нет валидных URL в источниках"

    if verdict == "confirmed" and confidence < 80:
        return False, f"Уровень 4: confirmed требует уверенности >= 80 (получено {confidence}%)"
    if verdict == "refuted" and confidence < 75:
        return False, f"Уровень 4: refuted требует уверенности >= 75 (получено {confidence}%)"
    if verdict == "insufficient" and confidence < 50:
        return False, f"Уровень 4: insufficient требует уверенности >= 50 (получено {confidence}%)"

    return True, "Уровень 4: все эвристические проверки пройдены"


def is_duplicate_in_facts(claim: str) -> bool:
    claim_norm = normalize(claim)
    for fact in get_facts_db().get("facts", []):
        existing = normalize(fact.get("claim", ""))
        if claim_norm == existing:
            return True
        new_words, old_words = set(claim_norm.split()), set(existing.split())
        if new_words and old_words:
            overlap = len(new_words & old_words) / max(len(new_words), len(old_words))
            if overlap > 0.9:
                return True
    return False


def log_moderation(claim: str, decision: str, reason: str, detail=None) -> None:
    log = read_json(MODERATION_LOG_FILE, {"entries": []})
    log["entries"].append({
        "timestamp": datetime.now().isoformat(),
        "claim": claim[:200],
        "decision": decision,
        "reason": reason,
        "detail": detail,
    })
    log["entries"] = log["entries"][-500:]
    write_json(MODERATION_LOG_FILE, log)


def save_deferred_fact(claim: str, result: dict, reason: str) -> None:
    """Внутренний файл для ручного разбора. Не публикуется, не отдаётся через API."""
    deferred = read_json(DEFERRED_FACTS_FILE, {"deferred": []})
    if any(item.get("claim", "").lower() == claim.lower() for item in deferred["deferred"]):
        return
    deferred["deferred"].append({
        "claim": claim,
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "sources": result.get("sources", []),
        "explanation": result.get("explanation", ""),
        "how_to_verify": result.get("how_to_verify", ""),
        "reason": reason,
        "deferred_at": datetime.now().isoformat(),
        "status": "pending_manual_review",
    })
    deferred["deferred"] = deferred["deferred"][-100:]
    write_json(DEFERRED_FACTS_FILE, deferred)


def moderate_and_add(text: str, result: dict) -> None:
    """Прогоняет ответ GigaChat через фильтры и добавляет факт в базу.
    Автопубликация в facts.json работает как раньше — содержимое просто
    не выводится на сайте и не отдаётся через /api/facts."""
    if not MODERATION_ENABLED:
        return

    with _moderation_lock:
        claim = text.strip()
        verdict = result.get("verdict")
        confidence = result.get("confidence", 0)
        sources = result.get("sources", [])
        explanation = result.get("explanation", "")
        bottom_line = result.get("bottom_line", "")
        how_to_verify = result.get("how_to_verify", "")

        if len(claim) < 15:
            log_moderation(claim, "rejected", "Уровень 1: текст слишком короткий")
            return
        if is_garbage_claim(claim):
            log_moderation(claim, "rejected", "Уровень 1: OCR-мусор или обрывок текста")
            return
        if verdict not in ("confirmed", "refuted", "insufficient"):
            log_moderation(claim, "rejected", f"Уровень 1: невалидный вердикт {verdict}")
            return
        if confidence < MODERATION_MIN_CONFIDENCE:
            log_moderation(claim, "rejected", f"Уровень 1: низкая уверенность {confidence}%")
            return
        if not sources:
            log_moderation(claim, "rejected", "Уровень 2: нет источников")
            return
        if is_duplicate_in_facts(claim):
            log_moderation(claim, "duplicate", "Уровень 3: такой факт уже есть в базе")
            return

        print(f"[MODERATION] проверка: {anonymize_text(claim)}")
        passed, reason = heuristic_quality_check(
            claim, verdict, confidence, sources, explanation, bottom_line
        )
        if not passed:
            log_moderation(claim, "rejected", reason)
            save_deferred_fact(claim, result, reason)
            print(f"[MODERATION] отложен: {reason}")
            return

        new_fact = {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "source": "Автомодерация (GigaChat)",
            "url": sources[0] if sources else "",
            "bottom_line": bottom_line,
            "explanation": explanation,
            "how_to_verify": how_to_verify,
            "added_by": "auto_moderation",
            "added_at": datetime.now().isoformat(),
        }

        with _facts_lock:
            db = read_json(FACTS_FILE, {"facts": [], "trusted_domains": [], "sources": []})
            db["facts"].append(new_fact)
            write_json(FACTS_FILE, db)
            set_facts_db(db)

        log_moderation(claim, "added", "Добавлено после успешной проверки качества")
        print(f"[MODERATION] факт добавлен: {anonymize_text(claim)}")


def save_pending_fact(text: str, result: dict) -> None:
    # Храним только хеш — пользовательский текст не сохраняем.
    entry_hash = hash_text(text)
    pending = read_json(PENDING_FACTS_FILE, {"pending": []})
    if any(item.get("hash") == entry_hash for item in pending["pending"]):
        return
    pending["pending"].append({
        "hash": entry_hash,
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "saved_at": datetime.now().isoformat(),
    })
    pending["pending"] = pending["pending"][-200:]
    write_json(PENDING_FACTS_FILE, pending)


TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
)


def recognize_text_from_image(file_storage) -> str:
    import pytesseract
    from PIL import Image

    raw = file_storage.read()
    if not raw:
        return ""
    image = Image.open(io.BytesIO(raw))

    for path in TESSERACT_PATHS:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

    return pytesseract.image_to_string(image, lang="rus+eng").strip()


def ocr_quality(text: str):
    if not text or len(text.strip()) < 5:
        return "poor", "Текст не распознан или слишком короткий"

    total = len(text)
    letters = sum(1 for c in text if c.isalpha())
    garbage = sum(1 for c in text if not c.isalnum() and c not in " .,!?;:-—–()«»\"'\n\r\t")

    if letters / total < 0.5:
        return "poor", "В тексте мало букв — возможно, плохое качество фото"
    if garbage / total > 0.15:
        return "poor", "В тексте много посторонних символов"
    return "good", ""


app = Flask(__name__, static_folder=".")
# За туннелем (CloudPub, ngrok, Cloudflare) реальный IP приходит
# в X-Forwarded-For, а не в request.remote_addr.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
CORS(app, origins=CORS_ORIGINS)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "facts_in_base": len(get_facts_db().get("facts", [])),
        "moderation_enabled": MODERATION_ENABLED,
        "gigachat_concurrency": GIGACHAT_CONCURRENCY,
        "queue": queue_snapshot(),
        "contact_email": CONTACT_EMAIL,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/facts")
def list_facts():
    facts = get_facts_db().get("facts", [])
    return jsonify({"total": len(facts)})


@app.route("/api/check", methods=["POST"])
def check():
    ip = request.remote_addr or "unknown"
    allowed, wait = check_rate_limit(ip)
    if not allowed:
        return jsonify({
            "verdict": "insufficient",
            "confidence": 0,
            "bottom_line": f"Слишком много запросов. Подождите {wait} сек.",
            "explanation": f"Rate limit: не более {RATE_LIMIT_MAX} запросов в минуту с одного IP.",
            "sources": [],
            "how_to_verify": "",
        }), 429

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Пустой текст"}), 400
    text = text[:5000]

    print(f"[CHECK] запрос: {anonymize_text(text)}")

    try:
        queue_enter()
    except TimeoutError:
        return jsonify({
            "verdict": "insufficient", "confidence": 0,
            "bottom_line": "Очередь переполнена.",
            "explanation": "Слишком много одновременных запросов. Попробуйте через минуту.",
            "sources": [], "how_to_verify": "",
        }), 503

    try:
        cached = cache_get(text)
        if cached:
            print("[CHECK] из кэша")
            cached["_from_cache"] = True
            return jsonify(cached)

        from_facts = search_in_facts(text)
        if from_facts:
            print(f"[CHECK] из базы: {from_facts['verdict']}")
            cache_set(text, from_facts)
            return jsonify(from_facts)

        try:
            print("[CHECK] GigaChat с web_search")
            result = call_gigachat(text)
            if "verdict" in result and "confidence" in result:
                cache_set(text, result)
                save_pending_fact(text, result)
                threading.Thread(target=moderate_and_add, args=(text, result), daemon=True).start()
                return jsonify(result)
            raise ValueError("неполный ответ GigaChat")
        except Exception as exc:
            print(f"[CHECK] ошибка GigaChat: {exc}")
            traceback.print_exc()

        return jsonify({
            "verdict": "insufficient", "confidence": 40,
            "bottom_line": "Недостаточно данных для однозначного вывода.",
            "explanation": "Мы не нашли подтверждений или опровержений в авторитетных источниках.",
            "sources": [],
            "how_to_verify": "Обратитесь к врачу или проверьте в нескольких источниках.",
        })
    finally:
        queue_leave()


@app.route("/api/check-image", methods=["POST"])
def check_image():
    if "image" not in request.files:
        return jsonify({"error": "Файл не получен"}), 400

    file = request.files["image"]

    try:
        recognized = recognize_text_from_image(file)
    except ImportError:
        return jsonify({
            "verdict": "insufficient", "confidence": 0,
            "bottom_line": "OCR не установлен.",
            "explanation": "Установите Tesseract (см. README).",
            "sources": [], "how_to_verify": "",
            "recognized_text": "",
        })
    except Exception as exc:
        return jsonify({
            "verdict": "insufficient", "confidence": 0,
            "bottom_line": "Ошибка распознавания.",
            "explanation": f"Не удалось распознать текст: {exc}",
            "sources": [], "how_to_verify": "",
            "recognized_text": "",
        })

    if not recognized or len(recognized) < 5:
        return jsonify({
            "verdict": "insufficient", "confidence": 30,
            "bottom_line": "Не удалось распознать текст на изображении.",
            "explanation": "Загрузите более чёткое фото.",
            "sources": [], "how_to_verify": "",
            "recognized_text": recognized or "",
        })

    print(f"[OCR] распознано: {anonymize_text(recognized)}")
    quality, quality_reason = ocr_quality(recognized)

    from_facts = search_in_facts(recognized)
    if from_facts:
        result = dict(from_facts)
    else:
        try:
            result = call_gigachat(recognized)
            threading.Thread(target=moderate_and_add, args=(recognized, result), daemon=True).start()
        except Exception as exc:
            print(f"[OCR] ошибка GigaChat: {exc}")
            return jsonify({
                "verdict": "insufficient", "confidence": 40,
                "bottom_line": "Не удалось проверить распознанный текст.",
                "explanation": f"Ошибка: {exc}",
                "sources": [], "how_to_verify": "",
                "recognized_text": recognized,
                "ocr_quality": quality,
            })

    result["recognized_text"] = recognized
    result["ocr_quality"] = quality
    result["ocr_quality_reason"] = quality_reason
    return jsonify(result)


@app.route("/api/queue/status")
def queue_status_route():
    return jsonify(queue_snapshot())


@app.route("/api/cache/stats")
def cache_stats():
    cache = read_json(CACHE_FILE, {})
    pending = read_json(PENDING_FACTS_FILE, {"pending": []})
    log = read_json(MODERATION_LOG_FILE, {"entries": []})
    deferred = read_json(DEFERRED_FACTS_FILE, {"deferred": []})

    entries = log.get("entries", [])
    return jsonify({
        "cached_items": len(cache),
        "pending_facts": len(pending.get("pending", [])),
        "total_processed": _queue_total,
        "facts_in_base": len(get_facts_db().get("facts", [])),
        "auto_added": sum(1 for e in entries if e.get("decision") == "added"),
        "auto_rejected": sum(1 for e in entries if e.get("decision") == "rejected"),
        "deferred_facts": len(deferred.get("deferred", [])),
    })


@app.route("/api/game/questions")
def game_questions():
    try:
        data = json.loads(GAME_QUESTIONS_FILE.read_text(encoding="utf-8"))
        return jsonify({"questions": data.get("questions", [])})
    except Exception as exc:
        print(f"[GAME] не удалось загрузить вопросы: {exc}")
        return jsonify({"questions": []})


if __name__ == "__main__":
    print("=" * 64)
    print("  МедФактЧек — бэкенд 7.7")
    print(f"  Адрес: http://{HOST}:{PORT}")
    print(f"  Rate limit: {RATE_LIMIT_MAX} / {RATE_LIMIT_WINDOW} сек")
    print(f"  Кэш: {CACHE_TTL // 3600} ч")
    print(f"  Автомодерация: {'вкл' if MODERATION_ENABLED else 'выкл'} (порог {MODERATION_MIN_CONFIDENCE}%)")
    print(f"  Параллельных запросов к GigaChat: {GIGACHAT_CONCURRENCY}")
    print(f"  Фактов в базе: {len(get_facts_db().get('facts', []))}")
    print("=" * 64)
    app.run(debug=DEBUG, port=PORT, host=HOST)