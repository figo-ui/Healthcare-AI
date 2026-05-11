"""
Gemini API integration for language detection and multilingual response.

Uses Google Gemini (gemini-2.0-flash) for:
  1. Detecting the language of user input (en / am / om + others)
  2. Generating localized medical responses when the LLM triage path
     is unavailable or as a supplementary multilingual layer.

The API key is read from settings.GEMINI_API_KEY (env: GEMINI_API_KEY).
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Circuit breaker ──────────────────────────────────────────────────────
_cb_lock = threading.Lock()
_cb_failures = 0
_cb_open = False
_cb_reset_at = 0.0
_CB_FAILURE_THRESHOLD = 5
_CB_RESET_SECONDS = 120


def _circuit_check() -> bool:
    """Return True if circuit is OPEN (calls should be blocked)."""
    global _cb_open, _cb_reset_at
    with _cb_lock:
        if _cb_open and time.monotonic() >= _cb_reset_at:
            _cb_open = False
            logger.info("Gemini circuit breaker: half-open, allowing probe")
        return _cb_open


def _circuit_record_success() -> None:
    global _cb_failures, _cb_open
    with _cb_lock:
        _cb_failures = 0
        _cb_open = False


def _circuit_record_failure() -> None:
    global _cb_failures, _cb_open, _cb_reset_at
    with _cb_lock:
        _cb_failures += 1
        if _cb_failures >= _CB_FAILURE_THRESHOLD:
            _cb_open = True
            _cb_reset_at = time.monotonic() + _CB_RESET_SECONDS
            logger.warning(
                "Gemini circuit breaker: OPEN after %d failures", _cb_failures
            )


# ── Rate limiter (10 calls / 60 s) ──────────────────────────────────────
_rl_lock = threading.Lock()
_rl_calls = None  # lazy init (collections.deque)


def _rate_limit_ok() -> bool:
    global _rl_calls
    from collections import deque as _deq
    with _rl_lock:
        if _rl_calls is None:
            _rl_calls = _deq()
        now = time.monotonic()
        while _rl_calls and _rl_calls[0] < now - 60:
            _rl_calls.popleft()
        if len(_rl_calls) >= 10:
            return False
        _rl_calls.append(now)
        return True


# ── Core API call ────────────────────────────────────────────────────────

def _gemini_available() -> bool:
    return bool(settings.GEMINI_API_KEY)


def _call_gemini(prompt: str, system_instruction: str = "", max_tokens: int = 512) -> Optional[str]:
    """Make a single Gemini API call. Returns the text response or None."""
    if not _gemini_available():
        return None
    if _circuit_check():
        logger.debug("Gemini circuit open, skipping call")
        return None
    if not _rate_limit_ok():
        logger.debug("Gemini rate limit reached, skipping call")
        return None

    api_key = settings.GEMINI_API_KEY
    model = settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        } if system_instruction else None,
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.1,
        },
    }
    # Remove None fields
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if text.strip():
            _circuit_record_success()
            return text.strip()
        _circuit_record_failure()
        return None
    except Exception as exc:
        logger.warning("Gemini API call failed: %s", exc)
        _circuit_record_failure()
        return None


# ── Language detection ───────────────────────────────────────────────────

# Map of Gemini-detected language names to our supported codes
_LANG_CODE_MAP = {
    "english": "en", "amharic": "am", "oromo": "om", "afaan oromoo": "om",
}


def detect_language_gemini(text: str, preferred: Optional[str] = None) -> Optional[str]:
    """Use Gemini to detect the language of *text*.

    Returns a supported language code (e.g. 'en', 'am', 'om') or None
    if detection fails. Falls back to langdetect if Gemini is unavailable.
    """
    if not text or not text.strip():
        return preferred if preferred in ("en", "am", "om") else None

    # If user has a preferred language set, trust it
    if preferred and preferred.lower() in {"en", "am", "om"}:
        return preferred.lower()

    prompt = (
        f"Detect the language of the following text. "
        f"Reply with ONLY the language name in English: English, Amharic, or Oromo. "
        f"If the text is too short or ambiguous, reply with the most likely language.\n\n"
        f'Text: """{text[:500]}"""'
    )

    result = _call_gemini(prompt, system_instruction="You are a language detection assistant. Reply with only the language name, nothing else.", max_tokens=20)
    if result:
        result_lower = result.strip().lower().rstrip(".")
        code = _LANG_CODE_MAP.get(result_lower)
        if code:
            logger.info("Gemini detected language: %s → %s", result.strip(), code)
            return code
        # Try partial match
        for name, code in _LANG_CODE_MAP.items():
            if name in result_lower or result_lower in name:
                logger.info("Gemini partial match: %s → %s", result.strip(), code)
                return code

    # Fallback to langdetect
    return _detect_langdetect_fallback(text)


def _detect_langdetect_fallback(text: str) -> Optional[str]:
    """Fallback language detection using langdetect library."""
    try:
        from langdetect import detect as ld_detect
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0  # deterministic results
        detected = ld_detect(text)
        _iso_map = {
            "en": "en", "am": "am", "om": "om",
        }
        code = _iso_map.get(detected, detected)
        if code in {"en", "am", "om"}:
            return code
        return "en"  # default fallback
    except Exception as exc:
        logger.debug("langdetect fallback failed: %s", exc)
        return None


# ── Multilingual response generation ────────────────────────────────────

def generate_multilingual_response(
    user_message: str,
    language: str = "en",
    rag_context: str = "",
    session_context: str = "",
) -> Optional[str]:
    """Generate a medical response in the specified language using Gemini.

    This is used as a supplementary path when the primary LLM (GitHub/Azure)
    is unavailable, or when we need guaranteed multilingual quality for
    Amharic, Oromo, and other local languages.
    """
    from .language_support import LANGUAGE_NAMES

    lang_name = LANGUAGE_NAMES.get(language, "English")

    system_instruction = (
        "You are a healthcare AI assistant. Provide helpful, safe medical guidance. "
        "Always include a disclaimer that this is not medical advice. "
        f"You MUST respond entirely in {lang_name}. "
        f"Do not use English unless the language is English. "
        f"Every sentence must be in {lang_name}."
    )

    prompt_parts = []
    if rag_context:
        prompt_parts.append(f"Retrieved medical knowledge:\n{rag_context}")
    if session_context:
        prompt_parts.append(f"Conversation history:\n{session_context}")
    prompt_parts.append(f"Patient message: {user_message}")
    prompt_parts.append(
        f"Respond in {lang_name}. Include: possible causes, risk level assessment, "
        f"recommended next steps, and warning signs if any. "
        f"End with: 'This is not medical advice — please see a doctor.' in {lang_name}."
    )

    prompt = "\n\n".join(prompt_parts)
    return _call_gemini(prompt, system_instruction=system_instruction, max_tokens=1024)


# ── Translation helper ──────────────────────────────────────────────────

def translate_text_gemini(text: str, target_language: str) -> Optional[str]:
    """Translate *text* into *target_language* using Gemini.

    Returns translated text or None on failure.
    """
    from .language_support import LANGUAGE_NAMES
    lang_name = LANGUAGE_NAMES.get(target_language, "English")

    prompt = f'Translate the following text to {lang_name}. Reply with ONLY the translation, nothing else.\n\n"""{text[:1000]}"""'
    return _call_gemini(prompt, system_instruction="You are a professional medical translator. Translate accurately preserving medical terminology.", max_tokens=1024)


# ── Health check ─────────────────────────────────────────────────────────

def gemini_available() -> bool:
    """Return True if Gemini API key is configured and circuit is not open."""
    return _gemini_available() and not _circuit_check()
