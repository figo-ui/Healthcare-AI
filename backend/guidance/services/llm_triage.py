import json
import logging
import os
import threading
import time
from collections import deque
from functools import lru_cache
from typing import Dict, List, Optional

from django.conf import settings

from .clinical_safety import apply_prediction_safety_overrides, build_safety_summary
from .schema import response_schema_prompt, validate_triage_response
from .search_router import build_search_prompt_context

logger = logging.getLogger(__name__)

# ── GitHub / Azure AI Inference Configuration ──────────────────────────────
_GITHUB_ENDPOINT = "https://models.github.ai/inference"
_GITHUB_MODEL = "meta/Llama-4-Maverick-17B-128E-Instruct-FP8"

# ── Circuit Breaker ────────────────────────────────────────────────────────
_cb_lock = threading.Lock()
_cb_failures = 0          # consecutive failure count
_cb_open = False          # True = circuit open (LLM calls blocked)
_cb_reset_at = 0.0        # epoch time when circuit resets
_CB_FAILURE_THRESHOLD = 3 # open after this many consecutive failures
_CB_RESET_SECONDS = 60    # seconds before circuit half-opens

# ── Token Bucket Rate Limiter (max 10 calls / 60 s) ───────────────────────
_rl_lock = threading.Lock()
_rl_calls: deque = deque()  # timestamps of recent calls
_RL_MAX_CALLS = 10
_RL_WINDOW_SECONDS = 60


def _circuit_check() -> bool:
    """Return True if the circuit is open (LLM should NOT be called)."""
    global _cb_open, _cb_reset_at
    with _cb_lock:
        if _cb_open and time.monotonic() >= _cb_reset_at:
            _cb_open = False  # half-open: allow one probe
            logger.info("LLM circuit breaker: half-open, allowing probe call")
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
                "LLM circuit breaker OPEN after %d consecutive failures. "
                "Will retry in %ds.", _cb_failures, _CB_RESET_SECONDS
            )


def _rate_limit_check() -> bool:
    """Return True if the rate limit is exceeded (LLM should NOT be called)."""
    now = time.monotonic()
    with _rl_lock:
        # Remove calls outside the window
        while _rl_calls and now - _rl_calls[0] > _RL_WINDOW_SECONDS:
            _rl_calls.popleft()
        if len(_rl_calls) >= _RL_MAX_CALLS:
            logger.warning(
                "LLM rate limit reached (%d calls in %ds window). "
                "Returning fallback.", _RL_MAX_CALLS, _RL_WINDOW_SECONDS
            )
            return True
        _rl_calls.append(now)
        return False

SYSTEM_PROMPT = (
    "You are an expert healthcare triage assistant. Analyze the patient's symptoms carefully and realistically. "
    "Return exactly one JSON object and no prose before or after it. "
    "Do not claim a final diagnosis — always include a disclaimer. "
    "Think like a clinician: consider the full symptom picture, context, and severity. "
    "Predict conditions that ACTUALLY match the symptoms — do not default to common conditions if they don't fit. "
    "If symptoms suggest a drug reaction, allergy, or adverse effect, predict that — not GERD or a cold. "
    "Assign risk_level based on clinical urgency: High = emergency (anaphylaxis, stroke, MI, sepsis), "
    "Medium = needs same-day care (drug reactions, infections, worsening symptoms), Low = monitor at home. "
    "List specific red_flags relevant to the predicted conditions. "
    "Give a clear, actionable recommendation. "
    "Use short, direct condition names and normalized probabilities that sum to 1."
)

SYSTEM_PROMPT_RAG = (
    "You are HealthAI, a friendly and concise healthcare chat assistant. "
    "Use the provided medical knowledge to give a brief, helpful answer. "
    "Write in a conversational tone — like a caring nurse, not a textbook. "
    "Keep responses short (2–4 sentences for greetings, 4–8 sentences for medical questions). "
    "Never use markdown headers (#), bold (**), or numbered lists. "
    "Just plain conversational paragraphs. "
    "Add a short disclaimer only for medical queries: 'This is not a diagnosis — please see a doctor.' "
    "Do not claim a final diagnosis. "
    "Do not say 'I'm not a doctor' — just give the disclaimer once. "
    "CRITICAL: If the user mentions a rash, itching, swelling, or reaction after taking medication, food, or a sting, "
    "always ask: 'Are you having trouble breathing, throat tightness, or lip/tongue swelling?' — "
    "these are signs of anaphylaxis and need emergency care immediately."
)


@lru_cache(maxsize=1)
def _get_client():
    """Create a cached Azure AI Inference client for GitHub Models."""
    token = os.getenv("GITHUB_TOKEN", "") or getattr(settings, "GITHUB_TOKEN", "")
    if not token:
        return None
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential

        client = ChatCompletionsClient(
            endpoint=_GITHUB_ENDPOINT,
            credential=AzureKeyCredential(token),
        )
        return client
    except ImportError:
        logger.warning("azure-ai-inference not installed — LLM triage unavailable")
        return None
    except Exception as exc:
        logger.error("Failed to create LLM client: %s", exc)
        return None


def llm_available() -> bool:
    """Check if the cloud LLM endpoint is reachable (has a token)."""
    client = _get_client()
    return client is not None


def _extract_json_payload(text: str) -> Optional[Dict]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _prompt_language(language: str) -> str:
    """Return a descriptive language instruction for the LLM prompt."""
    _LANG_PROMPTS = {
        "en": "English",
        "am": "Amharic (use Ethiopic script \u1200-\u137F)",
        "om": "Afaan Oromo (Latin script)",
    }
    return _LANG_PROMPTS.get(str(language).lower(), "English")


def _build_rag_context_block(rag_items: List[Dict]) -> str:
    """Format RAG retrieval hits into a context block for the LLM prompt."""
    if not rag_items:
        return ""
    parts = ["--- Retrieved Medical Knowledge ---"]
    for i, item in enumerate(rag_items[:5], 1):
        source = item.get("source", "unknown")
        text = item.get("text", "")[:600]
        score = item.get("score", 0)
        parts.append(f"[{i}] (source: {source}, relevance: {score:.2f})\n{text}")
    parts.append("--- End Retrieved Knowledge ---")
    return "\n\n".join(parts)


def predict_with_llm(
    symptom_text: str,
    top_k: int = 5,
    *,
    language: str = "en",
    search_context: Optional[Dict[str, object]] = None,
    rag_items: Optional[List[Dict]] = None,
) -> Dict:
    """Get structured triage predictions from the cloud LLM.

    Uses GitHub Models (Llama-4-Maverick-17B) via Azure AI Inference SDK.
    RAG context and search results are injected into the prompt so the LLM
    can ground its predictions in retrieved medical knowledge.
    """
    client = _get_client()
    if client is None:
        return {"available": False, "predictions": [], "risk_level": None, "raw_text": ""}

    # Circuit breaker + rate limiter checks
    if _circuit_check():
        logger.warning("LLM circuit breaker open — returning fallback for predict_with_llm")
        return {"available": False, "predictions": [], "risk_level": None, "raw_text": "", "error": "circuit_open"}
    if _rate_limit_check():
        return {"available": False, "predictions": [], "risk_level": None, "raw_text": "", "error": "rate_limited"}

    prompt_sections = [
        SYSTEM_PROMPT,
        f"IMPORTANT: Respond entirely in {_prompt_language(language)}. "
        f"All condition names, recommendations, and red flags must be in {_prompt_language(language)}.",
        "Use this schema exactly:",
        response_schema_prompt(),
    ]
    search_prompt = build_search_prompt_context(search_context)
    if search_prompt:
        prompt_sections.append(search_prompt)

    # Inject RAG knowledge context
    rag_block = _build_rag_context_block(rag_items or [])
    if rag_block:
        prompt_sections.append(
            "Use the following retrieved medical knowledge to inform your predictions:\n\n" + rag_block
        )

    system_content = "\n\n".join(prompt_sections)

    try:
        from azure.ai.inference.models import SystemMessage, UserMessage

        response = client.complete(
            messages=[
                SystemMessage(system_content),
                UserMessage(symptom_text),
            ],
            temperature=0.1,
            top_p=0.9,
            max_tokens=512,
            model=_GITHUB_MODEL,
            timeout=15,
        )
        decoded = response.choices[0].message.content
    except Exception as exc:
        logger.error("LLM API call failed: %s", exc)
        _circuit_record_failure()
        return {
            "available": False,
            "predictions": [],
            "risk_level": None,
            "raw_text": "",
            "error": str(exc),
        }

    payload = _extract_json_payload(decoded)
    if payload is None:
        _circuit_record_failure()
        return {
            "available": False,
            "predictions": [],
            "risk_level": None,
            "raw_text": decoded,
            "error": "Model did not return valid JSON",
        }

    structured = validate_triage_response(payload, raw_text=decoded)
    predictions = [
        {"condition": item.condition, "probability": item.probability}
        for item in structured.predictions
    ]
    predictions = apply_prediction_safety_overrides(symptom_text, predictions, top_k=top_k)
    _circuit_record_success()

    return {
        "available": True,
        "predictions": predictions[:top_k],
        "risk_level": structured.risk_level,
        "red_flags": structured.red_flags,
        "recommendation": structured.recommendation,
        "reasoning": structured.reasoning,
        "raw_text": decoded,
        "safety": build_safety_summary(symptom_text),
        "language": language,
        "search_context": search_context or {},
    }


def generate_rag_response(
    symptom_text: str,
    *,
    language: str = "en",
    rag_items: Optional[List[Dict]] = None,
    search_context: Optional[Dict[str, object]] = None,
    session_context: str = "",
) -> Dict:
    """Generate a natural language response grounded in RAG knowledge.

    This is the core of the restructured flow:
    User → Django → FAISS Retriever → LLM (with RAG context) → Response

    The LLM receives retrieved medical knowledge and produces a comprehensive,
    natural language answer instead of just structured JSON predictions.
    """
    client = _get_client()
    if client is None:
        return {"available": False, "response": "", "raw_text": ""}

    # Circuit breaker + rate limiter checks
    if _circuit_check():
        logger.warning("LLM circuit breaker open — returning fallback for generate_rag_response")
        return {"available": False, "response": "", "raw_text": "", "error": "circuit_open"}
    if _rate_limit_check():
        return {"available": False, "response": "", "raw_text": "", "error": "rate_limited"}

    # Build system prompt with RAG context
    system_sections = [
        SYSTEM_PROMPT_RAG,
        f"IMPORTANT: You MUST respond entirely in {_prompt_language(language)}. "
        f"Do not use English if the language is not English. "
        f"Every sentence of your response must be in {_prompt_language(language)}.",
    ]

    # Inject RAG knowledge
    rag_block = _build_rag_context_block(rag_items or [])
    if rag_block:
        system_sections.append(
            "Here is retrieved medical knowledge relevant to the user's query. "
            "Base your answer on this knowledge:\n\n" + rag_block
        )

    # Inject search context
    search_prompt = build_search_prompt_context(search_context)
    if search_prompt:
        system_sections.append(search_prompt)

    # Inject session context (prior conversation)
    if session_context:
        system_sections.append(
            "Previous conversation context:\n" + session_context
        )

    system_content = "\n\n".join(system_sections)

    try:
        from azure.ai.inference.models import SystemMessage, UserMessage

        response = client.complete(
            messages=[
                SystemMessage(system_content),
                UserMessage(symptom_text),
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=400,
            model=_GITHUB_MODEL,
            timeout=15,
        )
        text = response.choices[0].message.content
    except Exception as exc:
        logger.error("LLM RAG response call failed: %s", exc)
        _circuit_record_failure()
        return {"available": False, "response": "", "raw_text": "", "error": str(exc)}

    _circuit_record_success()
    return {
        "available": True,
        "response": text,
        "raw_text": text,
        "language": language,
        "model": _GITHUB_MODEL,
        "rag_items_used": len(rag_items or []),
    }
