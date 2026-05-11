import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib
from django.conf import settings

from .preprocess import clean_symptom_text


# Risk-level framing — minimal, dynamic phrases derived from risk severity
_RISK_FRAMES = {
    "high": "These symptoms need prompt attention. ",
    "medium": "Let me help you understand what's going on. ",
    "low": "Let me take a look at what might be going on. ",
}


@lru_cache(maxsize=1)
def _load_dialogue_artifacts():
    model_path = Path(settings.DIALOGUE_INTENT_MODEL_PATH)
    vectorizer_path = Path(settings.DIALOGUE_INTENT_VECTORIZER_PATH)
    templates_path = Path(settings.DIALOGUE_RESPONSE_TEMPLATES_PATH)
    consolidation_path = Path(getattr(settings, "DIALOGUE_INTENT_CONSOLIDATION_PATH", ""))

    model = None
    vectorizer = None
    templates: Dict[str, str] = {}
    consolidation: Dict[str, str] = {}

    if model_path.exists() and vectorizer_path.exists():
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

    if templates_path.exists():
        templates = json.loads(templates_path.read_text(encoding="utf-8"))

    if consolidation_path and consolidation_path.exists():
        consolidation = json.loads(consolidation_path.read_text(encoding="utf-8"))

    return model, vectorizer, templates, consolidation


def build_supportive_opening(symptom_text: str, risk_level: str) -> str:
    """Generate a supportive opening using curated medical knowledge (RAG).

    Only draws from authoritative sources (medquad, chatbot, grok) to ensure
    the opening provides genuine medical context — never raw data or
    low-quality matches.
    """
    from .rag import query_rag

    level = str(risk_level or "").lower()
    frame = _RISK_FRAMES.get(level, _RISK_FRAMES["low"])

    # Only use high-quality sources for the opening statement
    _OPENING_SOURCES = {"medquad", "kaggle_chatbot", "grok_dialogue", "grok_supervised"}

    rag_hits = query_rag(symptom_text, top_k=8)

    for hit in rag_hits:
        source = hit.get("source", "")
        if source not in _OPENING_SOURCES:
            continue
        answer = (hit.get("metadata") or {}).get("answer", "").strip()
        question = (hit.get("metadata") or {}).get("question", "").strip()
        score = hit.get("score", 0)

        if answer and score > 0.20:
            snippet = answer[:300].rstrip()
            if len(answer) > 300:
                snippet += "…"
            if question:
                return f"{frame}Regarding your concern — **{question}**\n{snippet}"
            return f"{frame}{snippet}"

    # Fallback: risk-level frame only
    return frame