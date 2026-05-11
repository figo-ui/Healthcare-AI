import csv
import io
import json
import logging
import random
import re
import time
from typing import Optional

import numpy as np

from django.conf import settings
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Count, Max, Min, Q
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.utils.crypto import constant_time_compare
from django.utils.dateparse import parse_date
from django.core.exceptions import ValidationError
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import (
    AuditLog,
    CaseSubmission,
    ChatMessage,
    ChatSession,
    HealthcareFacility,
    UserProfile,
)
from .serializers import (
    AdminUserUpdateSerializer,
    AnalyzeCaseSerializer,
    ChatMessageSerializer,
    ChatSessionSerializer,
    FacilitySearchSerializer,
    HealthcareFacilitySerializer,
    LoginSerializer,
    ProfileUpdateSerializer,
    RegisterSerializer,
    UserProfileSerializer,
    UserSerializer,
)
from .services.facilities import emergency_contacts, lookup_nearby_facilities
from .services.async_inference import async_case_status, submit_async_case_analysis
from .services.pipeline import run_case_analysis
from .services.dialogue_style import build_supportive_opening
from .services.preprocess import clean_symptom_text
from .services.language_support import build_assistant_summary, detect_language
from .services.pii_redaction import redact_phi_text
from .services.email_service import (
    send_emergency_alert_email,
    send_emergency_contact_alert,
    send_verification_email,
)
from .throttles import AnalyzeAnonRateThrottle, AnalyzeRateThrottle, AuthRateThrottle
from .authentication import CookieJWTAuthentication
import django_rq

logger = logging.getLogger(__name__)
GENERIC_CONDITION_RE = re.compile(r"^(condition\s+\d+|class_\d+)$", re.IGNORECASE)


def _tokens_for_user(user: User):
    refresh = RefreshToken.for_user(user)
    return {"refresh": str(refresh), "access": str(refresh.access_token)}


def _set_auth_cookies(response: Response, *, refresh_token: str, access_token: str) -> Response:
    common = {
        "httponly": True,
        "secure": bool(getattr(settings, "JWT_COOKIE_SECURE", True)),
        "samesite": getattr(settings, "JWT_COOKIE_SAMESITE", "Lax"),
        "domain": getattr(settings, "JWT_COOKIE_DOMAIN", None),
        "path": getattr(settings, "JWT_COOKIE_PATH", "/"),
    }
    response.set_cookie(
        getattr(settings, "JWT_ACCESS_COOKIE_NAME", "healthcare_access"),
        access_token,
        max_age=int(settings.SIMPLE_JWT["ACCESS_TOKEN_LIFETIME"].total_seconds()),
        **common,
    )
    response.set_cookie(
        getattr(settings, "JWT_REFRESH_COOKIE_NAME", "healthcare_refresh"),
        refresh_token,
        max_age=int(settings.SIMPLE_JWT["REFRESH_TOKEN_LIFETIME"].total_seconds()),
        **common,
    )
    return response


def _clear_auth_cookies(response: Response) -> Response:
    response.delete_cookie(
        getattr(settings, "JWT_ACCESS_COOKIE_NAME", "healthcare_access"),
        path=getattr(settings, "JWT_COOKIE_PATH", "/"),
        domain=getattr(settings, "JWT_COOKIE_DOMAIN", None),
        samesite=getattr(settings, "JWT_COOKIE_SAMESITE", "Lax"),
    )
    response.delete_cookie(
        getattr(settings, "JWT_REFRESH_COOKIE_NAME", "healthcare_refresh"),
        path=getattr(settings, "JWT_COOKIE_PATH", "/"),
        domain=getattr(settings, "JWT_COOKIE_DOMAIN", None),
        samesite=getattr(settings, "JWT_COOKIE_SAMESITE", "Lax"),
    )
    return response


def _resolve_user_from_identifier(identifier: str) -> Optional[User]:
    identifier = (identifier or "").strip()
    if not identifier:
        return None
    if "@" in identifier:
        return User.objects.filter(email__iexact=identifier).first()
    return User.objects.filter(username__iexact=identifier).first()


def _session_for_user(user: User, session_id: int) -> ChatSession:
    return get_object_or_404(ChatSession, id=session_id, user=user)


def _profile_for_user(user: User) -> UserProfile:
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


def _profile_medical_payload(profile: UserProfile) -> dict:
    if profile.medical_profile:
        return dict(profile.medical_profile)
    if profile.medical_history:
        return dict(profile.medical_history)
    return {}


def _assistant_summary(result: dict, symptom_text: str = "", language: str = "en") -> str:
    # For non-English languages, always use the localised summary builder
    # which produces a fully translated structured response
    if language in ("am", "om"):
        # First localise the result fields (translates recommendation, red_flags, etc.)
        from .services.language_support import localize_analysis_result
        localised = localize_analysis_result(result, language)
        return build_assistant_summary(localised, language)

    # English: prefer LLM natural response if available
    llm_response = result.get("llm_natural_response", "")
    if llm_response and llm_response.strip():
        return llm_response

    # English: prefer search-grounded summary when search results exist
    if (result.get("search_context") or {}).get("results"):
        return build_assistant_summary(result, language)

    conditions = result.get("probable_conditions", [])[:3]
    named_conditions = [
        item
        for item in conditions
        if not GENERIC_CONDITION_RE.fullmatch(str(item.get("condition", "")).strip())
        and str(item.get("condition", "")).strip().lower() != "unspecified clinical pattern"
    ]

    risk_level = str(result.get("risk_level", "Low")).lower()
    risk_score = float(result.get("risk_score", 0))
    recommendation = result.get("recommendation_text", "")
    red_flags = result.get("red_flags", [])

    supportive_opening = build_supportive_opening(
        symptom_text=symptom_text,
        risk_level=risk_level,
    )

    # Build a natural, conversational response
    parts = [supportive_opening]

    if named_conditions:
        top = named_conditions[0]
        top_name = top.get("condition", "Unknown")
        top_pct = float(top.get("probability", 0)) * 100
        if len(named_conditions) > 1:
            others = ", ".join(
                f"{c.get('condition', 'Unknown')} ({float(c.get('probability', 0)) * 100:.0f}%)"
                for c in named_conditions[1:3]
            )
            parts.append(
                f"Based on what you've described, the most likely possibility is **{top_name}** ({top_pct:.0f}% confidence), "
                f"with {others} also worth considering."
            )
        else:
            parts.append(
                f"Based on what you've described, this looks most consistent with **{top_name}** ({top_pct:.0f}% confidence)."
            )
    else:
        parts.append(
            "I wasn't able to identify a specific condition with high confidence from your description. "
            "More detail about your symptoms would help narrow this down."
        )

    # Risk context
    if risk_level == "high":
        parts.append(
            f"⚠️ Your symptoms suggest a **high-risk** situation (score {risk_score:.2f}). "
            "Please seek medical attention promptly."
        )
    elif risk_level in ("medium", "moderate"):
        parts.append(f"The overall risk level is **moderate** (score {risk_score:.2f}).")
    else:
        parts.append(f"The overall risk level appears **low** (score {risk_score:.2f}).")

    # Red flags
    if red_flags:
        parts.append(f"⚠️ Important warning signs to watch: {', '.join(red_flags[:3])}.")

    # RAG knowledge — supplement with relevant medical information from knowledge base
    rag_context = result.get("rag_context") or {}
    rag_hits = rag_context.get("items") or []
    rag_knowledge = _extract_rag_knowledge(rag_hits, max_answer_len=400) if rag_hits else ""
    if rag_knowledge:
        parts.append(f"**Related medical knowledge:**\n{rag_knowledge}")

    # Recommendation
    if recommendation:
        parts.append(f"**What to do next:** {recommendation}")

    parts.append(
        "\n*This is informational guidance only — not a medical diagnosis. "
        "Always consult a qualified healthcare professional for medical advice.*"
    )

    return "\n\n".join(p for p in parts if p.strip())


def _serialize_case_result(case: CaseSubmission) -> dict:
    inference = getattr(case, "inference", None)
    risk = getattr(case, "risk", None)
    fused_predictions = inference.fused_predictions if inference else []
    user_facing = [
        item
        for item in fused_predictions
        if not GENERIC_CONDITION_RE.fullmatch(str(item.get("condition", "")).strip())
    ]
    if not user_facing and fused_predictions:
        top_item = dict(fused_predictions[0])
        top_item["condition"] = "Unspecified clinical pattern"
        user_facing = [top_item]
    return {
        "case_id": case.id,
        "created_at": case.created_at.isoformat(),
        "symptom_text": case.symptom_text,
        "symptom_tags": case.symptom_tags,
        "status": case.status,
        "response_language": (case.metadata or {}).get("language", "en"),
        "probable_conditions": user_facing,
        "raw_probable_conditions": fused_predictions,
        "risk_level": risk.risk_level if risk else None,
        "risk_score": risk.risk_score if risk else None,
        "recommendation_text": risk.recommendation_text if risk else "",
        "red_flags": risk.red_flags if risk else [],
        "prevention_advice": risk.prevention_advice if risk else [],
    }

# ── Conversational Intent & Context-Aware Response System ───────────────────

class _Intent:
    GREETING = "greeting"
    IDENTITY = "identity"
    FOLLOW_UP_YES = "follow_up_yes"
    FOLLOW_UP_NO = "follow_up_no"
    FOLLOW_UP_DETAIL = "follow_up_detail"
    EMOTIONAL = "emotional"
    GRATITUDE = "gratitude"
    FAREWELL = "farewell"
    AFFIRMATION = "affirmation"
    SMALL_TALK = "small_talk"
    MEDICAL = "medical"
    INFORMATION_SEEKING = "information_seeking"


_SYMPTOM_KEYWORDS = re.compile(
    r"\b(pain|ache|hurt|sore|fever|cough|nausea|vomit|diarrhea|diarrhoea|bleed|bleeding"
    r"|rash|itch|swollen|swelling|dizzy|dizziness|fatigue|tired|weak|weakness"
    r"|shortness of breath|difficulty breathing|chest tightness"
    r"|headache|headach|headech|hedache|hedeache|migraine|stomach ache|abdominal|back pain|joint pain"
    r"|muscle ache|sore throat|earache|eye pain|skin lesion|burning sensation"
    r"|numbness|tingling|seizure|fainted|unconscious|palpitation"
    r"|discharge|blood in urine|high blood pressure|low blood pressure"
    r"|high sugar|diabetes symptoms|asthma attack|allergic reaction"
    r"|symptoms|diagnosis|diagnose|treatment|medication|medicine"
    r"|worse|worsening|spreading|coming back|won't go away|getting better"
    r"|side effect|reaction|lump|bruise|wound|injury|fracture"
    r"|infection|inflammation|congestion|phlegm|sputum"
    r"|had a|I have|I've|suffer|suffering)\b",
    re.IGNORECASE,
)

_EMOTIONAL_KEYWORDS = re.compile(
    r"\b(scared|afraid|worried|worry|anxious|anxiety|panic|terrified|frightened"
    r"|concerned|nervous|stressed|overwhelmed|helpless|hopeless|desperate"
    r"|confused|lost|don't know what to do|not sure what to do)\b",
    re.IGNORECASE,
)

_BODY_REGION_KEYWORDS = re.compile(
    r"\b(chest|head|stomach|back|neck|throat|ear|eye|eyes|knee|leg|arm|hand"
    r"|foot|feet|shoulder|hip|jaw|tooth|teeth|skin|lung|heart|liver|kidney)\b",
    re.IGNORECASE,
)

_GREETING_PREFIXES = ("hi", "hello", "hey", "howdy", "greetings", "what's up", "whats up", "sup")
_IDENTITY_PHRASES = (
    "are you a doctor", "are you human", "who are you", "what are you",
    "what can you do", "what do you do", "tell me about yourself",
    "tell me about you", "are you real", "are you ai", "are you a bot",
    "are you a robot", "are you chatgpt", "are you gpt",
)
_YES_WORDS = frozenset({"yes", "yeah", "yep", "yup", "correct", "right", "exactly", "mhm"})
_NO_WORDS = frozenset({"no", "nope", "nah", "not really", "not at all"})
_THANK_PREFIXES = ("thank", "thanks", "ty", "thx", "appreciate", "grateful")
_BYE_PREFIXES = ("bye", "goodbye", "see you", "take care", "gotta go", "talk later")
_AFFIRMATION_WORDS = frozenset({"ok", "okay", "sure", "alright", "great", "good", "fine", "got it", "understood", "makes sense"})
_HOW_PREFIXES = ("how are", "how do you", "how can", "how should", "how could")

# Information-seeking patterns: questions about conditions/treatments WITHOUT first-person symptom reporting
_INFO_SEEKING_PHRASES = re.compile(
    r"\b(what is|what are|what's|whats|tell me about|explain|describe|can you explain"
    r"|how does|how do|how is|why does|why do|why is|who gets|is it true"
    r"|difference between|causes of|symptoms of|treatment for|cure for"
    r"|prevention of|risk factors for|complications of|diagnosis of"
    r"|prognosis of|types of|stages of|signs of|how common is)\b",
    re.IGNORECASE,
)
# First-person symptom reporting signals — the user is describing THEIR symptoms
_FIRST_PERSON_SYMPTOM = re.compile(
    r"\b(I have|I've had|I'm having|I feel|I am feeling|I am experiencing|I've been"
    r"|my (head|chest|stomach|back|throat|ear|eye|skin|heart|joint|leg|arm)"
    r"|I suffer|I am suffering|my symptoms|I've noticed|I woke up with)\b",
    re.IGNORECASE,
)


def _get_session_context(session: ChatSession, limit: int = 6) -> list:
    """Return the last `limit` messages from the session for context."""
    msgs = list(
        session.messages.select_related("session")
        .order_by("-created_at")[:limit]
    )
    msgs.reverse()
    return msgs


def _has_medical_context(recent_messages: list) -> bool:
    """Check if the recent conversation contains medical analysis."""
    for msg in recent_messages:
        if msg.role == ChatMessage.Role.ASSISTANT:
            meta = msg.metadata or {}
            if meta.get("case_id") or meta.get("result"):
                return True
            if meta.get("conversational") and "symptom" in (msg.content or "").lower():
                return True
    return False


def _ml_classify_intent(text: str) -> tuple:
    """Use the trained dialogue classifier to predict intent.

    Returns (predicted_intent_label, confidence) or (None, 0.0) if
    the model is unavailable or confidence is too low.
    """
    try:
        from .services.dialogue_style import _load_dialogue_artifacts
        model, vectorizer, _, _ = _load_dialogue_artifacts()
        if model is None or vectorizer is None:
            return None, 0.0
        cleaned = clean_symptom_text(text)
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0]
        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])
        # Map numeric class ID to text label
        classes = list(model.classes_)
        if best_idx < len(classes):
            raw_id = str(classes[best_idx])
            # Load the ID→label mapping (cached after first load)
            label_map = _load_dialogue_class_map()
            label = label_map.get(raw_id, raw_id)
            return label, confidence
    except Exception:
        pass
    return None, 0.0


def _load_dialogue_class_map() -> dict:
    """Load and cache the dialogue classifier class ID → label mapping."""
    if not hasattr(_load_dialogue_class_map, "_cache"):
        try:
            from pathlib import Path
            map_path = Path(getattr(settings, "BASE_DIR", "")) / "models" / "dialogue_class_id_to_label.json"
            if map_path.exists():
                _load_dialogue_class_map._cache = json.loads(map_path.read_text(encoding="utf-8"))
            else:
                _load_dialogue_class_map._cache = {}
        except Exception:
            _load_dialogue_class_map._cache = {}
    return _load_dialogue_class_map._cache


def _map_dialogue_label_to_intent(label: str) -> str:
    """Map a dialogue model label to our _Intent categories."""
    if not label:
        return _Intent.SMALL_TALK
    lower = label.lower().strip()
    # Medical topic labels → MEDICAL
    medical_indicators = (
        "diabetes", "blood pressure", "heart", "cancer", "infection",
        "asthma", "allergy", "depression", "anxiety", "pain", "headache",
        "fever", "cough", "rash", "stomach", "diarrhea", "flu", "cold",
        "thyroid", "kidney", "liver", "skin", "pregnancy", "symptoms",
        "seizure", "stroke", "arthritis", "migraine", "pneumonia",
        "hepatitis", "hiv", "anemia", "fracture", "constipation",
        "insomnia", "vomit", "nausea", "burn", "swelling", "numbness",
        "dizziness", "fatigue", "bronchitis", "sinusitis", "acne",
        "eczema", "psoriasis", "herpes", "ulcer", "hernia",
        "osteoporosis", "cholesterol", "vaccine", "drug", "antibiotic",
        "clinical_qa", "information", "causes", "complications",
        "prevention", "exams and tests", "outlook", "symptoms",
        "frequency", "susceptibility", "inheritance", "genetic",
        "considerations", "stages", "research", "evidence_qa",
    )
    for indicator in medical_indicators:
        if indicator in lower:
            return _Intent.MEDICAL
    # Emotional
    emotional_indicators = ("anxiety", "depression", "stress", "bipolar", "postpartum")
    for indicator in emotional_indicators:
        if indicator in lower:
            return _Intent.EMOTIONAL
    return _Intent.SMALL_TALK


def _classify_intent(text: str, recent_messages: list) -> str:
    """Classify user message intent using regex + trained ML model + session context.

    Priority: regex strong signals → ML model supplement for ambiguous cases.
    The dialogue classifier (75.4% accuracy, 98.1% top-3) catches medical
    queries that regex misses (e.g., "Can antibiotics cause diarrhea?").
    """
    text = (text or "").strip()
    if not text:
        return _Intent.SMALL_TALK

    lower = text.lower()
    words = text.split()
    word_count = len(words)
    in_medical_context = _has_medical_context(recent_messages)

    # Strong signals: symptom keywords always → MEDICAL
    if _SYMPTOM_KEYWORDS.search(text):
        return _Intent.MEDICAL

    # Body-region keywords in short messages within medical context → elaboration
    if in_medical_context and word_count <= 5 and _BODY_REGION_KEYWORDS.search(text):
        return _Intent.FOLLOW_UP_DETAIL

    # Emotional keywords → EMOTIONAL
    if _EMOTIONAL_KEYWORDS.search(text):
        return _Intent.EMOTIONAL

    # Greetings — but only if the rest of the message is truly non-medical
    # "hi I have a headache" should NOT be treated as a greeting
    for prefix in _GREETING_PREFIXES:
        if lower.startswith(prefix) or lower == prefix:
            # Strip the greeting word and re-check for medical content
            rest = lower[len(prefix):].strip()
            if not rest:
                return _Intent.GREETING  # pure greeting like "hi"
            if _SYMPTOM_KEYWORDS.search(rest) or _BODY_REGION_KEYWORDS.search(rest):
                return _Intent.MEDICAL
            if _EMOTIONAL_KEYWORDS.search(rest):
                return _Intent.EMOTIONAL
            # Short rest with no medical signals → still a greeting
            if len(rest.split()) <= 3:
                return _Intent.GREETING
            # Longer rest — let it fall through to ML / default MEDICAL
            break

    # Identity questions
    for phrase in _IDENTITY_PHRASES:
        if phrase in lower:
            return _Intent.IDENTITY

    # How-questions — distinguish info-seeking from symptom reporting
    for prefix in _HOW_PREFIXES:
        if lower.startswith(prefix):
            if _FIRST_PERSON_SYMPTOM.search(text) or _SYMPTOM_KEYWORDS.search(text):
                return _Intent.MEDICAL
            return _Intent.INFORMATION_SEEKING

    # Gratitude
    for prefix in _THANK_PREFIXES:
        if lower.startswith(prefix):
            return _Intent.GRATITUDE

    # Farewell
    for prefix in _BYE_PREFIXES:
        if lower.startswith(prefix):
            return _Intent.FAREWELL

    # Yes/No in medical context → follow-up
    if in_medical_context:
        first_word = lower.split()[0] if lower else ""
        if first_word in _YES_WORDS or lower in _YES_WORDS:
            return _Intent.FOLLOW_UP_YES
        if first_word in _NO_WORDS or lower in _NO_WORDS:
            return _Intent.FOLLOW_UP_NO
        if word_count <= 4:
            return _Intent.FOLLOW_UP_DETAIL

    # Standalone yes/no (no medical context)
    if lower in _YES_WORDS or lower in _NO_WORDS:
        return _Intent.AFFIRMATION

    # Affirmation words
    if lower in _AFFIRMATION_WORDS:
        return _Intent.AFFIRMATION

    # Test messages
    if lower.startswith("test"):
        return _Intent.GREETING

    # Short messages without keywords (≤ 4 words)
    if word_count <= 4:
        if in_medical_context:
            return _Intent.FOLLOW_UP_DETAIL
        return _Intent.SMALL_TALK

    # Information-seeking detection: questions about conditions/treatments
    # WITHOUT first-person symptom reporting → lightweight RAG-based answer
    if _INFO_SEEKING_PHRASES.search(text):
        if _FIRST_PERSON_SYMPTOM.search(text):
            return _Intent.MEDICAL  # "I have diabetes, what are the complications?"
        if _BODY_REGION_KEYWORDS.search(text) or _SYMPTOM_KEYWORDS.search(text):
            return _Intent.INFORMATION_SEEKING  # "What are the symptoms of migraine?"
        return _Intent.INFORMATION_SEEKING  # "What is hypertension?"

    # Longer messages — check for health-adjacent questions
    question_starters = ("what", "when", "where", "why", "who", "how", "is", "are", "do", "can", "will", "would")
    first_word = lower.split()[0] if lower else ""
    if first_word in question_starters and word_count < 12:
        _health_adjacent = re.compile(
            r"\b(health|medical|doctor|hospital|clinic|medicine|vitamin|diet|exercise"
            r"|sleep|stress|mental health|wellness|prevention|vaccine|checkup)\b",
            re.IGNORECASE,
        )
        if _health_adjacent.search(text):
            return _Intent.INFORMATION_SEEKING
        return _Intent.SMALL_TALK

    # ── ML supplement: dialogue classifier for ambiguous cases ────────
    # When regex is uncertain (SMALL_TALK), use the trained dialogue model
    # to catch medical queries that lack explicit symptom keywords.
    # The model has 75.4% accuracy and 98.1% top-3 accuracy.
    ml_label, ml_confidence = _ml_classify_intent(text)
    if ml_label and ml_confidence >= 0.3:
        ml_intent = _map_dialogue_label_to_intent(ml_label)
        if ml_intent == _Intent.MEDICAL:
            # ML says medical — but distinguish info-seeking from symptom-reporting
            if _FIRST_PERSON_SYMPTOM.search(text):
                return _Intent.MEDICAL
            return _Intent.INFORMATION_SEEKING
        if ml_intent == _Intent.EMOTIONAL:
            return _Intent.EMOTIONAL

    # Default: if in medical context, treat as follow-up detail;
    # otherwise information-seeking (not full pipeline)
    if in_medical_context:
        return _Intent.FOLLOW_UP_DETAIL
    return _Intent.INFORMATION_SEEKING


def _is_conversational_message(text: str, session: ChatSession = None) -> bool:
    """Return True if the message should skip the ML pipeline and get a conversational reply.

    Three-way routing:
    - Conversational (greeting, identity, etc.) → _build_conversational_reply
    - INFORMATION_SEEKING → _build_informational_reply (RAG-based, no full pipeline)
    - MEDICAL → full ML pipeline (run_case_analysis)
    """
    intent = _classify_intent(text, _get_session_context(session) if session else [])
    return intent not in (_Intent.MEDICAL, _Intent.INFORMATION_SEEKING)


def _strip_rag_noise(text: str) -> str:
    """Remove metadata patterns that pollute RAG queries (Duration, Severity, Pre-existing)."""
    import re as _re
    # Strip appended metadata like ". Duration: Less than 24 hours", ". Severity: moderate (5/10)", ". Pre-existing: Diabetes"
    cleaned = _re.sub(r"\.\s*Duration:\s*[^.]+", "", text, flags=_re.IGNORECASE)
    cleaned = _re.sub(r"\.\s*Severity:\s*[^.]+", "", cleaned, flags=_re.IGNORECASE)
    cleaned = _re.sub(r"\.\s*Pre-existing:\s*[^.]+", "", cleaned, flags=_re.IGNORECASE)
    return cleaned.strip()


def _build_rag_query(text: str, recent: list) -> str:
    """Compose a RAG query from user text + session context for knowledge retrieval."""
    clean_text = _strip_rag_noise(text)
    parts = [clean_text]
    for msg in reversed(recent):
        if msg.role == ChatMessage.Role.ASSISTANT:
            # Use the last assistant message as context anchor for continuity
            parts.append(msg.content[:150])
            break
        if msg.role == ChatMessage.Role.USER:
            # Include the prior user message to capture the conversation thread
            parts.append(msg.content[:100])
            break
    return " ".join(parts)


def _extract_rag_knowledge(rag_hits: list, max_answer_len: int = 500) -> str:
    """Extract and format the best knowledge from RAG hits.

    Returns a formatted string with the top answer and optionally a second
    supporting answer, or empty string if nothing useful found.
    Only surfaces answers from high-quality sources.
    """
    if not rag_hits:
        return ""

    # Sources that produce user-facing-quality answers
    _HIGH_QUALITY_SOURCES = {
        "medquad", "kaggle_chatbot", "grok_dialogue", "grok_supervised",
        "triage", "triage_full", "kaggle_sym", "kaggle_desc", "kaggle_prec",
        "disease_symptom",
    }

    parts = []
    for i, hit in enumerate(rag_hits[:6]):
        source = hit.get("source", "")
        if source not in _HIGH_QUALITY_SOURCES:
            continue
        answer = (hit.get("metadata") or {}).get("answer", "").strip()
        question = (hit.get("metadata") or {}).get("question", "").strip()
        score = hit.get("score", 0)
        if not answer or score < 0.20:
            continue
        remaining = max_answer_len - sum(len(p) for p in parts)
        if remaining <= 50:
            break
        snippet = answer[:remaining].rstrip()
        if len(answer) > remaining:
            snippet += "…"
        if question and len(parts) == 0:
            parts.append(f"**{question}**\n{snippet}")
        elif question:
            parts.append(f"Related — **{question}**\n{snippet}")
        else:
            parts.append(snippet)
        if len(parts) >= 2:
            break

    return "\n\n".join(parts)


def _build_contextual_acknowledgment(text: str, intent: str, recent: list) -> str:
    """Build a brief, dynamic acknowledgment based on detected intent and context.

    No hardcoded templates — derives from the actual text and conversation state.
    """
    lower = (text or "").strip().lower()
    in_medical_context = _has_medical_context(recent)

    if intent == _Intent.EMOTIONAL:
        # Reference the specific emotion word detected in the user's text
        emotion_match = _EMOTIONAL_KEYWORDS.search(text)
        emotion_word = emotion_match.group(0) if emotion_match else "that"
        return f"I can hear that you're feeling {emotion_word}, and that's completely valid. "

    if intent == _Intent.GREETING:
        return ""

    if intent == _Intent.IDENTITY:
        return ""

    if intent == _Intent.GRATITUDE:
        return "I'm glad I could help. "

    if intent == _Intent.FAREWELL:
        return ""

    if intent == _Intent.FOLLOW_UP_YES:
        if in_medical_context:
            return "Thanks for confirming. "
        return ""

    if intent == _Intent.FOLLOW_UP_NO:
        if in_medical_context:
            return "Understood. "
        return ""

    if intent == _Intent.FOLLOW_UP_DETAIL:
        if in_medical_context:
            return "Thanks for the additional detail. "
        return ""

    if intent == _Intent.AFFIRMATION:
        return ""

    # SMALL_TALK or default
    return ""


def _build_follow_up_prompt(text: str, intent: str, recent: list, rag_hits: list) -> str:
    """Generate a natural follow-up question based on context and RAG results.

    Like a real doctor, asks targeted questions to gather more information.
    Derives keywords from the user's actual text and RAG results.
    """
    in_medical_context = _has_medical_context(recent)
    lower = (text or "").strip().lower()

    # Extract a topic keyword from the user's text for personalized follow-up
    topic = ""
    body_match = _BODY_REGION_KEYWORDS.search(text)
    symptom_match = _SYMPTOM_KEYWORDS.search(text)
    if body_match:
        topic = body_match.group(0)
    elif symptom_match:
        topic = symptom_match.group(0)

    if intent == _Intent.FAREWELL:
        return "Take care, and don't hesitate to come back if anything comes up."

    if intent == _Intent.GRATITUDE:
        return "If anything changes or you have more questions, I'm right here."

    if intent == _Intent.IDENTITY:
        return "What health concern can I help you with today?"

    if intent == _Intent.AFFIRMATION:
        return "Whenever you're ready, describe your symptoms and I'll analyze them."

    # If we have RAG knowledge, offer personalized analysis
    if rag_hits:
        best_q = (rag_hits[0].get("metadata") or {}).get("question", "").strip()
        if best_q and intent != _Intent.EMOTIONAL:
            if topic:
                return (
                    f"If you'd like a personalized assessment about your {topic}, "
                    f"describe your symptoms in detail and I'll run a full analysis."
                )
            return (
                "If you'd like a personalized assessment based on your specific situation, "
                "describe your symptoms in detail and I'll run a full analysis."
            )

    if in_medical_context:
        if intent == _Intent.FOLLOW_UP_YES:
            if topic:
                return (
                    f"Could you tell me more about your {topic} — when did it start, "
                    f"how severe is it, and does anything make it better or worse?"
                )
            return (
                "Could you tell me more — when did it start, how severe is it, "
                "and does anything make it better or worse?"
            )
        if intent == _Intent.FOLLOW_UP_DETAIL:
            if topic:
                return (
                    f"Could you describe the full picture about your {topic} — "
                    f"what you're feeling, when it started, and how severe it is on a scale of 1-10?"
                )
            return (
                "Could you describe the full picture — what you're feeling, "
                "when it started, and how severe it is on a scale of 1-10?"
            )
        if topic:
            return f"Can you describe what you're experiencing with your {topic} in more detail?"
        return "Can you describe what you're experiencing in more detail?"

    if intent == _Intent.EMOTIONAL:
        return "Can you tell me more about what you're going through? Even a rough description helps."

    # Default: prompt for symptoms
    return "Describe any symptoms you're experiencing and I'll analyze them for you."


def _quick_triage_prediction(text: str) -> dict:
    """Get a fast triage prediction from the trained triage classifier.

    The triage model has 97.4% accuracy and ECE 0.021 (well-calibrated).
    Returns {"condition": str, "confidence": float} or empty dict.
    """
    try:
        from .services.text_model import predict_text_probabilities
        result = predict_text_probabilities(text, top_k=1)
        preds = result.get("predictions", [])
        if preds:
            from .services.label_mapping import map_prediction_list
            mapped = map_prediction_list(preds[:1])
            if mapped and not GENERIC_CONDITION_RE.fullmatch(str(mapped[0].get("condition", "")).strip()):
                return {
                    "condition": str(mapped[0].get("condition", "")),
                    "confidence": float(mapped[0].get("probability", 0.0)),
                }
    except Exception:
        pass
    return {}


def _build_conversational_reply(
    text: str,
    language: str = "en",
    session: ChatSession = None,
) -> str:
    """Generate a context-aware response using all trained models.

    Flow: User → Django → FAISS Retriever → LLM (with RAG context) → Response

    When the cloud LLM is available, uses generate_rag_response to produce
    richer, more natural answers grounded in retrieved medical knowledge.
    Falls back to template-based composition when LLM is unavailable.
    """
    from .services.rag import query_rag
    from .services.llm_triage import generate_rag_response, llm_available

    recent = _get_session_context(session) if session else []
    intent = _classify_intent(text, recent)

    # ── Step 1: Query trained knowledge base (FAISS) ──────────────────────
    rag_query = _build_rag_query(text, recent)
    rag_hits = query_rag(rag_query, top_k=3)
    rag_knowledge = _extract_rag_knowledge(rag_hits)

    # ── Step 2: If LLM available, use RAG → LLM → Response path ──────────
    if llm_available() and rag_hits and intent not in (_Intent.GREETING,):
        session_context = "\n".join(
            f"{'Patient' if m.role == 'user' else 'Assistant'}: {(m.content or '')[:200]}"
            for m in recent[-4:]
        )
        llm_result = generate_rag_response(
            text,
            language=language,
            rag_items=rag_hits,
            session_context=session_context,
        )
        if llm_result.get("available") and llm_result.get("response", "").strip():
            return llm_result["response"]

    # ── Step 2b: Gemini fallback for multilingual conversational response ──
    if language != "en":
        try:
            from .services.gemini_service import generate_multilingual_response, gemini_available
            if gemini_available():
                rag_ctx = rag_knowledge[:600] if rag_knowledge else ""
                sess_ctx = "\n".join(
                    f"{'Patient' if m.role == 'user' else 'Assistant'}: {(m.content or '')[:150]}"
                    for m in recent[-4:]
                )
                gemini_reply = generate_multilingual_response(
                    text, language=language, rag_context=rag_ctx, session_context=sess_ctx,
                )
                if gemini_reply:
                    return gemini_reply
        except Exception:
            pass

    # ── Step 3: Fallback — template-based composition ────────────────────
    triage_hint = ""
    triage_meta = {}
    if intent in (_Intent.MEDICAL, _Intent.EMOTIONAL, _Intent.FOLLOW_UP_DETAIL,
                  _Intent.FOLLOW_UP_YES, _Intent.SMALL_TALK):
        triage_meta = _quick_triage_prediction(_strip_rag_noise(text))
        if triage_meta.get("condition") and triage_meta.get("confidence", 0) >= 0.15:
            cond = triage_meta["condition"]
            conf = triage_meta["confidence"]
            triage_hint = f"Based on your description, this may relate to **{cond}** (confidence: {conf:.0%}). "

    # ── Step 3: Build dynamic acknowledgment ──────────────────────────────
    acknowledgment = _build_contextual_acknowledgment(text, intent, recent)

    # ── Step 4: Build dynamic follow-up prompt ────────────────────────────
    follow_up = _build_follow_up_prompt(text, intent, recent, rag_hits)

    # ── Step 5: Compose response ──────────────────────────────────────────
    # For greetings with no prior context, introduce self
    if intent == _Intent.GREETING and not _has_medical_context(recent):
        _INTRO = {
            "en": "I'm HealthAI, your medical assistant. ",
            "am": "እኔ HealthAI ነኝ፣ የሕክምና ረዳትዎ። ምልክቶችዎን ይግለጹ፤ ሕክምናዊ ትንታኔ አቀርባለሁ። ",
            "om": "Ani HealthAI, gargaaraa fayyaa keetii. Mallattoolee kee ibsi; xiinxala fayyaa siif dhiyeessa. ",
        }
        intro = _INTRO.get(language, _INTRO["en"])
        has_medical_in_text = bool(_SYMPTOM_KEYWORDS.search(text) or _BODY_REGION_KEYWORDS.search(text))
        if rag_knowledge and has_medical_in_text:
            return f"{intro}{acknowledgment}Here's some health information that may be relevant:\n\n{rag_knowledge}\n\n{follow_up}"
        return f"{intro}{follow_up}"

    # For identity questions, describe capabilities dynamically
    if intent == _Intent.IDENTITY:
        _IDENTITY = {
            "en": (
                "I'm HealthAI — an AI medical assistant. I can analyze symptoms and identify possible conditions, "
                "assess risk levels (Low, Moderate, High), provide prevention and care recommendations, "
                "and locate nearby clinics and hospitals. I provide informational guidance, not a medical diagnosis. "
            ),
            "am": (
                "እኔ HealthAI ነኝ — AI የሕክምና ረዳት። ምልክቶችን ተንትኜ ሊሆኑ የሚችሉ ሕመሞችን መለየት፣ "
                "የአደጋ ደርጃ (ዝቅተኛ፣ መካከለኛ፣ ከፍተኛ) መገምገም፣ የጥንቃቄ እና ሕክምና ምክሮች መስጠት፣ "
                "እና ቅርብ ክሊኒኮችን ማግኘት እችላለሁ። ሕክምናዊ ምርመራ ሳይሆን መረጃዊ መመሪያ ነው የምሰጠው። "
            ),
            "om": (
                "Ani HealthAI — gargaaraa fayyaa AI. Mallattoolee xiinxalee dhukkuboota ta\'uu danda\'an adda baasuu, "
                "sadarkaa balaa (gadi, giddugaleessa, ol\'aanaa) madaaluu, gorsa ittisaa fi kunuunsaa kennuu, "
                "fi kilinikota dhiyoo argachuu danda\'a. Gorsa odeeffannoo malee, dhukkuba adda baasuu miti. "
            ),
        }
        identity_line = _IDENTITY.get(language, _IDENTITY["en"])
        if rag_knowledge:
            return f"{identity_line}\n\n{rag_knowledge}\n\n{follow_up}"
        return f"{identity_line}{follow_up}"

    # For follow-up intents in medical context, reference prior analysis
    if intent in (_Intent.FOLLOW_UP_DETAIL, _Intent.FOLLOW_UP_YES, _Intent.FOLLOW_UP_NO):
        prior_summary = _extract_prior_analysis_summary(recent)
        parts = [acknowledgment]
        if prior_summary:
            parts.append(f"Regarding our previous analysis — {prior_summary}")
        if rag_knowledge:
            parts.append(rag_knowledge)
        elif triage_hint:
            parts.append(triage_hint)
        parts.append(follow_up)
        return "\n\n".join(p for p in parts if p.strip())

    # For emotional intent, combine empathy with medical knowledge if available
    if intent == _Intent.EMOTIONAL:
        parts = [acknowledgment]
        if rag_knowledge:
            parts.append(
                "Here's some information that may help you understand what's going on:\n\n"
                + rag_knowledge
            )
        elif triage_hint:
            parts.append(triage_hint)
        parts.append(follow_up)
        return "\n\n".join(p for p in parts if p.strip())

    # For all other intents: acknowledgment + triage hint + RAG knowledge + follow-up
    if rag_knowledge or triage_hint:
        knowledge_block = f"{triage_hint}{rag_knowledge}" if triage_hint else rag_knowledge
        if not rag_knowledge and triage_hint:
            return f"{acknowledgment}{triage_hint}{follow_up}"
        return f"{acknowledgment}{knowledge_block}\n\n{follow_up}"

    # No RAG knowledge or triage — still provide acknowledgment + follow-up
    return f"{acknowledgment}{follow_up}"


def _build_informational_reply(
    text: str,
    language: str = "en",
    session: ChatSession = None,
) -> str:
    """Generate a knowledge-grounded response for information-seeking queries.

    Flow: User → Django → FAISS Retriever → LLM (with RAG context) → Response

    When the cloud LLM is available, uses generate_rag_response to produce
    a natural language answer grounded in RAG knowledge. Falls back to
    template-based RAG extraction when LLM is unavailable.
    """
    from .services.rag import query_rag
    from .services.llm_triage import generate_rag_response, llm_available

    recent = _get_session_context(session) if session else []
    intent = _classify_intent(text, recent)

    # ── Step 1: Check if user is following up on a previous analysis ──────
    prior_analysis = _extract_prior_analysis_summary(recent)

    # ── Step 2: Query RAG knowledge base (FAISS) ──────────────────────────
    rag_query = _build_rag_query(text, recent)
    rag_hits = query_rag(rag_query, top_k=5)
    rag_knowledge = _extract_rag_knowledge(rag_hits, max_answer_len=800)

    # ── Step 3: If LLM available, use RAG → LLM → Response path ──────────
    if llm_available() and rag_hits:
        session_context = "\n".join(
            f"{'Patient' if m.role == 'user' else 'Assistant'}: {(m.content or '')[:200]}"
            for m in recent[-4:]
        )
        llm_result = generate_rag_response(
            text,
            language=language,
            rag_items=rag_hits,
            session_context=session_context,
        )
        if llm_result.get("available") and llm_result.get("response", "").strip():
            return llm_result["response"]

    # ── Step 3b: Gemini fallback for multilingual informational response ──
    if language != "en":
        try:
            from .services.gemini_service import generate_multilingual_response, gemini_available
            if gemini_available():
                rag_ctx = rag_knowledge[:600] if rag_knowledge else ""
                sess_ctx = "\n".join(
                    f"{'Patient' if m.role == 'user' else 'Assistant'}: {(m.content or '')[:150]}"
                    for m in recent[-4:]
                )
                gemini_reply = generate_multilingual_response(
                    text, language=language, rag_context=rag_ctx, session_context=sess_ctx,
                )
                if gemini_reply:
                    return gemini_reply
        except Exception:
            pass

    # ── Step 4: Fallback — template-based RAG response ────────────────────
    triage_meta = _quick_triage_prediction(_strip_rag_noise(text))
    triage_hint = ""
    if triage_meta.get("condition") and triage_meta.get("confidence", 0) >= 0.15:
        cond = triage_meta["condition"]
        conf = triage_meta["confidence"]
        triage_hint = f"This topic relates to **{cond}** (relevance: {conf:.0%}). "

    # ── Step 5: Compose fallback response ─────────────────────────────────
    parts = []

    # Reference prior analysis if relevant
    if prior_analysis:
        parts.append(f"Building on our earlier discussion — {prior_analysis}")

    # Main knowledge answer
    if rag_knowledge:
        if triage_hint:
            parts.append(f"{triage_hint}\n{rag_knowledge}")
        else:
            parts.append(rag_knowledge)
    elif triage_hint:
        parts.append(triage_hint)

    # No knowledge found — honest fallback
    if not parts:
        topic = ""
        body_match = _BODY_REGION_KEYWORDS.search(text)
        symptom_match = _SYMPTOM_KEYWORDS.search(text)
        if body_match:
            topic = body_match.group(0)
        elif symptom_match:
            topic = symptom_match.group(0)
        if topic:
            parts.append(
                f"I don't have specific information about that {topic} topic in my knowledge base. "
                f"If you're experiencing symptoms related to your {topic}, describe them in detail "
                f"and I can run a full analysis."
            )
        else:
            parts.append(
                "I don't have specific information about that in my knowledge base. "
                "If you're experiencing symptoms, describe them and I'll analyze them for you."
            )

    # Follow-up prompt
    follow_up = _build_follow_up_prompt(text, intent, recent, rag_hits)
    parts.append(follow_up)

    # Disclaimer
    parts.append(
        "*This is informational health guidance only — not a medical diagnosis. "
        "Consult a qualified healthcare professional for medical advice.*"
    )

    return "\n\n".join(p for p in parts if p.strip())


def _extract_prior_analysis_summary(recent: list) -> str:
    """Extract a brief summary from a previous medical analysis in the session.

    Returns an empty string if no prior analysis exists, or a short summary
    referencing the conditions and risk level from the last analysis.
    """
    for msg in reversed(recent):
        if msg.role != ChatMessage.Role.ASSISTANT:
            continue
        meta = msg.metadata or {}
        result = meta.get("result")
        if not result:
            continue
        conditions = result.get("probable_conditions", [])[:2]
        risk_level = result.get("risk_level", "")
        if not conditions and not risk_level:
            continue
        condition_names = [
            c.get("condition", "") for c in conditions
            if not GENERIC_CONDITION_RE.fullmatch(str(c.get("condition", "")).strip())
        ]
        summary_parts = []
        if condition_names:
            summary_parts.append(f"we discussed {', '.join(condition_names[:2])}")
        if risk_level:
            summary_parts.append(f"risk was assessed as {risk_level.lower()}")
        return " and ".join(summary_parts) + ". "
    return ""


# ── End conversational intent helpers ────────────────────────────────────────


class HealthView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_classes = []  # no throttle — must not depend on Redis

    def get(self, request):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class QuickPromptsView(APIView):
    """
    GET /api/v1/quick-prompts/
    Public endpoint — returns symptom example prompts from the dialogue
    response templates so the frontend never has hardcoded strings.
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_classes = []

    def get(self, request):
        import json as _json
        path = getattr(settings, "DIALOGUE_RESPONSE_TEMPLATES_PATH", "")
        templates = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                templates = _json.load(f)
        except Exception:
            pass

        # Use symptom_report or general_health intent as quick prompts
        prompts = (
            templates.get("symptom_report")
            or templates.get("general_health")
            or templates.get("symptom_query")
            or []
        )
        # Fallback: pull first 6 entries from any intent that looks like examples
        if not prompts:
            for key, vals in templates.items():
                if isinstance(vals, list) and vals:
                    prompts = vals[:6]
                    break

        return Response({"prompts": prompts[:6]}, status=status.HTTP_200_OK)


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    throttle_classes = []  # uses AuthRateThrottle via settings scope

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # REQ-1: Create email verification token and send verification email
        from .models import EmailVerificationToken
        ev_token, _ = EmailVerificationToken.objects.get_or_create(user=user)
        send_verification_email(user, str(ev_token.token))

        # Audit log: track new registrations
        AuditLog.objects.create(
            actor=user,
            action="user_registered",
            target_type="user",
            target_id=str(user.id),
            metadata={"email": user.email, "ip": request.META.get("REMOTE_ADDR", "")},
        )

        tokens = _tokens_for_user(user)
        response = Response(
            {
                "user": UserSerializer(user).data,
                "profile": UserProfileSerializer(_profile_for_user(user)).data,
                "email_verification_sent": bool(user.email),
                "tokens": {"access": tokens["access"], "refresh": tokens["refresh"]},
            },
            status=status.HTTP_201_CREATED,
        )
        return _set_auth_cookies(
            response,
            refresh_token=tokens["refresh"],
            access_token=tokens["access"],
        )


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    throttle_classes = [AuthRateThrottle]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        identifier = serializer.validated_data["identifier"]
        password = serializer.validated_data["password"]

        user_obj = _resolve_user_from_identifier(identifier)
        username = user_obj.username if user_obj else identifier

        # axes lockout check happens inside authenticate via AxesStandaloneBackend
        user = authenticate(request=request, username=username, password=password)

        if user is None:
            # Record failed attempt for axes
            from axes.handlers.proxy import AxesProxyHandler
            AxesProxyHandler.user_login_failed(
                sender=self.__class__,
                credentials={"username": username},
                request=request,
            )
            return Response({"error": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)
        if not user.is_active:
            return Response({"error": "Account is disabled."}, status=status.HTTP_403_FORBIDDEN)

        # Signal successful login to axes so it resets failure count
        auth_login(request, user, backend="axes.backends.AxesStandaloneBackend")

        # Audit log: track successful logins
        AuditLog.objects.create(
            actor=user,
            action="user_login",
            target_type="user",
            target_id=str(user.id),
            metadata={"ip": request.META.get("REMOTE_ADDR", "")},
        )

        tokens = _tokens_for_user(user)
        response = Response(
            {
                "user": UserSerializer(user).data,
                "profile": UserProfileSerializer(_profile_for_user(user)).data,
                "tokens": {"access": tokens["access"], "refresh": tokens["refresh"]},
            },
            status=status.HTTP_200_OK,
        )
        return _set_auth_cookies(
            response,
            refresh_token=tokens["refresh"],
            access_token=tokens["access"],
        )


class CookieTokenRefreshView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser]

    def post(self, request):
        refresh_token = (
            request.data.get("refresh")
            or request.COOKIES.get(getattr(settings, "JWT_REFRESH_COOKIE_NAME", "healthcare_refresh"))
        )
        if not refresh_token:
            response = Response({"error": "Refresh token missing."}, status=status.HTTP_401_UNAUTHORIZED)
            return _clear_auth_cookies(response)
        try:
            refresh = RefreshToken(refresh_token)
        except Exception:
            response = Response({"error": "Invalid refresh token."}, status=status.HTTP_401_UNAUTHORIZED)
            return _clear_auth_cookies(response)

        if settings.SIMPLE_JWT.get("BLACKLIST_AFTER_ROTATION", False):
            try:
                refresh.blacklist()
            except Exception:
                pass

        new_refresh = RefreshToken.for_user(User.objects.get(id=refresh["user_id"]))
        new_access = str(new_refresh.access_token)
        response = Response(
            {
                "status": "refreshed",
                "tokens": {"access": new_access, "refresh": str(new_refresh)},
            },
            status=status.HTTP_200_OK,
        )
        return _set_auth_cookies(
            response,
            refresh_token=str(new_refresh),
            access_token=new_access,
        )


class LogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser]

    def post(self, request):
        refresh_token = (
            request.data.get("refresh")
            or request.COOKIES.get(getattr(settings, "JWT_REFRESH_COOKIE_NAME", "healthcare_refresh"))
        )
        if refresh_token:
            try:
                token = RefreshToken(refresh_token)
                token.blacklist()
            except Exception:
                response = Response({"error": "Invalid refresh token."}, status=status.HTTP_400_BAD_REQUEST)
                return _clear_auth_cookies(response)
        response = Response({"status": "logged_out"}, status=status.HTTP_200_OK)
        return _clear_auth_cookies(response)


# ── Social Login (OAuth via django-allauth) ─────────────────
class SocialLoginView(APIView):
    """
    Accepts a provider name + OAuth access_token (or id_token for Google)
    from the frontend, uses allauth to complete the social login, then
    returns our standard JWT tokens — just like LoginView.
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser]

    def post(self, request):
        provider = (request.data.get("provider") or "").strip().lower()
        access_token = (request.data.get("access_token") or "").strip()
        id_token = (request.data.get("id_token") or "").strip()

        if not provider:
            return Response({"error": "provider is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not access_token and not id_token:
            return Response({"error": "access_token or id_token is required."}, status=status.HTTP_400_BAD_REQUEST)

        from allauth.socialaccount.helpers import complete_social_login
        from allauth.socialaccount.models import SocialLogin, SocialApp
        from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
        from allauth.socialaccount.providers.github.views import GitHubOAuth2Adapter
        from allauth.socialaccount.providers.facebook.views import FacebookOAuth2Adapter
        from allauth.socialaccount.providers.microsoft.views import MicrosoftGraphOAuth2Adapter
        from allauth.socialaccount.providers.apple.views import AppleOAuth2Adapter

        ADAPTER_MAP = {
            "google": GoogleOAuth2Adapter,
            "github": GitHubOAuth2Adapter,
            "facebook": FacebookOAuth2Adapter,
            "microsoft": MicrosoftGraphOAuth2Adapter,
            "apple": AppleOAuth2Adapter,
        }

        adapter_cls = ADAPTER_MAP.get(provider)
        if not adapter_cls:
            return Response({"error": f"Unsupported provider: {provider}"}, status=status.HTTP_400_BAD_REQUEST)

        # Ensure a SocialApp is configured for this provider
        social_app = SocialApp.objects.filter(provider=provider).first()
        if not social_app:
            return Response(
                {"error": f"Social login for '{provider}' is not configured. Add OAuth keys in Django admin or .env."},
                status=status.HTTP_501_NOT_IMPLEMENTED,
            )

        try:
            adapter = adapter_cls(request)
            adapter.app = social_app
            token_obj = adapter.get_access_token_data(request, access_token or id_token)
            social_login = adapter.complete_login(request, social_app, token_obj)

            # Auto-signup: create user if they don't exist
            if not social_login.is_existing:
                social_login.lookup()
                if not social_login.is_existing:
                    # Create the user
                    email = social_login.account.extra_data.get("email", "")
                    name = social_login.account.extra_data.get("name", "")
                    first_name = social_login.account.extra_data.get("first_name", "") or name.split()[0] if name else ""
                    last_name = social_login.account.extra_data.get("last_name", "") or " ".join(name.split()[1:]) if name else ""
                    username = email.split("@")[0] if email else f"{provider}_{social_login.account.uid[:12]}"
                    # Ensure unique username
                    base = username
                    n = 1
                    while User.objects.filter(username=username).exists():
                        username = f"{base}_{n}"
                        n += 1
                    user = User.objects.create_user(
                        username=username,
                        email=email,
                        first_name=first_name,
                        last_name=last_name,
                    )
                    social_login.connect(request, user)

            user = social_login.user
            if not user.is_active:
                return Response({"error": "Account is disabled."}, status=status.HTTP_403_FORBIDDEN)

            # Generate JWT tokens (same as LoginView)
            tokens = _tokens_for_user(user)
            response = Response(
                {
                    "user": UserSerializer(user).data,
                    "profile": UserProfileSerializer(_profile_for_user(user)).data,
                    "tokens": {"access": tokens["access"], "refresh": tokens["refresh"]},
                },
                status=status.HTTP_200_OK,
            )
            return _set_auth_cookies(
                response,
                refresh_token=tokens["refresh"],
                access_token=tokens["access"],
            )

        except Exception as exc:
            logger.error("Social login failed for provider=%s: %s", provider, exc)
            return Response(
                {"error": f"Social login failed: {str(exc)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )


class SocialProvidersView(APIView):
    """Returns list of configured social login providers with client IDs for frontend OAuth."""
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        from allauth.socialaccount.models import SocialApp
        apps = SocialApp.objects.all()
        providers = []
        for app in apps:
            providers.append({
                "id": app.provider,
                "name": app.name or app.provider.capitalize(),
                "client_id": app.client_id,
                "configured": True,
            })
        return Response({"providers": providers})


class ProfileView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def get(self, request):
        profile = _profile_for_user(request.user)
        return Response(UserProfileSerializer(profile).data, status=status.HTTP_200_OK)

    def patch(self, request):
        serializer = ProfileUpdateSerializer(data=request.data, context={"request": request}, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        user = request.user
        profile = _profile_for_user(user)
        user_dirty_fields = []
        profile_dirty_fields = []

        for field in ("first_name", "last_name", "email"):
            if field in data:
                setattr(user, field, data[field])
                user_dirty_fields.append(field)
        if user_dirty_fields:
            user.save(update_fields=user_dirty_fields)

        for field in (
            "phone_number",
            "age",
            "gender",
            "address",
            "emergency_contact_name",
            "emergency_contact_phone",
            "preferred_language",
        ):
            if field in data:
                setattr(profile, field, data[field])
                profile_dirty_fields.append(field)
        if "medical_profile" in data:
            profile.medical_profile = data["medical_profile"]
            profile_dirty_fields.append("medical_profile")
            if "medical_history" not in data:
                profile.medical_history = data["medical_profile"]
                profile_dirty_fields.append("medical_history")
        if "medical_history" in data and "medical_profile" not in data:
            profile.medical_history = data["medical_history"]
            profile.medical_profile = data["medical_history"]
            profile_dirty_fields.extend(["medical_history", "medical_profile"])
        if profile_dirty_fields:
            profile.save(update_fields=profile_dirty_fields + ["updated_at"])

        # Audit log: track profile changes (excluding sensitive field values)
        all_dirty = user_dirty_fields + profile_dirty_fields
        if all_dirty:
            AuditLog.objects.create(
                actor=request.user,
                action="profile_updated",
                target_type="user",
                target_id=str(request.user.id),
                metadata={"fields_updated": all_dirty},
            )

        return Response(UserProfileSerializer(profile).data, status=status.HTTP_200_OK)


class ChatSessionListCreateView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser]

    def get(self, request):
        q = request.query_params.get("q", "").strip()
        date_from = parse_date(request.query_params.get("date_from", "") or "")
        date_to = parse_date(request.query_params.get("date_to", "") or "")

        sessions = ChatSession.objects.filter(user=request.user)
        if q:
            sessions = sessions.filter(
                Q(title__icontains=q)
                | Q(messages__content__icontains=q)
                | Q(cases__symptom_text__icontains=q)
            )
        if date_from:
            sessions = sessions.filter(created_at__date__gte=date_from)
        if date_to:
            sessions = sessions.filter(created_at__date__lte=date_to)

        sessions = sessions.distinct().order_by("-updated_at")
        return Response(ChatSessionSerializer(sessions, many=True).data, status=status.HTTP_200_OK)

    def post(self, request):
        title = str(request.data.get("title", "")).strip() or "Health Consultation"
        session = ChatSession.objects.create(user=request.user, title=title)
        return Response(ChatSessionSerializer(session).data, status=status.HTTP_201_CREATED)


class ChatSessionMessagesView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, session_id: int):
        session = _session_for_user(request.user, session_id=session_id)
        return Response(
            {
                "session": ChatSessionSerializer(session).data,
                "messages": ChatMessageSerializer(session.messages.all(), many=True).data,
            },
            status=status.HTTP_200_OK,
        )


class ChatAnalyzeView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser, MultiPartParser, FormParser]
    throttle_classes = [AnalyzeRateThrottle]

    @transaction.atomic
    def post(self, request, session_id: int):
        session = _session_for_user(request.user, session_id=session_id)
        serializer = AnalyzeCaseSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        # Strip frontend-appended metadata (Duration/Severity/Pre-existing) from symptom text
        payload["symptom_text"] = _strip_rag_noise(payload["symptom_text"])
        preferred_language = (
            payload.get("language_override")
            or getattr(request, "preferred_language", "")
            or str(_profile_for_user(request.user).preferred_language or "").strip().lower()
        )
        response_language = detect_language(payload["symptom_text"], preferred=preferred_language)

        # REQ-9: Redact PII from user message content before persisting to DB
        _redacted = redact_phi_text(payload["symptom_text"])
        stored_content = str(_redacted.get("redacted_text") or payload["symptom_text"])

        user_message = ChatMessage.objects.create(
            session=session,
            role=ChatMessage.Role.USER,
            content=stored_content,
            metadata={
                "symptom_tags": payload.get("symptom_tags", []),
                "language": response_language,
                "pii_redacted": bool(_redacted.get("entities")),
            },
        )

        # ── Intent-based routing ────────────────────────────────────────────
        # Three-way routing:
        #   1. Conversational (greeting, identity, etc.) → natural reply
        #   2. INFORMATION_SEEKING → RAG-based knowledge answer (no full pipeline)
        #   3. MEDICAL → full ML pipeline with risk assessment
        detected_intent = _classify_intent(
            payload["symptom_text"],
            _get_session_context(session),
        )

        # ── Path 1: Conversational intents (greeting, identity, etc.) ──────
        if _is_conversational_message(payload["symptom_text"], session=session):
            conv_reply = _build_conversational_reply(
                payload["symptom_text"],
                language=response_language,
                session=session,
            )
            # Collect ML model confidence metadata
            ml_label, ml_confidence = _ml_classify_intent(payload["symptom_text"])
            triage_meta = _quick_triage_prediction(payload["symptom_text"]) if detected_intent != _Intent.GREETING else {}
            assistant_message = ChatMessage.objects.create(
                session=session,
                role=ChatMessage.Role.ASSISTANT,
                content=conv_reply,
                metadata={
                    "conversational": True,
                    "intent": detected_intent,
                    "language": response_language,
                    "models_used": {
                        "dialogue_classifier": {
                            "label": ml_label,
                            "confidence": round(ml_confidence, 3),
                        },
                        "triage_classifier": triage_meta,
                        "rag": True,
                    },
                    "created_at": timezone.now().isoformat(),
                },
            )
            session.updated_at = timezone.now()
            if session.title == "Health Consultation":
                # Use redacted text to avoid PII leak in title
                session.title = (stored_content[:64] + "...") if len(stored_content) > 64 else stored_content
                session.save(update_fields=["title", "updated_at"])
            else:
                session.save(update_fields=["updated_at"])
            return Response(
                {
                    "session": ChatSessionSerializer(session).data,
                    "user_message": ChatMessageSerializer(user_message).data,
                    "assistant_message": ChatMessageSerializer(assistant_message).data,
                    "analysis": None,
                },
                status=status.HTTP_200_OK,
            )

        # ── Path 2: Information-seeking (lightweight RAG-based answer) ─────
        if detected_intent == _Intent.INFORMATION_SEEKING:
            info_reply = _build_informational_reply(
                payload["symptom_text"],
                language=response_language,
                session=session,
            )
            ml_label, ml_confidence = _ml_classify_intent(payload["symptom_text"])
            triage_meta = _quick_triage_prediction(payload["symptom_text"])
            assistant_message = ChatMessage.objects.create(
                session=session,
                role=ChatMessage.Role.ASSISTANT,
                content=info_reply,
                metadata={
                    "informational": True,
                    "intent": detected_intent,
                    "language": response_language,
                    "models_used": {
                        "dialogue_classifier": {
                            "label": ml_label,
                            "confidence": round(ml_confidence, 3),
                        },
                        "triage_classifier": triage_meta,
                        "rag": True,
                    },
                    "created_at": timezone.now().isoformat(),
                },
            )
            session.updated_at = timezone.now()
            if session.title == "Health Consultation":
                session.title = (stored_content[:64] + "...") if len(stored_content) > 64 else stored_content
                session.save(update_fields=["title", "updated_at"])
            else:
                session.save(update_fields=["updated_at"])
            return Response(
                {
                    "session": ChatSessionSerializer(session).data,
                    "user_message": ChatMessageSerializer(user_message).data,
                    "assistant_message": ChatMessageSerializer(assistant_message).data,
                    "analysis": None,
                },
                status=status.HTTP_200_OK,
            )

        # ── Path 3: MEDICAL → full ML pipeline (async for speed) ───────────

        profile = _profile_for_user(request.user)
        case_metadata = _profile_medical_payload(profile)
        case_metadata.update(payload.get("metadata", {}))
        case_metadata["language"] = response_language
        case_metadata["force_search"] = bool(payload.get("force_search", False))
        case_metadata["search_consent_given"] = bool(payload.get("search_consent_given", False))
        case_metadata["model_profile"] = payload.get("model_profile", "Clinical Balanced")
        if payload.get("mock_search_results"):
            case_metadata["mock_search_results"] = payload["mock_search_results"]

        case = CaseSubmission.objects.create(
            user=request.user,
            chat_session=session,
            symptom_text=payload["symptom_text"],
            symptom_tags=payload.get("symptom_tags", []),
            uploaded_image=payload.get("image"),
            consent_given=payload["consent_given"],
            location_lat=payload.get("location_lat"),
            location_lng=payload.get("location_lng"),
            facility_type_requested=payload.get("facility_type", ""),
            specialization_requested=payload.get("specialization", ""),
            search_radius_km=payload.get("search_radius_km", 5),
            metadata=case_metadata,
            status="processing",
        )

        # Run analysis — try async via RQ first, fall back to synchronous
        rq_available = False
        try:
            import django_rq
            django_rq.get_queue(getattr(settings, "RQ_ANALYSIS_QUEUE", "analysis"))
            rq_available = True
        except Exception:
            rq_available = False

        if rq_available:
            try:
                submit_async_case_analysis(case.id)
            except Exception as rq_exc:
                logger.warning("RQ submit failed (%s), falling back to sync for case_id=%s", rq_exc, case.id)
                rq_available = False

        if not rq_available:
            # Synchronous execution
            try:
                result = run_case_analysis(case)
            except Exception as pipe_exc:
                logger.exception("Chat analysis pipeline failed for case_id=%s: %s", case.id, pipe_exc)
                case.status = "failed"
                case.save(update_fields=["status"])
                return Response(
                    {
                        "error": (
                            "መረጃውን ማብራራት አልቻልንም። እባክዎ እንደገና ይሞክሩ፤ ምልክቶቹ ከባድ ከሆኑ ወይም እየባሱ ከሄዱ የሙያ ሕክምና እርዳታ ይፈልጉ።"
                            if response_language == "am"
                            else "We could not complete this analysis. Please retry, and seek professional care if symptoms are severe or worsening."
                        )
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            assistant_message = ChatMessage.objects.create(
                session=session,
                role=ChatMessage.Role.ASSISTANT,
                content=_assistant_summary(result, symptom_text=payload["symptom_text"], language=response_language),
                metadata={
                    "case_id": case.id,
                    "result": result,
                    "language": response_language,
                    "created_at": timezone.now().isoformat(),
                },
            )
            session.updated_at = timezone.now()
            if session.title == "Health Consultation":
                session.title = (stored_content[:64] + "...") if len(stored_content) > 64 else stored_content
                session.save(update_fields=["title", "updated_at"])
            else:
                session.save(update_fields=["updated_at"])
            return Response(
                {
                    "session": ChatSessionSerializer(session).data,
                    "user_message": ChatMessageSerializer(user_message).data,
                    "assistant_message": ChatMessageSerializer(assistant_message).data,
                    "analysis": result,
                },
                status=status.HTTP_200_OK,
            )

        # Async path — return immediately with polling info
        session.updated_at = timezone.now()
        if session.title == "Health Consultation":
            session.title = (stored_content[:64] + "...") if len(stored_content) > 64 else stored_content
            session.save(update_fields=["title", "updated_at"])
        else:
            session.save(update_fields=["updated_at"])

        return Response(
            {
                "session": ChatSessionSerializer(session).data,
                "user_message": ChatMessageSerializer(user_message).data,
                "status": "processing",
                "case_id": case.id,
                "status_token": str(case.status_token),
                "poll_url": f"/api/v1/analyze/{case.id}/stream/?token={case.status_token}",
            },
            status=status.HTTP_202_ACCEPTED,
        )


class AnalyzeCaseView(APIView):
    parser_classes = [JSONParser, MultiPartParser, FormParser]
    permission_classes = [permissions.AllowAny]
    # Use CookieJWTAuthentication so Bearer tokens and cookies are read.
    # AllowAny still permits anonymous submissions; authentication just
    # populates request.user when a valid token is present.
    authentication_classes = [CookieJWTAuthentication]
    throttle_classes = [AnalyzeAnonRateThrottle]

    @transaction.atomic
    def post(self, request):
        serializer = AnalyzeCaseSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        # Strip frontend-appended metadata (Duration/Severity/Pre-existing) from symptom text
        payload["symptom_text"] = _strip_rag_noise(payload["symptom_text"])
        user = request.user if request.user.is_authenticated else None
        preferred_language = payload.get("language_override") or getattr(request, "preferred_language", "")
        if user:
            preferred_language = preferred_language or str(_profile_for_user(user).preferred_language or "").strip().lower()
        response_language = detect_language(payload["symptom_text"], preferred=preferred_language)
        profile_metadata = {}
        if user:
            profile = _profile_for_user(user)
            profile_metadata = _profile_medical_payload(profile)
        profile_metadata.update(payload.get("metadata", {}))
        profile_metadata["language"] = response_language
        profile_metadata["force_search"] = bool(payload.get("force_search", False))
        profile_metadata["search_consent_given"] = bool(payload.get("search_consent_given", False))
        profile_metadata["model_profile"] = payload.get("model_profile", "Clinical Balanced")
        if payload.get("mock_search_results"):
            profile_metadata["mock_search_results"] = payload["mock_search_results"]

        case = CaseSubmission.objects.create(
            user=user,
            symptom_text=payload["symptom_text"],
            symptom_tags=payload.get("symptom_tags", []),
            uploaded_image=payload.get("image"),
            consent_given=payload["consent_given"],
            location_lat=payload.get("location_lat"),
            location_lng=payload.get("location_lng"),
            facility_type_requested=payload.get("facility_type", ""),
            specialization_requested=payload.get("specialization", ""),
            search_radius_km=payload.get("search_radius_km", 5),
            metadata=profile_metadata,
            status="processing",
        )

        if payload.get("async_mode", False):
            submit_async_case_analysis(case.id)
            return Response(
                {
                    "case_id": case.id,
                    "status": "queued",
                    "poll_url": f"/api/v1/analyze/{case.id}/?token={case.status_token}",
                    "status_token": str(case.status_token),
                    "response_language": response_language,
                },
                status=status.HTTP_202_ACCEPTED,
            )

        try:
            result = run_case_analysis(case)
            return Response(result, status=status.HTTP_200_OK)
        except Exception:
            logger.exception("Analysis pipeline failed for case_id=%s", case.id)
            case.status = "failed"
            case.save(update_fields=["status"])
            return Response(
                {
                    "error": (
                        "መረጃውን ማብራራት አልቻልንም። እባክዎ እንደገና ይሞክሩ፤ ምልክቶቹ ከባድ ከሆኑ ወይም እየባሱ ከሄዱ የሙያ ሕክምና እርዳታ ይፈልጉ።"
                        if response_language == "am"
                        else "We could not complete this analysis. Please retry, and seek professional care if symptoms are severe or worsening."
                    )
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AnalysisStatusView(APIView):
    permission_classes = [permissions.AllowAny]
    # Populate request.user from JWT so authenticated users can access their cases.
    authentication_classes = [CookieJWTAuthentication]

    def get(self, request, case_id: int):
        case = get_object_or_404(CaseSubmission, id=case_id)
        if case.user_id:
            if not request.user.is_authenticated or (request.user.id != case.user_id and not request.user.is_staff):
                return Response({"error": "Case not found."}, status=status.HTTP_404_NOT_FOUND)
        else:
            status_token = str(request.query_params.get("token", "")).strip()
            if not status_token or not constant_time_compare(status_token, str(case.status_token)):
                return Response({"error": "Case not found."}, status=status.HTTP_404_NOT_FOUND)

        task_status = async_case_status(case_id)
        payload = {
            "case_id": case.id,
            "status": case.status,
            "task_status": task_status,
        }
        if case.status == "completed":
            payload["result"] = _serialize_case_result(case)
        return Response(payload, status=status.HTTP_200_OK)


class ChatHistoryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        q = request.query_params.get("q", "").strip()
        date_from = parse_date(request.query_params.get("date_from", "") or "")
        date_to = parse_date(request.query_params.get("date_to", "") or "")

        sessions = ChatSession.objects.filter(user=request.user)
        if q:
            sessions = sessions.filter(
                Q(title__icontains=q) | Q(messages__content__icontains=q) | Q(cases__symptom_text__icontains=q)
            )
        if date_from:
            sessions = sessions.filter(updated_at__date__gte=date_from)
        if date_to:
            sessions = sessions.filter(updated_at__date__lte=date_to)
        sessions = sessions.distinct().order_by("-updated_at")

        payload = []
        for session in sessions:
            payload.append(
                {
                    "session": ChatSessionSerializer(session).data,
                    "messages": ChatMessageSerializer(session.messages.all(), many=True).data,
                    "cases": [_serialize_case_result(case) for case in session.cases.all()],
                }
            )
        return Response(payload, status=status.HTTP_200_OK)


class LocationNearbyView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        serializer = FacilitySearchSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        facilities = lookup_nearby_facilities(
            location_lat=data["location_lat"],
            location_lng=data["location_lng"],
            facility_type=data.get("facility_type", "hospital"),
            specialization=data.get("specialization", ""),
            radius_km=data.get("radius_km", 15),
            limit=10,
        )
        return Response({"facilities": facilities}, status=status.HTTP_200_OK)


class LocationDirectionsView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        origin_lat = request.query_params.get("origin_lat")
        origin_lng = request.query_params.get("origin_lng")
        destination_lat = request.query_params.get("destination_lat")
        destination_lng = request.query_params.get("destination_lng")
        place_id = request.query_params.get("place_id", "").strip()

        if place_id:
            url = f"https://www.google.com/maps/dir/?api=1&destination_place_id={place_id}"
            return Response({"maps_url": url}, status=status.HTTP_200_OK)

        if not all([origin_lat, origin_lng, destination_lat, destination_lng]):
            return Response(
                {"error": "Provide place_id or origin/destination coordinates."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={origin_lat},{origin_lng}&destination={destination_lat},{destination_lng}"
        )
        return Response({"maps_url": url}, status=status.HTTP_200_OK)


class EmergencyContactsView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        country_code = str(request.query_params.get("country_code", "")).strip().upper()
        return Response({"contacts": emergency_contacts(country_code=country_code)}, status=status.HTTP_200_OK)


class AdminUsersView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        q = request.query_params.get("q", "").strip()
        page = max(int(request.query_params.get("page", 1)), 1)
        page_size = min(int(request.query_params.get("page_size", 50)), 200)

        users = User.objects.all().select_related("profile").order_by("-date_joined")
        if q:
            users = users.filter(
                Q(username__icontains=q) | Q(email__icontains=q) |
                Q(first_name__icontains=q) | Q(last_name__icontains=q)
            )
        total = users.count()
        page_users = users[(page - 1) * page_size: page * page_size]
        return Response(
            {
                "count": total,
                "page": page,
                "page_size": page_size,
                "results": [UserSerializer(user).data for user in page_users],
            },
            status=status.HTTP_200_OK,
        )


class AdminUserDetailView(APIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def patch(self, request, user_id: int):
        user = get_object_or_404(User, id=user_id)
        serializer = AdminUserUpdateSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        dirty_fields = []
        for field in ("first_name", "last_name", "email", "is_active", "is_staff"):
            if field in data:
                setattr(user, field, data[field])
                dirty_fields.append(field)
        if dirty_fields:
            user.save(update_fields=dirty_fields)
            AuditLog.objects.create(
                actor=request.user,
                action="admin_update_user",
                target_type="user",
                target_id=str(user.id),
                metadata={"fields": dirty_fields},
            )
        return Response(UserSerializer(user).data, status=status.HTTP_200_OK)


class AdminFacilitiesView(APIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def get(self, request):
        facilities = HealthcareFacility.objects.all().order_by("name")
        return Response(HealthcareFacilitySerializer(facilities, many=True).data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = HealthcareFacilitySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        facility = serializer.save(source="manual")
        AuditLog.objects.create(
            actor=request.user,
            action="admin_create_facility",
            target_type="facility",
            target_id=str(facility.id),
        )
        return Response(HealthcareFacilitySerializer(facility).data, status=status.HTTP_201_CREATED)


class AdminFacilityDetailView(APIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def patch(self, request, facility_id: int):
        facility = get_object_or_404(HealthcareFacility, id=facility_id)
        serializer = HealthcareFacilitySerializer(instance=facility, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        facility = serializer.save()
        AuditLog.objects.create(
            actor=request.user,
            action="admin_update_facility",
            target_type="facility",
            target_id=str(facility.id),
        )
        return Response(HealthcareFacilitySerializer(facility).data, status=status.HTTP_200_OK)

    def delete(self, request, facility_id: int):
        facility = get_object_or_404(HealthcareFacility, id=facility_id)
        facility.delete()
        AuditLog.objects.create(
            actor=request.user,
            action="admin_delete_facility",
            target_type="facility",
            target_id=str(facility_id),
        )
        return Response(status=status.HTTP_204_NO_CONTENT)


class AdminAnalyticsView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        today = timezone.now().date()
        from datetime import timedelta
        seven_days_ago = today - timedelta(days=7)
        thirty_days_ago = today - timedelta(days=30)

        risk_breakdown = (
            CaseSubmission.objects.filter(risk__isnull=False)
            .values("risk__risk_level")
            .annotate(count=Count("id"))
            .order_by("risk__risk_level")
        )

        # Per-day activity for last 30 days
        daily_activity = (
            ChatMessage.objects.filter(
                role=ChatMessage.Role.USER,
                created_at__date__gte=thirty_days_ago,
            )
            .values("created_at__date")
            .annotate(messages=Count("id"), sessions=Count("session", distinct=True))
            .order_by("created_at__date")
        )
        daily_new_users = (
            User.objects.filter(date_joined__date__gte=thirty_days_ago)
            .values("date_joined__date")
            .annotate(new_users=Count("id"))
            .order_by("date_joined__date")
        )

        # Active users: users who sent a message in last 7 days
        active_user_ids = (
            ChatMessage.objects.filter(
                role=ChatMessage.Role.USER,
                created_at__date__gte=seven_days_ago,
            )
            .values_list("session__user_id", flat=True)
            .distinct()
        )
        active_users_count = len(active_user_ids)

        # Top 5 most active users (by message count, last 30 days)
        top_active_users = (
            User.objects.filter(
                chat_sessions__messages__role=ChatMessage.Role.USER,
                chat_sessions__messages__created_at__date__gte=thirty_days_ago,
            )
            .annotate(
                msg_count=Count("chat_sessions__messages", filter=Q(
                    chat_sessions__messages__role=ChatMessage.Role.USER,
                    chat_sessions__messages__created_at__date__gte=thirty_days_ago,
                ))
            )
            .order_by("-msg_count")[:5]
            .values("id", "username", "email", "first_name", "last_name", "msg_count")
        )

        # Most repetitive user questions (top 10, last 30 days)
        top_questions = (
            ChatMessage.objects.filter(
                role=ChatMessage.Role.USER,
                created_at__date__gte=thirty_days_ago,
                content__gt="",
            )
            .values("content")
            .annotate(frequency=Count("id"))
            .order_by("-frequency")[:10]
        )

        return Response(
            {
                "users_total": User.objects.count(),
                "users_active": User.objects.filter(is_active=True).count(),
                "users_active_7d": active_users_count,
                "users_new_today": User.objects.filter(date_joined__date=today).count(),
                "users_new_7d": User.objects.filter(date_joined__date__gte=seven_days_ago).count(),
                "chat_sessions_total": ChatSession.objects.count(),
                "chat_messages_total": ChatMessage.objects.count(),
                "messages_today": ChatMessage.objects.filter(created_at__date=today, role=ChatMessage.Role.USER).count(),
                "cases_total": CaseSubmission.objects.count(),
                "cases_today": CaseSubmission.objects.filter(created_at__date=today).count(),
                "cases_completed": CaseSubmission.objects.filter(status="completed").count(),
                "facilities_total": HealthcareFacility.objects.count(),
                "risk_breakdown": list(risk_breakdown),
                "daily_activity": [
                    {"date": str(d["created_at__date"]), "messages": d["messages"], "sessions": d["sessions"]}
                    for d in daily_activity
                ],
                "daily_new_users": [
                    {"date": str(d["date_joined__date"]), "new_users": d["new_users"]}
                    for d in daily_new_users
                ],
                "top_active_users": list(top_active_users),
                "top_questions": list(top_questions),
            },
            status=status.HTTP_200_OK,
        )


# ... (rest of the code remains the same)
class AdminConfigView(APIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def get(self, request):
        from django.conf import settings as proj_settings
        config = {
            "use_llm_triage": proj_settings.USE_LLM_TRIAGE,
            "llm_fallback_to_classical": proj_settings.LLM_FALLBACK_TO_CLASSICAL,
            "llm_rag_response": proj_settings.LLM_RAG_RESPONSE,
            "llm_base_model": proj_settings.TRIAGE_LLM_BASE_MODEL,
            "github_token_set": bool(proj_settings.GITHUB_TOKEN),
            "redis_url": proj_settings.REDIS_URL,
            "image_input_size": proj_settings.IMAGE_INPUT_SIZE,
            "case_retention_days": proj_settings.CASE_RETENTION_DAYS,
            "audit_log_retention_days": proj_settings.AUDIT_LOG_RETENTION_DAYS,
            "geoapify_key_set": bool(proj_settings.GEOAPIFY_API_KEY),
            "google_maps_key_set": bool(proj_settings.GOOGLE_MAPS_API_KEY),
            "debug": proj_settings.DEBUG,
        }
        return Response(config, status=status.HTTP_200_OK)

    def post(self, request):
        action = str(request.data.get("action", "")).strip()
        if action == "retrain_text_model":
            AuditLog.objects.create(
                actor=request.user,
                action="admin_retrain_requested",
                target_type="ml_model",
                target_id="text",
            )
            
            import django_rq
            import subprocess
            from django.conf import settings
            
            def run_retrain_script():
                subprocess.run(["python", str(settings.BASE_DIR / "scripts" / "preprocess_and_train.py")])
                
            django_rq.enqueue(run_retrain_script)
            
            return Response(
                {
                    "status": "queued",
                    "message": "Model retraining task pushed to the asynchronous worker queue successfully. Processing in background.",
                },
                status=status.HTTP_202_ACCEPTED,
            )
        return Response(
            {"error": "Unsupported action. Example: retrain_text_model"},
            status=status.HTTP_400_BAD_REQUEST,
        )


class ExportProfileView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        profile_data = UserProfileSerializer(_profile_for_user(request.user)).data
        cases = [_serialize_case_result(case) for case in request.user.cases.all()[:200]]
        return Response(
            {
                "exported_at": timezone.now().isoformat(),
                "user": UserSerializer(request.user).data,
                "profile": profile_data,
                "cases": cases,
            },
            status=status.HTTP_200_OK,
        )


class ExportChatHistoryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        output_format = (request.query_params.get("format", "json") or "json").strip().lower()
        session_id = request.query_params.get("session_id")
        sessions = ChatSession.objects.filter(user=request.user).order_by("-updated_at")
        if session_id:
            sessions = sessions.filter(id=session_id)

        if output_format == "csv":
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["session_id", "session_title", "role", "content", "created_at"])
            for session in sessions:
                for message in session.messages.all():
                    writer.writerow([session.id, session.title, message.role, message.content, message.created_at.isoformat()])
            response = HttpResponse(csv_buffer.getvalue(), content_type="text/csv")
            response["Content-Disposition"] = 'attachment; filename="chat_history.csv"'
            return response

        payload = []
        for session in sessions:
            payload.append(
                {
                    "session": ChatSessionSerializer(session).data,
                    "messages": ChatMessageSerializer(session.messages.all(), many=True).data,
                    "cases": [_serialize_case_result(case) for case in session.cases.all()],
                }
            )
        return Response(
            {
                "exported_at": timezone.now().isoformat(),
                "sessions": payload,
            },
            status=status.HTTP_200_OK,
        )


# ── REQ-7: SSE real-time job status stream ──────────────────────────────────

class AnalysisStatusSSEView(APIView):
    """
    Server-Sent Events endpoint for real-time async job status.
    GET /api/v1/analyze/<case_id>/stream/?token=<uuid>
    Streams status events until the job reaches a terminal state.
    """
    permission_classes = [permissions.AllowAny]
    # Populate request.user from JWT so authenticated users can stream their cases.
    authentication_classes = [CookieJWTAuthentication]

    def get(self, request, case_id: int):
        case = get_object_or_404(CaseSubmission, id=case_id)
        if case.user_id:
            if not request.user.is_authenticated or (
                request.user.id != case.user_id and not request.user.is_staff
            ):
                return Response({"error": "Case not found."}, status=status.HTTP_404_NOT_FOUND)
        else:
            status_token = str(request.query_params.get("token", "")).strip()
            if not status_token or not constant_time_compare(status_token, str(case.status_token)):
                return Response({"error": "Case not found."}, status=status.HTTP_404_NOT_FOUND)

        def _event_stream():
            terminal = {"completed", "failed", "unknown"}
            poll_interval = float(getattr(settings, "SSE_POLL_INTERVAL_SECONDS", 1.5))
            max_wait = float(getattr(settings, "SSE_MAX_WAIT_SECONDS", 620))
            elapsed = 0.0
            last_status = None
            while elapsed < max_wait:
                current_case = CaseSubmission.objects.filter(id=case_id).only("status", "async_job_id", "chat_session_id").first()
                if current_case is None:
                    yield f"data: {json.dumps({'status': 'unknown', 'case_id': case_id})}\n\n"
                    break
                task_status = async_case_status(case_id)
                if task_status != last_status:
                    payload: dict = {"case_id": case_id, "status": task_status}
                    if task_status == "completed":
                        payload["result"] = _serialize_case_result(current_case)
                        # Include the assistant message content for the chat UI
                        assistant_msg = ChatMessage.objects.filter(
                            session_id=current_case.chat_session_id,
                            metadata__case_id=case_id,
                            role=ChatMessage.Role.ASSISTANT,
                        ).first()
                        if assistant_msg:
                            payload["assistant_message"] = ChatMessageSerializer(assistant_msg).data
                        # Include full analysis for the analysis card
                        inference = getattr(current_case, "inference", None)
                        risk = getattr(current_case, "risk", None)
                        if inference or risk:
                            payload["analysis"] = _serialize_case_result(current_case)
                    yield f"data: {json.dumps(payload)}\n\n"
                    last_status = task_status
                if task_status in terminal:
                    break
                time.sleep(poll_interval)
                elapsed += poll_interval
            else:
                yield f"data: {json.dumps({'case_id': case_id, 'status': 'timeout'})}\n\n"

        response = StreamingHttpResponse(_event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response


# ── REQ-5: Admin user management (list + detail already exist) ───────────────
# AdminAuditLogView — expose audit log to admin UI

class AdminAuditLogView(APIView):
    """GET /api/v1/admin/audit-log/ — paginated audit log for admin dashboard."""
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        from .models import AuditLog as _AuditLog
        qs = _AuditLog.objects.select_related("actor").order_by("-created_at")
        actor_id = request.query_params.get("actor_id")
        action = request.query_params.get("action", "").strip()
        if actor_id:
            qs = qs.filter(actor_id=actor_id)
        if action:
            qs = qs.filter(action__icontains=action)
        page_size = min(int(request.query_params.get("page_size", 50)), 200)
        page = max(int(request.query_params.get("page", 1)), 1)
        total = qs.count()
        entries = qs[(page - 1) * page_size: page * page_size]
        results = [
            {
                "id": entry.id,
                "actor": entry.actor.username if entry.actor else None,
                "action": entry.action,
                "target_type": entry.target_type,
                "target_id": entry.target_id,
                "metadata": entry.metadata,
                "created_at": entry.created_at.isoformat(),
            }
            for entry in entries
        ]
        return Response({"total": total, "page": page, "page_size": page_size, "results": results})


# ── REQ-1: Email verification ────────────────────────────────────────────────

class VerifyEmailView(APIView):
    """
    GET /api/v1/auth/verify-email/?token=<uuid>
    POST /api/v1/auth/verify-email/  {"token": "<uuid>"}
    Marks the user's email as verified.
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser, FormParser]

    def _verify(self, token_str):
        from .models import EmailVerificationToken
        token_str = str(token_str).strip()
        if not token_str:
            return Response({"error": "Token is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            ev = EmailVerificationToken.objects.select_related("user").get(token=token_str)
        except EmailVerificationToken.DoesNotExist:
            return Response({"error": "Invalid or expired token."}, status=status.HTTP_400_BAD_REQUEST)
        except ValidationError:
            return Response({"error": "Invalid token format."}, status=status.HTTP_400_BAD_REQUEST)
        if ev.is_verified:
            return Response({"status": "already_verified"}, status=status.HTTP_200_OK)
        ev.verified_at = timezone.now()
        ev.save(update_fields=["verified_at"])
        AuditLog.objects.create(
            actor=ev.user,
            action="email_verified",
            target_type="user",
            target_id=str(ev.user_id),
        )
        return Response({"status": "verified", "email": ev.user.email}, status=status.HTTP_200_OK)

    def get(self, request):
        return self._verify(request.query_params.get("token", ""))

    def post(self, request):
        return self._verify(request.data.get("token", ""))


class ResendVerificationEmailView(APIView):
    """
    POST /api/v1/auth/resend-verification/
    Resends the verification email for the authenticated user.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        from .models import EmailVerificationToken
        user = request.user
        ev_token, _ = EmailVerificationToken.objects.get_or_create(user=user)
        if ev_token.is_verified:
            return Response({"status": "already_verified"}, status=status.HTTP_200_OK)
        sent = send_verification_email(user, str(ev_token.token))
        return Response({"status": "sent" if sent else "queued"}, status=status.HTTP_200_OK)


# ── REQ-1: Password Reset ────────────────────────────────────────────────────

class PasswordResetRequestView(APIView):
    """
    POST /api/v1/auth/password-reset/
    Body: { "email": "user@example.com" }
    Sends a password reset link. Always returns 200 to prevent email enumeration.
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser]
    throttle_classes = []  # Axes + auth throttle applied at settings level

    def post(self, request):
        from .models import PasswordResetToken
        from .services.email_service import send_password_reset_email
        email = str(request.data.get("email", "")).strip().lower()
        if email:
            user = User.objects.filter(email__iexact=email, is_active=True).first()
            if user:
                # Invalidate any existing unused tokens
                PasswordResetToken.objects.filter(user=user, used_at__isnull=True).delete()
                token = PasswordResetToken.objects.create(user=user)
                send_password_reset_email(user, str(token.token))
                AuditLog.objects.create(
                    actor=None,
                    action="password_reset_requested",
                    target_type="user",
                    target_id=str(user.id),
                )
        # Always return 200 — never reveal whether email exists
        return Response(
            {"status": "ok", "message": "If that email is registered, a reset link has been sent."},
            status=status.HTTP_200_OK,
        )


class PasswordResetConfirmView(APIView):
    """
    POST /api/v1/auth/password-reset/confirm/
    Body: { "token": "<uuid>", "new_password": "NewPass123" }
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = [JSONParser]

    def post(self, request):
        from .models import PasswordResetToken
        token_str = str(request.data.get("token", "")).strip()
        new_password = str(request.data.get("new_password", "")).strip()

        if not token_str or not new_password:
            return Response(
                {"error": "token and new_password are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Password strength validation
        import re as _re
        if len(new_password) < 8:
            return Response({"error": "Password must be at least 8 characters."}, status=status.HTTP_400_BAD_REQUEST)
        if not _re.search(r"[A-Z]", new_password):
            return Response({"error": "Password must include an uppercase letter."}, status=status.HTTP_400_BAD_REQUEST)
        if not _re.search(r"[a-z]", new_password):
            return Response({"error": "Password must include a lowercase letter."}, status=status.HTTP_400_BAD_REQUEST)
        if not _re.search(r"[0-9]", new_password):
            return Response({"error": "Password must include a number."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            reset_token = PasswordResetToken.objects.select_related("user").get(token=token_str)
        except PasswordResetToken.DoesNotExist:
            return Response({"error": "Invalid or expired reset token."}, status=status.HTTP_400_BAD_REQUEST)
        except ValidationError:
            return Response({"error": "Invalid token format."}, status=status.HTTP_400_BAD_REQUEST)

        if reset_token.is_used:
            return Response({"error": "This reset link has already been used."}, status=status.HTTP_400_BAD_REQUEST)
        if reset_token.is_expired:
            return Response({"error": "This reset link has expired. Please request a new one."}, status=status.HTTP_400_BAD_REQUEST)

        user = reset_token.user
        user.set_password(new_password)
        user.save(update_fields=["password"])
        reset_token.used_at = timezone.now()
        reset_token.save(update_fields=["used_at"])

        # Blacklist all existing refresh tokens for this user
        try:
            from rest_framework_simplejwt.token_blacklist.models import OutstandingToken
            for outstanding in OutstandingToken.objects.filter(user=user):
                try:
                    outstanding.blacklistedtoken  # already blacklisted
                except Exception:
                    from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken
                    BlacklistedToken.objects.get_or_create(token=outstanding)
        except Exception:
            pass

        AuditLog.objects.create(
            actor=user,
            action="password_reset_completed",
            target_type="user",
            target_id=str(user.id),
        )
        return Response({"status": "ok", "message": "Password updated successfully."}, status=status.HTTP_200_OK)


# ── REQ-5: Admin dialogue template editor ───────────────────────────────────

class AdminDialogueTemplatesView(APIView):
    """
    GET  /api/v1/admin/dialogue-templates/  — list all response templates
    POST /api/v1/admin/dialogue-templates/  — update a template key
    Body: { "intent": "<intent_key>", "templates": ["response1", "response2"] }
    """
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [JSONParser]

    def _load_templates(self) -> dict:
        import json as _json
        path = getattr(settings, "DIALOGUE_RESPONSE_TEMPLATES_PATH", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            return {}

    def _save_templates(self, data: dict) -> bool:
        import json as _json
        path = getattr(settings, "DIALOGUE_RESPONSE_TEMPLATES_PATH", "")
        try:
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def get(self, request):
        templates = self._load_templates()
        return Response({"templates": templates}, status=status.HTTP_200_OK)

    def post(self, request):
        intent = str(request.data.get("intent", "")).strip()
        new_templates = request.data.get("templates")
        if not intent:
            return Response({"error": "intent is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(new_templates, list) or not all(isinstance(t, str) for t in new_templates):
            return Response({"error": "templates must be a list of strings."}, status=status.HTTP_400_BAD_REQUEST)

        data = self._load_templates()
        data[intent] = new_templates
        if not self._save_templates(data):
            return Response({"error": "Unable to save templates file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        AuditLog.objects.create(
            actor=request.user,
            action="admin_update_dialogue_template",
            target_type="dialogue_template",
            target_id=intent,
            metadata={"template_count": len(new_templates)},
        )
        return Response({"status": "updated", "intent": intent, "templates": new_templates}, status=status.HTTP_200_OK)


# ── REQ-5: Admin model performance metrics ──────────────────────────────────

class AdminModelMetricsView(APIView):
    """
    GET /api/v1/admin/model-metrics/
    Returns text and image model evaluation metrics from stored JSON files.
    """
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        import json as _json
        from pathlib import Path as _Path

        base = _Path(settings.BASE_DIR) / "models"

        def _load(path: _Path) -> dict:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return _json.load(f)
            except Exception:
                return {}

        text_metrics = _load(base / "text_training_metrics.json")
        text_eval = _load(base / "evaluation" / "text_evaluation_summary.json")
        image_metrics = _load(base / "image_training_metrics.json")
        dialogue_metrics = _load(base / "dialogue_training_metrics.json")

        return Response(
            {
                "text_model": {
                    "training": text_metrics,
                    "evaluation": text_eval,
                },
                "image_model": {
                    "training": image_metrics,
                },
                "dialogue_model": {
                    "training": dialogue_metrics,
                },
            },
            status=status.HTTP_200_OK,
        )

class AdminRetrainModelView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        import subprocess
        # Async trigger for the retraining script
        script_path = settings.BASE_DIR / "scripts" / "preprocess_and_train.py"
        try:
            subprocess.Popen(["python", str(script_path)])
            return Response({"status": "Retraining job queued and started in background."}, status=status.HTTP_202_ACCEPTED)
        except Exception as e:
            return Response({"error": f"Failed to start retraining: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── Admin: User Activity Management ──────────────────────────────────────────

class AdminUserActivityView(APIView):
    """
    GET /api/v1/admin/user-activity/
    Returns all users with their activity stats: session count, message count,
    last activity timestamp, and activity status (active/recent/inactive).
    Supports search, filtering by activity status, and pagination.
    """
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        from datetime import timedelta
        today = timezone.now().date()
        seven_days_ago = today - timedelta(days=7)
        thirty_days_ago = today - timedelta(days=30)

        q = request.query_params.get("q", "").strip()
        activity_filter = request.query_params.get("activity", "").strip().lower()
        page = max(int(request.query_params.get("page", 1)), 1)
        page_size = min(int(request.query_params.get("page_size", 50)), 200)

        users = User.objects.all().select_related("profile").order_by("-date_joined")

        if q:
            users = users.filter(
                Q(username__icontains=q) | Q(email__icontains=q) |
                Q(first_name__icontains=q) | Q(last_name__icontains=q)
            )

        # Annotate with activity stats
        users = users.annotate(
            session_count=Count("chat_sessions", distinct=True),
            message_count=Count(
                "chat_sessions__messages",
                filter=Q(chat_sessions__messages__role=ChatMessage.Role.USER),
                distinct=True,
            ),
            last_message_time=Max("chat_sessions__messages__created_at"),
        )

        # Filter by activity status
        if activity_filter == "active":
            users = users.filter(last_message_time__date__gte=seven_days_ago)
        elif activity_filter == "recent":
            users = users.filter(
                last_message_time__date__gte=thirty_days_ago,
                last_message_time__date__lt=seven_days_ago,
            )
        elif activity_filter == "inactive":
            users = users.filter(
                Q(last_message_time__date__lt=thirty_days_ago) | Q(last_message_time__isnull=True)
            )

        total = users.count()
        page_users = users[(page - 1) * page_size: page * page_size]

        results = []
        for u in page_users:
            last_msg = u.last_message_time
            if last_msg and last_msg.date() >= seven_days_ago:
                activity_status = "active"
            elif last_msg and last_msg.date() >= thirty_days_ago:
                activity_status = "recent"
            else:
                activity_status = "inactive"

            results.append({
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "first_name": u.first_name,
                "last_name": u.last_name,
                "is_active": u.is_active,
                "is_staff": u.is_staff,
                "date_joined": u.date_joined.isoformat() if u.date_joined else None,
                "last_login": u.last_login.isoformat() if u.last_login else None,
                "session_count": u.session_count,
                "message_count": u.message_count,
                "last_activity": last_msg.isoformat() if last_msg else None,
                "activity_status": activity_status,
            })

        return Response(
            {"count": total, "page": page, "page_size": page_size, "results": results},
            status=status.HTTP_200_OK,
        )


class AdminDailyActivityView(APIView):
    """
    GET /api/v1/admin/daily-activity/
    Returns per-day breakdown of user messages, sessions, new users,
    and case submissions for the last N days (default 30).
    """
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        from datetime import timedelta
        days = min(int(request.query_params.get("days", 30)), 365)
        start_date = timezone.now().date() - timedelta(days=days)

        daily_messages = (
            ChatMessage.objects.filter(
                role=ChatMessage.Role.USER,
                created_at__date__gte=start_date,
            )
            .values("created_at__date")
            .annotate(count=Count("id"))
            .order_by("created_at__date")
        )
        daily_sessions = (
            ChatSession.objects.filter(created_at__date__gte=start_date)
            .values("created_at__date")
            .annotate(count=Count("id"))
            .order_by("created_at__date")
        )
        daily_new_users = (
            User.objects.filter(date_joined__date__gte=start_date)
            .values("date_joined__date")
            .annotate(count=Count("id"))
            .order_by("date_joined__date")
        )
        daily_cases = (
            CaseSubmission.objects.filter(created_at__date__gte=start_date)
            .values("created_at__date")
            .annotate(count=Count("id"))
            .order_by("created_at__date")
        )

        # Merge into a single date-indexed dict
        activity_map = {}
        for d in daily_messages:
            date_str = str(d["created_at__date"])
            activity_map.setdefault(date_str, {"messages": 0, "sessions": 0, "new_users": 0, "cases": 0})
            activity_map[date_str]["messages"] = d["count"]
        for d in daily_sessions:
            date_str = str(d["created_at__date"])
            activity_map.setdefault(date_str, {"messages": 0, "sessions": 0, "new_users": 0, "cases": 0})
            activity_map[date_str]["sessions"] = d["count"]
        for d in daily_new_users:
            date_str = str(d["date_joined__date"])
            activity_map.setdefault(date_str, {"messages": 0, "sessions": 0, "new_users": 0, "cases": 0})
            activity_map[date_str]["new_users"] = d["count"]
        for d in daily_cases:
            date_str = str(d["created_at__date"])
            activity_map.setdefault(date_str, {"messages": 0, "sessions": 0, "new_users": 0, "cases": 0})
            activity_map[date_str]["cases"] = d["count"]

        # Fill in missing dates with zeros
        from datetime import timedelta as _td
        current = start_date
        today = timezone.now().date()
        while current <= today:
            date_str = str(current)
            activity_map.setdefault(date_str, {"messages": 0, "sessions": 0, "new_users": 0, "cases": 0})
            current += _td(days=1)

        sorted_activity = [
            {"date": k, **v} for k, v in sorted(activity_map.items())
        ]

        return Response(
            {"days": days, "daily_activity": sorted_activity},
            status=status.HTTP_200_OK,
        )


class AdminTopQuestionsView(APIView):
    """
    GET /api/v1/admin/top-questions/
    Returns the most frequently asked user questions.
    Supports grouping by similarity (exact match) and time range filtering.
    """
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        from datetime import timedelta
        days = min(int(request.query_params.get("days", 30)), 365)
        limit = min(int(request.query_params.get("limit", 20)), 100)
        start_date = timezone.now().date() - timedelta(days=days)

        top_questions = (
            ChatMessage.objects.filter(
                role=ChatMessage.Role.USER,
                created_at__date__gte=start_date,
                content__gt="",
            )
            .values("content")
            .annotate(
                frequency=Count("id"),
                first_asked=Min("created_at"),
                last_asked=Max("created_at"),
                unique_users=Count("session__user", distinct=True),
            )
            .order_by("-frequency")[:limit]
        )

        results = [
            {
                "question": q["content"][:200],
                "frequency": q["frequency"],
                "unique_users": q["unique_users"],
                "first_asked": q["first_asked"].isoformat() if q["first_asked"] else None,
                "last_asked": q["last_asked"].isoformat() if q["last_asked"] else None,
            }
            for q in top_questions
        ]

        return Response(
            {"days": days, "total_questions": ChatMessage.objects.filter(
                role=ChatMessage.Role.USER,
                created_at__date__gte=start_date,
                content__gt="",
            ).count(), "top_questions": results},
            status=status.HTTP_200_OK,
        )


# ── Language Detection & Multilingual Support ──────────────────────────────

class DetectLanguageView(APIView):
    """
    POST /api/v1/detect-language/
    Detect the language of user-provided text using Gemini API + heuristics.
    Returns: detected language code, language name, and confidence hint.
    """
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser]

    def post(self, request):
        text = str(request.data.get("text", "")).strip()
        preferred = str(request.data.get("preferred", "")).strip().lower() or None

        if not text:
            return Response(
                {"error": "Text is required for language detection."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        from .services.language_support import detect_language, LANGUAGE_NAMES, SUPPORTED_LANGUAGES

        detected_code = detect_language(text, preferred=preferred)
        lang_name = LANGUAGE_NAMES.get(detected_code, "English")

        return Response({
            "detected_language": detected_code,
            "language_name": lang_name,
            "supported": detected_code in SUPPORTED_LANGUAGES,
            "original_text_preview": text[:100],
        }, status=status.HTTP_200_OK)


class SupportedLanguagesView(APIView):
    """
    GET /api/v1/supported-languages/
    Returns the list of all supported languages with names and flags.
    No authentication required — used by the frontend language switcher.
    """
    permission_classes = []
    authentication_classes = []

    def get(self, request):
        from .services.language_support import LANGUAGE_NAMES, SUPPORTED_LANGUAGES

        _FLAGS = {
            "en": "🇺🇸", "am": "🇪🇹", "om": "🇪🇹",
        }

        languages = [
            {
                "code": code,
                "name": LANGUAGE_NAMES.get(code, code),
                "flag": _FLAGS.get(code, "🌐"),
            }
            for code in sorted(SUPPORTED_LANGUAGES)
        ]

        gemini_ready = False
        try:
            from .services.gemini_service import gemini_available
            gemini_ready = gemini_available()
        except Exception:
            pass

        return Response({
            "languages": languages,
            "total": len(languages),
            "gemini_detection_available": gemini_ready,
        }, status=status.HTTP_200_OK)
