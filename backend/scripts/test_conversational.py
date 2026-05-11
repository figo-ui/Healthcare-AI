"""Test the conversational reply pipeline WITHOUT Django.

Directly imports RAG and replicates the pure-logic functions
from views.py, bypassing Django ORM entirely.
"""
import os
import sys
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # backend/
sys.path.insert(0, str(ROOT))

# ── Import RAG (no Django dependency) ───────────────────────────────
from guidance.services.rag import query_rag, _build_index

print("Loading RAG index...")
idx = _build_index()
print(f"  Docs: {len(idx.docs):,}")
sources = {}
for d in idx.docs:
    sources[d.source] = sources.get(d.source, 0) + 1
for k, v in sorted(sources.items()):
    print(f"    {k}: {v:,}")

# ── Replicate key constants from views.py ───────────────────────────
_SYMPTOM_KEYWORDS = re.compile(
    r"\b(headache|fever|cough|pain|nausea|dizzy|fatigue|rash|swelling|"
    r"bleeding|breath|chest|stomach|vomit|diarrh|constipat|sore throat|"
    r"body ache|joint|muscle|chills|appetite|insomnia|anxiety|depress|"
    r"migraine|allerg|asthma|diabet|blood pressure|cancer|infect|flu|"
    r"cold|heart|lung|kidney|liver|skin|eye|ear|throat|back|neck|"
    r"numb|tingl|burn|itch|lump|bump|bruise|wound|cut|bite|sting|"
    r"pregnan|period|menstrual|menopaus|urinat|bladder|bowel|"
    r"cholesterol|blood sugar|hypertension|hypotension|arrhythm|palpitat|"
    r"seizure|epileps|stroke|tumor|lesion|cyst|polyp|fracture|sprain|"
    r"strain|swollen|inflam|chronic|acute|symptom|condition|disease|"
    r"illness|sick|unwell|dizzy|faint|confus|memory|vision|"
    r"hearing|sleep|weight|appetite|thirst|urinat|sweat|tremor|"
    r"spasm|cramp|stiff|weak|tired|exhaust|letharg|malaise)\b",
    re.IGNORECASE,
)

_EMOTIONAL_KEYWORDS = re.compile(
    r"\b(anxious|worried|scared|afraid|fear|terrif|panic|nervous|"
    r"stressed|overwhelm|sad|depressed|hopeless|helpless|lonely|"
    r"isolated|frustrated|angry|upset|crying|tears|grief|mourning|"
    r"loss|confused|lost|desperate|suicidal|self.harm|hurt myself|"
    r"can't cope|don't know what to do|at my wits end)\b",
    re.IGNORECASE,
)


class _Intent:
    GREETING = "GREETING"
    IDENTITY = "IDENTITY"
    FOLLOW_UP_YES = "FOLLOW_UP_YES"
    FOLLOW_UP_NO = "FOLLOW_UP_NO"
    FOLLOW_UP_DETAIL = "FOLLOW_UP_DETAIL"
    EMOTIONAL = "EMOTIONAL"
    GRATITUDE = "GRATITUDE"
    FAREWELL = "FAREWELL"
    AFFIRMATION = "AFFIRMATION"
    SMALL_TALK = "SMALL_TALK"
    MEDICAL = "MEDICAL"


def _ml_classify_intent(text: str) -> tuple:
    """Use the trained dialogue classifier to predict intent."""
    try:
        import joblib
        model_path = ROOT / "models" / "dialogue_classifier_calibrated.joblib"
        vec_path = ROOT / "models" / "dialogue_tfidf_vectorizer.joblib"
        if model_path.exists() and vec_path.exists():
            model = joblib.load(model_path)
            vectorizer = joblib.load(vec_path)
            vec = vectorizer.transform([text])
            proba = model.predict_proba(vec)[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            classes = list(model.classes_)
            if best_idx < len(classes):
                label = str(classes[best_idx])
                return label, confidence
    except Exception as e:
        print(f"  [ML classifier error: {e}]")
    return None, 0.0


def _map_dialogue_label_to_intent(label: str) -> str:
    if not label:
        return _Intent.SMALL_TALK
    lower = label.lower().strip()
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
        "prevention", "exams and tests", "outlook",
        "frequency", "susceptibility", "inheritance", "genetic",
        "considerations", "stages", "research", "evidence_qa",
    )
    for indicator in medical_indicators:
        if indicator in lower:
            return _Intent.MEDICAL
    emotional_indicators = ("anxiety", "depression", "stress", "bipolar", "postpartum")
    for indicator in emotional_indicators:
        if indicator in lower:
            return _Intent.EMOTIONAL
    return _Intent.SMALL_TALK


def _classify_intent(text: str, recent_messages: list) -> str:
    lower = (text or "").strip().lower()
    if not lower:
        return _Intent.SMALL_TALK
    if re.match(r"^(hi|hello|hey|good\s*(morning|afternoon|evening)|greetings|howdy|yo|sup|what'?s\s*up|hola|salam|merhaba)\b", lower):
        return _Intent.GREETING
    if re.search(r"\b(who\s+are\s+you|what\s+are\s+you|your\s+name|what\s+can\s+you\s+do|what\s+do\s+you\s+do|are\s+you\s+a\s+(doctor|bot|ai|human|robot|machine))\b", lower):
        return _Intent.IDENTITY
    if re.search(r"\b(bye|goodbye|see\s+you|take\s+care|good\s+night|farewell|later|ciao|gotta\s+go)\b", lower):
        return _Intent.FAREWELL
    if re.search(r"\b(thank|thanks|thx|appreciate|grateful|helpful)\b", lower):
        return _Intent.GRATITUDE
    if _EMOTIONAL_KEYWORDS.search(text):
        return _Intent.EMOTIONAL
    if re.match(r"^(yes|yeah|yep|sure|ok|okay|right|correct|exactly|absolutely|indeed|affirmative|uh[\s-]*huh|mhm|mm[\s-]*hmm)\b", lower):
        return _Intent.FOLLOW_UP_YES
    if re.match(r"^(no|nope|nah|not\s+really|negative|n|I\s+don'?t\s+think\s+so)\b", lower):
        return _Intent.FOLLOW_UP_NO
    if re.search(r"\b(more\s+detail|tell\s+me\s+more|explain\s+more|what\s+else|anything\s+else|side\s+effect|how\s+long|when\s+did|how\s+often|what\s+about|can\s+you\s+also|also\s+I|I\s+also|additionally|furthermore|more\s+about)\b", lower):
        return _Intent.FOLLOW_UP_DETAIL
    if re.match(r"^(true|correct|right|exactly|precisely|that'?s\s+right|sounds\s+right|makes\s+sense|I\s+see|got\s+it|understood)\b", lower):
        return _Intent.AFFIRMATION
    if _SYMPTOM_KEYWORDS.search(text):
        return _Intent.MEDICAL
    if re.search(r"\b(how\s+are\s+you|what'?s\s+up|how'?s\s+it\s+going|nice\s+weather|how\s+do\s+you\s+do|good\s+day|bad\s+day)\b", lower):
        return _Intent.SMALL_TALK
    # ML supplement: dialogue classifier for ambiguous cases
    ml_label, ml_confidence = _ml_classify_intent(text)
    if ml_label and ml_confidence >= 0.3:
        ml_intent = _map_dialogue_label_to_intent(ml_label)
        if ml_intent == _Intent.MEDICAL:
            return _Intent.MEDICAL
        if ml_intent == _Intent.EMOTIONAL:
            return _Intent.EMOTIONAL
    return _Intent.SMALL_TALK


def _build_rag_query(text: str, recent: list) -> str:
    parts = [text]
    for msg in reversed(recent):
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", str(msg))
        if role == "assistant":
            parts.append(content[:150])
            break
        if role == "user":
            parts.append(content[:100])
            break
    return " ".join(parts)


def _extract_rag_knowledge(rag_hits: list, max_answer_len: int = 500) -> str:
    if not rag_hits:
        return ""
    parts = []
    for i, hit in enumerate(rag_hits[:2]):
        answer = (hit.get("metadata") or {}).get("answer", "").strip()
        question = (hit.get("metadata") or {}).get("question", "").strip()
        score = hit.get("score", 0)
        if not answer or score < 0.03:
            continue
        remaining = max_answer_len - sum(len(p) for p in parts)
        if remaining <= 50:
            break
        snippet = answer[:remaining].rstrip()
        if len(answer) > remaining:
            snippet += "…"
        if question and i == 0:
            parts.append(f"**{question}**\n{snippet}")
        elif question:
            parts.append(f"Related — **{question}**\n{snippet}")
        else:
            parts.append(snippet)
    return "\n\n".join(parts)


def _has_medical_context(recent: list) -> bool:
    for msg in recent[-6:]:
        content = getattr(msg, "content", str(msg))
        if _SYMPTOM_KEYWORDS.search(content):
            return True
    return False


def _build_contextual_acknowledgment(text: str, intent: str, recent: list) -> str:
    if intent == _Intent.EMOTIONAL:
        emotion_match = _EMOTIONAL_KEYWORDS.search(text)
        emotion_word = emotion_match.group(0) if emotion_match else "that"
        return f"I can hear that you're feeling {emotion_word}, and that's completely valid. "
    if intent in (_Intent.GREETING, _Intent.IDENTITY, _Intent.FAREWELL, _Intent.AFFIRMATION):
        return ""
    if intent == _Intent.GRATITUDE:
        return "I'm glad I could help. "
    if intent in (_Intent.FOLLOW_UP_YES, _Intent.FOLLOW_UP_NO, _Intent.FOLLOW_UP_DETAIL):
        if _has_medical_context(recent):
            return "Thanks for confirming. " if intent == _Intent.FOLLOW_UP_YES else "Understood. "
        return ""
    return ""


def _build_follow_up_prompt(text: str, intent: str, recent: list, rag_hits: list) -> str:
    in_medical_context = _has_medical_context(recent)
    if intent == _Intent.FAREWELL:
        return "Take care, and don't hesitate to come back if anything comes up."
    if intent == _Intent.GRATITUDE:
        return "If anything changes or you have more questions, I'm right here."
    if intent == _Intent.IDENTITY:
        return "What health concern can I help you with today?"
    if intent == _Intent.AFFIRMATION:
        return "Whenever you're ready, describe your symptoms and I'll analyze them."
    if rag_hits:
        best_q = (rag_hits[0].get("metadata") or {}).get("question", "").strip()
        if best_q and intent != _Intent.EMOTIONAL:
            return "If you'd like a personalized assessment based on your specific situation, describe your symptoms in detail and I'll run a full analysis."
    if in_medical_context:
        if intent == _Intent.FOLLOW_UP_YES:
            return "Could you tell me more — when did it start, how severe is it, and does anything make it better or worse?"
        if intent == _Intent.FOLLOW_UP_DETAIL:
            return "Could you describe the full picture — what you're feeling, when it started, and how severe it is on a scale of 1-10?"
        return "Can you describe what you're experiencing in more detail?"
    if intent == _Intent.EMOTIONAL:
        return "Can you tell me more about what you're going through? Even a rough description helps."
    return "Describe any symptoms you're experiencing and I'll analyze them for you."


def _build_conversational_reply(text: str, language: str = "en", session=None) -> str:
    recent = []
    intent = _classify_intent(text, recent)
    rag_query = _build_rag_query(text, recent)
    rag_hits = query_rag(rag_query, top_k=3)
    rag_knowledge = _extract_rag_knowledge(rag_hits)
    acknowledgment = _build_contextual_acknowledgment(text, intent, recent)
    follow_up = _build_follow_up_prompt(text, intent, recent, rag_hits)

    if intent == _Intent.GREETING and not _has_medical_context(recent):
        _INTRO = {
            "en": "I'm HealthAI, your medical assistant. ",
            "am": "እኔ HealthAI ነኝ፣ የሕክምና ረዳትዎ። ",
            "es": "Soy HealthAI, tu asistente médico. ",
            "fr": "Je suis HealthAI, votre assistant médical. ",
            "ar": "أنا HealthAI، مساعدك الطبي. ",
        }
        intro = _INTRO.get(language, _INTRO["en"])
        if rag_knowledge:
            return f"{intro}{acknowledgment}Here's some health information that may be relevant:\n\n{rag_knowledge}\n\n{follow_up}"
        return f"{intro}{follow_up}"

    if intent == _Intent.IDENTITY:
        capabilities = [
            "analyze symptoms and identify possible conditions",
            "assess risk levels (Low, Moderate, High)",
            "provide prevention and care recommendations",
            "locate nearby clinics and hospitals",
        ]
        cap_text = ", ".join(capabilities[:-1]) + f", and {capabilities[-1]}"
        identity_line = f"I'm HealthAI — an AI medical assistant. I can {cap_text}. "
        disclaimer = "I provide informational guidance, not a medical diagnosis. "
        if rag_knowledge:
            return f"{identity_line}{disclaimer}\n\n{rag_knowledge}\n\n{follow_up}"
        return f"{identity_line}{disclaimer}{follow_up}"

    if rag_knowledge:
        return f"{acknowledgment}{rag_knowledge}\n\n{follow_up}"

    return f"{acknowledgment}{follow_up}"


# ═══════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1. Intent Classification")
print("=" * 70)

intent_tests = [
    "Hello",
    "Who are you?",
    "I have a headache and feel dizzy",
    "I feel sad and anxious",
    "What causes high blood pressure?",
    "Thank you so much",
    "Goodbye",
    "Yes, that's right",
    "No, not really",
    "What about side effects?",
    "How are you?",
    "Can antibiotics cause diarrhea?",
    "My stomach hurts after eating",
]

for text in intent_tests:
    intent = _classify_intent(text, [])
    ml_label, ml_conf = _ml_classify_intent(text)
    ml_str = f"ML:{ml_label}({ml_conf:.2f})" if ml_label else "ML:none"
    print(f"  [{intent:20s}] <- {text}  [{ml_str}]")

print("\n" + "=" * 70)
print("2. Full Conversational Replies (RAG-powered)")
print("=" * 70)

reply_tests = [
    ("Hello", "en"),
    ("Who are you?", "en"),
    ("I have a headache and feel dizzy", "en"),
    ("I feel sad and anxious", "en"),
    ("What causes high blood pressure?", "en"),
    ("Thank you", "en"),
    ("Goodbye", "en"),
    ("Yes, that's right", "en"),
    ("Can antibiotics cause diarrhea?", "en"),
    ("My stomach hurts after eating", "en"),
]

for text, lang in reply_tests:
    reply = _build_conversational_reply(text, language=lang, session=None)
    intent = _classify_intent(text, [])
    print(f"\n  [{intent}] Q: {text}")
    for i in range(0, len(reply), 120):
        print(f"     {reply[i:i+120]}")
    print()

print("=" * 70)
print("All tests complete!")
