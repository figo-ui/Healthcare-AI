import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


AMHARIC_SCRIPT_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F]")
ETHIOPIC_EXT_RE   = re.compile(r"[\u2D80-\u2DDF\uAB01-\uAB2F]")
WHITESPACE_RE     = re.compile(r"\s+")

SUPPORTED_LANGUAGES = {
    "en",  # English
    "am",  # Amharic
    "om",  # Afaan Oromoo
}

# Human-readable names for UI display
LANGUAGE_NAMES = {
    "en": "English",
    "am": "አማርኛ (Amharic)",
    "om": "Afaan Oromoo",
}

# ---------------------------------------------------------------------------
# Static UI strings  {key: {lang: text}}
# ---------------------------------------------------------------------------
COMMON_STRINGS: Dict[str, Dict[str, str]] = {
    "disclaimer": {
        "en": "This is not medical advice.",
        "am": "ይህ የሕክምና ምክር አይደለም።",
        "om": "Kun gorsa fayyaa miti.",
    },
    "seek_professional_care": {
        "en": "Seek professional care.",
        "am": "የሙያ ሕክምና እርዳታ ይፈልጉ።",
        "om": "Gargaarsa ogummaa barbaadi.",
    },
    "risk_low":    {"en": "Low",    "am": "ዝቅተኛ",    "om": "Gadi"},
    "risk_medium": {"en": "Medium", "am": "መካከለኛ",   "om": "Giddugaleessa"},
    "risk_high":   {"en": "High",   "am": "ከፍተኛ",    "om": "Ol'aanaa"},
    "current_sources_checked": {
        "en": "Current sources checked",
        "am": "የተመረመሩ ወቅታዊ ምንጮች",
        "om": "Madda ammaa ilaalame",
    },
    "freshness_note": {
        "en": "This answer used current search results because the query needs up-to-date information.",
        "am": "ይህ መልስ ወቅታዊ መረዳ ⍝ለምፈለጉ አዳዲስ የፍለጋ ውጤቶችን ተጠቀሞ።",
        "om": "Deebiin kun bu'aa barbaachaa ammaa fayyadame sababii gaaffichi odeeffannoo haaraa barbaadaaf.",
    },
    "monitor": {
        "en": "Monitor symptoms and seek care if they worsen or persist.",
        "am": "ምልክቶችን ይከታተሉ፡ ከባድ ከሆኑ ወይም ከቀጠሉ ሕክምና ይፈልጉ።",
        "om": "Mallattoolee hordofi; yoo hammaatan ykn itti fufsiisuuf gargaarsa barbaadi.",
    },
    "same_day": {
        "en": "Arrange same-day or next-day clinical care.",
        "am": "በዚያው ቀን ወይም በሚቀጥለው ቀን ክሊኒካዊ እንክብካቤ ያዘጋጁ።",
        "om": "Kunuunsa kilinikaa guyyaa sanaa ykn guyyaa itti aanuuf qopheessi.",
    },
    "urgent": {
        "en": "Seek urgent in-person or emergency care now.",
        "am": "አሁኑኑ አስቸኳይ የቀጥታ ወይም የድንገተኛ ሕክምና ይፈልጉ።",
        "om": "Amma gargaarsa hatattamaa ykn yeroo ariifachiisaa barbaadi.",
    },
    "assistant_open_low": {
        "en": "Thanks for sharing your symptoms.",
        "am": "ምልክቶችዎን ⍝ለገለጹ እናመሰጋናለን።",
        "om": "Mallattoolee kee waan qooddeef galatoomi.",
    },
    "assistant_open_medium": {
        "en": "Thank you for explaining your symptoms clearly.",
        "am": "ምልክቶችዎን በግልፅ ሁናተ ⍝ለገለጹ እናመሰጋናለን።",
        "om": "Mallattoolee kee ifatti ibsiteef galatoomi.",
    },
    "assistant_open_high": {
        "en": "Thank you for sharing these details quickly.",
        "am": "እነዚህን ዝርዝሮች በፍጥነት ⍝ለካፈሉ እናመሰጋናለን።",
        "om": "Bal'inni kana ariitiidhaan qooddeef galatoomi.",
    },
    "possible_causes": {
        "en": "Possible causes from your symptoms",
        "am": "ከምልክቶችዎ የሚገመቱ ምክንያቶች",
        "om": "Sababa mallattoowwan keetii irraa ta'uu danda'u",
    },
    "risk_level_line": {
        "en": "Current risk level",
        "am": "የአሁኑ የአደጋ ደርጃ",
        "om": "Sadarkaa balaa ammaa",
    },
    "recommended_next_step": {
        "en": "Recommended next step",
        "am": "የሚመከረው ቀጣይ እርምጃ",
        "om": "Tarkaanfii itti aanu kan gorsamee",
    },
    "probable_conditions_label": {
        "en": "Probable conditions",
        "am": "የሚሆኑ ሕመሞች",
        "om": "Dhukkuboota ta'uu danda'an",
    },
    "confidence_label": {
        "en": "confidence",
        "am": "እርግጠኝነት",
        "om": "amantaa",
    },
    "red_flags_label": {
        "en": "Warning signs",
        "am": "የማስጠንቀቂያ ምልክቶች",
        "om": "Mallattoolee akeekkachiisaa",
    },
    "recommendation_label": {
        "en": "What to do next",
        "am": "ቀጥሎ ምን ማድረግ",
        "om": "Itti aansee maal gochuu",
    },
    "not_a_diagnosis": {
        "en": "This is not a diagnosis — please see a doctor.",
        "am": "ይህ ምርመራ አይደለም — እባክዎ ሐኪም ያማክሩ።",
        "om": "Kun dhukkuba adda baasuu miti — maaloo dokitara barbaadi.",
    },
    "seek_emergency_now": {
        "en": "Seek emergency care immediately.",
        "am": "አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ።",
        "om": "Amma gargaarsa hatattamaa barbaadi.",
    },
    "high_risk_warning": {
        "en": "Your symptoms suggest a high-risk situation. Please seek medical attention promptly.",
        "am": "ምልክቶችዎ ከፍተኛ አደጋ ያሳያሉ። እባክዎ ወዲያውኑ ሕክምና ይፈልጉ።",
        "om": "Mallattooleen kee haala balaa ol'aanaa agarsiisu. Maaloo hatattamaan gargaarsa fayyaa barbaadi.",
    },
    "low_risk_message": {
        "en": "Your symptoms appear low risk. Monitor and seek care if they worsen.",
        "am": "ምልክቶችዎ ዝቅተኛ አደጋ ያሳያሉ። ይከታተሉ፤ ከባሱ ሕክምና ይፈልጉ።",
        "om": "Mallattooleen kee balaa gadi agarsiisu. Hordofi; yoo hammaatan gargaarsa barbaadi.",
    },
    "medium_risk_message": {
        "en": "Your symptoms suggest moderate risk. Arrange a clinic visit soon.",
        "am": "ምልክቶችዎ መካከለኛ አደጋ ያሳያሉ። ቶሎ ክሊኒክ ይሂዱ።",
        "om": "Mallattooleen kee balaa giddugaleessa agarsiisu. Dafanii kilinika deemi.",
    },
}

# ---------------------------------------------------------------------------
# Comprehensive phrase translation map
# ---------------------------------------------------------------------------
_PHRASE_MAP: Dict[str, Dict[str, str]] = {
    # Emergency patterns
    "possible stroke pattern": {
        "am": "የሚቻካይ ስትሮክ ምልክት",
        "om": "mallattoo stroke ta'uu danda'u",
    },
    "possible cardiac emergency pattern": {
        "am": "የሚቻካይ የልብ ድንገተኛ ምልክት",
        "om": "mallattoo hatattamaa onnee ta'uu danda'u",
    },
    "possible sepsis pattern": {
        "am": "የሚቻካይ ሴፕሲስ ምልክት",
        "om": "mallattoo sepsis ta'uu danda'u",
    },
    "possible anaphylaxis pattern": {
        "am": "የሚቻካይ አናፊላክሲስ ምልክት",
        "om": "mallattoo anaphylaxis ta'uu danda'u",
    },
    "possible kidney infection pattern": {
        "am": "የሚቻካይ የኩላሊት ኢንፌክሽን ምልክት",
        "om": "mallattoo dhukkuba kalee ta'uu danda'u",
    },
    "possible drug reaction pattern": {
        "am": "የሚቻካይ የመድሃኒት ምላሽ ምልክት",
        "om": "mallattoo deebii qorichaa ta'uu danda'u",
    },
    "possible urinary infection": {
        "am": "የሚቻካይ የሽንት ኢንፌክሽን",
        "om": "dhukkuba fincaan ta'uu danda'u",
    },
    "possible pneumonia pattern": {
        "am": "የሚቻካይ ሳምባ ምች ምልክት",
        "om": "mallattoo pneumonia ta'uu danda'u",
    },
    "possible hypertensive pattern": {
        "am": "የሚቻካይ ከፍተኛ ደም ግፊት ምልክት",
        "om": "mallattoo dhiibbaa dhiigaa ol'aanaa ta'uu danda'u",
    },
    "acute onset": {
        "am": "ድንገተኛ መጀመሪያ",
        "om": "jalqaba hatattamaa",
    },
    "pediatric high-risk age group (infant)": {
        "am": "ህፃን ልጅ ከፍተኛ አደጋ ዕድሜ ቡድን",
        "om": "garee umrii balaa ol'aanaa (daa'ima)",
    },
    "pediatric fever risk": {
        "am": "የህፃን ትኩሳት አደጋ",
        "om": "balaa ho'a daa'imaa",
    },
    # Actions
    "call emergency services": {
        "am": "የድንገተኛ አደጋ አገልግሎቶችን ይደውሉ",
        "om": "tajaajila hatattamaa bilbili",
    },
    "go to the emergency department": {
        "am": "ወደ ድንገተኛ ክፍል ይሂዱ",
        "om": "kutaa hatattamaa deemi",
    },
    "seek immediate emergency care": {
        "am": "አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ",
        "om": "amma gargaarsa hatattamaa barbaadi",
    },
    "seek urgent care": {
        "am": "አስቸኳይ ሕክምና ይፈልጉ",
        "om": "gargaarsa hatattamaa barbaadi",
    },
    "seek urgent in-person medical care": {
        "am": "አስቸኳይ የቀጥታ ሕክምና ይፈልጉ",
        "om": "gargaarsa fayyaa hatattamaa barbaadi",
    },
    "seek urgent": {
        "am": "አስቸኳይ ይፈልጉ",
        "om": "hatattamaa barbaadi",
    },
    "seek professional care": {
        "am": "የሙያ ሕክምና እርዳታ ይፈልጉ",
        "om": "gargaarsa ogummaa barbaadi",
    },
    "seek care": {
        "am": "ሕክምና ይፈልጉ",
        "om": "gargaarsa barbaadi",
    },
    "arrange a same-day": {
        "am": "በዚያው ቀን ያዘጋጁ",
        "om": "guyyaa sanaa qopheessi",
    },
    "arrange same-day": {
        "am": "በዚያው ቀን ያዘጋጁ",
        "om": "guyyaa sanaa qopheessi",
    },
    "monitor symptoms": {
        "am": "ምልክቶችን ይከታተሉ",
        "om": "mallattoolee hordofi",
    },
    "monitor and seek care": {
        "am": "ይከታተሉ እና ሕክምና ይፈልጉ",
        "om": "hordofi fi gargaarsa barbaadi",
    },
    # Conditions
    "heart attack": {
        "am": "የልብ ድካም",
        "om": "dhukkuba onnee",
    },
    "stroke": {
        "am": "ስትሮክ",
        "om": "stroke",
    },
    "anaphylaxis": {
        "am": "አናፊላክሲስ",
        "om": "anaphylaxis",
    },
    "sepsis": {
        "am": "ሴፕሲስ",
        "om": "sepsis",
    },
    "pneumonia": {
        "am": "ሳምባ ምች",
        "om": "pneumonia",
    },
    # Descriptors
    "emergency pattern detected": {
        "am": "ድንገተኛ ምልክት ተገኝቷል",
        "om": "mallattoo hatattamaa argame",
    },
    "emergency": {
        "am": "ድንገተኛ",
        "om": "hatattamaa",
    },
    "high-risk": {
        "am": "ከፍተኛ አደጋ",
        "om": "balaa ol'aanaa",
    },
    "high risk": {
        "am": "ከፍተኛ አደጋ",
        "om": "balaa ol'aanaa",
    },
    "moderate risk": {
        "am": "መካከለኛ አደጋ",
        "om": "balaa giddugaleessa",
    },
    "low risk": {
        "am": "ዝቅተኛ አደጋ",
        "om": "balaa gadi",
    },
    "immediately": {
        "am": "አሁኑኑ",
        "om": "amma",
    },
    "same-day": {
        "am": "በዚያው ቀን",
        "om": "guyyaa sanaa",
    },
    "next-day": {
        "am": "በሚቀጥለው ቀን",
        "om": "guyyaa itti aanu",
    },
    "today": {
        "am": "ዛሬ",
        "om": "har'a",
    },
    "clinic visit": {
        "am": "ክሊኒክ ጉብኝት",
        "om": "daawwannaa kilinikaa",
    },
    "clinical evaluation": {
        "am": "ክሊኒካዊ ምዘና",
        "om": "madaallii kilinikaa",
    },
    "urine testing": {
        "am": "የሽንት ምርመራ",
        "om": "qorannoo fincaanii",
    },
    "stop the suspected medication": {
        "am": "የተጠረጠረውን መድሃኒት ያቁሙ",
        "om": "qoricha shakkame dhaabi",
    },
    "breathing difficulty": {
        "am": "የትንፋሽ ችግር",
        "om": "rakkoo hafuura",
    },
    "swelling worsens": {
        "am": "እብጠቱ ከባሱ",
        "om": "dhiitaan hammaate",
    },
    "low confidence result": {
        "am": "ዝቅተኛ እርግጠኝነት ውጤት",
        "om": "bu'aa amantaa gadi",
    },
    "clinical consultation": {
        "am": "ክሊኒካዊ ምክክር",
        "om": "mari'annaa kilinikaa",
    },
    "proper evaluation": {
        "am": "ትክክለኛ ምዘና",
        "om": "madaallii sirrii",
    },
    "warning signs present": {
        "am": "የማስጠንቀቂያ ምልክቶች አሉ",
        "om": "mallattoolee akeekkachiisaa jiru",
    },
    "professional assessment": {
        "am": "ሙያዊ ምዘና",
        "om": "madaallii ogummaa",
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _t(key: str, lang: str) -> str:
    """Return a localised string, falling back to English."""
    bucket = COMMON_STRINGS.get(key, {})
    return bucket.get(lang) or bucket.get("en", "")


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

def detect_language(text: str, preferred: Optional[str] = None) -> str:
    """Detect the language of *text*, optionally honouring a *preferred* hint.

    Detection priority:
      1. Explicit *preferred* hint (if it's a supported language)
      2. Fast script/heuristic checks (Amharic script, Oromo markers)
      3. Gemini API detection (accurate for en/am/om + 18 other languages)
      4. langdetect library fallback
      5. Default to "en"
    """
    if preferred and preferred.lower() in SUPPORTED_LANGUAGES:
        return preferred.lower()

    text = str(text or "")
    if not text.strip():
        return preferred.lower() if preferred and preferred.lower() in SUPPORTED_LANGUAGES else "en"

    # Fast heuristic: Amharic script
    if AMHARIC_SCRIPT_RE.search(text) or ETHIOPIC_EXT_RE.search(text):
        return "am"

    # Fast heuristic: Oromo markers
    oromo_markers = {"galatoomi", "barbaadi", "fayyaa", "danda", "itti", "guyyaa", "kana"}
    lower = text.lower()
    if any(marker in lower for marker in oromo_markers):
        return "om"

    # Gemini API detection (primary for all other languages)
    try:
        from .gemini_service import detect_language_gemini
        detected = detect_language_gemini(text, preferred=preferred)
        if detected and detected in SUPPORTED_LANGUAGES:
            return detected
    except Exception as exc:
        logger.debug("Gemini language detection skipped: %s", exc)

    # langdetect fallback
    try:
        from langdetect import detect as ld_detect
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        detected = ld_detect(text)
        _iso_map = {"en": "en", "am": "am", "om": "om"}
        code = _iso_map.get(detected, detected)
        if code in SUPPORTED_LANGUAGES:
            return code
    except Exception:
        pass

    return "en"


# ---------------------------------------------------------------------------
# normalize_text_for_models
# ---------------------------------------------------------------------------

def normalize_text_for_models(text: str, language: str) -> str:
    """Normalise text for downstream ML models."""
    text = str(text or "")
    return WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# translate_dynamic_text
# ---------------------------------------------------------------------------

def translate_dynamic_text(text: str, language: str) -> str:
    """Phrase-level translation of dynamic recommendation/flag text.

    Applies longest-match substitution so multi-word phrases take priority.
    For English returns the original unchanged.
    """
    if language == "en" or not language:
        return text

    result = str(text or "")

    # Static phrase-map translation (am, om)
    if language in ("am", "om"):
        for phrase, translations in sorted(_PHRASE_MAP.items(), key=lambda kv: -len(kv[0])):
            target = translations.get(language)
            if not target:
                continue
            if phrase.lower() in result.lower():
                result = re.sub(re.escape(phrase), target, result, flags=re.IGNORECASE)
        return result

    return result


# ---------------------------------------------------------------------------
# localize_analysis_result
# ---------------------------------------------------------------------------

def localize_analysis_result(result: Dict, language: str) -> Dict:
    """Localise all user-facing strings in an analysis result dict.

    Returns a shallow copy — original is not mutated.
    Translates: risk_level_label, disclaimer_text, recommendation_text,
    red_flags, prevention_advice, risk_factors, and adds a localised
    assistant_summary field.
    """
    if language == "en":
        return result

    localised = dict(result)

    # Risk level label
    risk_level = str(result.get("risk_level", "Low"))
    risk_key = "risk_{}".format(risk_level.lower())
    localised_risk = _t(risk_key, language)
    if localised_risk:
        localised["risk_level_label"] = localised_risk

    # Disclaimer
    disclaimer = _t("disclaimer", language)
    if disclaimer:
        localised["disclaimer_text"] = disclaimer

    # Recommendation text
    rec = str(result.get("recommendation_text", ""))
    if rec:
        localised["recommendation_text"] = translate_dynamic_text(rec, language)

    # Red flags
    red_flags = result.get("red_flags", [])
    if red_flags and isinstance(red_flags, list):
        localised["red_flags"] = [translate_dynamic_text(str(f), language) for f in red_flags]

    # Prevention advice
    prevention = result.get("prevention_advice", [])
    if prevention and isinstance(prevention, list):
        localised["prevention_advice"] = [translate_dynamic_text(str(p), language) for p in prevention]

    # Risk factors
    risk_factors = result.get("risk_factors", [])
    if risk_factors and isinstance(risk_factors, list):
        localised["risk_factors"] = [translate_dynamic_text(str(f), language) for f in risk_factors]

    # Add a fully localised assistant summary
    localised["localised_summary"] = build_assistant_summary(localised, language)

    return localised


# ---------------------------------------------------------------------------
# build_assistant_summary
# ---------------------------------------------------------------------------

def build_assistant_summary(result: Dict, language: str = "en") -> str:
    """Build a complete, localised assistant summary from an analysis result.

    This is the primary function used when language != "en". It produces
    a structured, human-readable response entirely in the target language.
    """
    risk_level = str(result.get("risk_level", "Low"))
    risk_key = "risk_{}".format(risk_level.lower())
    risk_label = _t(risk_key, language) or risk_level

    # Opening sentence
    open_key = "assistant_open_{}".format(risk_level.lower())
    opening = _t(open_key, language) or _t("assistant_open_low", language)

    # Probable conditions
    conditions: List[Dict] = result.get("probable_conditions", [])[:3]
    condition_names: List[str] = [
        str(c.get("condition", "")).strip()
        for c in conditions
        if str(c.get("condition", "")).strip()
    ]

    # Localised labels
    risk_line_label = _t("risk_level_line", language)
    possible_causes_label = _t("possible_causes", language)
    next_step_label = _t("recommended_next_step", language)
    confidence_label = _t("confidence_label", language)
    not_diagnosis = _t("not_a_diagnosis", language)

    # Recommendation (already translated if localize_analysis_result was called)
    recommendation = str(result.get("recommendation_text", ""))
    localised_rec = translate_dynamic_text(recommendation, language) if recommendation else ""

    # Red flags (already translated)
    red_flags = result.get("red_flags", [])
    red_flags_label = _t("red_flags_label", language)

    # Action string based on risk level
    risk_lower = risk_level.lower()
    if risk_lower == "high":
        action = _t("urgent", language)
    elif risk_lower in ("medium", "moderate"):
        action = _t("same_day", language)
    else:
        action = _t("monitor", language)

    # Build the summary
    parts: List[str] = []

    if opening:
        parts.append(opening)

    # Conditions with confidence
    if condition_names and possible_causes_label:
        cond_parts = []
        for c in conditions[:3]:
            name = str(c.get("condition", "")).strip()
            prob = float(c.get("probability", 0)) * 100
            if name:
                cond_parts.append("{} ({:.0f}% {})".format(name, prob, confidence_label))
        if cond_parts:
            parts.append("{}: {}.".format(possible_causes_label, ", ".join(cond_parts)))

    # Risk level
    if risk_line_label:
        parts.append("{}: {}.".format(risk_line_label, risk_label))

    # Red flags
    if red_flags and red_flags_label:
        flags_text = ", ".join(str(f) for f in red_flags[:3])
        parts.append("{}: {}.".format(red_flags_label, flags_text))

    # Action
    if action:
        parts.append(action)

    # Recommendation
    if localised_rec and next_step_label:
        parts.append("{}: {}".format(next_step_label, localised_rec))
    elif localised_rec:
        parts.append(localised_rec)

    # Disclaimer
    if not_diagnosis:
        parts.append(not_diagnosis)

    return " ".join(p for p in parts if p.strip())
