import re
from typing import Dict, List, Optional

from .clinical_safety import build_safety_summary
from .clinical_protocol import MANDATORY_DISCLAIMER

# ── Temporal urgency patterns ──────────────────────────────────────────────
_ACUTE_ONSET_PATTERNS = [
    "started today", "sudden", "acute", "just now", "minutes ago", "hours ago",
    "suddenly", "abruptly", "all of a sudden", "out of nowhere", "this morning",
    "this afternoon", "this evening", "tonight", "right now", "immediately",
]
_CHRONIC_PATTERNS = [
    "for weeks", "chronic", "months", "for months", "for years", "long-standing",
    "ongoing", "persistent for", "recurring", "recurrent", "years ago",
]


SEVERITY_PRIOR = {
    "eczema": 0.35,
    "psoriasis": 0.45,
    "acne": 0.20,
    "fungal infection": 0.55,
    "contact dermatitis": 0.30,
    "cellulitis": 0.75,
    "melanoma": 0.95,
    "malaria": 0.75,
    # Safety override conditions — must have high severity priors
    "stroke": 0.95,
    "paralysis (brain hemorrhage)": 0.95,
    "heart attack": 0.95,
    "unstable angina": 0.90,
    "possible nstemi / stemi": 0.90,
    "pulmonary embolism": 0.90,
    "acute pulmonary embolism (disorder)": 0.90,
    "possible aortic dissection": 0.95,
    "sepsis caused by virus (disorder)": 0.90,
    "possible meningitis": 0.85,
    "possible ectopic pregnancy": 0.90,
    "possible appendicitis": 0.75,
    "possible diabetic ketoacidosis": 0.90,
    "anaphylaxis": 0.95,
    "spontaneous pneumothorax": 0.80,
    "possible gastrointestinal bleed": 0.85,
    "peptic ulcer disease": 0.70,
    "possible bowel obstruction": 0.80,
    "epiglottitis": 0.85,
    "bronchospasm / acute asthma exacerbation": 0.80,
    "bronchial asthma": 0.65,
    "pneumonia": 0.65,
    "urinary tract infection": 0.50,
    "cystitis": 0.45,
    "escherichia coli urinary tract infection": 0.50,
    "drug reaction": 0.60,
    "adverse drug reaction": 0.65,
    "allergy": 0.45,
    "allergic dermatitis": 0.40,
    "urticaria": 0.40,
    "hypertension": 0.55,
    "essential hypertension": 0.50,
    "hypertensive crisis": 0.80,
    "peripheral neuropathy": 0.45,
    "diabetic neuropathy": 0.55,
    "neuropathy": 0.45,
    "carpal tunnel syndrome": 0.30,
    "osteoarthritis": 0.30,
    "rheumatoid arthritis": 0.45,
    "arthritis": 0.35,
    "gout": 0.45,
    "common cold": 0.15,
    "viral pharyngitis": 0.20,
    "acute viral pharyngitis (disorder)": 0.20,
    "viral sinusitis (disorder)": 0.20,
    "panic attack": 0.25,
    "chronic fatigue syndrome": 0.30,
    "anemia": 0.35,
    "hypothyroidism": 0.30,
    "depression": 0.30,
    "iron deficiency anemia": 0.35,
    "lumbar strain": 0.25,
    "back pain": 0.25,
    "muscle strain": 0.20,
    "herniated disc": 0.45,
    "sciatica": 0.40,
    "insomnia": 0.20,
    "generalized anxiety disorder": 0.25,
    "anxiety disorder": 0.25,
}

FORCED_HIGH_CONDITIONS = {
    "melanoma",
}

# ── Red-flag terms split by clinical severity ──────────────────────────
# Emergency red flags: symptoms that indicate life-threatening conditions
# and should always escalate risk to High and trigger urgent care.
EMERGENCY_RED_FLAG_TERMS = [
    "chest pain",
    "trouble breathing",
    "shortness of breath",
    "fainting",
    "severe bleeding",
    "unconscious",
    "slurred speech",
    "one side weakness",
    "one-side weakness",
    "one sided weakness",
    "unilateral weakness",
    "facial droop",
    "face droop",
    "throat swelling",
    "lip swelling",
    "tongue swelling",
    "anaphylaxis",
]

# Warning red flags: symptoms that warrant prompt (same-day) evaluation
# but are NOT inherently life-threatening on their own.
WARNING_RED_FLAG_TERMS = [
    "high fever",
    "rapid swelling",
    "confusion",
    "wheezing",
    "after medication",
    "drug reaction",
]

RED_FLAG_TERMS = EMERGENCY_RED_FLAG_TERMS + WARNING_RED_FLAG_TERMS

GENERIC_CONDITION_RE = re.compile(r"^(condition\s+\d+|class_\d+)$", re.IGNORECASE)

DISCLAIMER_TEXT = MANDATORY_DISCLAIMER


def detect_red_flags(symptom_text: str) -> List[str]:
    text = symptom_text.lower()
    flags = [term for term in RED_FLAG_TERMS if term in text]
    if _stroke_pattern(text):
        flags.append("possible stroke pattern")
    if _acute_coronary_pattern(text):
        flags.append("possible cardiac emergency pattern")
    if _sepsis_pattern(text):
        flags.append("possible sepsis pattern")
    if _kidney_fever_pattern(text):
        flags.append("possible kidney infection pattern")
    if _drug_reaction_pattern(text):
        flags.append("possible drug reaction pattern")
    safety_summary = build_safety_summary(symptom_text)
    flags.extend(safety_summary["red_flags"])
    flags.extend(safety_summary.get("warning_flags", []))
    return sorted(set(flags))


def _classify_red_flags(flags: List[str]) -> tuple[list[str], list[str]]:
    """Split red flags into emergency and warning categories.

    Emergency flags indicate life-threatening conditions (stroke, MI, sepsis,
    anaphylaxis, GI bleed, etc.) and should always escalate to High risk.
    Warning flags indicate conditions needing same-day care but not
    necessarily emergency services.
    """
    emergency_keywords = {
        "stroke pattern", "cardiac emergency pattern", "sepsis pattern",
        "anaphylaxis pattern", "pulmonary embolism pattern",
        "aortic dissection pattern", "gastrointestinal bleed pattern",
        "bowel obstruction pattern", "airway emergency pattern",
        "pneumothorax pattern", "diabetic ketoacidosis pattern",
        "ectopic pregnancy pattern", "appendicitis pattern",
        "severe asthma pattern",
    }
    warning_keywords = {
        "hypertensive pattern", "kidney infection pattern",
        "urinary infection", "pneumonia pattern",
        "drug reaction pattern", "neuropathy pattern",
    }
    emergency = []
    warning = []
    for flag in flags:
        flag_lower = flag.lower()
        # Check if flag matches any emergency keyword
        is_emergency = any(kw in flag_lower for kw in emergency_keywords)
        # Also check against the raw emergency terms list
        if not is_emergency:
            is_emergency = flag_lower in [t.lower() for t in EMERGENCY_RED_FLAG_TERMS]
        # Check if flag matches warning keywords
        is_warning = any(kw in flag_lower for kw in warning_keywords)
        if not is_warning:
            is_warning = flag_lower in [t.lower() for t in WARNING_RED_FLAG_TERMS]

        if is_emergency:
            emergency.append(flag)
        elif is_warning:
            warning.append(flag)
        else:
            # Unknown flags default to warning (conservative but not alarmist)
            warning.append(flag)
    return emergency, warning


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _risk_level(score: float) -> str:
    if score > 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"


def _stroke_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    neuro_terms = [
        "slurred speech",
        "one side weakness",
        "one-side weakness",
        "one sided weakness",
        "unilateral weakness",
        "facial droop",
        "face droop",
        "difficulty speaking",
        "cannot speak",
    ]
    return any(term in text for term in neuro_terms)


def _acute_coronary_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    has_chest_pain = "chest pain" in text or "chest pressure" in text or "pressure in chest" in text
    has_associated = any(
        term in text
        for term in [
            "trouble breathing",
            "shortness of breath",
            "sweating",
            "sweat",
            "left arm pain",
            "jaw pain",
        ]
    )
    return has_chest_pain and has_associated


def _sepsis_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    has_infection = any(term in text for term in ["fever", "chills", "infection"])
    has_systemic = any(term in text for term in ["confusion", "unconscious", "low blood pressure", "very weak"])
    return has_infection and has_systemic


def _kidney_fever_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    has_fever = "fever" in text or "chills" in text
    has_kidney_zone = any(term in text for term in ["kidney", "flank", "lower back", "back pain"])
    persistent_hint = any(term in text for term in ["long time", "for days", "for weeks", "persistent"])
    urinary_hint = any(term in text for term in ["burning urination", "frequent urination", "dysuria", "urine"])
    return has_fever and has_kidney_zone and (persistent_hint or urinary_hint)


def _drug_reaction_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    has_skin = any(term in text for term in ["rash", "itching", "itchy", "hives", "urticaria", "skin reaction", "swelling"])
    has_drug = any(term in text for term in [
        "after medication", "after drug", "after pill", "after medicine",
        "medication", "side effect", "drug reaction", "drug rash",
        "antibiotic", "penicillin", "adverse", "pill", "medicine",
    ])
    return has_skin and has_drug


def _temporal_urgency_modifier(symptom_text: str) -> tuple[float, Optional[str]]:
    """Return (score_boost, flag_label) based on temporal onset language.

    Acute onset → +0.15 boost and 'acute onset' red flag.
    Chronic/weeks → -0.05 modifier (reduces urgency slightly).
    """
    text = symptom_text.lower()
    for pattern in _ACUTE_ONSET_PATTERNS:
        if pattern in text:
            return 0.15, "acute onset"
    for pattern in _CHRONIC_PATTERNS:
        if pattern in text:
            return -0.05, None
    return 0.0, None


def _vital_signs_risk(
    heart_rate: Optional[int] = None,
    blood_pressure_systolic: Optional[int] = None,
    blood_pressure_diastolic: Optional[int] = None,
    temperature_celsius: Optional[float] = None,
    spo2_percent: Optional[int] = None,
) -> tuple[float, str, List[str]]:
    """Assess risk from vital signs.

    Returns (risk_boost, risk_level_override_or_empty, vital_red_flags).
    """
    boost = 0.0
    level_override = ""
    flags: List[str] = []

    if spo2_percent is not None:
        if spo2_percent < 94:
            boost = max(boost, 0.5)
            level_override = "High"
            flags.append(f"low oxygen saturation (SpO2 {spo2_percent}%)")

    if blood_pressure_systolic is not None:
        if blood_pressure_systolic > 180:
            boost = max(boost, 0.5)
            level_override = "High"
            flags.append(f"hypertensive crisis (systolic {blood_pressure_systolic} mmHg)")
        elif blood_pressure_systolic < 80:
            boost = max(boost, 0.5)
            level_override = "High"
            flags.append(f"critically low blood pressure (systolic {blood_pressure_systolic} mmHg)")

    if heart_rate is not None:
        if heart_rate > 150:
            boost = max(boost, 0.5)
            level_override = "High"
            flags.append(f"severe tachycardia (HR {heart_rate} bpm)")
        elif heart_rate < 40:
            boost = max(boost, 0.5)
            level_override = "High"
            flags.append(f"severe bradycardia (HR {heart_rate} bpm)")

    if temperature_celsius is not None:
        if temperature_celsius > 39.5:
            boost = max(boost, 0.15)
            if not level_override:
                level_override = "Medium"
            flags.append(f"high fever ({temperature_celsius:.1f}°C)")

    return boost, level_override, flags


def _pediatric_adjustment(
    age: Optional[int],
    symptom_text: str,
    risk_score: float,
    risk_level: str,
    red_flags: List[str],
    risk_factors: List[str],
) -> tuple[float, str, List[str], List[str]]:
    """Apply pediatric risk adjustments.

    Returns (adjusted_risk_score, adjusted_risk_level, updated_red_flags, updated_risk_factors).
    """
    if age is None:
        return risk_score, risk_level, red_flags, risk_factors

    text = symptom_text.lower()
    updated_flags = list(red_flags)
    updated_factors = list(risk_factors)

    if age < 2:
        if "pediatric high-risk age group (infant)" not in updated_flags:
            updated_flags.append("pediatric high-risk age group (infant)")
        updated_factors.append("Infant age (<2 years) significantly increases clinical risk.")
        risk_score = _clip(risk_score + 0.15)

    if age < 5 and any(term in text for term in ["fever", "high temperature", "hot", "temperature"]):
        risk_score = _clip(risk_score + 0.20)
        if "pediatric fever risk" not in updated_flags:
            updated_flags.append("pediatric fever risk")
        updated_factors.append("Fever in a child under 5 years requires prompt evaluation.")
        if risk_level == "Low":
            risk_level = "Medium"

    if age > 65:
        if risk_score < 0.4:
            risk_score = max(risk_score, 0.4)
        if "elderly patient vulnerability" not in updated_factors:
            updated_factors.append("Elderly age (>65) increases vulnerability and clinical risk.")

    return risk_score, risk_level, updated_flags, updated_factors


def _lower_uti_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    return any(
        term in text
        for term in [
            "burning urination",
            "burning micturition",
            "painful urination",
            "dysuria",
            "frequent urination",
            "frequent urine",
            "urinary frequency",
            "lower abdomen pain",
            "suprapubic",
        ]
    )


def _recommendation(
    risk_level: str,
    red_flags: List[str],
    confidence_band: str,
    symptom_text: str,
    emergency_flags: List[str] | None = None,
    warning_flags: List[str] | None = None,
) -> str:
    emergency_flags = emergency_flags or []
    warning_flags = warning_flags or []

    # ── Emergency patterns: always highest priority ──
    if "possible stroke pattern" in red_flags or "possible cardiac emergency pattern" in red_flags or "possible sepsis pattern" in red_flags:
        return "Emergency pattern detected: seek immediate emergency care now."

    # ── Emergency red flags present (but no named emergency pattern) ──
    if emergency_flags and risk_level == "High":
        return "Urgent: seek immediate in-person medical care or emergency services."

    # ── Specific clinical patterns ──
    if _kidney_fever_pattern(symptom_text):
        return (
            "Possible urinary/kidney involvement pattern: arrange a same-day in-person medical evaluation "
            "for urine tests and clinical examination."
        )
    if _lower_uti_pattern(symptom_text):
        return "Likely urinary symptom pattern: arrange a same-day or next-day clinic visit for urine testing and treatment guidance."
    if _drug_reaction_pattern(symptom_text):
        return "Possible drug reaction pattern: stop the suspected medication and arrange same-day clinical evaluation. Seek emergency care if breathing difficulty or swelling worsens."

    # ── Warning red flags (not emergency): same-day care, not ER ──
    if warning_flags and risk_level == "Medium":
        return "Warning signs present: arrange a same-day clinic visit for professional assessment."

    # ── High risk without specific emergency flags ──
    if risk_level == "High":
        return "High-risk assessment: seek urgent in-person medical care today."

    # ── Low confidence ──
    if confidence_band == "low":
        return "Low confidence result: arrange a clinical consultation for proper evaluation."

    # ── Medium risk ──
    if risk_level == "Medium":
        return "Schedule a same-day or next-day clinic visit for professional assessment."

    # ── Low risk ──
    return "Low-risk pattern: monitor symptoms and seek care if symptoms worsen or persist."


def _risk_factors(
    *,
    top_condition: str,
    top_prob: float,
    red_flags: List[str],
    vulnerability: float,
    uncertainty: float,
    disagreement: float,
) -> List[str]:
    if GENERIC_CONDITION_RE.fullmatch(top_condition.strip()):
        top_condition_text = "non-specific symptom pattern"
    else:
        top_condition_text = top_condition

    factors: List[str] = [
        f"Primary predicted condition pattern: {top_condition_text} ({top_prob * 100:.1f}% confidence).",
    ]
    if red_flags:
        factors.append(f"Red-flag symptoms detected: {', '.join(red_flags)}.")
    if vulnerability >= 0.3:
        factors.append("Patient vulnerability context increases caution level (age/comorbid factors).")
    if uncertainty >= 0.5:
        factors.append("Model uncertainty is elevated; clinical confirmation is important.")
    if disagreement >= 0.4:
        factors.append("Text and image predictions disagree meaningfully; treat result as lower certainty.")
    return factors


def _prevention_advice(risk_level: str, top_condition: str) -> List[str]:
    condition = top_condition.lower()

    if risk_level == "High":
        advice = [
            "Do not delay: proceed to emergency or urgent care services immediately.",
            "Do not drive yourself — have someone take you or call emergency services.",
            "Avoid eating, drinking, or taking any new medications until a clinician evaluates you.",
        ]
        if "infection" in condition:
            advice.append("Maintain strict hygiene to prevent spreading a possible infection.")
        return advice[:5]

    # Medium / Low risk
    advice = [
        "Stay hydrated and track symptom progression at least twice daily.",
        "Avoid self-medication without clinician guidance, especially antibiotics/steroids.",
        "Seek in-person evaluation if symptoms worsen, persist, or new severe symptoms appear.",
    ]
    if "infection" in condition:
        advice.insert(0, "Maintain strict hygiene and avoid sharing personal items.")
    if "dermatitis" in condition or "eczema" in condition:
        advice.insert(0, "Use gentle skin care products and avoid known irritants/allergens.")
    if risk_level == "Medium":
        advice.insert(0, "Arrange a same-day or next-day clinic visit for proper evaluation.")
    return advice[:5]


def compute_risk(
    fused_predictions: List[Dict[str, float]],
    uncertainty: float,
    disagreement: float,
    symptom_text: str,
    confidence_band: str,
    vulnerability: float = 0.0,
    *,
    age: Optional[int] = None,
    heart_rate: Optional[int] = None,
    blood_pressure_systolic: Optional[int] = None,
    blood_pressure_diastolic: Optional[int] = None,
    temperature_celsius: Optional[float] = None,
    spo2_percent: Optional[int] = None,
):
    top_condition = fused_predictions[0]["condition"] if fused_predictions else "unspecified"
    top_prob = float(fused_predictions[0]["probability"]) if fused_predictions else 0.0
    severity_prior = SEVERITY_PRIOR.get(str(top_condition).lower(), 0.45)
    severity_component = _clip(severity_prior * top_prob)

    red_flags = detect_red_flags(symptom_text)
    emergency_flags, warning_flags = _classify_red_flags(red_flags)
    safety_summary = build_safety_summary(symptom_text)

    # Weight red flags by severity: emergency flags contribute much more
    emergency_component = _clip(len(emergency_flags) * 0.45)
    warning_component = _clip(len(warning_flags) * 0.15)
    redflag_component = _clip(emergency_component + warning_component)
    vulnerability_component = _clip(vulnerability)
    uncertainty_component = _clip(uncertainty)
    disagreement_component = _clip(disagreement)

    risk_score = _clip(
        0.40 * severity_component
        + 0.25 * redflag_component
        + 0.15 * vulnerability_component
        + 0.12 * uncertainty_component
        + 0.08 * disagreement_component
    )

    # ── Item 1: Temporal urgency modifier ─────────────────────────────────
    temporal_boost, temporal_flag = _temporal_urgency_modifier(symptom_text)
    if temporal_boost != 0.0:
        risk_score = _clip(risk_score + temporal_boost)
    if temporal_flag and temporal_flag not in red_flags:
        red_flags = sorted(set(red_flags + [temporal_flag]))
        emergency_flags, warning_flags = _classify_red_flags(red_flags)

    # ── Item 2: Vital signs risk ───────────────────────────────────────────
    vital_boost, vital_level_override, vital_flags = _vital_signs_risk(
        heart_rate=heart_rate,
        blood_pressure_systolic=blood_pressure_systolic,
        blood_pressure_diastolic=blood_pressure_diastolic,
        temperature_celsius=temperature_celsius,
        spo2_percent=spo2_percent,
    )
    if vital_boost > 0:
        risk_score = _clip(risk_score + vital_boost)
    if vital_flags:
        red_flags = sorted(set(red_flags + vital_flags))
        emergency_flags, warning_flags = _classify_red_flags(red_flags)

    risk_level = _risk_level(risk_score)

    # Apply vital signs level override
    if vital_level_override == "High":
        risk_score = max(risk_score, 0.75)
        risk_level = "High"
    elif vital_level_override == "Medium" and risk_level == "Low":
        risk_score = max(risk_score, 0.38)
        risk_level = "Medium"

    forced_risk = safety_summary.get("risk_level")
    if forced_risk == "High":
        risk_score = max(risk_score, 0.85)
        risk_level = "High"
    elif forced_risk == "Medium" and risk_level == "Low":
        risk_score = max(risk_score, 0.38)
        risk_level = "Medium"
    if _stroke_pattern(symptom_text) or _acute_coronary_pattern(symptom_text) or _sepsis_pattern(symptom_text):
        risk_score = max(risk_score, 0.85)
        risk_level = "High"
    if str(top_condition).lower() in FORCED_HIGH_CONDITIONS and top_prob >= 0.6:
        risk_score = max(risk_score, 0.75)
        risk_level = "High"
    if "malaria" in str(top_condition).lower() and any(term in symptom_text.lower() for term in ["fever", "chills", "headache"]):
        risk_score = max(risk_score, 0.4)
        risk_level = "Medium"
    if risk_level == "Low" and _kidney_fever_pattern(symptom_text):
        risk_score = max(risk_score, 0.38)
        risk_level = "Medium"
    if risk_level == "Low" and _lower_uti_pattern(symptom_text):
        risk_score = max(risk_score, 0.34)
        risk_level = "Medium"
    if risk_level == "Low" and _drug_reaction_pattern(symptom_text):
        risk_score = max(risk_score, 0.38)
        risk_level = "Medium"

    # ── Consistency enforcement (BEFORE recommendation) ──────────────
    # This guarantees risk_level, risk_score, and needs_urgent_care are
    # final and consistent BEFORE we compute recommendation/advice.
    risk_level, risk_score, needs_urgent_care = _ensure_consistency(
        risk_level=risk_level,
        risk_score=risk_score,
        emergency_flags=emergency_flags,
        warning_flags=warning_flags,
    )

    # ── Now compute outputs using the FINAL risk_level ────────────────
    recommendation = _recommendation(
        risk_level=risk_level,
        red_flags=red_flags,
        confidence_band=confidence_band,
        symptom_text=symptom_text,
        emergency_flags=emergency_flags,
        warning_flags=warning_flags,
    )
    # Safety summary recommendation overrides only when it's more specific
    if safety_summary.get("recommendation") and forced_risk in ("High", "Medium"):
        recommendation = safety_summary["recommendation"]

    risk_factors = _risk_factors(
        top_condition=str(top_condition),
        top_prob=top_prob,
        red_flags=red_flags,
        vulnerability=vulnerability_component,
        uncertainty=uncertainty_component,
        disagreement=disagreement_component,
    )
    prevention_advice = _prevention_advice(risk_level=risk_level, top_condition=str(top_condition))

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "severity_component": round(severity_component, 4),
        "redflag_component": round(redflag_component, 4),
        "vulnerability_component": round(vulnerability_component, 4),
        "uncertainty_component": round(uncertainty_component, 4),
        "disagreement_component": round(disagreement_component, 4),
        "recommendation_text": recommendation,
        "disclaimer_text": DISCLAIMER_TEXT,
        "needs_urgent_care": needs_urgent_care,
        "red_flags": red_flags,
        "risk_factors": risk_factors,
        "prevention_advice": prevention_advice,
    }


def _ensure_consistency(
    *,
    risk_level: str,
    risk_score: float,
    emergency_flags: List[str],
    warning_flags: List[str],
) -> tuple[str, float, bool]:
    """Enforce consistency between risk_level, risk_score, and needs_urgent_care.

    Rules:
    - Emergency red flags ALWAYS force High risk (score >= 0.75) and urgent care.
    - Warning red flags alone do NOT force urgent care — they align with Medium risk.
    - needs_urgent_care is True ONLY when risk_level is High.
    - risk_score must be within the band implied by risk_level.
    """
    # Emergency flags → force High
    if emergency_flags:
        risk_level = "High"
        risk_score = max(risk_score, 0.75)

    # Ensure score matches level
    if risk_level == "High" and risk_score < 0.66:
        risk_score = max(risk_score, 0.75)
    elif risk_level == "Medium" and risk_score < 0.33:
        risk_score = max(risk_score, 0.38)
    elif risk_level == "Low" and risk_score >= 0.33:
        risk_score = min(risk_score, 0.32)

    # needs_urgent_care only for High risk (emergency conditions)
    needs_urgent_care = risk_level == "High"

    return risk_level, risk_score, needs_urgent_care
