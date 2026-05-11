import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from django.conf import settings

from .clinical_safety import apply_prediction_safety_overrides
from .preprocess import clean_symptom_text


KEYWORD_PRIOR_MAP = {
    "eczema": ["itchy", "dry skin", "rash", "red patches"],
    "psoriasis": ["scaly", "silvery", "thick plaques"],
    "acne": ["pimples", "blackheads", "whiteheads", "oily skin"],
    "fungal infection": ["ring", "itching", "peeling", "spread"],
    "contact dermatitis": ["allergy", "soap", "cosmetic", "irritation"],
    "drug reaction": ["after medication", "drug reaction", "side effect", "drug rash", "medication rash"],
    "adverse drug reaction": ["adverse", "drug reaction", "medication reaction", "antibiotic rash"],
    "allergic dermatitis": ["allergy", "rash", "itching", "hives", "urticaria"],
    "gastroenteritis": ["stomach pain", "nausea", "vomiting", "diarrhea", "after eating"],
    "peptic ulcer": ["stomach pain", "burning stomach", "heartburn", "after eating"],
    "food poisoning": ["nausea", "vomiting", "diarrhea", "after eating", "food"],
    "viral pharyngitis": ["fever", "cough", "sore throat", "child"],
    "upper respiratory infection": ["fever", "cough", "runny nose", "congestion", "child"],
}

URINARY_KEYWORD_BOOST = {
    "Escherichia coli urinary tract infection": [
        "burning urination",
        "burning micturition",
        "urinary",
        "urine",
        "kidney pain",
        "flank pain",
    ],
    "Cystitis": [
        "burning urination",
        "burning micturition",
        "urine",
        "bladder",
        "lower abdomen",
    ],
    "Diabetic renal disease (disorder)": [
        "kidney",
        "renal",
        "flank",
    ],
}

UTI_RELATED_CLASS_HINTS = [
    "urinary tract infection",
    "cystitis",
    "pyeloneph",
    "kidney infection",
    "escherichia coli urinary tract infection",
    "renal infection",
]

RESPIRATORY_TERMS = [
    "cough",
    "sore throat",
    "runny nose",
    "shortness of breath",
    "trouble breathing",
    "breath",
    "wheezing",
    "loss of taste",
    "loss of smell",
]

STROKE_TERMS = [
    "slurred speech",
    "one side weakness",
    "one-side weakness",
    "one sided weakness",
    "unilateral weakness",
    "facial droop",
    "face droop",
    "cannot speak",
    "difficulty speaking",
]

CARDIAC_TERMS = [
    "chest pain",
    "pressure in chest",
    "chest pressure",
    "pain in chest",
    "left arm pain",
    "sweating",
    "sweat",
]

SEVERE_RESP_TERMS = [
    "trouble breathing",
    "shortness of breath",
    "breathless",
    "cannot breathe",
]

UTI_TERMS = [
    "burning urination",
    "burning micturition",
    "painful urination",
    "frequent urine",
    "frequent urination",
    "urinary frequency",
    "urine often",
    "lower abdomen pain",
    "suprapubic",
    "dysuria",
    "urinate",
    "urination",
    "urine",
]

KIDNEY_TERMS = [
    "kidney",
    "flank",
    "lower back",
    "back pain",
    "loin",
]

FEVER_TERMS = [
    "fever",
    "high temperature",
    "temperature",
    "chills",
]

DERMATITIS_TERMS = [
    "itchy",
    "itching",
    "rash",
    "red skin",
    "red rash",
]

EXPOSURE_TERMS = [
    "chemical",
    "soap",
    "detergent",
    "cosmetic",
    "cream",
    "lotion",
    "after exposure",
    "allergy",
    "irritation",
]

DRUG_REACTION_TERMS = [
    "after medication",
    "after drug",
    "after pill",
    "after medicine",
    "medication",
    "side effect",
    "drug reaction",
    "drug rash",
    "antibiotic",
    "penicillin",
    "adverse",
    "pill",
    "medicine",
]

GI_TERMS = [
    "stomach pain",
    "stomach ache",
    "abdominal pain",
    "belly pain",
    "nausea",
    "vomiting",
    "diarrhea",
    "bloating",
    "indigestion",
    "heartburn",
    "after eating",
    "after food",
    "loss of appetite",
]

PEDIATRIC_TERMS = [
    "my child",
    "my baby",
    "my infant",
    "my toddler",
    "my kid",
    "pediatric",
    "children",
]

PNEUMONIA_TERMS = [
    "cough",
    "fever",
    "chest pain",
    "pain when breathing",
    "trouble breathing",
    "shortness of breath",
    "phlegm",
]

URI_TERMS = [
    "sore throat",
    "runny nose",
    "mild cough",
    "low fever",
    "congestion",
    "sneezing",
]

PANIC_TERMS = [
    "fear",
    "panic",
    "fast heartbeat",
    "palpitations",
    "chest tightness",
    "feeling anxious",
]

MALARIA_TERMS = [
    "malaria",
    "mosquito",
    "fever",
    "chills",
    "headache",
]


@lru_cache(maxsize=1)
def _load_text_artifacts():
    model_path = Path(settings.TEXT_MODEL_PATH)
    vectorizer_path = Path(settings.TFIDF_VECTORIZER_PATH)
    labels_path = Path(settings.TEXT_LABELS_PATH)
    svd_path_str = str(getattr(settings, "TEXT_SVD_PATH", "")).strip()
    svd_path = Path(svd_path_str) if svd_path_str else None
    consolidation_path_str = str(getattr(settings, "TRIAGE_LABEL_CONSOLIDATION_PATH", "")).strip()
    consolidation_path = Path(consolidation_path_str) if consolidation_path_str else None

    if not model_path.exists() or not vectorizer_path.exists():
        return None, None, [], None, {}

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    svd = joblib.load(svd_path) if svd_path and svd_path.exists() else None
    labels: List[str] = []

    if labels_path.exists():
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            labels = [raw[str(i)] for i in range(len(raw))]
        elif isinstance(raw, list):
            labels = [str(v) for v in raw]
    elif hasattr(model, "classes_"):
        labels = [str(v) for v in model.classes_]
    else:
        # CalibratedClassifierCV - classes_ is on the inner estimator
        inner = getattr(model, "estimator", None)
        if inner and hasattr(inner, "classes_"):
            labels = [str(v) for v in inner.classes_]

    # Load label consolidation mapping (consolidated -> original)
    consolidation = {}
    if consolidation_path and consolidation_path.exists():
        raw_map = json.loads(consolidation_path.read_text(encoding="utf-8"))
        # Invert: original -> consolidated (for display, we show consolidated names)
        consolidation = raw_map

    return model, vectorizer, labels, svd, consolidation


def _normalize_distribution(scores: Dict[str, float]) -> List[Dict[str, float]]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        return []

    preds = [
        {"condition": k, "probability": round(max(v, 0.0) / total, 4)}
        for k, v in scores.items()
    ]
    preds.sort(key=lambda item: item["probability"], reverse=True)
    return preds


def _contains_any(text: str, terms: List[str]) -> bool:
    return any(term in text for term in terms)


def _find_label_indices(labels: List[str], exact_names: List[str], partial_names: List[str]) -> List[int]:
    indices: List[int] = []
    lower_labels = [str(label).strip().lower() for label in labels]
    for idx, name in enumerate(lower_labels):
        if name in [item.lower() for item in exact_names]:
            indices.append(idx)
            continue
        if any(partial in name for partial in [item.lower() for item in partial_names]):
            indices.append(idx)
    return sorted(set(indices))


def _apply_probability_floor(
    adjusted: np.ndarray,
    labels: List[str],
    *,
    exact_names: List[str],
    partial_names: List[str],
    floor: float,
) -> None:
    indices = _find_label_indices(labels, exact_names=exact_names, partial_names=partial_names)
    # Filter out-of-bounds indices
    indices = [idx for idx in indices if 0 <= idx < len(adjusted)]
    if not indices:
        return

    floor = max(0.0, min(0.95, float(floor)))
    share = floor / len(indices)
    non_target_total = float(np.sum(adjusted)) - float(np.sum(adjusted[indices]))
    if non_target_total <= 0:
        adjusted[:] = 0.0
        for idx in indices:
            adjusted[idx] = share
        return

    scale = max(0.0, 1.0 - floor) / non_target_total
    for idx in range(len(adjusted)):
        if idx not in indices:
            adjusted[idx] *= scale
    for idx in indices:
        adjusted[idx] = max(adjusted[idx], share)


def _heuristic_prediction(symptom_text: str, top_k: int):
    text = clean_symptom_text(symptom_text)
    scores: Dict[str, float] = {condition: 0.05 for condition in KEYWORD_PRIOR_MAP.keys()}
    for condition, keywords in KEYWORD_PRIOR_MAP.items():
        for keyword in keywords:
            if keyword in text:
                scores[condition] += 1.0

    predictions = _normalize_distribution(scores)[:top_k]
    confidence = predictions[0]["probability"] if predictions else 0.0
    return {
        "predictions": predictions,
        "confidence": round(confidence, 4),
        "model_version": "text-heuristic-v1",
    }


def _keyword_boost_distribution(
    clean_text: str,
    labels: List[str],
    probs: List[float],
) -> List[float]:
    if not labels or not probs:
        return probs

    label_to_idx = {str(label).strip(): idx for idx, label in enumerate(labels)}
    lower_labels = [str(label).strip().lower() for label in labels]
    adjusted = np.asarray([float(p) for p in probs], dtype=np.float64)
    adjusted = np.clip(adjusted, 1e-12, 1.0)

    kidney_pattern = _contains_any(clean_text, FEVER_TERMS) and _contains_any(clean_text, KIDNEY_TERMS)
    lower_uti_pattern = _contains_any(clean_text, UTI_TERMS)
    stroke_pattern = _contains_any(clean_text, STROKE_TERMS)
    cardiac_pattern = _contains_any(clean_text, CARDIAC_TERMS) and (
        _contains_any(clean_text, SEVERE_RESP_TERMS)
        or "sweating" in clean_text
        or "sweat" in clean_text
    )
    dermatitis_pattern = _contains_any(clean_text, DERMATITIS_TERMS) and _contains_any(clean_text, EXPOSURE_TERMS)
    drug_reaction_pattern = _contains_any(clean_text, DERMATITIS_TERMS) and _contains_any(clean_text, DRUG_REACTION_TERMS)
    gi_pattern = _contains_any(clean_text, GI_TERMS) and not _contains_any(clean_text, UTI_TERMS + ["urinate", "urination", "urine", "dysuria"])
    pediatric_pattern = _contains_any(clean_text, PEDIATRIC_TERMS) and _contains_any(clean_text, FEVER_TERMS + ["cough", "sore throat", "runny nose"])
    fatigue_pattern = _contains_any(clean_text, ["tired", "fatigue", "exhausted", "no energy", "always tired", "fatigued", "chronic fatigue"])
    pneumonia_pattern = (
        "cough" in clean_text
        and _contains_any(clean_text, FEVER_TERMS)
        and ("chest pain" in clean_text or _contains_any(clean_text, SEVERE_RESP_TERMS) or "pain when breathing" in clean_text)
    )
    uri_pattern = (
        _contains_any(clean_text, ["sore throat", "runny nose"])
        and ("cough" in clean_text or "mild cough" in clean_text or "low fever" in clean_text)
    )
    panic_pattern = _contains_any(clean_text, ["chest tightness", "fast heartbeat", "palpitations"]) and _contains_any(
        clean_text, ["fear", "panic", "anxious"]
    )
    malaria_pattern = (
        ("malaria" in clean_text or "mosquito" in clean_text)
        and _contains_any(clean_text, ["fever", "chills", "headache"])
    )
    has_respiratory = _contains_any(clean_text, RESPIRATORY_TERMS)

    for condition, keywords in URINARY_KEYWORD_BOOST.items():
        idx = label_to_idx.get(condition)
        if idx is None:
            continue
        hits = sum(1 for keyword in keywords if keyword in clean_text)
        if hits > 0:
            adjusted[idx] *= min(6.0, 1.0 + 1.2 * hits)
        if kidney_pattern:
            adjusted[idx] *= 8.0

    if kidney_pattern:
        for idx, name in enumerate(lower_labels):
            if any(hint in name for hint in UTI_RELATED_CLASS_HINTS):
                adjusted[idx] *= 5.0

    if lower_uti_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Urinary tract infection"],
            partial_names=["urinary tract infection"],
            floor=0.42,
        )
        # Suppress non-UTI predictions when urination symptoms are clear
        for idx, name in enumerate(lower_labels):
            if any(w in name for w in ["gerd", "gastroesophageal", "scombroid", "pregnancy"]):
                adjusted[idx] *= 0.02

    if kidney_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Urinary tract infection"],
            partial_names=["urinary tract infection", "pyeloneph", "renal infection"],
            floor=0.55,
        )

    if stroke_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Stroke", "Paralysis (brain hemorrhage)"],
            partial_names=["stroke", "brain hemorrhage", "paralysis"],
            floor=0.72,
        )

    if cardiac_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Heart attack"],
            partial_names=["heart attack"],
            floor=0.58,
        )

    if dermatitis_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Allergy", "Drug Reaction"],
            partial_names=["allergy", "drug reaction"],
            floor=0.58,
        )
        for idx, name in enumerate(lower_labels):
            if "fungal infection" in name:
                adjusted[idx] *= 0.2

    if drug_reaction_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Drug Reaction", "Adverse drug reaction", "Anaphylaxis", "Allergy"],
            partial_names=["drug reaction", "adverse drug", "allergy", "anaphylax", "urticaria", "drug overdose"],
            floor=0.50,
        )
        # Suppress clearly wrong predictions for drug reactions
        for idx, name in enumerate(lower_labels):
            if any(w in name for w in ["gerd", "gastroesophageal", "whooping", "common cold"]):
                adjusted[idx] *= 0.1

    if gi_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Gastroenteritis", "Peptic ulcer disease", "GERD"],
            partial_names=["gastroenteritis", "peptic ulcer", "gastritis", "gerd", "food poisoning", "stomach", "abdominal"],
            floor=0.45,
        )
        # If urination terms present, this is NOT a GI issue — suppress GI predictions
        if _contains_any(clean_text, UTI_TERMS + ["urinate", "urination", "urine", "dysuria"]):
            for idx, name in enumerate(lower_labels):
                if any(w in name for w in ["gerd", "gastroesophageal", "food poisoning"]):
                    adjusted[idx] *= 0.05

    if pediatric_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Acute Otitis Media", "Viral sinusitis (disorder)", "Common Cold", "Pneumonia"],
            partial_names=["otitis media", "viral pharyngitis", "viral sinusitis", "common cold", "pneumonia", "bronchiolitis", "croup"],
            floor=0.40,
        )

    if fatigue_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Anemia", "Hypothyroidism", "Depression", "Chronic fatigue syndrome"],
            partial_names=["anemia", "hypothyroid", "depression", "chronic fatigue", "iron deficiency", "thyroid"],
            floor=0.45,
        )

    # Suppress clearly irrelevant predictions based on symptom context
    if _contains_any(clean_text, ["tired", "fatigue", "exhausted", "no energy", "always tired"]):
        for idx, name in enumerate(lower_labels):
            if any(w in name for w in ["scombroid", "food poisoning", "pregnancy", "alzheimer", "dementia", "thrombosis"]):
                adjusted[idx] *= 0.05

    if pneumonia_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Pneumonia"],
            partial_names=["pneumonia"],
            floor=0.52,
        )

    if uri_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Common Cold", "Viral pharyngitis", "Acute viral pharyngitis (disorder)", "Viral sinusitis (disorder)"],
            partial_names=["common cold", "viral pharyngitis", "viral sinusitis"],
            floor=0.62,
        )

    if panic_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Panic attack"],
            partial_names=["panic attack"],
            floor=0.48,
        )

    if malaria_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Malaria"],
            partial_names=["malaria"],
            floor=0.52,
        )

    if not has_respiratory:
        for idx, name in enumerate(lower_labels):
            if "covid" in name:
                adjusted[idx] *= 0.08
            if "acute respiratory failure" in name or "respiratory distress" in name:
                adjusted[idx] *= 0.12

    if "chest pain" not in clean_text:
        for idx, name in enumerate(lower_labels):
            if "heart attack" in name:
                adjusted[idx] *= 0.2

    if not any(term in clean_text for term in ["leg swelling", "calf pain", "calf swelling", "deep vein", "dvt"]):
        for idx, name in enumerate(lower_labels):
            if "deep venous thrombosis" in name:
                adjusted[idx] *= 0.15

    total = float(np.sum(np.clip(adjusted, 0.0, None)))
    if total <= 0:
        return probs

    return [float(max(v, 0.0) / total) for v in adjusted]


_GENERIC_RE = re.compile(r"^(condition\s+\d+|class_\d+)$", re.IGNORECASE)


def _rag_informed_predictions(symptom_text: str, ml_predictions: List[Dict], top_k: int = 5) -> List[Dict]:
    """When ML model is uncertain, use RAG retrieval to suggest relevant conditions.

    RAG results often contain disease names and symptom descriptions that are
    far more relevant than the ML model's low-confidence guesses. This function
    extracts condition names from RAG hits and blends them with ML predictions.
    """
    from .rag import query_rag, should_use_rag

    if not should_use_rag(symptom_text):
        return ml_predictions

    rag_items = query_rag(symptom_text, top_k=6)
    if not rag_items:
        return ml_predictions

    # Extract condition names from RAG text (e.g., "Disease: Hypertension", "Possible condition: Arthritis")
    # Only match structured patterns from kaggle_sym, triage, disease_symptom sources
    condition_pattern = re.compile(
        r"(?:disease|possible condition|diagnosis)[\s:]*\s*([A-Z][A-Za-z\s/\-()]+?)(?:\n|$)",
        re.IGNORECASE,
    )
    rag_conditions: Dict[str, float] = {}

    for item in rag_items:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        score = float(item.get("score", 0))
        # Weight by source quality and relevance score
        weight = min(1.0, max(0.1, score / 1.5))

        for match in condition_pattern.finditer(text):
            cond = match.group(1).strip()
            # Filter out generic labels
            if _GENERIC_RE.fullmatch(cond):
                continue
            if len(cond) < 3 or len(cond) > 60:
                continue
            if cond.lower() in ("the", "this", "that", "a", "an"):
                continue
            rag_conditions[cond] = rag_conditions.get(cond, 0) + weight

    if not rag_conditions:
        return ml_predictions

    # Normalize RAG condition scores
    total_rag = sum(rag_conditions.values())
    if total_rag <= 0:
        return ml_predictions

    # ML confidence determines blend ratio:
    # High ML confidence (>30%) → mostly ML, RAG supplements
    # Low ML confidence (<15%) → mostly RAG, ML supplements
    ml_top1 = ml_predictions[0]["probability"] if ml_predictions else 0
    if ml_top1 > 0.30:
        ml_weight, rag_weight = 0.75, 0.25
    elif ml_top1 > 0.15:
        ml_weight, rag_weight = 0.50, 0.50
    else:
        ml_weight, rag_weight = 0.25, 0.75

    # Compute blocked words for symptom context — prevents irrelevant conditions
    _irrelevant_map = {
        "tired": ["scombroid", "food poisoning", "pregnancy", "alzheimer", "dementia", "thrombosis"],
        "fatigue": ["scombroid", "food poisoning", "pregnancy", "alzheimer", "dementia", "thrombosis"],
        "exhausted": ["scombroid", "food poisoning", "pregnancy", "alzheimer", "dementia", "thrombosis"],
        "no energy": ["scombroid", "food poisoning", "pregnancy", "alzheimer", "dementia", "thrombosis"],
        "urinate": ["gerd", "gastroesophageal", "heartburn", "scombroid"],
        "urination": ["gerd", "gastroesophageal", "heartburn", "scombroid"],
        "urine": ["gerd", "gastroesophageal", "heartburn", "scombroid"],
        "dysuria": ["gerd", "gastroesophageal", "heartburn", "scombroid"],
        "numb": ["scombroid", "food poisoning", "pregnancy", "sepsis"],
        "tingling": ["scombroid", "food poisoning", "pregnancy", "sepsis"],
    }
    lower_symptom = symptom_text.lower()
    blocked_words = set()
    for key, words in _irrelevant_map.items():
        if key in lower_symptom:
            blocked_words.update(words)

    # Build blended predictions — also suppress irrelevant ML predictions
    ml_conds = {}
    for p in ml_predictions:
        cond = p["condition"]
        cond_lower = cond.lower()
        # Skip conditions containing blocked words
        if any(blocked in cond_lower for blocked in blocked_words):
            continue
        ml_conds[cond] = p["probability"] * ml_weight

    # Add RAG conditions (only those not already in ML predictions and not blocked)
    for cond, score in sorted(rag_conditions.items(), key=lambda x: -x[1]):
        norm_score = (score / total_rag) * rag_weight
        # Skip conditions containing blocked words
        cond_lower = cond.lower()
        if any(blocked in cond_lower for blocked in blocked_words):
            continue
        if cond in ml_conds:
            ml_conds[cond] = max(ml_conds[cond], norm_score)
        else:
            ml_conds[cond] = norm_score

    # Normalize to sum to 1
    total = sum(ml_conds.values())
    if total <= 0:
        return ml_predictions

    blended = [
        {"condition": cond, "probability": round(score / total, 4)}
        for cond, score in ml_conds.items()
    ]
    blended.sort(key=lambda x: x["probability"], reverse=True)
    return blended[:top_k]


def predict_text_probabilities(symptom_text: str, top_k: int = 5):
    model, vectorizer, labels, svd, consolidation = _load_text_artifacts()
    if model is None or vectorizer is None:
        return _heuristic_prediction(symptom_text, top_k=top_k)

    clean_text = clean_symptom_text(symptom_text)
    matrix = vectorizer.transform([clean_text])
    if svd is not None:
        matrix = svd.transform(matrix)
    probs = model.predict_proba(matrix)[0]
    model_labels = labels if labels else [str(v) for v in getattr(model, "classes_", [])]
    probs = _keyword_boost_distribution(clean_text, model_labels, list(probs))

    if not model_labels:
        model_labels = [str(i) for i in range(len(probs))]

    predictions = [
        {"condition": model_labels[idx], "probability": round(float(prob), 4)}
        for idx, prob in enumerate(probs)
    ]
    predictions.sort(key=lambda item: item["probability"], reverse=True)
    predictions = apply_prediction_safety_overrides(clean_text, predictions, top_k=top_k)
    # Filter out generic "Condition XXX" labels — they carry no clinical meaning
    named = [p for p in predictions if not _GENERIC_RE.fullmatch(str(p.get("condition", "")).strip())]
    if named:
        predictions = named
    # When ML is uncertain, blend with RAG-extracted conditions for relevance
    predictions = _rag_informed_predictions(clean_text, predictions, top_k=top_k)
    predictions = predictions[:top_k]
    confidence = predictions[0]["probability"] if predictions else 0.0

    return {
        "predictions": predictions,
        "confidence": round(confidence, 4),
        "model_version": "text-ml-v2",
    }
