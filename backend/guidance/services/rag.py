from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import faiss  # required — install via: pip install faiss-cpu

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_READY = PROJECT_ROOT / "data" / "ready"

# ── Unified ────────────────────────────────────────────────────────────
MEDQUAD_PATH = DATA_READY / "unified" / "ULTIMATE_CONVERSATIONAL_QA.csv"
TRIAGE_PATH = DATA_READY / "unified" / "ULTIMATE_TRIAGE_KNOWLEDGE.csv"
IMAGING_LABELS_PATH = DATA_READY / "unified" / "ULTIMATE_IMAGING_LABELS.csv"

# ── Kaggle symptom ─────────────────────────────────────────────────────
SYMPTOM_DESC_PATH = DATA_READY / "kaggle_symptom" / "symptom_Description.csv"
SYMPTOM_PREC_PATH = DATA_READY / "kaggle_symptom" / "symptom_precaution.csv"
SYMPTOM_DATASET_PATH = DATA_READY / "kaggle_symptom" / "dataset.csv"
SYMPTOM_SEVERITY_PATH = DATA_READY / "kaggle_symptom" / "Symptom-severity.csv"
DISEASE_SYMPTOM_PROCESSED_PATH = DATA_READY / "kaggle_symptom" / "disease_symptom_processed.csv"

# ── Kaggle chatbot ─────────────────────────────────────────────────────
KAGGLE_CHATBOT_TRAIN_PATH = DATA_READY / "kaggle_chatbot" / "train.csv"
KAGGLE_CHATBOT_VAL_PATH = DATA_READY / "kaggle_chatbot" / "val.csv"

# ── Dialogue ───────────────────────────────────────────────────────────
DIALOGUE_FULL_PATH = DATA_READY / "dialogue" / "full.csv"

# ── Grok ───────────────────────────────────────────────────────────────
GROK_DIALOGUE_PATH = DATA_READY / "grok" / "triage_dialogue_reasoning.csv"
GROK_SUPERVISED_PATH = DATA_READY / "grok" / "triage_supervised.csv"

# ── Fitzpatrick (dermatology) ──────────────────────────────────────────
FITZPATRICK_META_PATH = DATA_READY / "fitzpatrick" / "metadata.csv"

# ── MIMIC-IV ───────────────────────────────────────────────────────────
MIMIC_CONDITIONS_PATH = DATA_READY / "mimic" / "conditions.csv"
MIMIC_MEDICATIONS_PATH = DATA_READY / "mimic" / "medications.csv"
MIMIC_OBSERVATIONS_PATH = DATA_READY / "mimic" / "observations.csv"
MIMIC_PROCEDURES_PATH = DATA_READY / "mimic" / "procedures.csv"

# ── Synthea ─────────────────────────────────────────────────────────────
SYNTHEA_CONDITIONS_PATH = DATA_READY / "synthea" / "conditions.csv"
SYNTHEA_MEDICATIONS_PATH = DATA_READY / "synthea" / "medications.csv"
SYNTHEA_OBSERVATIONS_PATH = DATA_READY / "synthea" / "observations.csv"
SYNTHEA_PROCEDURES_PATH = DATA_READY / "synthea" / "procedures.csv"
SYNTHEA_CAREPLANS_PATH = DATA_READY / "synthea" / "careplans.csv"
SYNTHEA_ALLERGIES_PATH = DATA_READY / "synthea" / "allergies.csv"
SYNTHEA_IMMUNIZATIONS_PATH = DATA_READY / "synthea" / "immunizations.csv"
SYNTHEA_ENCOUNTERS_PATH = DATA_READY / "synthea" / "encounters.csv"

# ── Triage (full dataset) ──────────────────────────────────────────────
TRIAGE_FULL_PATH = DATA_READY / "triage" / "full.csv"

# ── UCI ────────────────────────────────────────────────────────────────
UCI_HEART_PATH = DATA_READY / "uci" / "heart_failure.csv"

RAG_TRIGGER_TERMS = {
    "treatment",
    "treat",
    "therapy",
    "management",
    "prevention",
    "prevent",
    "guideline",
    "recommendation",
    "medication",
    "dosage",
    "next step",
    "symptom",
    "pain",
    "ache",
    "hurt",
    "hurting",
    "fever",
    "cough",
    "headache",
    "dizzy",
    "dizziness",
    "nausea",
    "fatigue",
    "tired",
    "exhausted",
    "energy",
    "rash",
    "swelling",
    "bleeding",
    "breathing",
    "heart",
    "chest",
    "stomach",
    "back pain",
    "back hurt",
    "joint",
    "muscle",
    "anxiety",
    "anxious",
    "depression",
    "depressed",
    "diabetes",
    "blood pressure",
    "cancer",
    "infection",
    "allergy",
    "asthma",
    "flu",
    "cold",
    "numb",
    "numbness",
    "tingling",
    "urinate",
    "urination",
    "urine",
    "burning",
    "itch",
    "itching",
    "sleep",
    "insomnia",
    "weak",
    "weakness",
    "vomit",
    "diarrhea",
    "constipation",
    "sore",
    "throat",
    "nose",
    "congestion",
    "wheeze",
    "wheezing",
    "skin",
    "eye",
    "vision",
    "ear",
    "hearing",
    "tooth",
    "dental",
    "pregnan",
    "period",
    "menstrual",
    "lump",
    "bump",
    "bruise",
    "dizzy",
    "faint",
    "seizure",
    "convulsion",
    "tremor",
    "shaking",
    "memory",
    "forget",
    "confusion",
    "weight",
    "appetite",
    "thirst",
    "frequent",
    "after eating",
    "after food",
    "after medication",
    "side effect",
    "drug",
    "pill",
    "antibiotic",
    "medicine",
}


@dataclass(frozen=True)
class RagDoc:
    text: str
    source: str
    metadata: Dict[str, str]


@dataclass
class RagIndex:
    vectorizer: TfidfVectorizer
    matrix: Union[csr_matrix, np.ndarray]  # sparse by default, dense only for FAISS
    docs: List[RagDoc]
    faiss_index: Optional[object] = None


def should_use_rag(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in RAG_TRIGGER_TERMS)


def _load_medquad(max_docs: int) -> List[RagDoc]:
    if not MEDQUAD_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with MEDQUAD_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = str(row.get("question", "") or row.get("user_text", "")).strip()
            answer = str(row.get("answer", "") or row.get("assistant_text", "")).strip()
            if not question or not answer:
                continue
            text = f"Q: {question}\nA: {answer}"
            docs.append(
                RagDoc(
                    text=text,
                    source="medquad",
                    metadata={"question": question[:500], "answer": answer[:2000]},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_triage(max_docs: int) -> List[RagDoc]:
    """Load symptom→condition mapping from ULTIMATE_TRIAGE_KNOWLEDGE.csv."""
    if not TRIAGE_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with TRIAGE_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symptoms = str(row.get("symptom_text", "")).strip()
            condition = str(row.get("condition", "")).strip()
            if not symptoms or not condition:
                continue
            text = f"Symptoms: {symptoms}\nPossible condition: {condition}"
            docs.append(
                RagDoc(
                    text=text,
                    source="triage",
                    metadata={"question": f"What condition is associated with: {symptoms[:200]}?", "answer": condition[:500]},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_kaggle_symptoms(max_docs: int) -> List[RagDoc]:
    """Load disease descriptions and precautions from Kaggle symptom datasets."""
    docs: List[RagDoc] = []

    # Disease descriptions
    if SYMPTOM_DESC_PATH.exists():
        with SYMPTOM_DESC_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                disease = str(row.get("Disease", "")).strip()
                description = str(row.get("Description", "")).strip()
                if not disease or not description:
                    continue
                text = f"Disease: {disease}\nDescription: {description}"
                docs.append(
                    RagDoc(
                        text=text,
                        source="kaggle_desc",
                        metadata={"question": f"What is {disease}?", "answer": description[:1000]},
                    )
                )
                if len(docs) >= max_docs:
                    break

    # Disease precautions
    if SYMPTOM_PREC_PATH.exists() and len(docs) < max_docs:
        with SYMPTOM_PREC_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                disease = str(row.get("Disease", "")).strip()
                precautions = [
                    str(row.get(f"Precaution_{i}", "")).strip()
                    for i in range(1, 5)
                    if str(row.get(f"Precaution_{i}", "")).strip()
                ]
                if not disease or not precautions:
                    continue
                prec_text = ", ".join(precautions)
                text = f"Disease: {disease}\nPrecautions: {prec_text}"
                docs.append(
                    RagDoc(
                        text=text,
                        source="kaggle_prec",
                        metadata={"question": f"What precautions for {disease}?", "answer": prec_text[:500]},
                    )
                )
                if len(docs) >= max_docs:
                    break

    # Disease→symptom mapping
    if SYMPTOM_DATASET_PATH.exists() and len(docs) < max_docs:
        with SYMPTOM_DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                disease = str(row.get("Disease", "")).strip()
                symptoms = [
                    str(row.get(f"Symptom_{i}", "")).strip()
                    for i in range(1, 18)
                    if str(row.get(f"Symptom_{i}", "")).strip()
                ]
                if not disease or not symptoms:
                    continue
                sym_text = ", ".join(symptoms)
                text = f"Disease: {disease}\nSymptoms: {sym_text}"
                docs.append(
                    RagDoc(
                        text=text,
                        source="kaggle_sym",
                        metadata={"question": f"What are symptoms of {disease}?", "answer": sym_text[:500]},
                    )
                )
                if len(docs) >= max_docs:
                    break

    return docs


def _load_kaggle_chatbot(max_docs: int) -> List[RagDoc]:
    """Load real patient→doctor Q&A pairs from Kaggle chatbot dataset.

    These are conversational medical exchanges — exactly the kind of data
    that makes the AI respond like a human doctor rather than a script.
    Only rows with label=1 (relevant answer) are included.
    """
    docs: List[RagDoc] = []
    paths = [KAGGLE_CHATBOT_TRAIN_PATH, KAGGLE_CHATBOT_VAL_PATH]

    for path in paths:
        if not path.exists() or len(docs) >= max_docs:
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                question = str(row.get("question", "")).strip()
                answer = str(row.get("answer", "")).strip()
                label = str(row.get("label", "1")).strip()
                # Only include rows where the answer is relevant (label=1)
                if not question or not answer or label == "-1":
                    continue
                text = f"Q: {question}\nA: {answer}"
                docs.append(
                    RagDoc(
                        text=text,
                        source="kaggle_chatbot",
                        metadata={"question": question[:500], "answer": answer[:2000]},
                    )
                )
                if len(docs) >= max_docs:
                    break

    return docs


def _load_dialogue(max_docs: int) -> List[RagDoc]:
    """Load dialogue intent classification data from dialogue/full.csv."""
    if not DIALOGUE_FULL_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with DIALOGUE_FULL_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text_val = str(row.get("text", "")).strip()
            intent = str(row.get("intent", "")).strip()
            if not text_val or not intent:
                continue
            rag_text = f"Patient message: {text_val}\nIntent: {intent}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="dialogue",
                    metadata={"question": text_val[:500], "answer": f"This type of message typically relates to {intent} concerns."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_grok_dialogue(max_docs: int) -> List[RagDoc]:
    """Load Grok-generated triage dialogue with reasoning."""
    if not GROK_DIALOGUE_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with GROK_DIALOGUE_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_text = str(row.get("user_text", "")).strip()
            assistant_text = str(row.get("assistant_text", "")).strip()
            intent = str(row.get("intent", "")).strip()
            if not user_text or not assistant_text:
                continue
            rag_text = f"Q: {user_text}\nA: {assistant_text}"
            if intent:
                rag_text += f"\nIntent: {intent}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="grok_dialogue",
                    metadata={"question": user_text[:500], "answer": assistant_text[:2000]},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_grok_supervised(max_docs: int) -> List[RagDoc]:
    """Load Grok supervised triage data with condition, urgency, and reasoning."""
    if not GROK_SUPERVISED_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with GROK_SUPERVISED_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symptoms = str(row.get("symptom_text", "")).strip()
            condition = str(row.get("condition", "")).strip()
            urgency = str(row.get("urgency", "")).strip()
            reasoning = str(row.get("reasoning", "")).strip()
            if not symptoms or not condition:
                continue
            rag_text = f"Symptoms: {symptoms}\nCondition: {condition}\nUrgency: {urgency}"
            if reasoning:
                rag_text += f"\nReasoning: {reasoning}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="grok_supervised",
                    metadata={"question": f"What condition for: {symptoms[:200]}?", "answer": f"{condition} (urgency: {urgency})"},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_fitzpatrick(max_docs: int) -> List[RagDoc]:
    """Load Fitzpatrick dermatology dataset — skin condition labels and categories."""
    if not FITZPATRICK_META_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with FITZPATRICK_META_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row.get("label", "")).strip()
            three_part = str(row.get("three_partition_label", "")).strip()
            nine_part = str(row.get("nine_partition_label", "")).strip()
            if not label:
                continue
            rag_text = f"Skin condition: {label}"
            if three_part:
                rag_text += f"\nCategory: {three_part}"
            if nine_part:
                rag_text += f"\nSubcategory: {nine_part}"
            category_desc = f" ({three_part})" if three_part else ""
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="fitzpatrick",
                    metadata={"question": f"What is {label}?", "answer": f"{label} is a dermatological condition{category_desc}. It falls under the category of {three_part or 'skin disorder'}."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_mimic_conditions(max_docs: int) -> List[RagDoc]:
    """Load MIMIC-IV clinical condition descriptions."""
    if not MIMIC_CONDITIONS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with MIMIC_CONDITIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            description = str(row.get("description", "")).strip()
            if not description or description in seen:
                continue
            seen.add(description)
            rag_text = f"Clinical condition: {description}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="mimic_condition",
                    metadata={"question": f"What is {description}?", "answer": f"{description} is a clinical condition documented in hospital records."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_mimic_medications(max_docs: int) -> List[RagDoc]:
    """Load MIMIC-IV medication records — deduplicated by drug name."""
    if not MIMIC_MEDICATIONS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with MIMIC_MEDICATIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            drug = str(row.get("drug", "")).strip()
            route = str(row.get("route", "")).strip()
            dose_val = str(row.get("dose_val_rx", "")).strip()
            dose_unit = str(row.get("dose_unit_rx", "")).strip()
            if not drug or drug in seen:
                continue
            seen.add(drug)
            rag_text = f"Medication: {drug}"
            if route:
                rag_text += f"\nRoute: {route}"
            if dose_val:
                rag_text += f"\nTypical dose: {dose_val} {dose_unit}".strip()
            route_info = f", administered {route}" if route else ""
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="mimic_medication",
                    metadata={"question": f"What is the medication {drug}?", "answer": f"{drug} is a medication used in hospital settings{route_info}."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_mimic_observations(max_docs: int) -> List[RagDoc]:
    """Load MIMIC-IV lab observation types — deduplicated by test_name."""
    if not MIMIC_OBSERVATIONS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with MIMIC_OBSERVATIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            test_name = str(row.get("test_name", "")).strip()
            comments = str(row.get("comments", "")).strip()
            spec_type = str(row.get("spec_type_desc", "")).strip()
            if not test_name or test_name in seen:
                continue
            seen.add(test_name)
            rag_text = f"Lab test: {test_name}"
            if spec_type:
                rag_text += f"\nSpecimen type: {spec_type}"
            if comments:
                rag_text += f"\nTypical result: {comments}"
            spec_info = f" using {spec_type} specimen" if spec_type else ""
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="mimic_observation",
                    metadata={"question": f"What is the lab test {test_name}?", "answer": f"{test_name} is a laboratory test{spec_info} used in clinical diagnostics."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_synthea_conditions(max_docs: int) -> List[RagDoc]:
    """Load Synthea condition descriptions — deduplicated by DESCRIPTION."""
    if not SYNTHEA_CONDITIONS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with SYNTHEA_CONDITIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            description = str(row.get("DESCRIPTION", "")).strip()
            if not description or description in seen:
                continue
            seen.add(description)
            rag_text = f"Condition: {description}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="synthea_condition",
                    metadata={"question": f"What is {description}?", "answer": f"{description} is a medical condition recorded in patient health records."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_synthea_medications(max_docs: int) -> List[RagDoc]:
    """Load Synthea medication descriptions — deduplicated by DESCRIPTION."""
    if not SYNTHEA_MEDICATIONS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with SYNTHEA_MEDICATIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            description = str(row.get("DESCRIPTION", "")).strip()
            base_cost = str(row.get("BASE_COST", "")).strip()
            if not description or description in seen:
                continue
            seen.add(description)
            rag_text = f"Medication: {description}"
            if base_cost:
                rag_text += f"\nBase cost: ${base_cost}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="synthea_medication",
                    metadata={"question": f"What is the medication {description}?", "answer": f"{description} is a medication prescribed in clinical practice."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_synthea_observations(max_docs: int) -> List[RagDoc]:
    """Load Synthea observation types — deduplicated by DESCRIPTION."""
    if not SYNTHEA_OBSERVATIONS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with SYNTHEA_OBSERVATIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            description = str(row.get("DESCRIPTION", "")).strip()
            value = str(row.get("VALUE", "")).strip()
            units = str(row.get("UNITS", "")).strip()
            obs_type = str(row.get("TYPE", "")).strip()
            if not description or description in seen:
                continue
            seen.add(description)
            rag_text = f"Observation: {description}"
            if value and units:
                rag_text += f"\nTypical value: {value} {units}"
            elif value:
                rag_text += f"\nTypical value: {value}"
            if obs_type:
                rag_text += f"\nType: {obs_type}"
            value_info = f" (typical: {value} {units})" if value and units else ""
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="synthea_observation",
                    metadata={"question": f"What is the medical observation {description}?", "answer": f"{description} is a clinical observation{value_info} used in patient assessment."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_synthea_procedures(max_docs: int) -> List[RagDoc]:
    """Load Synthea procedure descriptions — deduplicated by DESCRIPTION."""
    if not SYNTHEA_PROCEDURES_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with SYNTHEA_PROCEDURES_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            description = str(row.get("DESCRIPTION", "")).strip()
            reason_desc = str(row.get("REASONDESCRIPTION", "")).strip()
            if not description or description in seen:
                continue
            seen.add(description)
            rag_text = f"Procedure: {description}"
            if reason_desc:
                rag_text += f"\nReason: {reason_desc}"
            reason_info = f", typically performed for {reason_desc}" if reason_desc else ""
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="synthea_procedure",
                    metadata={"question": f"What is the medical procedure {description}?", "answer": f"{description} is a medical procedure{reason_info}."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_synthea_careplans(max_docs: int) -> List[RagDoc]:
    """Load Synthea care plan descriptions — deduplicated by DESCRIPTION."""
    if not SYNTHEA_CAREPLANS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with SYNTHEA_CAREPLANS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            description = str(row.get("DESCRIPTION", "")).strip()
            reason_desc = str(row.get("REASONDESCRIPTION", "")).strip()
            if not description or description in seen:
                continue
            seen.add(description)
            rag_text = f"Care plan: {description}"
            if reason_desc:
                rag_text += f"\nFor condition: {reason_desc}"
            reason_info = f" for managing {reason_desc}" if reason_desc else ""
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="synthea_careplan",
                    metadata={"question": f"What is the care plan {description}?", "answer": f"{description} is a care plan{reason_info} in clinical practice."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_synthea_allergies(max_docs: int) -> List[RagDoc]:
    """Load Synthea allergy descriptions — deduplicated by DESCRIPTION."""
    if not SYNTHEA_ALLERGIES_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with SYNTHEA_ALLERGIES_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            description = str(row.get("DESCRIPTION", "")).strip()
            if not description or description in seen:
                continue
            seen.add(description)
            rag_text = f"Allergy: {description}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="synthea_allergy",
                    metadata={"question": f"What is the allergy {description}?", "answer": f"{description} is a documented allergic reaction in patient records."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_synthea_immunizations(max_docs: int) -> List[RagDoc]:
    """Load Synthea immunization descriptions — deduplicated by DESCRIPTION."""
    if not SYNTHEA_IMMUNIZATIONS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with SYNTHEA_IMMUNIZATIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            description = str(row.get("DESCRIPTION", "")).strip()
            if not description or description in seen:
                continue
            seen.add(description)
            rag_text = f"Immunization: {description}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="synthea_immunization",
                    metadata={"question": f"What is the immunization {description}?", "answer": f"{description} is an immunization administered in clinical practice."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_triage_full(max_docs: int) -> List[RagDoc]:
    """Load full triage dataset (symptom→condition pairs)."""
    if not TRIAGE_FULL_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with TRIAGE_FULL_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symptoms = str(row.get("symptom_text", "")).strip()
            condition = str(row.get("condition", "")).strip()
            if not symptoms or not condition:
                continue
            rag_text = f"Symptoms: {symptoms}\nCondition: {condition}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="triage_full",
                    metadata={"question": f"What condition is associated with {symptoms[:200]}?", "answer": f"Based on the symptoms described ({symptoms[:150]}), the associated condition is {condition}."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_uci_heart(max_docs: int) -> List[RagDoc]:
    """Load UCI heart failure clinical records — aggregated risk factor summary."""
    if not UCI_HEART_PATH.exists():
        return []
    docs: List[RagDoc] = []
    # Aggregate statistics instead of individual patient records
    risk_counts = {"anaemia": 0, "diabetes": 0, "high_blood_pressure": 0, "smoking": 0}
    total = 0
    fatal = 0
    ef_values = []
    with UCI_HEART_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            if str(row.get("anaemia", "")).strip() == "1":
                risk_counts["anaemia"] += 1
            if str(row.get("diabetes", "")).strip() == "1":
                risk_counts["diabetes"] += 1
            if str(row.get("high_blood_pressure", "")).strip() == "1":
                risk_counts["high_blood_pressure"] += 1
            if str(row.get("smoking", "")).strip() == "1":
                risk_counts["smoking"] += 1
            if str(row.get("DEATH_EVENT", "")).strip() == "1":
                fatal += 1
            ef = str(row.get("ejection_fraction", "")).strip()
            if ef:
                try:
                    ef_values.append(int(ef))
                except ValueError:
                    pass
    if total == 0:
        return []
    avg_ef = sum(ef_values) / len(ef_values) if ef_values else 0
    fatality_rate = fatal / total
    rag_text = (
        f"Heart failure clinical study ({total} patients): "
        f"{risk_counts['anaemia']} had anaemia, "
        f"{risk_counts['diabetes']} had diabetes, "
        f"{risk_counts['high_blood_pressure']} had high blood pressure, "
        f"{risk_counts['smoking']} were smokers. "
        f"Average ejection fraction: {avg_ef:.0f}%. "
        f"Fatality rate: {fatality_rate:.1%}."
    )
    docs.append(
        RagDoc(
            text=rag_text,
            source="uci_heart",
            metadata={
                "question": "What are the risk factors for heart failure mortality?",
                "answer": (
                    f"In a study of {total} heart failure patients, key risk factors included "
                    f"anaemia ({risk_counts['anaemia']}/{total}), "
                    f"diabetes ({risk_counts['diabetes']}/{total}), "
                    f"high blood pressure ({risk_counts['high_blood_pressure']}/{total}), "
                    f"and smoking ({risk_counts['smoking']}/{total}). "
                    f"The fatality rate was {fatality_rate:.1%} with average ejection fraction of {avg_ef:.0f}%."
                ),
            },
        )
    )
    return docs


def _load_imaging_labels(max_docs: int) -> List[RagDoc]:
    """Load imaging condition labels and categories from ULTIMATE_IMAGING_LABELS.csv."""
    if not IMAGING_LABELS_PATH.exists():
        return []
    docs: List[RagDoc] = []
    seen: set = set()
    with IMAGING_LABELS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = str(row.get("condition", "")).strip()
            category = str(row.get("category", "")).strip()
            if not condition or condition in seen:
                continue
            seen.add(condition)
            rag_text = f"Imaging condition: {condition}"
            if category:
                rag_text += f"\nCategory: {category}"
            category_info = f" in the {category} category" if category else ""
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="imaging_labels",
                    metadata={"question": f"What is {condition}?", "answer": f"{condition} is a medical condition identifiable through medical imaging{category_info}."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_symptom_severity(max_docs: int) -> List[RagDoc]:
    """Load symptom severity weights from Kaggle."""
    if not SYMPTOM_SEVERITY_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with SYMPTOM_SEVERITY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symptom = str(row.get("Symptom", "")).strip()
            weight = str(row.get("weight", "")).strip()
            if not symptom:
                continue
            severity = "mild" if int(weight) <= 3 else "moderate" if int(weight) <= 6 else "severe"
            rag_text = f"Symptom: {symptom} — {severity} severity"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="symptom_severity",
                    metadata={"question": f"What is the severity of {symptom}?", "answer": f"{symptom} is generally considered {severity} in clinical assessment."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_disease_symptom_processed(max_docs: int) -> List[RagDoc]:
    """Load processed disease→symptom mappings from Kaggle."""
    if not DISEASE_SYMPTOM_PROCESSED_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with DISEASE_SYMPTOM_PROCESSED_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symptoms = str(row.get("symptom_text", "")).strip()
            condition = str(row.get("condition", "")).strip()
            if not symptoms or not condition:
                continue
            rag_text = f"Disease: {condition}\nSymptoms: {symptoms}"
            docs.append(
                RagDoc(
                    text=rag_text,
                    source="disease_symptom",
                    metadata={"question": f"What are the symptoms of {condition}?", "answer": f"Common symptoms of {condition} include {symptoms[:300]}."},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


# ── Source quality tiers for result ranking ────────────────────────────────
# Higher tier = more authoritative / better for user-facing answers.
_SOURCE_QUALITY_BOOST = {
    # Tier 1 — curated medical Q&A (most informative for patients)
    "medquad": 0.20,
    "triage": 0.15,
    "triage_full": 0.12,
    "kaggle_sym": 0.10,
    "kaggle_desc": 0.10,
    "kaggle_prec": 0.10,
    # Tier 2 — structured knowledge (good for condition matching)
    "disease_symptom": 0.08,
    "symptom_severity": 0.05,
    "grok_supervised": 0.05,
    "grok_dialogue": 0.03,
    # Tier 3 — chatbot Q&A (often noisy, lower priority)
    "kaggle_chatbot": 0.02,
    # Tier 3 — clinical EHR (useful for condition matching, less explanatory)
    "synthea_condition": 0.02,
    "synthea_medication": 0.02,
    "synthea_observation": 0.02,
    "synthea_procedure": 0.02,
    "synthea_careplan": 0.02,
    "synthea_allergy": 0.02,
    "synthea_immunization": 0.02,
    "mimic_condition": 0.02,
    "mimic_medication": 0.02,
    "mimic_observation": 0.02,
    # Tier 4 — supplementary (low boost, should not dominate)
    "dialogue": 0.00,
    "fitzpatrick": 0.00,
    "imaging_labels": 0.00,
    "symptom_severity": -0.05,
    "uci_heart": 0.00,
}


@lru_cache(maxsize=1)
def _build_index() -> RagIndex:
    import pickle
    import hashlib

    # ── Cache path ─────────────────────────────────────────────────────────
    cache_enabled = os.getenv("RAG_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
    cache_path = PROJECT_ROOT / "backend" / "models" / "rag_index_cache.pkl"

    if cache_enabled and cache_path.exists():
        try:
            # Check if cache is newer than the newest data CSV
            cache_mtime = cache_path.stat().st_mtime
            data_paths = [
                MEDQUAD_PATH, TRIAGE_PATH, KAGGLE_CHATBOT_TRAIN_PATH,
                TRIAGE_FULL_PATH, SYMPTOM_DESC_PATH, SYMPTOM_DATASET_PATH,
            ]
            newest_data = max(
                (p.stat().st_mtime for p in data_paths if p.exists()),
                default=0.0,
            )
            if cache_mtime > newest_data:
                with open(cache_path, "rb") as f:
                    index = pickle.load(f)
                logger.info("RAG index loaded from cache (%s)", cache_path.name)
                return index
            else:
                logger.info("RAG cache is stale — rebuilding index")
        except Exception as e:
            logger.warning("Failed to load RAG cache (%s) — rebuilding", e)

    max_docs = int(os.getenv("RAG_MAX_DOCS", "20000"))
    # ── Tier 1: Authoritative QA knowledge (highest retrieval boost) ───
    medquad_docs = _load_medquad(max_docs=max_docs)
    triage_docs = _load_triage(max_docs=max_docs // 4)
    kaggle_docs = _load_kaggle_symptoms(max_docs=500)
    chatbot_docs = _load_kaggle_chatbot(max_docs=max_docs // 2)
    # ── Tier 2: Supplementary triage & symptom data ─────────────────────
    triage_full_docs = _load_triage_full(max_docs=max_docs // 2)
    severity_docs = _load_symptom_severity(max_docs=500)
    disease_symptom_docs = _load_disease_symptom_processed(max_docs=5000)
    # ── Tier 3: Clinical EHR (capped for build speed) ──────────────────
    mimic_cond_docs = _load_mimic_conditions(max_docs=2000)
    synthea_cond_docs = _load_synthea_conditions(max_docs=2000)
    # ── Tier 4: Auxiliary (minimal, for diversity) ─────────────────────
    fitzpatrick_docs = _load_fitzpatrick(max_docs=500)
    imaging_docs = _load_imaging_labels(max_docs=500)
    grok_dialogue_docs = _load_grok_dialogue(max_docs=500)
    uci_heart_docs = _load_uci_heart(max_docs=500)

    docs = (
        medquad_docs + triage_docs + kaggle_docs + chatbot_docs
        + triage_full_docs + severity_docs + disease_symptom_docs
        + mimic_cond_docs + synthea_cond_docs
        + fitzpatrick_docs + imaging_docs + grok_dialogue_docs + uci_heart_docs
    )
    if not docs:
        return RagIndex(vectorizer=TfidfVectorizer(), matrix=csr_matrix(np.zeros((0, 1))), docs=[])

    texts = [doc.text for doc in docs]
    # Limit max_features to keep FAISS index memory-feasible (< 4 GiB)
    # 143K docs × 10K features × 4 bytes ≈ 5.3 GiB — still too much for 8GB RAM
    # Use SVD to reduce to 512 dimensions: 143K × 512 × 4 ≈ 275 MiB
    max_features = min(5000, int(os.getenv("RAG_MAX_FEATURES", "5000")))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    matrix = vectorizer.fit_transform(texts).astype(np.float32)  # stays sparse

    faiss_index = None
    if faiss is not None and matrix.shape[0] > 0:
        # Use TruncatedSVD to reduce dimensions for FAISS (memory-efficient)
        n_components = min(256, min(matrix.shape[1], matrix.shape[0] - 1))
        try:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced = svd.fit_transform(matrix).astype(np.float32)  # (143K, 512) ≈ 275 MiB
            faiss.normalize_L2(reduced)
            faiss_index = faiss.IndexFlatIP(n_components)
            faiss_index.add(reduced)
            # Store SVD for query-time reduction
            matrix = reduced  # replace sparse with reduced dense
            vectorizer._rag_svd = svd  # attach SVD to vectorizer for query path
        except Exception as e:
            logger.warning("FAISS SVD reduction failed (%s), falling back to sparse cosine", e)
            faiss_index = None

    index = RagIndex(vectorizer=vectorizer, matrix=matrix, docs=docs, faiss_index=faiss_index)

    # ── Persist cache to disk ──────────────────────────────────────────────
    if cache_enabled:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(index, f, protocol=4)
            logger.info("RAG index cached to %s (%d docs)", cache_path.name, len(docs))
        except Exception as e:
            logger.warning("Failed to save RAG cache: %s", e)

    return index


def query_rag(query: str, top_k: int = 4) -> List[Dict[str, object]]:
    query = _strip_rag_noise(query)
    if not query.strip():
        return []
    index = _build_index()
    if not index.docs:
        return []

    query_vec = index.vectorizer.transform([query]).astype(np.float32)

    # ── FAISS path (SVD-reduced, fast) ──────────────────────────────────
    if index.faiss_index is not None:
        svd = getattr(index.vectorizer, "_rag_svd", None)
        if svd is not None:
            q_reduced = svd.transform(query_vec).astype(np.float32)
            faiss.normalize_L2(q_reduced)
        else:
            q_dense = query_vec.toarray()
            q_reduced = _normalize(q_dense)
        # Over-retrieve to allow re-ranking by source quality
        n_fetch = min(top_k * 4, len(index.docs))
        scores, indices = index.faiss_index.search(q_reduced, n_fetch)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = index.docs[int(idx)]
            boost = _SOURCE_QUALITY_BOOST.get(doc.source, 0.0)
            adjusted = float(score) + boost
            # Filter out low-relevance noise (score < 0.3 after boost)
            if adjusted < 0.3:
                continue
            hits.append({"text": doc.text, "source": doc.source, "score": adjusted, "metadata": doc.metadata})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]

    # ── Sparse path (sklearn cosine_similarity works on sparse) ────────
    scores = cosine_similarity(query_vec, index.matrix)[0]
    # Apply source quality boost
    for i, doc in enumerate(index.docs):
        boost = _SOURCE_QUALITY_BOOST.get(doc.source, 0.0)
        scores[i] += boost
    ranked = np.argsort(scores)[::-1][: min(top_k, len(index.docs))]
    results = []
    for idx in ranked:
        doc = index.docs[int(idx)]
        score = float(scores[int(idx)])
        # Filter out low-relevance noise (score < 0.3)
        if score < 0.3:
            continue
        results.append(
            {"text": doc.text, "source": doc.source, "score": score, "metadata": doc.metadata}
        )
    return results


def rebuild_rag_index() -> Dict[str, object]:
    """Force rebuild the RAG index (clears LRU cache and rebuilds from all data)."""
    _build_index.cache_clear()
    index = _build_index()
    source_counts: Dict[str, int] = {}
    for doc in index.docs:
        source_counts[doc.source] = source_counts.get(doc.source, 0) + 1
    return {
        "total_docs": len(index.docs),
        "sources": source_counts,
        "faiss_enabled": index.faiss_index is not None,
        "matrix_shape": list(index.matrix.shape) if index.matrix.shape[0] > 0 else [0, 0],
    }


def _strip_rag_noise(text: str) -> str:
    """Remove metadata patterns that pollute RAG queries (Duration, Severity, Pre-existing)."""
    import re as _re
    cleaned = _re.sub(r"\.\s*Duration:\s*[^.]+", "", text, flags=_re.IGNORECASE)
    cleaned = _re.sub(r"\.\s*Severity:\s*[^.]+", "", cleaned, flags=_re.IGNORECASE)
    cleaned = _re.sub(r"\.\s*Pre-existing:\s*[^.]+", "", cleaned, flags=_re.IGNORECASE)
    return cleaned.strip()


def build_rag_context(query: str, *, top_k: int = 4, search_context: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    clean_query = _strip_rag_noise(query)
    hits = query_rag(clean_query, top_k=top_k)
    return {
        "enabled": bool(hits),
        "query": clean_query,
        "items": hits,
        "sources": sorted({item["source"] for item in hits}),
        "external_search": search_context or {},
    }
