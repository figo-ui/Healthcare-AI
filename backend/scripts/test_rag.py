"""Quick test: verify RAG loads ALL datasets — no Django needed."""
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # backend/

DATA_PATHS = {
    # Unified
    "medquad": ROOT / "data" / "ready" / "unified" / "ULTIMATE_CONVERSATIONAL_QA.csv",
    "triage_knowledge": ROOT / "data" / "ready" / "unified" / "ULTIMATE_TRIAGE_KNOWLEDGE.csv",
    "imaging_labels": ROOT / "data" / "ready" / "unified" / "ULTIMATE_IMAGING_LABELS.csv",
    # Kaggle symptom
    "symptom_desc": ROOT / "data" / "ready" / "kaggle_symptom" / "symptom_Description.csv",
    "symptom_prec": ROOT / "data" / "ready" / "kaggle_symptom" / "symptom_precaution.csv",
    "symptom_dataset": ROOT / "data" / "ready" / "kaggle_symptom" / "dataset.csv",
    "symptom_severity": ROOT / "data" / "ready" / "kaggle_symptom" / "Symptom-severity.csv",
    "disease_symptom": ROOT / "data" / "ready" / "kaggle_symptom" / "disease_symptom_processed.csv",
    # Kaggle chatbot
    "chatbot_train": ROOT / "data" / "ready" / "kaggle_chatbot" / "train.csv",
    "chatbot_val": ROOT / "data" / "ready" / "kaggle_chatbot" / "val.csv",
    # Dialogue
    "dialogue_full": ROOT / "data" / "ready" / "dialogue" / "full.csv",
    # Grok
    "grok_dialogue": ROOT / "data" / "ready" / "grok" / "triage_dialogue_reasoning.csv",
    "grok_supervised": ROOT / "data" / "ready" / "grok" / "triage_supervised.csv",
    # Fitzpatrick
    "fitzpatrick": ROOT / "data" / "ready" / "fitzpatrick" / "metadata.csv",
    # MIMIC-IV
    "mimic_conditions": ROOT / "data" / "ready" / "mimic" / "conditions.csv",
    "mimic_medications": ROOT / "data" / "ready" / "mimic" / "medications.csv",
    "mimic_observations": ROOT / "data" / "ready" / "mimic" / "observations.csv",
    "mimic_procedures": ROOT / "data" / "ready" / "mimic" / "procedures.csv",
    # Synthea
    "synthea_conditions": ROOT / "data" / "ready" / "synthea" / "conditions.csv",
    "synthea_medications": ROOT / "data" / "ready" / "synthea" / "medications.csv",
    "synthea_observations": ROOT / "data" / "ready" / "synthea" / "observations.csv",
    "synthea_procedures": ROOT / "data" / "ready" / "synthea" / "procedures.csv",
    "synthea_careplans": ROOT / "data" / "ready" / "synthea" / "careplans.csv",
    "synthea_allergies": ROOT / "data" / "ready" / "synthea" / "allergies.csv",
    "synthea_immunizations": ROOT / "data" / "ready" / "synthea" / "immunizations.csv",
    # Triage
    "triage_full": ROOT / "data" / "ready" / "triage" / "full.csv",
    # UCI
    "uci_heart": ROOT / "data" / "ready" / "uci" / "heart_failure.csv",
}

print("=" * 60)
print("RAG Dataset Inventory (ALL sources)")
print("=" * 60)

total_available = 0
for name, path in DATA_PATHS.items():
    if not path.exists():
        print(f"  [MISSING] {name}: {path}")
        continue
    with path.open("r", encoding="utf-8", newline="") as f:
        count = sum(1 for _ in csv.DictReader(f))
    total_available += count
    print(f"  {name}: {count:,} rows  ({path.name})")

print(f"\n  Total available: {total_available:,} rows")

# Now test actual RAG loading via the module
print("\n" + "=" * 60)
print("Building Full RAG Index (all sources)")
print("=" * 60)

BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from guidance.services.rag import rebuild_rag_index, query_rag

result = rebuild_rag_index()
print(f"  Total docs indexed: {result['total_docs']:,}")
print(f"  FAISS enabled: {result['faiss_enabled']}")
print(f"  Matrix shape: {result['matrix_shape']}")
print(f"\n  Sources breakdown:")
for source, count in sorted(result["sources"].items(), key=lambda x: -x[1]):
    print(f"    {source}: {count:,}")

# Test queries
queries = [
    "I have a headache and feel dizzy",
    "What is diabetes?",
    "I'm worried about chest pain",
    "How to prevent flu",
    "My stomach hurts after eating",
    "I feel sad and anxious",
    "What causes high blood pressure",
    "I have a rash on my arm",
    "Can antibiotics cause diarrhea?",
    "Is it normal to feel tired all the time?",
    "What medication is used for heart failure?",
    "What is Otitis media?",
    "What are symptoms of Fungal infection?",
    "I have a skin rash and itching",
    "What is the typical dose of Furosemide?",
]

print("\n" + "=" * 60)
print("Sample RAG Queries (top-3 from full index)")
print("=" * 60)

for q in queries:
    hits = query_rag(q, top_k=3)
    print(f"\n  Q: {q}")
    for hit in hits:
        score = hit["score"]
        source = hit["source"]
        answer = (hit.get("metadata") or {}).get("answer", "")[:120]
        if score > 0.03:
            print(f"    [{score:.3f}] ({source}) {answer}")
