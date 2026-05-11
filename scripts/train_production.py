#!/usr/bin/env python3
"""
Production-Ready Training Pipeline v2

Improvements over v1:
  1. Label consolidation: case-fold merge + group rare classes (<30) into category buckets
  2. Transformer encoder (distilbert-base-uncased) instead of TF-IDF + LR
  3. Temperature scaling calibration for reliable confidence scores
  4. Minimum 30 samples per class threshold
  5. Top-k accuracy reporting

Usage:
  python scripts/train_production.py --only triage
  python scripts/train_production.py --only triage,dialogue
  python scripts/train_production.py                  # train all
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "data" / "ready"
MODELS = ROOT / "backend" / "models"

# ── Medical category mapping for rare-condition grouping ─────────────────
CATEGORY_KEYWORDS = {
    "respiratory": [
        "pneumonia", "bronchitis", "asthma", "copd", "pulmonary", "respiratory",
        "sinusitis", "pharyngitis", "laryngitis", "urti", "otitis", "rhinitis",
        "cough", "influenza", "flu", "tuberculosis", "emphysema", "pneumothorax",
        "bronchospasm", "covid", "sars", "croup", "pleural", "lung",
    ],
    "cardiovascular": [
        "angina", "infarction", "heart", "atrial", "fibrillation", "pericarditis",
        "embolism", "dvt", "thrombosis", "hypertension", "hypotension", "aortic",
        "cardiomyopathy", "arrhythmia", "pvt", "psvt", "stem", "nstemi",
        "coronary", "cardiac", "endocarditis", "myocarditis", "vascular",
    ],
    "gastrointestinal": [
        "gerd", "ulcer", "gastritis", "colitis", "appendicitis", "pancreatitis",
        "cholecystitis", "diverticulitis", "hepatitis", "cirrhosis", "bowel",
        "diarrhea", "constipation", "hemorrhoid", "esophageal", "intestinal",
        "gastroenteritis", "ibd", "ibs", "crohn", "celiac", "gallbladder",
        "pancreatic", "gi", "rectal", "anal", "fecal",
    ],
    "neurological": [
        "stroke", "seizure", "epilepsy", "migraine", "headache", "vertigo",
        "neuropathy", "sclerosis", "parkinson", "alzheimer", "dementia",
        "guillain", "myasthenia", "bell", "neuralgia", "meningitis", "encephalitis",
        "tremor", "ataxia", "dystonic", "conversion", "clonus",
    ],
    "musculoskeletal": [
        "fracture", "sprain", "arthritis", "osteoporosis", "back pain",
        "tendinitis", "bursitis", "fibromyalgia", "muscle", "joint", "spinal",
        "herniated", "disc", "rotator cuff", "clavicle", "rib", "lumbar",
        "cervical", "spondylosis", "osteomyelitis", "tenosynovitis",
    ],
    "dermatological": [
        "dermatitis", "eczema", "psoriasis", "cellulitis", "abscess", "rash",
        "urticaria", "acne", "melanoma", "carcinoma", "keratosis", "lesion",
        "skin", "fungal", "herpes", "shingles", "impetigo", "scabies",
        "laceration", "burn", "wound", "blister", "folliculitis",
    ],
    "infectious": [
        "hiv", "aids", "sepsis", "infection", "bacterial", "viral", "fungal",
        "parasitic", "malaria", "dengue", "typhoid", "rabies", "tetanus",
        "measles", "mumps", "rubella", "chickenpox", "mononucleosis",
        "strep", "staphylococcal", "clostridium", "anaphylaxis",
    ],
    "mental_health": [
        "anxiety", "depression", "panic", "ptsd", "bipolar", "schizophrenia",
        "insomnia", "substance", "alcohol", "overdose", "suicidal", "eating disorder",
        "adhd", "autism", "ocd", "phobia", "delirium", "psychosis",
    ],
    "endocrine": [
        "diabetes", "thyroid", "hyperthyroid", "hypothyroid", "adrenal",
        "cushing", "addison", "acromegaly", "metabolic", "obesity", "pcod",
        "pcos", "gout", "hyperlipidemia",
    ],
    "renal_urological": [
        "kidney", "renal", "uti", "urinary", "nephrolithiasis", "stone",
        "cystitis", "prostatitis", "prostate", "bladder", "incontinence",
        "nephritis", "hematuria",
    ],
    "reproductive": [
        "pregnancy", "prenatal", "miscarriage", "ectopic", "endometriosis",
        "fibroid", "ovarian", "pelvic", "menstrual", "vaginal", "cervical",
        "uterine", "breast", "infertility", "preeclampsia",
    ],
    "hematological": [
        "anemia", "leukemia", "lymphoma", "bleeding", "thrombocytopenia",
        "hemophilia", "sickle", "thalassemia", "polycythemia", "neutropenia",
    ],
    "autoimmune": [
        "lupus", "sle", "rheumatoid", "scleroderma", "vasculitis", "sarcoidosis",
        "crohn", "celiac", "sjogren", "hashimoto", "goodpasture",
    ],
    "ent": [
        "ear", "nose", "throat", "tonsillitis", "epistaxis", "hearing",
        "tinnitus", "vertigo", "swimmer", "sinus", "nasal", "allergic",
        "conductive", "sensorineural",
    ],
    "ophthalmological": [
        "eye", "conjunctivitis", "glaucoma", "cataract", "retinal", "vision",
        "uveitis", "keratitis", "optic", "strabismus", "amblyopia",
    ],
}


def consolidate_labels(labels_series: pd.Series, min_samples: int = 30) -> tuple:
    """Consolidate labels: case-fold merge, then group rare classes into category buckets."""
    # Step 1: Case-fold merge
    folded = labels_series.str.strip().str.title()
    # Fix specific known variants
    fold_map = {
        "Covid-19": "COVID-19",
        "Urti": "URTI",
        "Gerd": "GERD",
        "Sle": "SLE",
        "Psvt": "PSVT",
        "Hiv (Initial Infection)": "HIV (Initial Infection)",
        "Aids": "AIDS",
    }
    folded = folded.replace(fold_map)
    
    # Step 2: Count after merge
    counts = folded.value_counts()
    
    # Step 3: Group rare labels (< min_samples) into category buckets
    rare_labels = set(counts[counts < min_samples].index)
    
    def map_rare(label):
        if label not in rare_labels:
            return label
        label_lower = label.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in label_lower:
                    return f"other_{category}"
        return "other_general"
    
    consolidated = folded.map(map_rare)
    
    # Step 4: Re-merge any new duplicates created by grouping
    final_counts = consolidated.value_counts()
    
    # Build mapping dict for reference
    mapping = {}
    for orig, cons in zip(labels_series, consolidated):
        if orig != cons:
            mapping[orig] = cons
    
    return consolidated, mapping, final_counts


def consolidate_dialogue_intents(intent_series: pd.Series, min_samples: int = 30, max_cap: int = 5000) -> tuple:
    """Consolidate dialogue intents: drop rare intents, cap mega-classes for balance."""
    counts = intent_series.value_counts()
    rare_intents = set(counts[counts < min_samples].index)
    
    # Keep only non-rare intents - drop samples with rare intents
    def map_intent(intent):
        if intent in rare_intents:
            return None  # Will be filtered out
        return intent
    
    consolidated = intent_series.map(map_intent)
    
    # Build mapping dict for reference (only for kept intents)
    mapping = {}
    for orig, cons in zip(intent_series, consolidated):
        if cons is None:
            mapping[orig] = "__dropped__"
    
    final_counts = consolidated.dropna().value_counts()
    return consolidated, mapping, final_counts


# ══════════════════════════════════════════════════════════════════════════
# 1. TEXT TRIAGE CLASSIFIER (Transformer)
# ══════════════════════════════════════════════════════════════════════════
def train_triage(args):
    print("\n" + "=" * 70)
    print("  TRAINING: Text Triage Classifier (Production v2)")
    print("=" * 70)
    t0 = time.time()

    # Load data
    train_df = pd.read_csv(READY / "triage" / "train.csv")
    val_df = pd.read_csv(READY / "triage" / "val.csv")
    test_df = pd.read_csv(READY / "triage" / "test.csv")

    # Consolidate labels
    print("  Consolidating labels...")
    train_df["condition_cons"], triage_map, train_counts = consolidate_labels(
        train_df["condition"], min_samples=args.min_samples
    )
    val_df["condition_cons"], _, _ = consolidate_labels(val_df["condition"], min_samples=args.min_samples)
    test_df["condition_cons"], _, _ = consolidate_labels(test_df["condition"], min_samples=args.min_samples)

    num_classes_train = train_df["condition_cons"].nunique()
    print(f"  Classes after consolidation: {num_classes_train} (was 371)")
    print(f"  Top 10: {dict(train_counts.head(10))}")

    # Filter val/test to only labels seen in train
    train_labels_set = set(train_df["condition_cons"].unique())
    val_df = val_df[val_df["condition_cons"].isin(train_labels_set)].copy()
    test_df = test_df[test_df["condition_cons"].isin(train_labels_set)].copy()
    print(f"  Val/Test after filtering to train labels: {len(val_df)}/{len(test_df)}")

    # Prepare text
    WHITESPACE_RE = re.compile(r"\s+")
    def clean_text(value):
        text = str(value).strip().lower()
        text = text.replace("|", " ").replace("_", " ").replace("-", " ")
        text = WHITESPACE_RE.sub(" ", text)
        return text

    X_train = train_df["symptom_text"].apply(clean_text).values
    y_train = train_df["condition_cons"].values
    X_val = val_df["symptom_text"].apply(clean_text).values
    y_val = val_df["condition_cons"].values
    X_test = test_df["symptom_text"].apply(clean_text).values
    y_test = test_df["condition_cons"].values

    # Try transformer, fall back to improved TF-IDF
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    try:
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False

    out_dir = MODELS
    out_dir.mkdir(parents=True, exist_ok=True)

    use_transformer = HAS_TORCH and HAS_TRANSFORMERS and not args.no_transformer
    if use_transformer:
        print("  WARNING: Transformer fine-tuning on CPU is very slow (hours). Consider --no-transformer.")
        metrics = _train_triage_transformer(
            X_train, y_train, X_val, y_val, X_test, y_test,
            num_classes_train, out_dir, args
        )
    else:
        print("  Using improved TF-IDF + LR with label consolidation + calibration")
        metrics = _train_triage_tfidf(
            X_train, y_train, X_val, y_val, X_test, y_test,
            num_classes_train, out_dir, args
        )

    # Save label consolidation mapping
    (out_dir / "triage_label_consolidation.json").write_text(
        json.dumps(triage_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics["training_time_sec"] = round(time.time() - t0, 1)
    metrics["label_consolidation"] = True
    metrics["original_classes"] = 371
    metrics["consolidated_classes"] = num_classes_train
    (out_dir / "triage_production_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  DONE: Triage model saved - Time: {metrics['training_time_sec']}s")
    return metrics


def _train_triage_transformer(X_train, y_train, X_val, y_val, X_test, y_test,
                               _num_classes_unused, out_dir, args):
    """Train triage with DistilBERT + classification head + temperature scaling."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModel
    from sklearn.preprocessing import LabelEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using transformer on {device}")

    model_name = args.transformer_model
    print(f"  Model: {model_name}")

    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)  # Only fit on train labels - val/test already filtered
    y_train_enc = le.transform(y_train)
    # Map val/test labels through encoder (unknown labels -> skip)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    num_classes = len(le.classes_)
    print(f"  LabelEncoder classes: {num_classes}")
    labels_list = list(le.classes_)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = args.max_len

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text, max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.LongTensor([self.labels[idx]])[0],
            }

    train_ds = TextDataset(X_train, y_train_enc, tokenizer, max_len)
    val_ds = TextDataset(X_val, y_val_enc, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model: DistilBERT + pooling + classifier
    class TriageTransformer(nn.Module):
        def __init__(self, model_name, num_classes, dropout=0.3):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            hidden_size = self.bert.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token
            cls_output = outputs.last_hidden_state[:, 0, :]
            return self.classifier(cls_output)

    model = TriageTransformer(model_name, num_classes).to(device)

    # Class weights
    class_counts = np.bincount(y_train_enc, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    epochs = args.triage_epochs

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                out = model(input_ids, attention_mask)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "triage_transformer_best.pt")
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} - loss: {train_loss/len(train_loader):.4f} - val_f1: {val_f1:.4f} - val_acc: {val_acc:.4f}")

        if patience_counter >= 7:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Temperature scaling calibration ──
    print("  Calibrating with temperature scaling...")
    model.load_state_dict(torch.load(out_dir / "triage_transformer_best.pt", map_location=device))
    temperature = _calibrate_temperature(model, val_ds, num_classes, device, tokenizer, max_len)

    # Test evaluation
    model.eval()
    test_ds = TextDataset(X_test, y_test_enc, tokenizer, max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            out = model(input_ids, attention_mask)
            # Apply temperature scaling
            logits = out / temperature
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Top-3 accuracy
    all_probs_arr = np.array(all_probs)
    top3_preds = np.argsort(all_probs_arr, axis=1)[:, -3:]
    top3_correct = sum(
        1 for i, label in enumerate(all_labels) if label in top3_preds[i]
    )
    top3_acc = top3_correct / len(all_labels)

    # Expected Calibration Error
    ece = _compute_ece(all_probs_arr, np.array(all_labels), n_bins=10)

    print(f"\n  --- Triage Test Results ---")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"  Test Macro-F1:     {test_f1:.4f}")
    print(f"  Test Weighted-F1:  {test_f1_weighted:.4f}")
    print(f"  Top-3 Accuracy:    {top3_acc:.4f}")
    print(f"  ECE (calibrated):  {ece:.4f}")

    # Save artifacts
    torch.save(model.state_dict(), out_dir / "triage_transformer_final.pt")
    (out_dir / "triage_labels.json").write_text(
        json.dumps(labels_list, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "triage_temperature.json").write_text(
        json.dumps({"temperature": round(float(temperature), 4)}, indent=2), encoding="utf-8"
    )

    # Save tokenizer reference
    (out_dir / "triage_model_config.json").write_text(
        json.dumps({
            "model_name": model_name,
            "max_len": max_len,
            "num_classes": num_classes,
            "architecture": "distilbert_cls",
        }, indent=2), encoding="utf-8"
    )

    return {
        "architecture": "distilbert_cls",
        "model_name": model_name,
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_weighted_f1": round(float(test_f1_weighted), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "ece_calibrated": round(float(ece), 4),
        "temperature": round(float(temperature), 4),
        "best_val_macro_f1": round(float(best_val_f1), 4),
        "epochs_trained": int(epoch),
    }


def _train_triage_tfidf(X_train, y_train, X_val, y_val, X_test, y_test,
                         num_classes, out_dir, args):
    """Fallback: improved TF-IDF + LogisticRegression with calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    import joblib

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    labels_list = list(le.classes_)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, C=1.0)),
    ])

    print("  Training TF-IDF + LR (with label consolidation)...")
    pipeline.fit(X_train, y_train_enc)

    # Calibrate
    print("  Calibrating with Platt scaling...")
    calibrated = CalibratedClassifierCV(pipeline.named_steps["clf"], cv="prefit", method="sigmoid")
    calibrated.fit(pipeline.named_steps["tfidf"].transform(X_val), y_val_enc)

    # Test
    X_test_tfidf = pipeline.named_steps["tfidf"].transform(X_test)
    y_pred = calibrated.predict(X_test_tfidf)
    y_prob = calibrated.predict_proba(X_test_tfidf)

    test_f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test_enc, y_pred)
    test_f1_weighted = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)

    # Top-3
    top3_preds = np.argsort(y_prob, axis=1)[:, -3:]
    top3_correct = sum(1 for i, label in enumerate(y_test_enc) if label in top3_preds[i])
    top3_acc = top3_correct / len(y_test_enc)

    ece = _compute_ece(y_prob, y_test_enc, n_bins=10)

    print(f"\n  --- Triage TF-IDF Test Results ---")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"  Test Macro-F1:     {test_f1:.4f}")
    print(f"  Top-3 Accuracy:    {top3_acc:.4f}")
    print(f"  ECE (calibrated):  {ece:.4f}")

    # Save
    joblib.dump(calibrated, out_dir / "triage_classifier_calibrated.joblib")
    joblib.dump(pipeline.named_steps["tfidf"], out_dir / "triage_tfidf_vectorizer.joblib")
    (out_dir / "triage_labels.json").write_text(
        json.dumps(labels_list, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "architecture": "tfidf_lr_calibrated",
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_weighted_f1": round(float(test_f1_weighted), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "ece_calibrated": round(float(ece), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. DIALOGUE INTENT CLASSIFIER (Transformer)
# ══════════════════════════════════════════════════════════════════════════
def train_dialogue(args):
    print("\n" + "=" * 70)
    print("  TRAINING: Dialogue Intent Classifier (Production v2)")
    print("=" * 70)
    t0 = time.time()

    train_df = pd.read_csv(READY / "dialogue" / "train.csv")
    val_df = pd.read_csv(READY / "dialogue" / "val.csv")
    test_df = pd.read_csv(READY / "dialogue" / "test.csv")

    # Consolidate intents
    print("  Consolidating intents...")
    train_df["intent_cons"], dial_map, train_counts = consolidate_dialogue_intents(
        train_df["intent"], min_samples=args.min_samples
    )
    val_df["intent_cons"], _, _ = consolidate_dialogue_intents(val_df["intent"], min_samples=args.min_samples)
    test_df["intent_cons"], _, _ = consolidate_dialogue_intents(test_df["intent"], min_samples=args.min_samples)

    # Drop rows with None intents (rare intents filtered out)
    train_df = train_df.dropna(subset=["intent_cons"]).copy()
    val_df = val_df.dropna(subset=["intent_cons"]).copy()
    test_df = test_df.dropna(subset=["intent_cons"]).copy()

    num_classes_train = train_df["intent_cons"].nunique()
    print(f"  Intents after consolidation: {num_classes_train} (was 395)")
    print(f"  Train rows after dropping rare: {len(train_df)}")
    print(f"  Top 10: {dict(train_counts.head(10))}")

    # Filter val/test to only intents seen in train
    train_intents_set = set(train_df["intent_cons"].unique())
    val_df = val_df[val_df["intent_cons"].isin(train_intents_set)].copy()
    test_df = test_df[test_df["intent_cons"].isin(train_intents_set)].copy()
    print(f"  Val/Test after filtering to train intents: {len(val_df)}/{len(test_df)}")

    # Merge overly generic intents and undersample dominant classes
    # "information" and "general" are too vague - merge into "general_info"
    intent_merge_map = {"information": "general_info", "general": "general_info"}
    train_df["intent_cons"] = train_df["intent_cons"].replace(intent_merge_map)
    val_df["intent_cons"] = val_df["intent_cons"].replace(intent_merge_map)
    test_df["intent_cons"] = test_df["intent_cons"].replace(intent_merge_map)

    # Undersample classes with > 5000 samples to cap at 5000
    MAX_PER_CLASS = 5000
    frames = []
    for intent, group in train_df.groupby("intent_cons"):
        if len(group) > MAX_PER_CLASS:
            frames.append(group.sample(n=MAX_PER_CLASS, random_state=42))
        else:
            frames.append(group)
    train_df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
    print(f"  After undersampling: {len(train_df)} train rows")
    print(f"  Intent distribution: {dict(train_df['intent_cons'].value_counts().head(15))}")

    # Use only user text for intent prediction (intent should be derivable from user query)
    WHITESPACE_RE = re.compile(r"\s+")
    def clean_text(value):
        text = str(value).strip().lower()
        text = text.replace("|", " ").replace("_", " ").replace("-", " ")
        text = WHITESPACE_RE.sub(" ", text)
        return text

    def make_dialogue_text(row):
        text = clean_text(row.get("text", row.get("user_text", row.get("user", ""))))
        return text

    X_train = train_df.apply(make_dialogue_text, axis=1).values
    y_train = train_df["intent_cons"].values
    X_val = val_df.apply(make_dialogue_text, axis=1).values
    y_val = val_df["intent_cons"].values
    X_test = test_df.apply(make_dialogue_text, axis=1).values
    y_test = test_df["intent_cons"].values

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, AutoModel
        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False

    out_dir = MODELS
    out_dir.mkdir(parents=True, exist_ok=True)

    use_transformer = HAS_TRANSFORMERS and not args.no_transformer
    if use_transformer:
        print("  WARNING: Transformer fine-tuning on CPU is very slow (hours). Consider --no-transformer.")
        metrics = _train_dialogue_transformer(
            X_train, y_train, X_val, y_val, X_test, y_test,
            num_classes_train, out_dir, args
        )
    else:
        print("  Using improved TF-IDF + LR with intent consolidation + calibration")
        metrics = _train_dialogue_tfidf(
            X_train, y_train, X_val, y_val, X_test, y_test,
            num_classes_train, out_dir, args
        )

    # Save consolidation mapping
    (out_dir / "dialogue_intent_consolidation.json").write_text(
        json.dumps(dial_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics["training_time_sec"] = round(time.time() - t0, 1)
    metrics["label_consolidation"] = True
    metrics["original_intents"] = 395
    metrics["consolidated_intents"] = num_classes_train
    (out_dir / "dialogue_production_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  DONE: Dialogue model saved - Time: {metrics['training_time_sec']}s")
    return metrics


def _train_dialogue_transformer(X_train, y_train, X_val, y_val, X_test, y_test,
                                 _num_classes_unused, out_dir, args):
    """Train dialogue with DistilBERT + classification head + temperature scaling."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModel
    from sklearn.preprocessing import LabelEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using transformer on {device}")

    model_name = args.transformer_model
    print(f"  Model: {model_name}")

    le = LabelEncoder()
    le.fit(y_train)  # Only fit on train labels - val/test already filtered
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    num_classes = len(le.classes_)
    print(f"  LabelEncoder intents: {num_classes}")
    labels_list = list(le.classes_)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = args.max_len

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text, max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.LongTensor([self.labels[idx]])[0],
            }

    train_ds = TextDataset(X_train, y_train_enc, tokenizer, max_len)
    val_ds = TextDataset(X_val, y_val_enc, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class DialogueTransformer(nn.Module):
        def __init__(self, model_name, num_classes, dropout=0.3):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            hidden_size = self.bert.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            return self.classifier(cls_output)

    model = DialogueTransformer(model_name, num_classes).to(device)

    class_counts = np.bincount(y_train_enc, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_val_f1 = 0
    patience_counter = 0
    epochs = args.dialogue_epochs

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                out = model(input_ids, attention_mask)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "dialogue_transformer_best.pt")
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} - loss: {train_loss/len(train_loader):.4f} - val_f1: {val_f1:.4f} - val_acc: {val_acc:.4f}")

        if patience_counter >= 7:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Temperature scaling
    print("  Calibrating with temperature scaling...")
    model.load_state_dict(torch.load(out_dir / "dialogue_transformer_best.pt", map_location=device))
    temperature = _calibrate_temperature(model, val_ds, num_classes, device, tokenizer, max_len)

    # Test evaluation
    model.eval()
    test_ds = TextDataset(X_test, y_test_enc, tokenizer, max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            out = model(input_ids, attention_mask)
            logits = out / temperature
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    all_probs_arr = np.array(all_probs)
    top3_preds = np.argsort(all_probs_arr, axis=1)[:, -3:]
    top3_correct = sum(1 for i, label in enumerate(all_labels) if label in top3_preds[i])
    top3_acc = top3_correct / len(all_labels)

    ece = _compute_ece(all_probs_arr, np.array(all_labels), n_bins=10)

    print(f"\n  --- Dialogue Test Results ---")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"  Test Macro-F1:     {test_f1:.4f}")
    print(f"  Test Weighted-F1:  {test_f1_weighted:.4f}")
    print(f"  Top-3 Accuracy:    {top3_acc:.4f}")
    print(f"  ECE (calibrated):  {ece:.4f}")

    torch.save(model.state_dict(), out_dir / "dialogue_transformer_final.pt")
    (out_dir / "dialogue_labels.json").write_text(
        json.dumps(labels_list, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "dialogue_temperature.json").write_text(
        json.dumps({"temperature": round(float(temperature), 4)}, indent=2), encoding="utf-8"
    )
    (out_dir / "dialogue_model_config.json").write_text(
        json.dumps({
            "model_name": model_name,
            "max_len": max_len,
            "num_classes": num_classes,
            "architecture": "distilbert_cls",
        }, indent=2), encoding="utf-8"
    )

    return {
        "architecture": "distilbert_cls",
        "model_name": model_name,
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_weighted_f1": round(float(test_f1_weighted), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "ece_calibrated": round(float(ece), 4),
        "temperature": round(float(temperature), 4),
        "best_val_macro_f1": round(float(best_val_f1), 4),
        "epochs_trained": int(epoch),
    }


def _train_dialogue_tfidf(X_train, y_train, X_val, y_val, X_test, y_test,
                           num_classes, out_dir, args):
    """Fallback: improved TF-IDF + LR with calibration for dialogue."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    import joblib

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    labels_list = list(le.classes_)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1, C=1.0)),
    ])

    print("  Training TF-IDF + LR (with intent consolidation + calibration)...")
    pipeline.fit(X_train, y_train_enc)

    calibrated = CalibratedClassifierCV(pipeline.named_steps["clf"], cv="prefit", method="sigmoid")
    calibrated.fit(pipeline.named_steps["tfidf"].transform(X_val), y_val_enc)

    X_test_tfidf = pipeline.named_steps["tfidf"].transform(X_test)
    y_pred = calibrated.predict(X_test_tfidf)
    y_prob = calibrated.predict_proba(X_test_tfidf)

    test_f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test_enc, y_pred)
    test_f1_weighted = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)

    top3_preds = np.argsort(y_prob, axis=1)[:, -3:]
    top3_correct = sum(1 for i, label in enumerate(y_test_enc) if label in top3_preds[i])
    top3_acc = top3_correct / len(y_test_enc)

    ece = _compute_ece(y_prob, y_test_enc, n_bins=10)

    print(f"\n  --- Dialogue TF-IDF Test Results ---")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"  Test Macro-F1:     {test_f1:.4f}")
    print(f"  Top-3 Accuracy:    {top3_acc:.4f}")
    print(f"  ECE (calibrated):  {ece:.4f}")

    joblib.dump(calibrated, out_dir / "dialogue_classifier_calibrated.joblib")
    joblib.dump(pipeline.named_steps["tfidf"], out_dir / "dialogue_tfidf_vectorizer.joblib")
    (out_dir / "dialogue_labels.json").write_text(
        json.dumps(labels_list, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "architecture": "tfidf_lr_calibrated",
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_weighted_f1": round(float(test_f1_weighted), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "ece_calibrated": round(float(ece), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# CALIBRATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════
def _calibrate_temperature(model, val_dataset, num_classes, device, tokenizer, max_len):
    """Learn optimal temperature for temperature scaling using NLL on validation set."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Collect logits and labels
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            out = model(input_ids, attention_mask)
            all_logits.append(out.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Optimize temperature
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_loss():
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    final_temp = temperature.item()
    print(f"  Optimal temperature: {final_temp:.4f}")
    return final_temp


def _compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    for bin_idx in range(n_bins):
        bin_lower = bin_idx / n_bins
        bin_upper = (bin_idx + 1) / n_bins
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)

    return float(ece)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Production Training Pipeline v2")
    parser.add_argument("--only", default="", help="Comma-separated: triage,dialogue")
    parser.add_argument("--transformer-model", default="distilbert-base-uncased",
                        help="HuggingFace transformer model name")
    parser.add_argument("--max-len", type=int, default=128, help="Max token length")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--triage-epochs", type=int, default=15, help="Max triage epochs")
    parser.add_argument("--dialogue-epochs", type=int, default=10, help="Max dialogue epochs")
    parser.add_argument("--min-samples", type=int, default=30, help="Min samples per class before grouping")
    parser.add_argument("--no-transformer", action="store_true", help="Force TF-IDF mode (faster on CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    selected = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else ["triage", "dialogue"]

    print("=" * 60)
    print("  PRODUCTION TRAINING PIPELINE v2 - Consolidated")
    print("=" * 60)
    print(f"  Models to train: {selected}")
    print(f"  Transformer: {args.transformer_model}")
    print(f"  Min samples/class: {args.min_samples}")
    print(f"  Data source: {READY}")
    print(f"  Output: {MODELS}")

    all_metrics = {}
    total_t0 = time.time()

    if "triage" in selected:
        all_metrics["triage"] = train_triage(args)

    if "dialogue" in selected:
        all_metrics["dialogue"] = train_dialogue(args)

    total_time = round(time.time() - total_t0, 1)
    all_metrics["total_training_time_sec"] = total_time
    (MODELS / "production_run_summary.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("  PRODUCTION TRAINING COMPLETE")
    print("=" * 70)
    for name, m in all_metrics.items():
        if isinstance(m, dict):
            f1 = m.get("test_macro_f1", "N/A")
            acc = m.get("test_accuracy", "N/A")
            top3 = m.get("top3_accuracy", "N/A")
            ece = m.get("ece_calibrated", "N/A")
            print(f"  {name:12s} - acc: {acc}, f1: {f1}, top3: {top3}, ece: {ece}")
    print(f"  Total time: {total_time}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
