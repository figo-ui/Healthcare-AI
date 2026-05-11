#!/usr/bin/env python3
"""
Unified Training Pipeline — trains all models from data/ready/

Models trained:
  1. Text Triage Classifier (symptom_text → condition)
  2. Dialogue Intent Classifier (text → intent)
  3. Image Classifier (DermaMNIST 64×64 → skin condition)

Usage:
  python scripts/train_all.py                  # train all
  python scripts/train_all.py --only triage    # train specific model
  python scripts/train_all.py --only triage,dialogue
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "data" / "ready"
MODELS = ROOT / "backend" / "models"

WHITESPACE_RE = re.compile(r"\s+")
REASON_CLAUSE_RE = re.compile(r"\breason:\s*[^|]+")
COMORBIDITY_CLAUSE_RE = re.compile(r"\bcomorbidities:\s*[^|]+")
MEDICATION_CLAUSE_RE = re.compile(r"\bcurrent medications?:\s*[^|]+")
ALLERGY_CLAUSE_RE = re.compile(r"\bknown allergies?:\s*[^|]+")
ADMIN_CLAUSE_RE = re.compile(
    r"\b("
    r"encounter for symptom(?: \(procedure\))?|"
    r"general examination of patient(?: \(procedure\))?|"
    r"patient encounter procedure|"
    r"well child visit(?: \(procedure\))?|"
    r"death certification|"
    r"symptoms reported:|"
    r"hypertension follow up encounter"
    r")\b"
)


# ── Shared Utilities ─────────────────────────────────────────────────────
def clean_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("|", " ")
    text = REASON_CLAUSE_RE.sub(" ", text)
    text = COMORBIDITY_CLAUSE_RE.sub(" ", text)
    text = MEDICATION_CLAUSE_RE.sub(" ", text)
    text = ALLERGY_CLAUSE_RE.sub(" ", text)
    text = ADMIN_CLAUSE_RE.sub(" ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text


def strip_target_leakage(symptom_text: str, condition: str) -> str:
    text = clean_text(symptom_text)
    target = clean_text(condition)
    target_compact = re.sub(r"\s*\(.*?\)\s*", " ", str(condition).strip().lower())
    target_compact = clean_text(target_compact)
    for term in {target, target_compact}:
        if term and len(term) >= 4:
            text = re.sub(rf"\b{re.escape(term)}\b", " ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def distribution_report(y, out_path: Path, name: str) -> None:
    counts = pd.Series(y, dtype="string").value_counts()
    payload = {
        "name": name,
        "rows": int(len(y)),
        "classes": int(counts.shape[0]),
        "min_class_count": int(counts.min()),
        "median_class_count": float(counts.median()),
        "max_class_count": int(counts.max()),
        "max_min_ratio": round(float(counts.max() / max(1, counts.min())), 4),
        "top_20": {str(k): int(v) for k, v in counts.head(20).to_dict().items()},
        "bottom_20": {str(k): int(v) for k, v in counts.tail(20).to_dict().items()},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class XGBMulticlassWrapper:
    def __init__(self, **params):
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        self.model = XGBClassifier(**params)
        self.encoder = LabelEncoder()
        self.classes_: list = []

    def fit(self, X, y):
        y_enc = self.encoder.fit_transform(np.asarray(y, dtype=str))
        self.classes_ = [str(v) for v in self.encoder.classes_]
        class_ids = np.unique(y_enc)
        class_weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=y_enc)
        weight_map = {int(k): float(v) for k, v in zip(class_ids, class_weights)}
        sample_weight = np.array([weight_map[int(v)] for v in y_enc], dtype=np.float32)
        self.model.fit(X, y_enc, sample_weight=sample_weight)
        return self

    def predict(self, X):
        pred_ids = self.model.predict(X)
        return self.encoder.inverse_transform(pred_ids.astype(int))

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def rebalance_hybrid_smote_svd(
    X_sparse, y, *,
    majority_labels, majority_cap_fraction, minority_target,
    random_state, svd_components, smote_seed_floor=6,
):
    y_series = pd.Series(y, dtype="string")
    svd = TruncatedSVD(n_components=int(svd_components), random_state=random_state)
    X_dense = svd.fit_transform(X_sparse)

    total_rows = len(y_series)
    cap_count = max(1, int(total_rows * majority_cap_fraction))
    under_strategy = {}
    for label in majority_labels:
        n_label = int((y_series == label).sum())
        if n_label > cap_count:
            under_strategy[label] = cap_count

    X_res, y_res = X_dense, y_series
    if under_strategy:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
        X_res, y_res = rus.fit_resample(X_res, y_res)

    counts = Counter(y_res)
    seed_strategy = {str(cls): int(smote_seed_floor) for cls, n in counts.items() if int(n) < int(smote_seed_floor)}
    if seed_strategy:
        ros = RandomOverSampler(sampling_strategy=seed_strategy, random_state=random_state)
        X_res, y_res = ros.fit_resample(X_res, y_res)

    counts = Counter(y_res)
    smote_strategy = {
        str(cls): int(minority_target)
        for cls, n in counts.items()
        if int(n) >= int(smote_seed_floor) and int(n) < int(minority_target)
    }
    if smote_strategy:
        smote = SMOTE(
            sampling_strategy=smote_strategy,
            k_neighbors=max(1, int(smote_seed_floor) - 1),
            random_state=random_state,
        )
        X_res, y_res = smote.fit_resample(X_res, y_res)

    return X_res, np.asarray(y_res, dtype=object), svd


# ══════════════════════════════════════════════════════════════════════════
# 1. TEXT TRIAGE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════
def train_triage(args):
    print("\n" + "=" * 70)
    print("  TRAINING: Text Triage Classifier")
    print("=" * 70)
    t0 = time.time()

    data_dir = READY / "triage"
    out_dir = MODELS
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Clean and strip leakage
    for df in [train_df, val_df, test_df]:
        df["symptom_text"] = [
            strip_target_leakage(text, cond)
            for text, cond in zip(df["symptom_text"], df["condition"])
        ]
        df["condition"] = df["condition"].astype(str).str.strip()

    # Filter empty
    for df in [train_df, val_df, test_df]:
        mask = (df["symptom_text"] != "") & (df["condition"] != "")
        df.drop(df[~mask].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    x_train, y_train = train_df["symptom_text"].values, train_df["condition"].values
    x_val, y_val = val_df["symptom_text"].values, val_df["condition"].values
    x_test, y_test = test_df["symptom_text"].values, test_df["condition"].values

    # TF-IDF
    print("  Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    X_train_vec = tfidf.fit_transform(x_train)
    X_val_vec = tfidf.transform(x_val)
    X_test_vec = tfidf.transform(x_test)

    # Rebalance
    print("  Rebalancing with hybrid SMOTE+SVD...")
    # Detect majority labels
    train_counts = pd.Series(y_train).value_counts()
    majority_labels = train_counts[train_counts > len(y_train) * 0.05].index.tolist()
    if not majority_labels:
        majority_labels = train_counts.head(2).index.tolist()

    X_train_reb, y_train_reb, svd = rebalance_hybrid_smote_svd(
        X_train_vec, y_train,
        majority_labels=majority_labels,
        majority_cap_fraction=args.majority_cap,
        minority_target=args.minority_target,
        random_state=args.seed,
        svd_components=args.svd_components,
    )
    X_val_model = svd.transform(X_val_vec)
    X_test_model = svd.transform(X_test_vec)

    distribution_report(y_train, out_dir / "class_distribution_train_raw.json", "train_raw")
    distribution_report(y_train_reb, out_dir / "class_distribution_train_rebalanced.json", "train_rebalanced")

    # Train classifier
    print(f"  Training {args.model_type.upper()} classifier...")
    if args.model_type == "lr":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train_reb, y_train_reb)

        if args.calibrate:
            print("  Calibrating with sigmoid (Platt scaling)...")
            try:
                from sklearn.frozen import FrozenEstimator
                clf = CalibratedClassifierCV(
                    estimator=FrozenEstimator(clf), method="sigmoid", cv="prefit",
                )
            except ImportError:
                clf = CalibratedClassifierCV(base_estimator=clf, method="sigmoid", cv="prefit")
            clf.fit(X_val_model, y_val)
    else:
        if not HAS_XGB:
            print("  ⚠ XGBoost not installed, falling back to LR")
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
            clf.fit(X_train_reb, y_train_reb)
        else:
            clf = XGBMulticlassWrapper(
                objective="multi:softprob",
                n_estimators=800, learning_rate=0.05, max_depth=8,
                min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
                reg_lambda=2.0, eval_metric="mlogloss", tree_method="hist",
                random_state=args.seed, n_jobs=-1,
            )
            clf.fit(X_train_reb, y_train_reb)

    # Evaluate
    print("  Evaluating...")
    y_pred = clf.predict(X_test_model)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Test Accuracy:     {acc:.4f}            ║")
    print(f"  ║  Test Macro-F1:     {f1_macro:.4f}            ║")
    print(f"  ║  Test Weighted-F1:  {f1_weighted:.4f}            ║")
    print(f"  ╚══════════════════════════════════════╝\n")

    # Save artifacts
    joblib.dump(clf, out_dir / "text_classifier.joblib")
    joblib.dump(tfidf, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(svd, out_dir / "tfidf_svd.joblib")

    labels = [str(v) for v in getattr(clf, "classes_", [])]
    (out_dir / "text_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    metrics = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "samples": int(len(train_df) + len(val_df) + len(test_df)),
        "classes": int(len(labels)),
        "model_type": args.model_type,
        "rebalance_mode": "hybrid_smote_svd",
        "majority_labels": majority_labels,
        "majority_cap_fraction": round(float(args.majority_cap), 4),
        "minority_target": int(args.minority_target),
        "calibrated": bool(args.calibrate and args.model_type == "lr"),
        "uses_svd": True,
        "training_time_sec": round(time.time() - t0, 1),
    }
    (out_dir / "text_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"  ✅ Triage model saved to {out_dir}")
    print(f"     Time: {metrics['training_time_sec']}s")
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# 2. DIALOGUE INTENT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════
INTENT_TEMPLATES = {
    "symptom": "Thank you for sharing your symptoms clearly.",
    "treatment": "I understand your concern, and I will walk you through practical care options.",
    "diagn": "That is an important question, and we can review it step by step.",
    "risk": "Your concern is valid, and it helps to review risk factors carefully.",
    "prevent": "Prevention is a strong step, and we can focus on actions you can take today.",
    "outlook": "I know this can feel stressful, and I will keep the guidance clear and direct.",
    "cause": "Let me help you understand what might be causing this.",
    "information": "I'll provide you with clear, evidence-based information.",
    "complications": "Let me explain the possible complications and what to watch for.",
    "exams and tests": "Here's what you can expect from diagnostic testing.",
}


def template_for_intent(intent: str) -> str:
    text = intent.lower()
    for key, template in INTENT_TEMPLATES.items():
        if key in text:
            return template
    return "Thanks for sharing the details; I will provide clear guidance based on your input."


def train_dialogue(args):
    print("\n" + "=" * 70)
    print("  TRAINING: Dialogue Intent Classifier")
    print("=" * 70)
    t0 = time.time()

    data_dir = READY / "dialogue"
    out_dir = MODELS
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Clean
    for df in [train_df, val_df, test_df]:
        df["text"] = df["text"].astype(str).map(clean_text)
        df["intent"] = df["intent"].astype(str).str.strip().str.lower()

    x_train, y_train = train_df["text"].values, train_df["intent"].values
    x_val, y_val = val_df["text"].values, val_df["intent"].values
    x_test, y_test = test_df["text"].values, test_df["intent"].values

    # TF-IDF + LogisticRegression pipeline
    print("  Fitting TF-IDF + LogisticRegression...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)
    X_train_vec = tfidf.fit_transform(x_train)
    X_val_vec = tfidf.transform(x_val)
    X_test_vec = tfidf.transform(x_test)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred_test = clf.predict(X_test_vec)
    y_pred_val = clf.predict(X_val_vec)

    test_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, y_pred_test)
    val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
    val_acc = accuracy_score(y_val, y_pred_val)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Val  Accuracy:     {val_acc:.4f}            ║")
    print(f"  ║  Val  Macro-F1:     {val_f1:.4f}            ║")
    print(f"  ║  Test Accuracy:     {test_acc:.4f}            ║")
    print(f"  ║  Test Macro-F1:     {test_f1:.4f}            ║")
    print(f"  ╚══════════════════════════════════════╝\n")

    # Save
    joblib.dump(clf, out_dir / "dialogue_intent_classifier.joblib")
    joblib.dump(tfidf, out_dir / "dialogue_intent_vectorizer.joblib")

    labels = [str(v) for v in clf.classes_]
    (out_dir / "dialogue_intent_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    templates = {label: template_for_intent(label) for label in labels}
    (out_dir / "dialogue_response_templates.json").write_text(json.dumps(templates, indent=2), encoding="utf-8")

    metrics = {
        "accuracy": round(float(test_acc), 4),
        "macro_f1": round(float(test_f1), 4),
        "val_accuracy": round(float(val_acc), 4),
        "val_macro_f1": round(float(val_f1), 4),
        "samples": int(len(train_df) + len(val_df) + len(test_df)),
        "intent_classes": int(len(labels)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "training_time_sec": round(time.time() - t0, 1),
    }
    (out_dir / "dialogue_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"  ✅ Dialogue model saved to {out_dir}")
    print(f"     Time: {metrics['training_time_sec']}s")
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# 3. IMAGE CLASSIFIER (PyTorch CNN)
# ══════════════════════════════════════════════════════════════════════════
def train_image(args):
    print("\n" + "=" * 70)
    print("  TRAINING: Image Classifier (DermaMNIST)")
    print("=" * 70)
    t0 = time.time()

    data_dir = READY / "imaging"
    out_dir = MODELS
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if PyTorch is available
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    if not HAS_TORCH:
        print("  ⚠ PyTorch not installed — saving data splits only, skipping CNN training")
        # Just copy the npy files and existing model metrics
        for split in ["train", "val", "test"]:
            for suffix in ["images.npy", "labels.npy"]:
                src = data_dir / f"{split}_{suffix}"
                if src.exists():
                    import shutil
                    shutil.copy2(src, out_dir / f"dermamnist_{split}_{suffix}")
        return {"status": "skipped_no_pytorch"}

    # Load data
    train_images = np.load(data_dir / "train_images.npy")
    train_labels = np.load(data_dir / "train_labels.npy")
    val_images = np.load(data_dir / "val_images.npy")
    val_labels = np.load(data_dir / "val_labels.npy")
    test_images = np.load(data_dir / "test_images.npy")
    test_labels = np.load(data_dir / "test_labels.npy")

    print(f"  Train: {train_images.shape} | Val: {val_images.shape} | Test: {test_images.shape}")
    num_classes = int(train_labels.max()) + 1
    print(f"  Classes: {num_classes}")

    # Normalize images to [0, 1] and convert to CHW
    def prepare_images(imgs):
        if imgs.max() > 1:
            imgs = imgs.astype(np.float32) / 255.0
        # Handle different shapes
        if imgs.ndim == 4 and imgs.shape[-1] in [1, 3]:
            imgs = imgs.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        elif imgs.ndim == 3:
            imgs = imgs[:, np.newaxis, :, :]  # HW -> CHW
        return imgs

    train_images = prepare_images(train_images)
    val_images = prepare_images(val_images)
    test_images = prepare_images(test_images)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(train_images)
    y_train = torch.LongTensor(train_labels)
    X_val = torch.FloatTensor(val_images)
    y_val = torch.LongTensor(val_labels)
    X_test = torch.FloatTensor(test_images)
    y_test = torch.LongTensor(test_labels)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # CNN model
    class DermaCNN(nn.Module):
        def __init__(self, num_classes, in_channels=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    in_channels = train_images.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = DermaCNN(num_classes, in_channels=in_channels).to(device)

    # Class weights for imbalanced data
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    epochs = args.image_epochs

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "dermacnn_best.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} — loss: {train_loss/len(train_loader):.4f} — val_f1: {val_f1:.4f} — val_acc: {val_acc:.4f}")

        if patience_counter >= 15:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test evaluation with best model
    model.load_state_dict(torch.load(out_dir / "dermacnn_best.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test.to(device)
        out = model(X_test_dev)
        y_pred = out.argmax(dim=1).cpu().numpy()
        y_true = y_test.numpy()

    test_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    test_acc = accuracy_score(y_true, y_pred)
    test_f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Test Accuracy:     {test_acc:.4f}            ║")
    print(f"  ║  Test Macro-F1:     {test_f1:.4f}            ║")
    print(f"  ║  Test Weighted-F1:  {test_f1_weighted:.4f}            ║")
    print(f"  ╚══════════════════════════════════════╝\n")

    # Save label map
    label_names_path = out_dir / "image_labels.json"
    if label_names_path.exists():
        pass  # keep existing
    else:
        labels_list = [str(i) for i in range(num_classes)]
        label_names_path.write_text(json.dumps(labels_list, indent=2), encoding="utf-8")

    metrics = {
        "dataset": "dermamnist_64",
        "architecture": "dermacnn",
        "train_samples": int(len(train_labels)),
        "val_samples": int(len(val_labels)),
        "test_samples": int(len(test_labels)),
        "classes": int(num_classes),
        "best_val_macro_f1": round(float(best_val_f1), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_weighted_f1": round(float(test_f1_weighted), 4),
        "test_accuracy": round(float(test_acc), 4),
        "epochs_trained": int(epoch),
        "training_time_sec": round(time.time() - t0, 1),
    }
    (out_dir / "image_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"  ✅ Image model saved to {out_dir}")
    print(f"     Time: {metrics['training_time_sec']}s")
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline")
    parser.add_argument("--only", default="", help="Comma-separated models to train: triage,dialogue,image")
    parser.add_argument("--model-type", choices=["lr", "xgb"], default="lr", help="Triage classifier type")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate triage LR with Platt scaling")
    parser.add_argument("--majority-cap", type=float, default=0.12, help="Majority class cap fraction")
    parser.add_argument("--minority-target", type=int, default=300, help="Minority class oversample target")
    parser.add_argument("--svd-components", type=int, default=384, help="SVD components for rebalancing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image-epochs", type=int, default=50, help="Max epochs for image CNN")
    args = parser.parse_args()

    selected = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else ["triage", "dialogue", "image"]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         UNIFIED TRAINING PIPELINE — data/ready/         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Models to train: {selected}")
    print(f"  Data source: {READY}")
    print(f"  Output: {MODELS}")

    all_metrics = {}
    total_t0 = time.time()

    if "triage" in selected:
        all_metrics["triage"] = train_triage(args)

    if "dialogue" in selected:
        all_metrics["dialogue"] = train_dialogue(args)

    if "image" in selected:
        all_metrics["image"] = train_image(args)

    total_time = round(time.time() - total_t0, 1)

    # Save combined metrics
    all_metrics["total_training_time_sec"] = total_time
    (MODELS / "training_run_summary.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    for name, m in all_metrics.items():
        if isinstance(m, dict):
            f1 = m.get("macro_f1", m.get("test_macro_f1", "N/A"))
            acc = m.get("accuracy", m.get("test_accuracy", "N/A"))
            print(f"  {name:12s} — acc: {acc}, f1: {f1}")
    print(f"  Total time: {total_time}s")
    print(f"  Summary: {MODELS / 'training_run_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
