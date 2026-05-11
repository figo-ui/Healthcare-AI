#!/usr/bin/env python3
"""
Preprocess ALL raw/processed datasets into a single consolidated `data/ready/` folder.

Output structure:
  data/ready/
    triage/
      train.csv, val.csv, test.csv, full.csv, label_map.json, stats.json
    dialogue/
      train.csv, val.csv, test.csv, full.csv, label_map.json, stats.json
    imaging/
      train_images.npy, val_images.npy, test_images.npy,
      train_labels.npy, val_labels.npy, test_labels.npy,
      metadata.json, label_map.json
    mimic/
      encounters.csv, conditions.csv, patients.csv, medications.csv,
      observations.csv, procedures.csv, stats.json
    fitzpatrick/
      metadata.csv, train.csv, val.csv, test.csv, label_map.json, stats.json
    grok/
      triage_supervised.csv, dialogue_reasoning.csv, stats.json
    kaggle_symptom/
      disease_symptom.csv, symptom_severity.csv, symptom_precaution.csv, stats.json
    kaggle_chatbot/
      train.csv, val.csv, stats.json
    uci/
      heart_disease.csv, heart_failure.csv, diabetes.csv, kidney_disease.csv, stats.json
    synthea/
      conditions.csv, encounters.csv, patients.csv, observations.csv,
      medications.csv, procedures.csv, stats.json
"""

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
UNIFIED = DATA / "unified"
V1 = DATA / "dataset_v1.0"
READY = DATA / "ready"

WHITESPACE_RE = re.compile(r"\s+")

# ── Helpers ───────────────────────────────────────────────────────────────
def clean_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text


def save_split(df, out_dir, label_col="condition", test_size=0.15, val_size=0.1, random_state=42):
    """Split into train/val/test and save CSVs + label_map + stats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure string labels
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    # Filter classes with fewer than 5 samples (need enough for 3 splits)
    counts = df[label_col].value_counts()
    keep_classes = counts[counts >= 5].index
    dropped = len(df) - df[label_col].isin(keep_classes).sum()
    df = df[df[label_col].isin(keep_classes)].reset_index(drop=True)
    if dropped > 0:
        print(f"  Dropped {dropped} rows from classes with <5 samples")

    # Encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df[label_col])
    label_map = {int(i): str(cls) for i, cls in enumerate(le.classes_)}

    # Stratified split
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, random_state=random_state,
        stratify=df[label_col],
    )
    relative_val = val_size / (test_size + val_size)
    # Check if temp split can be stratified
    temp_counts = temp_df[label_col].value_counts()
    can_stratify_temp = bool((temp_counts >= 2).all())
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative_val, random_state=random_state,
        stratify=temp_df[label_col] if can_stratify_temp else None,
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    df.to_csv(out_dir / "full.csv", index=False)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    stats = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "num_classes": len(le.classes_),
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "top_10": {str(k): int(v) for k, v in counts.head(10).to_dict().items()},
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  → {out_dir.relative_to(DATA)}: {len(df)} rows, {len(le.classes_)} classes")
    return stats


def copy_file(src, dst):
    """Copy a single file, creating parent dirs."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ── 1. Triage (Symptom → Condition) ──────────────────────────────────────
def preprocess_triage():
    print("\n═══ Triage Dataset ═══")
    out = READY / "triage"

    # Primary: dataset_v1.0/triage (already split, high quality)
    v1_full = V1 / "triage" / "full.csv"
    if v1_full.exists():
        df = pd.read_csv(v1_full)
        # Merge in ULTIMATE_TRIAGE_KNOWLEDGE (deduplicated)
        ult = UNIFIED / "ULTIMATE_TRIAGE_KNOWLEDGE.csv"
        if ult.exists():
            ult_df = pd.read_csv(ult)
            if {"symptom_text", "condition"}.issubset(ult_df.columns):
                combined = pd.concat([df, ult_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["symptom_text", "condition"], keep="first")
                df = combined.reset_index(drop=True)
                print(f"  Merged ULTIMATE_TRIAGE_KNOWLEDGE: {len(df)} total rows")

        # Merge in grok triage_supervised
        grok_tri = RAW / "grok" / "triage_supervised.csv"
        if grok_tri.exists():
            grok_df = pd.read_csv(grok_tri)
            if {"question", "answer"}.issubset(grok_df.columns):
                grok_renamed = grok_df.rename(columns={"question": "symptom_text", "answer": "condition"})
                grok_renamed = grok_renamed[["symptom_text", "condition"]].dropna()
                before = len(df)
                combined = pd.concat([df, grok_renamed], ignore_index=True)
                combined = combined.drop_duplicates(subset=["symptom_text", "condition"], keep="first")
                df = combined.reset_index(drop=True)
                print(f"  Merged grok triage: +{len(df) - before} rows")

        # Merge in processed expanded_symptom_condition
        esc = PROCESSED / "expanded_symptom_condition_clean_processed.csv"
        if esc.exists():
            esc_df = pd.read_csv(esc)
            if {"symptom_text", "condition"}.issubset(esc_df.columns):
                before = len(df)
                combined = pd.concat([df, esc_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["symptom_text", "condition"], keep="first")
                df = combined.reset_index(drop=True)
                print(f"  Merged expanded_symptom_condition: +{len(df) - before} rows")

        # Merge in kaggle processed
        kag = RAW / "kaggle" / "processed" / "integrated_important_plus_kaggle_processed_min5.csv"
        if kag.exists():
            kag_df = pd.read_csv(kag)
            if {"symptom_text", "condition"}.issubset(kag_df.columns):
                before = len(df)
                combined = pd.concat([df, kag_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["symptom_text", "condition"], keep="first")
                df = combined.reset_index(drop=True)
                print(f"  Merged kaggle integrated: +{len(df) - before} rows")

        # Clean
        df["symptom_text"] = df["symptom_text"].astype(str).str.strip()
        df["condition"] = df["condition"].astype(str).str.strip()
        df = df[(df["symptom_text"] != "") & (df["condition"] != "")].reset_index(drop=True)

        # Filter rare classes (<2 samples can't be stratified)
        counts = df["condition"].value_counts()
        keep = counts[counts >= 2].index
        df = df[df["condition"].isin(keep)].reset_index(drop=True)

        # Keep only needed columns
        df = df[["symptom_text", "condition"]]
        save_split(df, out, label_col="condition")
    else:
        print("  ⚠ No dataset_v1.0/triage/full.csv found — skipping")


# ── 2. Dialogue (User Text → Intent) ─────────────────────────────────────
def preprocess_dialogue():
    print("\n═══ Dialogue Dataset ═══")
    out = READY / "dialogue"

    # Primary: dataset_v1.0/dialogue
    v1_train = V1 / "dialogue" / "train.csv"
    if v1_train.exists():
        dfs = []
        for split in ["train", "val", "test"]:
            p = V1 / "dialogue" / f"{split}.csv"
            if p.exists():
                dfs.append(pd.read_csv(p))
        df = pd.concat(dfs, ignore_index=True)

        # Merge in ULTIMATE_CONVERSATIONAL_QA
        ult = UNIFIED / "ULTIMATE_CONVERSATIONAL_QA.csv"
        if ult.exists():
            ult_df = pd.read_csv(ult, nrows=500000)  # cap to avoid memory issues
            # Detect columns
            if {"short_question", "short_answer"}.issubset(ult_df.columns):
                ult_renamed = ult_df.rename(columns={"short_question": "text", "short_answer": "answer"})
                # Infer intent from tags or label if available
                if "tags" in ult_df.columns:
                    ult_renamed["intent"] = ult_df["tags"].fillna("general").astype(str).str.strip("[]'\"")
                elif "label" in ult_df.columns:
                    ult_renamed["intent"] = ult_df["label"].fillna("general").astype(str)
                else:
                    ult_renamed["intent"] = "general"
                ult_renamed = ult_renamed[["text", "intent"]].dropna()
                ult_renamed["text"] = ult_renamed["text"].map(clean_text)
                dfs.append(ult_renamed)
                print(f"  Merged ULTIMATE_CONVERSATIONAL_QA")

        # Merge in grok dialogue reasoning
        grok_dia = RAW / "grok" / "triage_dialogue_reasoning.csv"
        if grok_dia.exists():
            grok_df = pd.read_csv(grok_dia)
            if {"user_text", "intent"}.issubset(grok_df.columns):
                grok_renamed = grok_df.rename(columns={"user_text": "text"})[["text", "intent"]].dropna()
                grok_renamed["text"] = grok_renamed["text"].map(clean_text)
                dfs.append(grok_renamed)
                print(f"  Merged grok dialogue_reasoning")

        # Merge in medquad dialogue legacy
        medq = RAW / "dialogue_legacy" / "medquad_plus_kaggle_unknown_positive_general.csv"
        if medq.exists():
            medq_df = pd.read_csv(medq, nrows=200000)
            # Detect columns
            user_col = next((c for c in ["short_question", "question", "user_text", "text"] if c in medq_df.columns), None)
            intent_col = next((c for c in ["intent", "label", "tags"] if c in medq_df.columns), None)
            if user_col:
                medq_renamed = medq_df.rename(columns={user_col: "text"})
                medq_renamed["intent"] = medq_df[intent_col].fillna("general").astype(str) if intent_col else "general"
                medq_renamed = medq_renamed[["text", "intent"]].dropna()
                medq_renamed["text"] = medq_renamed["text"].map(clean_text)
                dfs.append(medq_renamed)
                print(f"  Merged medquad dialogue")

        # Merge in kaggle chatbot
        kcb = RAW / "kaggle" / "raw" / "medical-chatbot-dataset" / "train_data_chatbot.csv"
        if kcb.exists():
            kcb_df = pd.read_csv(kcb, nrows=200000)
            if {"short_question", "tags"}.issubset(kcb_df.columns):
                kcb_renamed = kcb_df.rename(columns={"short_question": "text"})
                kcb_renamed["intent"] = kcb_df["tags"].fillna("general").astype(str).str.strip("[]'\"")
                kcb_renamed = kcb_renamed[["text", "intent"]].dropna()
                kcb_renamed["text"] = kcb_renamed["text"].map(clean_text)
                dfs.append(kcb_renamed)
                print(f"  Merged kaggle chatbot")

        # Combine all
        df = pd.concat(dfs, ignore_index=True)
        df["text"] = df["text"].astype(str).str.strip()
        df["intent"] = df["intent"].astype(str).str.strip().str.lower()
        df = df[(df["text"] != "") & (df["intent"] != "")].reset_index(drop=True)
        df = df.drop_duplicates(subset=["text", "intent"], keep="first").reset_index(drop=True)

        # Filter rare intents
        counts = df["intent"].value_counts()
        keep = counts[counts >= 2].index
        df = df[df["intent"].isin(keep)].reset_index(drop=True)

        df = df[["text", "intent"]]
        save_split(df, out, label_col="intent")
    else:
        print("  ⚠ No dataset_v1.0/dialogue found — skipping")


# ── 3. Imaging (DermaMNIST) ──────────────────────────────────────────────
def preprocess_imaging():
    print("\n═══ Imaging Dataset ═══")
    out = READY / "imaging"
    out.mkdir(parents=True, exist_ok=True)

    v1_img = V1 / "imaging"
    if v1_img.exists() and (v1_img / "train_images.npy").exists():
        for split in ["train", "val", "test"]:
            imgs = np.load(v1_img / f"{split}_images.npy")
            labels = np.load(v1_img / f"{split}_labels.npy")
            np.save(out / f"{split}_images.npy", imgs)
            np.save(out / f"{split}_labels.npy", labels)
            print(f"  Copied {split}: images {imgs.shape}, labels {labels.shape}")

        # Copy metadata
        if (v1_img / "imaging_metadata.json").exists():
            copy_file(v1_img / "imaging_metadata.json", out / "metadata.json")

        # Create label_map from model labels
        model_labels = ROOT / "backend" / "models" / "image_labels.json"
        if model_labels.exists():
            copy_file(model_labels, out / "label_map.json")

        stats = {
            "source": "dermamnist_28 + improved_dermcnn",
            "train_samples": int(np.load(v1_img / "train_labels.npy").shape[0]),
            "val_samples": int(np.load(v1_img / "val_labels.npy").shape[0]),
            "test_samples": int(np.load(v1_img / "test_labels.npy").shape[0]),
            "num_classes": 7,
        }
        with open(out / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    else:
        print("  ⚠ No dataset_v1.0/imaging found — skipping")


# ── 4. MIMIC-IV Clinical Records ─────────────────────────────────────────
def preprocess_mimic():
    print("\n═══ MIMIC-IV Dataset ═══")
    out = READY / "mimic"
    out.mkdir(parents=True, exist_ok=True)

    mimic_file = RAW / "clinical" / "MIMIC_IV_Transcript.csv"
    if not mimic_file.exists():
        print("  ⚠ No MIMIC-IV transcript found — skipping")
        return

    df = pd.read_csv(mimic_file, low_memory=False)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Detect and save useful sub-tables
    col_map = {
        "encounters": [c for c in ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
                                     "admission_location", "discharge_location", "insurance", "marital_status",
                                     "race", "gender", "anchor_age"] if c in df.columns],
        "conditions": [c for c in ["subject_id", "hadm_id", "description", "drg_type", "drg_severity",
                                    "drg_mortality"] if c in df.columns],
        "medications": [c for c in ["subject_id", "hadm_id", "drug", "formulary_drug_cd", "dose_val_rx",
                                     "dose_unit_rx", "route"] if c in df.columns],
        "observations": [c for c in ["subject_id", "hadm_id", "test_name", "org_name", "ab_name",
                                      "comments", "spec_type_desc"] if c in df.columns],
        "procedures": [c for c in ["subject_id", "hadm_id", "order_type", "order_subtype",
                                    "transaction_type", "eventtype", "careunit"] if c in df.columns],
    }

    stats = {"total_rows": len(df), "total_columns": len(df.columns)}
    for name, cols in col_map.items():
        if cols:
            sub = df[cols].dropna(how="all").reset_index(drop=True)
            sub.to_csv(out / f"{name}.csv", index=False)
            stats[f"{name}_rows"] = len(sub)
            stats[f"{name}_columns"] = len(cols)
            print(f"  Saved {name}.csv: {len(sub)} rows")

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ── 5. Fitzpatrick17k Imaging Metadata ───────────────────────────────────
def preprocess_fitzpatrick():
    print("\n═══ Fitzpatrick17k Dataset ═══")
    out = READY / "fitzpatrick"

    fitz_file = PROCESSED / "fitzpatrick17k_processed_v1.csv"
    if not fitz_file.exists():
        print("  ⚠ No fitzpatrick17k_processed_v1.csv found — skipping")
        return

    df = pd.read_csv(fitz_file)
    print(f"  Loaded {len(df)} rows")

    # Keep relevant columns
    keep_cols = [c for c in ["md5hash", "fitzpatrick_scale", "fitzpatrick_centaur", "label",
                              "nine_partition_label", "three_partition_label", "local_path",
                              "download_status", "split", "domain"] if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    # Save full metadata
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metadata.csv", index=False)

    # Split by existing split column if available
    if "split" in df.columns:
        for split_name in ["train", "val", "test"]:
            split_df = df[df["split"] == split_name]
            if len(split_df) > 0:
                split_df.to_csv(out / f"{split_name}.csv", index=False)
                print(f"  Saved {split_name}.csv: {len(split_df)} rows")

    # Label map
    if "label" in df.columns:
        labels = sorted(df["label"].dropna().unique().tolist())
        label_map = {i: str(l) for i, l in enumerate(labels)}
        with open(out / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)

    stats = {
        "total_rows": len(df),
        "num_labels": int(df["label"].nunique()) if "label" in df.columns else 0,
        "downloaded_count": int((df["download_status"] == "downloaded").sum()) if "download_status" in df.columns else 0,
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ── 6. Grok Reasoning Data ───────────────────────────────────────────────
def preprocess_grok():
    print("\n═══ Grok Reasoning Dataset ═══")
    out = READY / "grok"
    out.mkdir(parents=True, exist_ok=True)

    stats = {}
    for fname in ["triage_supervised.csv", "triage_dialogue_reasoning.csv"]:
        src = RAW / "grok" / fname
        if src.exists():
            df = pd.read_csv(src)
            df.to_csv(out / fname, index=False)
            stats[fname] = {"rows": len(df), "columns": list(df.columns)}
            print(f"  Copied {fname}: {len(df)} rows")

    if stats:
        with open(out / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)


# ── 7. Kaggle Symptom Dataset ────────────────────────────────────────────
def preprocess_kaggle_symptom():
    print("\n═══ Kaggle Symptom Dataset ═══")
    out = READY / "kaggle_symptom"
    out.mkdir(parents=True, exist_ok=True)

    src_dir = RAW / "kaggle" / "raw" / "disease-symptom-description-dataset"
    if not src_dir.exists():
        print("  ⚠ No kaggle disease-symptom dataset found — skipping")
        return

    stats = {}
    for fname in ["dataset.csv", "Symptom-severity.csv", "symptom_Description.csv", "symptom_precaution.csv"]:
        src = src_dir / fname
        if src.exists():
            df = pd.read_csv(src)
            df.to_csv(out / fname, index=False)
            stats[fname] = {"rows": len(df), "columns": list(df.columns)}
            print(f"  Copied {fname}: {len(df)} rows")

    # Also copy the processed version
    proc = RAW / "kaggle" / "processed" / "kaggle_disease_symptom_processed.csv"
    if proc.exists():
        df = pd.read_csv(proc)
        df.to_csv(out / "disease_symptom_processed.csv", index=False)
        stats["disease_symptom_processed.csv"] = {"rows": len(df), "columns": list(df.columns)}
        print(f"  Copied disease_symptom_processed.csv: {len(df)} rows")

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ── 8. Kaggle Chatbot Dataset ────────────────────────────────────────────
def preprocess_kaggle_chatbot():
    print("\n═══ Kaggle Chatbot Dataset ═══")
    out = READY / "kaggle_chatbot"

    src_dir = RAW / "kaggle" / "raw" / "medical-chatbot-dataset"
    if not src_dir.exists():
        print("  ⚠ No kaggle medical-chatbot dataset found — skipping")
        return

    out.mkdir(parents=True, exist_ok=True)
    stats = {}

    for fname, out_name in [("train_data_chatbot.csv", "train.csv"), ("validation_data_chatbot.csv", "val.csv")]:
        src = src_dir / fname
        if src.exists():
            df = pd.read_csv(src)
            # Standardize columns
            rename = {}
            if "short_question" in df.columns:
                rename["short_question"] = "question"
            if "short_answer" in df.columns:
                rename["short_answer"] = "answer"
            if rename:
                df = df.rename(columns=rename)
            df.to_csv(out / out_name, index=False)
            stats[out_name] = {"rows": len(df), "columns": list(df.columns)}
            print(f"  Saved {out_name}: {len(df)} rows")

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ── 9. UCI Datasets ──────────────────────────────────────────────────────
def preprocess_uci():
    print("\n═══ UCI Datasets ═══")
    out = READY / "uci"
    out.mkdir(parents=True, exist_ok=True)

    uci_dir = RAW / "open" / "extracted"
    stats = {}

    datasets = {
        "heart_disease": "uci-heart-disease",
        "heart_failure": "uci-heart-failure-clinical-records",
        "diabetes": "uci-diabetes",
        "kidney_disease": "uci-chronic-kidney-disease",
    }

    for label, folder in datasets.items():
        folder_path = uci_dir / folder
        if not folder_path.exists():
            print(f"  ⚠ {folder} not found — skipping")
            continue

        # Find CSV files in the folder
        csvs = list(folder_path.rglob("*.csv"))
        if csvs:
            # Use the largest CSV as the main dataset
            main_csv = max(csvs, key=lambda p: p.stat().st_size)
            try:
                df = pd.read_csv(main_csv)
                df.to_csv(out / f"{label}.csv", index=False)
                stats[label] = {"rows": len(df), "columns": list(df.columns), "source_file": main_csv.name}
                print(f"  Saved {label}.csv: {len(df)} rows from {main_csv.name}")
            except Exception as e:
                print(f"  ⚠ Error reading {main_csv}: {e}")

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ── 10. Synthea COVID-19 ─────────────────────────────────────────────────
def preprocess_synthea():
    print("\n═══ Synthea COVID-19 Dataset ═══")
    out = READY / "synthea"
    out.mkdir(parents=True, exist_ok=True)

    src_dir = RAW / "open" / "extracted" / "synthea-10k-covid19-csv" / "10k_synthea_covid19_csv"
    if not src_dir.exists():
        print("  ⚠ No synthea-10k-covid19 found — skipping")
        return

    stats = {}
    for csv_file in src_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            df.to_csv(out / csv_file.name, index=False)
            stats[csv_file.name] = {"rows": len(df), "columns": list(df.columns)}
            print(f"  Copied {csv_file.name}: {len(df)} rows")
        except Exception as e:
            print(f"  ⚠ Error reading {csv_file.name}: {e}")

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ── 11. Unified Datasets (copy as-is) ────────────────────────────────────
def preprocess_unified():
    print("\n═══ Unified Datasets ═══")
    out = READY / "unified"
    out.mkdir(parents=True, exist_ok=True)

    stats = {}
    for csv_file in UNIFIED.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, nrows=10)  # just read header for stats
            full_rows = sum(1 for _ in open(csv_file, encoding="utf-8", errors="replace")) - 1
            copy_file(csv_file, out / csv_file.name)
            stats[csv_file.name] = {"rows": full_rows, "columns": list(df.columns)}
            print(f"  Copied {csv_file.name}: ~{full_rows} rows")
        except Exception as e:
            print(f"  ⚠ Error with {csv_file.name}: {e}")

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PREPROCESSING ALL DATASETS → data/ready/")
    print("=" * 70)

    READY.mkdir(parents=True, exist_ok=True)

    preprocess_triage()
    preprocess_dialogue()
    preprocess_imaging()
    preprocess_mimic()
    preprocess_fitzpatrick()
    preprocess_grok()
    preprocess_kaggle_symptom()
    preprocess_kaggle_chatbot()
    preprocess_uci()
    preprocess_synthea()
    preprocess_unified()

    # Write master summary
    print("\n═══ Writing Master Summary ═══")
    summary = {}
    for sub_dir in sorted(READY.iterdir()):
        if sub_dir.is_dir():
            stats_file = sub_dir / "stats.json"
            if stats_file.exists():
                with open(stats_file, encoding="utf-8") as f:
                    summary[sub_dir.name] = json.load(f)

    with open(READY / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All datasets preprocessed into: {READY}")
    print(f"   Manifest written to: {READY / 'manifest.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
