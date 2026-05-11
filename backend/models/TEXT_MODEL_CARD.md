# Text Model Card

## Overview

Runtime text triage uses a classical symptom model:

- `TF-IDF`
- optional `TruncatedSVD`
- `LogisticRegression` or `XGBoost`

Primary artifacts:

- `text_classifier.joblib`
- `tfidf_vectorizer.joblib`
- `text_labels.json`
- `tfidf_svd.joblib` (optional)

## Intended Use

- symptom-to-condition ranking
- risk-routing support when combined with safety overrides
- fallback path when the optional LLM adapter is unavailable

## Not Intended Use

- final diagnosis
- unsupervised clinical decision-making
- specialist pathways without dedicated validation

## Training Data Requirements

- symptom-only narratives
- no target leakage (`Reason:`, medication lists, allergies, administrative encounter text)
- avoid generic labels such as `Condition 275`

## Evaluation Status

Current checked-in metrics are in `text_training_metrics.json`.

### Internal (rebalanced train/test split)

- model: `SGDClassifier`
- accuracy: `0.9531`
- macro-F1: `0.8323`
- classes: `202`
- train samples: `34,896` (rebalanced from 254 raw → 200 per class)
- test samples: `8,725`
- vectorizer features: `7,350`

### External evaluation (Synthea CSV, 8 overlapping classes)

See `evaluation/text_evaluation_summary.json`:

- accuracy_top1: `0.7567`
- macro_F1_top1: `0.0968`
- top3_accuracy: `0.7604`
- ECE: `0.7251` (severe miscalibration)
- Brier score: `0.9618`

Interpretation: internal metrics are inflated by the rebalanced split. The external evaluation reveals the model is **severely miscalibrated** and performs poorly on rare classes (macro-F1 near zero). The keyword-boost layer in `text_model.py` compensates for common clinical patterns, but the base classifier needs retraining with more diverse, real-world symptom narratives.

## Runtime Controls

- safety override layer can promote emergency differentials
- bilingual normalization maps supported Amharic phrases into model-friendly English keywords
- real-time search does not change the local classifier; it augments context only

## Related Tooling

- preprocessing: `backend/scripts/preprocess_and_train.py`
- external-style evaluation: `backend/scripts/evaluate_text_model.py`
- regression suite: `backend/scripts/run_triage_regression.py`
- strict-JSON LLM dataset prep: `backend/scripts/prepare_triage_llm_dataset.py`
