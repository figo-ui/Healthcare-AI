# AI Healthcare Assistant Backend

Django REST API backend for the multilingual healthcare-assistant with cloud LLM + FAISS RAG pipeline.

## Architecture

```
User → Django → Intent Router
                 ├─ Conversational → FAISS → LLM → Natural Language Response
                 ├─ Info-Seeking   → FAISS → LLM → Knowledge-Grounded Response
                 └─ Medical        → FAISS → LLM (structured triage)
                                      → Classical ML → Fusion → Risk → Clinical Report
                                      → FAISS → LLM → Natural Language Summary
```

- **LLM:** Llama-4-Maverick-17B-128E-Instruct-FP8 via GitHub Models (Azure AI Inference SDK)
- **RAG:** FAISS index over 143,775 medical documents (TF-IDF + SVD(512) dimensionality reduction)
- **Classical ML fallback** when LLM is unavailable

> This is clinical decision support only, not autonomous diagnosis.

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install "django-allauth==65.16.1"
```

Copy `.env.example` to `.env` and set:

```bash
copy .env.example .env
```

Required values in `.env`:
- `DB_PASSWORD` — PostgreSQL password
- `GITHUB_TOKEN` — GitHub PAT with `models` scope (for cloud LLM)

Run Redis locally before using async analysis:

```bash
redis-server
```

Apply migrations and start Django:

```bash
python manage.py migrate
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234
python manage.py runserver 127.0.0.1:8000 --skip-checks --noreload
```

Start the RQ worker in a second shell (optional, for async jobs):

```bash
python scripts/run_rq_worker.py
```

## Auth Model

- JWT access/refresh tokens issued into HttpOnly cookies
- Frontend requests must send `credentials: "include"`
- Refresh: `POST /api/v1/auth/refresh/` (rotates cookies)
- Logout: clears both auth cookies

## API Notes

- Health: `GET /api/v1/health/`
- Session analyze: `POST /api/v1/chat/sessions/{session_id}/analyze/`
  - Requires `consent_given=true` in request body
  - Uses `MultiPartParser` / `FormParser` (send as form-data)
- Search is only attempted when the query looks freshness-sensitive and `search_consent_given=true`
- Before external search, symptom text is redacted with Presidio + regex fallbacks

## Key Runtime Features

### RAG Pipeline (FAISS)

- 143,775 documents from 12+ medical datasets indexed with TF-IDF → SVD(512) → FAISS
- Source-quality boosting: Tier 1 (MedQuAD, Chatbot QA, Triage Knowledge) > Tier 2 (Triage Full, Symptom desc) > Tier 3 (MIMIC, Synthea) > Tier 4 (Dialogue, Imaging, Grok)
- First request builds the index (~30–60s), subsequent requests are fast (cached in memory)

### LLM Integration

- Cloud-hosted Llama-4-Maverick-17B via GitHub Models API
- Two modes:
  - **`predict_with_llm`** — structured JSON triage (conditions, probabilities, risk, red flags)
  - **`generate_rag_response`** — natural language response grounded in RAG context
- Controlled by env vars: `USE_LLM_TRIAGE=true`, `LLM_RAG_RESPONSE=true`
- Falls back to classical ML + template responses when LLM is unavailable

### Bilingual behavior

- `langdetect` + Amharic-script heuristics determine response language
- Amharic symptom phrases are normalized into English clinical keywords
- Risk labels, disclaimers, and recommendations are localized

### Async analysis

- Async requests queued through Redis/RQ
- Anonymous polling requires the returned `status_token`
- Authenticated users can poll their own cases without the token

### Search routing

- Freshness triggers: `latest`, `current`, `guideline`, `outbreak`, `2026`, `Ethiopia`, `malaria`
- PubMed first, DuckDuckGo fallback
- Search results attached under `search_context`

## Model Artifacts

Expected under `backend/models/`:

| File | Description |
|---|---|
| `triage_classifier_calibrated.joblib` | Triage classifier (97.4% acc, 371 conditions) |
| `triage_tfidf_vectorizer.joblib` | TF-IDF vectorizer (7,649 features) |
| `triage_labels.json` | 371 condition labels |
| `dialogue_intent_classifier.joblib` | Intent classifier (75.4% acc) |
| `dialogue_intent_vectorizer.joblib` | Intent TF-IDF vectorizer |
| `dialogue_intent_labels.json` | Intent label map |
| `dialogue_intent_consolidation.json` | Consolidated intent mappings |
| `dialogue_response_templates.json` | Response templates |
| `dermacnn_best.pt` | Skin lesion CNN (PyTorch, EfficientNet-B3) |
| `image_labels.json` | 7 skin lesion classes |
| `triage_llm_adapter/` | Cloud LLM preflight config |

If artifacts are missing, the backend falls back to safer demo behavior where possible.

## Training

### Text model

```bash
python scripts/step3_train_triage_v4.py
```

### Dialogue model

```bash
python scripts/step3_train_dialogue_v3.py
```

### Image model

Local retraining is CPU-heavy; prefer Kaggle for real training:

```bash
python scripts/step3_train_image_v2.py
```

## Regression + Retention

```bash
python scripts/run_triage_regression.py
python scripts/purge_phi_data.py --case-retention-days 30 --audit-retention-days 365
```
