# HealthAI — AI-Powered Medical Assistant

> **Academic Defence Project** | Advisor: Mr. Alemu Gudeta | Contact: 0919778608  
> Repository: https://github.com/figo-ui/AI-assistant.git

A full-stack, bilingual (English / Amharic) AI-powered healthcare assistant that analyzes symptoms, predicts possible conditions, scores risk levels, and guides users to the nearest healthcare facility. Powered by a cloud LLM (Llama-4-Maverick-17B) with FAISS-based Retrieval-Augmented Generation over 143,775 medical documents.

> This system is **clinical decision support only** — not a substitute for professional medical advice.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Every-Session Startup](#every-session-startup)
7. [Environment Variables](#environment-variables)
8. [Project Structure](#project-structure)
9. [AI Pipeline](#ai-pipeline)
10. [ML Models](#ml-models)
11. [API Reference](#api-reference)
12. [Admin Panel](#admin-panel)
13. [Docker Deployment](#docker-deployment)
14. [Retraining Models](#retraining-models)
15. [Dataset Overview](#dataset-overview)
16. [Security](#security)
17. [Troubleshooting](#troubleshooting)
18. [Login Credentials](#login-credentials)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           FRONTEND  (HTML / CSS / Vanilla JavaScript)        │
│  Auth · Dashboard · Chat · Symptom Wizard · Map · Admin      │
│  Served directly by Django — no build tools required         │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP + JWT (HttpOnly cookies)
┌──────────────────────────▼──────────────────────────────────┐
│          BACKEND  (Django 5.2 + Django REST Framework)       │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐   │
│  │   Auth   │  │  Chat    │  │   12-Step ML Pipeline    │   │
│  │  & JWT   │  │ Sessions │  │  (triage + image + RAG)  │   │
│  └──────────┘  └──────────┘  └─────────────────────────┘   │
│                                                              │
│  PostgreSQL · Redis cache · RQ async workers                 │
└──────────────────────────────────────────────────────────────┘
```

**Request flow:**

```
User message → Intent Router
                ├─ Conversational → FAISS → LLM → Natural language response
                ├─ Info-Seeking   → FAISS → LLM → Knowledge-grounded answer
                └─ Medical        → FAISS → LLM (structured triage JSON)
                                     → Classical ML → Fusion → Risk scoring
                                     → Clinical safety overrides
                                     → FAISS → LLM → Natural language summary
                                     → Nearby facility lookup
```

Every response flows through FAISS retrieval → LLM generation. The LLM always receives RAG context from 143K medical documents, ensuring responses are grounded in verified medical knowledge.

---

## Prerequisites — Install These First

| Tool | Version | Download Link |
|---|---|---|
| **Python** | 3.10 or 3.11 | https://www.python.org/downloads/ |
| **PostgreSQL** | 14 – 17 | https://www.postgresql.org/download/windows/ |
| **Git** | any | https://git-scm.com/download/win |

> **Redis** is installed automatically in the steps below — no separate download needed.

---

## Full Installation — Copy & Paste

### Step 1 — Clone the repository

```bash


---

### Step 2 — Create and activate the Python virtual environment

```bash
cd backend
python -m venv .venv
```

**Activate (Windows):**
```bash
.venv\Scripts\activate
```

**Activate (macOS / Linux):**
```bash
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt.

---

### Step 3 — Install all Python dependencies

```bash
pip install -r requirements.txt
pip install "django-allauth==65.16.1"
```

> The second command installs `django-allauth` which is required for authentication.

---

### Step 4 — Install Redis (Windows portable — no admin required)

Run these commands from **any terminal** (not inside the backend folder):

```bash
curl -L -o redis.zip https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip
powershell -command "Expand-Archive -Path redis.zip -DestinationPath C:\Redis -Force"
del redis.zip
```

---

### Step 5 — Set up the environment file

```bash
copy .env.example .env
```

Open `backend\.env` in any text editor and set:

```
DB_PASSWORD=your-postgres-password-here
GITHUB_TOKEN=your-github-personal-access-token
```

- **DB_PASSWORD** — your PostgreSQL password
- **GITHUB_TOKEN** — a GitHub PAT with `models` scope from https://github.com/settings/tokens (required for the cloud LLM)

Everything else is pre-configured and ready to use.

---

### Step 6 — Create the PostgreSQL database

Open **pgAdmin** (installed with PostgreSQL) and run this in the Query Tool:

```sql
CREATE DATABASE "chat-bot";
```

Or use psql from the terminal:

```bash
psql -U postgres -c "CREATE DATABASE \"chat-bot\";"
```

---

### Step 7 — Start PostgreSQL

**Windows — open Services:**
```
Win + R  →  type: services.msc  →  Enter
Find "postgresql-x64-17"  →  Right-click  →  Start
```

Or from an **Administrator** terminal:
```bash
net start postgresql-x64-17
```

---

### Step 8 — Start Redis (open a new terminal, keep it running)

```bash
C:\Redis\redis-server.exe C:\Redis\redis.windows.conf
```

> Keep this terminal open. Redis must be running while the app is running.

---

### Step 9 — Run database migrations

```bash
cd backend
.venv\Scripts\activate
python manage.py migrate
```

---

### Step 10 — Create the admin user

```bash
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234
```

---

### Step 11 — Start the server

```bash
python manage.py runserver 127.0.0.1:8000 --skip-checks --noreload
```

---

### Step 12 — Open the app

Open your browser and go to:

```
http://127.0.0.1:8000
```

Login with:
- **Email:** `admin@example.com`
- **Password:** `Admin1234`

---

## All Commands in One Block (Windows)

```bash
# 1. Clone
git clone https://github.com/figo-ui/AI-assistant.git
cd AI-assistant\backend

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install "django-allauth==65.16.1"

# 4. Environment file
copy .env.example .env
# → Open .env and set DB_PASSWORD and GITHUB_TOKEN

# 5. Install Redis (run once)
curl -L -o redis.zip https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip
powershell -command "Expand-Archive -Path redis.zip -DestinationPath C:\Redis -Force"
del redis.zip

# 6. Create database (in pgAdmin or psql)
# CREATE DATABASE "chat-bot";

# 7. Start PostgreSQL (in Services or as Admin)
# net start postgresql-x64-17

# 8. Start Redis — open a NEW terminal and run:
# C:\Redis\redis-server.exe C:\Redis\redis.windows.conf

# 9. Migrate database
python manage.py migrate

# 10. Create admin user
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234

# 11. Start the server
python manage.py runserver 127.0.0.1:8000 --skip-checks --noreload

# 12. Open browser → http://127.0.0.1:8000
```

---

## Every Time You Start the Project

You need to run these 3 things each session:

**Terminal 1 — Redis:**
```bash
C:\Redis\redis-server.exe C:\Redis\redis.windows.conf
```

**Terminal 2 — Django server:**
```bash
cd AI-assistant\backend
.venv\Scripts\activate
python manage.py runserver 127.0.0.1:8000 --skip-checks --noreload
```

**PostgreSQL** must be running (start via Services or pgAdmin).

---

## Project Structure

```
AI-assistant/
│
├── UI/                              Frontend (HTML / CSS / JavaScript)
│   ├── index.html                   Main application
│   ├── auth.html                    Login & Register
│   ├── admin.html                   Admin panel
│   └── src/
│       ├── css/                     Stylesheets
│       └── js/
│           ├── api.js               API client (JWT auth, all endpoints)
│           ├── app.js               App router and navigation
│           ├── auth.js              Login / register logic
│           └── features/
│               ├── chat.js          AI chat with analysis cards
│               ├── symptoms.js      3-step symptom wizard
│               ├── facilities.js    Map + clinic finder (Geoapify)
│               ├── dashboard.js     Dashboard with live data
│               └── emergency.js     Emergency contacts modal
│
├── backend/                         Django REST API
│   ├── guidance/
│   │   ├── models.py                Database models
│   │   ├── views.py                 All API endpoints
│   │   ├── serializers.py           Request/response validation
│   │   ├── urls.py                  URL routing
│   │   └── services/
│   │       ├── pipeline.py          12-step ML inference pipeline
│   │       ├── llm_triage.py        Cloud LLM (Llama-4) — structured triage + RAG response
│   │       ├── text_model.py        Classical symptom → condition classifier
│   │       ├── image_model.py       Skin lesion CNN inference
│   │       ├── fusion.py            Text + image prediction fusion
│   │       ├── risk.py              Risk scoring engine
│   │       ├── clinical_safety.py   Emergency pattern overrides
│   │       ├── rag.py               FAISS-based RAG (143K docs, SVD-reduced index)
│   │       ├── facilities.py        Geoapify + Google Places lookup
│   │       ├── language_support.py  English / Amharic bilingual
│   │       ├── pii_redaction.py     PHI redaction (Presidio)
│   │       └── search_router.py     PubMed / DuckDuckGo search
│   ├── models/                      Trained ML model files
│   │   ├── triage_classifier_calibrated.joblib    Triage model (97.4% acc)
│   │   ├── triage_tfidf_vectorizer.joblib          Triage vectorizer
│   │   ├── triage_labels.json                      371 condition labels
│   │   ├── dialogue_classifier_calibrated.joblib   Intent model (75.4% acc)
│   │   ├── dialogue_tfidf_vectorizer.joblib         Intent vectorizer
│   │   ├── dialogue_response_templates.json         Response templates
│   │   └── dermacnn_best.pt                         Skin CNN (PyTorch)
│   ├── scripts/                     Training & utility scripts
│   ├── .env.example                 Environment template
│   ├── .env                         Your local config (not committed)
│   └── requirements.txt
│
├── data/
│   └── ready/                       Pre-processed datasets (143K+ documents)
│       ├── triage/                  74,631 symptom-condition pairs
│       ├── dialogue/                109,142 intent-labelled turns
│       ├── unified/                 133K QA pairs + 45K triage knowledge
│       ├── mimic/                   MIMIC-IV clinical records
│       ├── synthea/                 Synthea synthetic EHR data
│       ├── kaggle_symptom/          Disease descriptions & precautions
│       ├── kaggle_chatbot/          59K medical Q&A pairs
│       ├── fitzpatrick/             Dermatology labels
│       ├── grok/                    AI-generated triage reasoning
│       └── uci/                     Heart failure clinical records
│
├── README.md                        This file
├── USER_GUIDE.md                    How to use the app (for users)
├── PROJECT_DEFENCE_BRIEF.md         Technical defence document
└── docker-compose.yml               Docker setup (optional)
```

---

## AI Pipeline — How It Works

### 1. Intent Classification

User messages are classified by a trained dialogue intent classifier (75.4% accuracy, 395 intent classes) into one of: **Greeting**, **Medical**, **Information-Seeking**, **Emotional**, **Follow-Up**, or **Small Talk**.

### 2. FAISS Retrieval (RAG)

Every query is searched against a **FAISS index** of 143,775 medical documents using TF-IDF → SVD(512) dimensionality reduction. The top-k hits are ranked with source-quality boosting:

| Source Tier | Datasets | Boost |
|---|---|---|
| **Tier 1** — Authoritative | MedQuAD (133K QA), Chatbot QA (59K), Triage Knowledge (45K) | +0.08 – +0.10 |
| **Tier 2** — Supplementary | Triage Full (74K), Symptom descriptions, Disease mappings | +0.05 |
| **Tier 3** — Clinical EHR | MIMIC-IV, Synthea (conditions, meds, observations) | +0.02 |
| **Tier 4** — Auxiliary | Dialogue labels, Imaging labels, Grok reasoning, Fitzpatrick | 0.00 |

### 3. LLM Generation (Llama-4-Maverick-17B)

Retrieved documents are injected into the LLM prompt as context. The LLM then:

- **Conversational path** — Generates a natural, empathetic response grounded in medical knowledge
- **Information-seeking path** — Produces a detailed, knowledge-grounded answer
- **Medical path** — Outputs structured JSON triage predictions (condition, probability, urgency, red flags) AND a final natural language summary

### 4. Classical ML Fallback

When the LLM is unavailable, the system falls back to:
- **Triage classifier** (97.4% accuracy, 371 conditions) for condition prediction
- **Template-based response** composition using RAG hits

### 5. Full Medical Pipeline

For symptom analysis, the 12-step pipeline runs:

1. Language detection & translation
2. PII/PHI redaction
3. FAISS RAG retrieval
4. LLM structured triage (with RAG context)
5. Classical ML triage (fallback/ensemble)
6. Image model inference (if image uploaded)
7. Text + image fusion
8. Risk level computation
9. Clinical safety overrides (emergency patterns)
10. Clinical report generation
11. LLM natural language summary (with RAG context)
12. Nearby facility lookup & localization

---

## API Endpoints

Base URL: `http://127.0.0.1:8000/api/v1/`

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health/` | None | Server health check |
| POST | `/auth/register/` | None | Create account |
| POST | `/auth/login/` | None | Login → JWT tokens |
| POST | `/auth/refresh/` | Cookie | Rotate JWT tokens |
| POST | `/auth/logout/` | JWT | Logout |
| GET | `/auth/social/providers/` | None | Configured OAuth providers |
| POST | `/auth/social/login/` | None | Social login (Google etc.) |
| POST | `/auth/verify-email/` | None | Verify email address |
| POST | `/auth/password-reset/` | None | Request password reset |
| POST | `/auth/password-reset/confirm/` | None | Confirm password reset |
| GET/PATCH | `/profile/` | JWT | View / update profile |
| GET/POST | `/chat/sessions/` | JWT | List / create sessions |
| GET | `/chat/sessions/{id}/messages/` | JWT | Session messages |
| POST | `/chat/sessions/{id}/analyze/` | JWT | Run symptom analysis |
| GET | `/chat/history/` | JWT | Full chat history |
| GET | `/chat/export/` | JWT | Export chat as CSV/JSON |
| GET | `/quick-prompts/` | None | Suggested prompts |
| GET | `/location/nearby/` | None | Find nearby facilities |
| GET | `/location/emergency/` | None | Emergency contacts |
| GET | `/location/directions/` | None | Get directions |
| GET | `/admin/analytics/` | Staff | Platform statistics |
| GET | `/admin/users/` | Staff | User management |
| GET | `/admin/model-metrics/` | Staff | ML model performance |
| POST | `/admin/retrain/` | Staff | Trigger model retrain |
| GET | `/admin/audit-log/` | Staff | Audit log |
| GET | `/admin/config/` | Staff | System configuration |
| GET | `/admin/dialogue-templates/` | Staff | Response templates |

---

## Trained Models — Already Included

No retraining needed to run the demo. All model files are in `backend/models/`.

| Model | File | Performance |
|---|---|---|
| Triage classifier | `triage_classifier_calibrated.joblib` | 97.4% accuracy, 371 conditions |
| Triage vectorizer | `triage_tfidf_vectorizer.joblib` | 7,649 features |
| Dialogue intent | `dialogue_classifier_calibrated.joblib` | 75.4% accuracy, 395 intents |
| Skin lesion CNN | `dermacnn_best.pt` | 69% accuracy, 7 classes |
| Cloud LLM | GitHub Models API | Llama-4-Maverick-17B-128E-Instruct-FP8 |

---

## Retraining Models (Optional)

Only needed if you want to update the models with new data.

```bash
cd backend
.venv\Scripts\activate

# Retrain triage text classifier (~2 min on CPU)
python scripts/step3_train_triage_v4.py

# Retrain dialogue intent classifier
python scripts/step3_train_dialogue_v3.py

# Retrain image model (slow on CPU — use Kaggle GPU instead)
python scripts/step3_train_image_v2.py

# Run regression tests after retraining
python scripts/run_triage_regression.py
```

---

## Dataset Utilization

Datasets serve **dual purposes**: training ML models offline AND populating the FAISS RAG index at runtime.

| Dataset | Rows | Trained Model | RAG Source |
|---|---|---|---|
| Triage | 74,631 | Triage classifier (97.4% acc) | ✅ `triage_full` |
| Dialogue | 109,142 | Intent classifier (75.4% acc) | ✅ `dialogue` |
| MedQuAD | 133,985 | — | ✅ `medquad` (Tier 1) |
| Triage Knowledge | 45,780 | — | ✅ `triage_knowledge` (Tier 1) |
| Kaggle Chatbot | 59,504 | — | ✅ `chatbot_qa` (Tier 1) |
| Kaggle Symptom | 4,920 + 304 | Keyword boosting | ✅ `symptom_desc`, `disease_symptom` |
| MIMIC-IV | 232,158 | — | ✅ `mimic_*` (Tier 3) |
| Synthea | 1.6M+ | — | ✅ `synthea_*` (Tier 3) |
| Fitzpatrick | 3,815 | Image model training | ✅ `fitzpatrick` |
| Grok | 460 | — | ✅ `grok_*` |
| UCI Heart | 299 | — | ✅ `uci_heart` |
| Imaging Labels | 16,577 | Image model training | ✅ `imaging_labels` |

**Total RAG documents:** 143,775 (FAISS-indexed with SVD(512) dimensionality reduction)

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'allauth'`**
```bash
pip install "django-allauth==65.16.1"
```

**`OperationalError: connection refused` (PostgreSQL)**
- Start PostgreSQL via Services (`services.msc`) or pgAdmin
- Check `DB_PASSWORD` in `backend\.env` matches your PostgreSQL password

**`Redis is not reachable` warning**
- Start Redis: `C:\Redis\redis-server.exe C:\Redis\redis.windows.conf`
- The app works without Redis but async jobs and caching are disabled

**First request is slow (30–60 seconds)**
- Normal — the FAISS RAG index (143K docs) builds lazily on the first request
- Subsequent requests are fast (index is cached in memory)

**LLM not responding / `GITHUB_TOKEN` error**
- Ensure `GITHUB_TOKEN` is set in `backend\.env` with a valid GitHub PAT
- Get a token at https://github.com/settings/tokens (needs `models` scope)
- The system falls back to classical ML + template responses when LLM is unavailable

**`X has N features, but model expects M features`**
- The triage model and vectorizer are from different training runs. Fix:
```bash
python scripts/step3_train_triage_v4.py
```

**Login returns `Invalid credentials`**
- Reset the admin password:
```bash
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234 --reset-password
```

**Port 8000 already in use**
```bash
# Use a different port
python manage.py runserver 127.0.0.1:8080 --skip-checks --noreload
# Then open http://127.0.0.1:8080
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 5.2, Django REST Framework 3.16 |
| Auth | JWT (SimpleJWT) + HttpOnly cookies, django-allauth |
| LLM | Llama-4-Maverick-17B via GitHub Models (Azure AI Inference SDK) |
| RAG | FAISS + TF-IDF + TruncatedSVD(512), 143K documents |
| ML — Text | scikit-learn (TF-IDF + LinearSVC, calibrated) |
| ML — Image | PyTorch (CNN, EfficientNet-B3 backbone) |
| ML — Dialogue | scikit-learn (TF-IDF + SGDClassifier) |
| PII Redaction | Microsoft Presidio |
| Database | PostgreSQL 17 |
| Cache / Queue | Redis + django-rq |
| Maps | Leaflet.js + Geoapify Places API (OpenStreetMap) |
| Frontend | Vanilla HTML / CSS / JavaScript (no build step) |

---

## Login Credentials

| Role | Email | Password | Access |
|---|---|---|---|
| Admin | `admin@example.com` | `Admin1234` | Full admin panel + all features |
| User | Register on the login page | Your choice | All user features |

Admin panel: `http://127.0.0.1:8000/admin.html`

---

*This system is clinical decision support only — not a substitute for professional medical advice.*
