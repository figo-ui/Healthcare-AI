# HealthAI — Complete Installation Guide

Every command needed to go from zero to a running system, in order.

---

## Prerequisites — Install These First

Before running any commands, install these tools manually:

| Tool | Download Link | Notes |
|---|---|---|
| **Python 3.10 or 3.11** | https://www.python.org/downloads/ | ✅ Check "Add to PATH" during install |
| **PostgreSQL 14–17** | https://www.postgresql.org/download/windows/ | Remember the password you set |
| **Git** | https://git-scm.com/download/win | Use default options |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/figo-ui/AI-assistant.git
cd AI-assistant
```

---

## Step 2 — Install Redis (Windows Portable)

Redis has no official Windows installer. Download and extract the portable version:

```bash
# Download Redis portable zip
curl -L -o redis.zip https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip

# Extract to C:\Redis
powershell -Command "Expand-Archive -Path redis.zip -DestinationPath C:\Redis -Force"

# Clean up
del redis.zip
```

> **Or manually:** Download from https://github.com/microsoftarchive/redis/releases and extract to `C:\Redis`

---

## Step 3 — Create the PostgreSQL Database

Open **pgAdmin** (installed with PostgreSQL) or run in a terminal:

```bash
# Connect to PostgreSQL and create the database
psql -U postgres -c "CREATE DATABASE \"chat-bot\";"
```

If `psql` is not in your PATH, find it at:
```
C:\Program Files\PostgreSQL\17\bin\psql.exe
```

---

## Step 4 — Set Up Python Virtual Environment

```bash
# Navigate to the backend folder
cd backend

# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (macOS / Linux)
# source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)` at the start.

---

## Step 5 — Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install django-allauth (required — may not install automatically)
pip install "django-allauth==65.16.1"
```

> This step takes 3–10 minutes depending on your internet speed. It installs Django, scikit-learn, PyTorch, TensorFlow, and all other dependencies.

---

## Step 6 — Configure Environment Variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Now open `backend/.env` in any text editor and set these two values:

```env
DB_PASSWORD=your-postgres-password-here
DJANGO_SECRET_KEY=any-long-random-string-at-least-50-characters
```

Everything else is pre-configured and works out of the box.

---

## Step 7 — Start PostgreSQL

**Windows — Option A (pgAdmin):**
Open pgAdmin — it starts the PostgreSQL server automatically.

**Windows — Option B (Services):**
```bash
# Run as Administrator
net start postgresql-x64-17
```

**Windows — Option C (PowerShell as Admin):**
```powershell
Start-Service postgresql-x64-17
```

---

## Step 8 — Start Redis

Open a **new terminal window** and run:

```bash
C:\Redis\redis-server.exe C:\Redis\redis.windows.conf
```

Keep this terminal open. You should see:
```
[*] The server is now ready to accept connections on port 6379
```

---

## Step 9 — Run Database Migrations

Back in the `backend/` folder with the venv active:

```bash
python manage.py migrate
```

Expected output ends with:
```
Running migrations:
  Applying guidance.0001_initial... OK
  ...
```

---

## Step 10 — Create the Admin User

```bash
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234
```

Expected output:
```
Created admin user username='admin' email='admin@example.com'.
```

---

## Step 11 — Start the Django Server

```bash
python manage.py runserver 127.0.0.1:8000 --skip-checks --noreload
```

Expected output:
```
Django version 5.2.13, using settings 'healthcare_ai.settings'
Starting development server at http://127.0.0.1:8000/
```

---

## Step 12 — Open the App

Open your browser and go to:

```
http://127.0.0.1:8000
```

Login with:
- **Email:** `admin@example.com`
- **Password:** `Admin1234`

---

## All Commands in One Block (Copy-Paste)

Run these in order after installing Python, PostgreSQL, and Git:

```bash
# 1. Clone
git clone https://github.com/figo-ui/AI-assistant.git
cd AI-assistant

# 2. Redis (run once)
curl -L -o redis.zip https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip
powershell -Command "Expand-Archive -Path redis.zip -DestinationPath C:\Redis -Force"
del redis.zip

# 3. Create database (PostgreSQL must be running)
psql -U postgres -c "CREATE DATABASE \"chat-bot\";"

# 4. Python setup
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install "django-allauth==65.16.1"

# 5. Configure
copy .env.example .env
# ↑ Then open .env and set DB_PASSWORD and DJANGO_SECRET_KEY

# 6. Initialize database
python manage.py migrate
python manage.py seed_admin --username admin --email admin@example.com --password Admin1234

# 7. Start Redis (in a separate terminal)
# C:\Redis\redis-server.exe C:\Redis\redis.windows.conf

# 8. Start server
python manage.py runserver 127.0.0.1:8000 --skip-checks --noreload
```

---

## Every Time You Restart Your Computer

You need to restart these services after each reboot:

### Terminal 1 — Redis
```bash
C:\Redis\redis-server.exe C:\Redis\redis.windows.conf
```

### Terminal 2 — Django Server
```bash
cd AI-assistant\backend
.venv\Scripts\activate
python manage.py runserver 127.0.0.1:8000 --skip-checks --noreload
```

### Terminal 3 — RQ Worker (optional, for async analysis)
```bash
cd AI-assistant\backend
.venv\Scripts\activate
python scripts/run_rq_worker.py
```

> PostgreSQL starts automatically on Windows if you installed it as a service. If not, start it via pgAdmin or `net start postgresql-x64-17` (as Administrator).

---

## Verify Everything is Working

Run this health check after starting the server:

```bash
curl http://127.0.0.1:8000/api/v1/health/
```

Expected response:
```json
{"status": "ok"}
```

Test login:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/login/ ^
  -H "Content-Type: application/json" ^
  -d "{\"identifier\":\"admin@example.com\",\"password\":\"Admin1234\"}"
```

Expected: a JSON response containing `"tokens": {"access": "...", "refresh": "..."}`

---

## Optional — Retrain the ML Models

The trained models are already included. Only run this if you want to update them:

```bash
cd AI-assistant\backend
.venv\Scripts\activate

# Retrain triage text classifier (~2 minutes on CPU)
python scripts/step3_train_triage_v4.py

# Retrain dialogue intent classifier
python scripts/step3_train_dialogue_v3.py

# Run regression tests to verify model quality
python scripts/run_triage_regression.py
```

---

## Optional — Configure Social Login

To enable Google / GitHub / Microsoft / Facebook login:

```bash
# 1. Add your OAuth credentials to .env
#    SOCIAL_GOOGLE_CLIENT_ID=your-client-id
#    SOCIAL_GOOGLE_SECRET=your-secret

# 2. Register them in the database
python manage.py seed_social_apps
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'allauth'`
```bash
pip install "django-allauth==65.16.1"
```

### `psycopg2.OperationalError: Connection refused`
PostgreSQL is not running. Start it:
```bash
# As Administrator
net start postgresql-x64-17
```
Or open pgAdmin — it starts the server automatically.

### `Redis is not reachable` warning
Start Redis in a separate terminal:
```bash
C:\Redis\redis-server.exe C:\Redis\redis.windows.conf
```
The app still works without Redis but async jobs and caching are disabled.

### First request is slow (5–10 seconds)
Normal — the ML models load on the first request. Subsequent requests are fast.

### `X has N features, but model expects M features`
The triage model and vectorizer are from different training runs. Retrain:
```bash
python scripts/step3_train_triage_v4.py
```

### `Permission denied: '.'` in server logs
The `TEXT_SVD_PATH` is set to an empty string which resolves to `.`. This is harmless — the SVD is not used. Confirm `TEXT_SVD_PATH=` (empty) in your `.env`.

### Port 8000 already in use
```bash
# Find and kill the process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| Disk | 5 GB free | 10 GB free |
| CPU | Any dual-core | Quad-core |
| OS | Windows 10 | Windows 10/11 |
| Python | 3.10 | 3.10 or 3.11 |
| PostgreSQL | 14 | 17 |

---

*HealthAI — AI Healthcare Assistant | Academic Project 2026*
