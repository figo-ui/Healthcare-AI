# HealthAI — User Guide

> A complete walkthrough of every feature in the system, written for end users and evaluators.

---

## Table of Contents

1. [Getting Started — Create an Account](#1-getting-started--create-an-account)
2. [Dashboard Overview](#2-dashboard-overview)
3. [AI Health Assistant (Chat)](#3-ai-health-assistant-chat)
4. [Symptom Analysis Wizard](#4-symptom-analysis-wizard)
5. [Clinic Finder — Map](#5-clinic-finder--map)
6. [Emergency Button](#6-emergency-button)
7. [User Profile](#7-user-profile)
8. [Admin Panel](#8-admin-panel-staff-only)
9. [Language Support — Amharic](#9-language-support--amharic)
10. [Security Features](#10-security-features)

---

## 1. Getting Started — Create an Account

### Opening the App

Open your browser and go to:

```
http://127.0.0.1:8000
```

You will be redirected to the login page automatically.

---

### Registering a New Account

1. On the login page, click **"Create account"** at the bottom
2. Fill in:
   - **Full Name** — your first and last name
   - **Email Address** — used to log in
   - **Password** — minimum 8 characters (the strength meter shows Weak / Fair / Good / Strong)
3. Click **Create Account**
4. You are logged in immediately and redirected to the Dashboard

> A verification email is sent to your address. The app works without verifying, but verifying unlocks email alerts for high-risk cases.

---

### Logging In

1. Enter your **email** and **password**
2. Click **Sign In**
3. After 5 wrong attempts, your account is locked for 15 minutes (brute-force protection)

---

### Social Login (if configured)

The login page shows buttons for Google, Apple, Microsoft, GitHub, and Facebook. These work when the administrator has configured OAuth keys. If a provider is not configured, clicking it shows a message explaining this.

---

### Forgot Password

Click **"Forgot password?"** on the login page → enter your email → a reset link is sent.

---

## 2. Dashboard Overview

After logging in you land on the **Dashboard**. It shows:

| Tile | What it shows |
|---|---|
| **Welcome** | Greeting with your name and time of day |
| **Heart Rate** | 72 bpm (demo value — connects to wearable data if integrated) |
| **SpO2** | Blood oxygen saturation percentage |
| **Blood Pressure** | Systolic / diastolic reading |
| **Health Trends** | Line chart — switch between Week / Month / Year view |
| **Alerts** | Your 3 most recent chat sessions shown as activity items |
| **Daily Steps** | Step count with weekly trend |
| **Sleep Score** | Sleep quality score out of 100 |
| **Calories** | Daily calorie tracking |

The **Alerts** section is live — it loads your actual recent conversations from the backend and shows how long ago each one happened.

---

### Sidebar Navigation

The left sidebar has four main sections:

| Icon | Section | What it does |
|---|---|---|
| 🏠 Dashboard | Dashboard | Health overview |
| 💬 AI Assistant | Chat | Talk to the AI |
| 🛡️ Symptom Check | Symptoms | Step-by-step analysis wizard |
| 📍 Clinic Finder | Facilities | Map of nearby healthcare |

At the bottom of the sidebar:
- **Emergency button** (red, pulsing) — shows emergency contacts immediately
- **Your name and email** — with a logout button (arrow icon)

---

## 3. AI Health Assistant (Chat)

Click **AI Assistant** in the sidebar.

### Starting a Conversation

The chat opens with a greeting from HealthAI. You can:
- Click one of the **quick reply buttons** (Check my symptoms, Review medications, Health trends, Dietary advice)
- Or type anything in the text box at the bottom

### Typing a Message

1. Click the text area at the bottom: *"Describe your symptoms or ask a health question..."*
2. Type your message — in **English or Amharic**
3. Press **Enter** or click the **Send** button (→)
4. The AI responds within a few seconds

**Example messages you can try:**
```
I have a headache and fever for 2 days
I feel chest pain and shortness of breath
What is malaria?
የደረት ህመም አለኝ (Amharic: I have chest pain)
```

### What the AI Returns

For symptom descriptions, the response includes:

- **Probable Conditions** — a ranked list with percentage probabilities
  - e.g. *"Viral pharyngitis (34.2%), Common Cold (28.1%), Influenza (15.7%)"*
- **Risk Level** — Low / Medium / High with a visual meter bar
- **Risk Score** — a number from 0.0 to 1.0
- **Recommendation** — what to do next (monitor at home / see a doctor / go to emergency)
- **Red Flags** — warning signs that need immediate attention (shown in red)
- **Prevention Advice** — steps to reduce risk

### Conversation History

The left panel inside the chat shows all your previous conversations. Click any session to reload it and continue where you left off.

Click the **+** button to start a fresh conversation.

### Attaching an Image

Click the **paperclip** (📎) button to attach a photo of a skin condition. The AI will analyze it alongside your text description and combine both results.

---

## 4. Symptom Analysis Wizard

Click **Symptom Check** in the sidebar for a more structured, step-by-step analysis.

### Step 1 — Describe Symptoms

- Type your symptoms in the text area
- Or click the **quick chips** (Headache, Chest pain, Fatigue, Dizziness, Nausea) to add them automatically
- Click **Continue Analysis →**

### Step 2 — Analysis Parameters

Provide additional context to improve accuracy:

| Field | Options |
|---|---|
| **Duration** | Less than 24 hours / 1–3 days / 4–7 days / More than a week |
| **Severity** | Slider from 1 (Mild) to 10 (Severe) |
| **Pre-existing Conditions** | None / Hypertension / Diabetes / Asthma / Heart condition |

Click **Generate Report**

### Step 3 — Results

The report shows:
- A **risk meter** bar (green = Low, yellow = Medium, red = High)
- **Risk level and score**
- **Probable conditions** with probabilities
- **Recommendation** text
- **Red flags** (if any) in a red warning box
- **Prevention advice** list
- A mandatory disclaimer: *"If symptoms worsen or you experience severe pain, seek emergency care immediately"*

Two action buttons:
- **New Analysis** — start over
- **Find Nearby Clinic** — jumps to the Clinic Finder map

---

## 5. Clinic Finder — Map

Click **Clinic Finder** in the sidebar.

### Automatic Location Detection

When the page loads, the browser asks for your location permission:
- Click **Allow** → the map zooms to your position and shows nearby facilities automatically
- Click **Block** → a message appears: *"Type your city or area name in the search box and press Enter"*

### Searching by City Name

If GPS is blocked or unavailable:
1. Click the search box at the top left
2. Type your city or area — e.g. `Haramaya`, `Dire Dawa`, `Addis Ababa`, `Harar`
3. Press **Enter**
4. The map geocodes your location and shows nearby facilities

### Filter by Facility Type

Click the filter chips to narrow results:

| Chip | Shows |
|---|---|
| **All** | All healthcare facilities |
| **Emergency** | Emergency hospitals only |
| **Dental** | Dental clinics |
| **Pharmacy** | Pharmacies |
| **Pediatric** | Children's clinics |

### Reading the Results

Each facility card shows:
- **Name** of the facility
- **Distance** in kilometers
- **Phone number** (if available)
- **Source** — OpenStreetMap (real data) or Registry (local database)

Click a card → the map zooms to that facility and opens a popup with:
- Full address
- Phone number
- **Get Directions** link → opens Google Maps with directions from your location

### Re-detect Location

Click the **📍 locate button** on the map (bottom right) to re-trigger GPS detection at any time.

### Search Radius

The default search radius is **15 km** — wide enough to find facilities in rural areas. For example, searching from Haramaya finds hospitals in Harar (~16 km away).

---

## 6. Emergency Button

The red **Emergency** button in the sidebar is always visible.

Click it → a modal appears with:
- Local emergency phone numbers for your country
- For Ethiopia: **Ambulance 907**, **Police 991**, **Fire 939**
- Tap any number to call directly from your phone

The system detects your country from the browser and shows the correct numbers. If detection fails, it shows international defaults (112, 911).

---

## 7. User Profile

Click your **name** at the bottom of the sidebar, or navigate to Profile.

### What You Can Update

| Field | Description |
|---|---|
| First / Last Name | Display name |
| Email | Login email |
| Phone Number | Contact number |
| Age | Used to adjust risk scoring (age ≥ 65 increases vulnerability score) |
| Gender | Optional |
| Address | Optional |
| Preferred Language | English or Amharic |
| Emergency Contact Name | Person to notify on high-risk cases |
| Emergency Contact Phone | Their phone number |
| Medical History | Pre-existing conditions (used to adjust risk scoring) |

### Exporting Your Data

From the profile page you can export:
- Your full profile as JSON
- Your complete chat history as CSV or JSON

---

## 8. Admin Panel (Staff Only)

Go to `http://127.0.0.1:8000/admin.html`

Login with staff credentials:
- Email: `admin@example.com`
- Password: `Admin1234`

Non-staff users are redirected to the main app automatically.

### Overview Tab

Shows live platform statistics:
- Total consultations
- ML model accuracy
- Average response time
- Active users
- Critical alerts count
- System uptime

### Health Alerts Tab

Lists all high-risk cases flagged by the system — shows the risk level, symptom summary, and timestamp.

### AI Models Tab

Shows current model performance metrics:
- Model name and version
- Accuracy and F1 score
- Training date and dataset size

**Retraining Console** — click **Start Retraining Cycle** to trigger a model retrain via the backend API.

### Users Tab

Table of all registered users showing:
- ID, name, email
- Role (User / Admin)
- Status (Active / Disabled)
- Last login date

### System Health Tab

Shows the status of all infrastructure services:
- API Server
- Database (PostgreSQL)
- ML Pipeline
- Redis Cache

### Audit Log Tab

Every admin action is logged here:
- Action type (login, create, update, delete, emergency_flagged)
- Description
- Timestamp

---

## 9. Language Support — Amharic

The system automatically detects whether you are writing in English or Amharic.

### How it works

- If your message contains **Ethiopic script** (ሀ–ፐ range) → Amharic mode
- If `langdetect` identifies Amharic → Amharic mode
- Otherwise → English mode

### What changes in Amharic mode

- The AI response is written in Amharic
- Risk labels are translated: Low → ዝቅተኛ, Medium → መካከለኛ, High → ከፍተኛ
- Recommendations and disclaimers are localized
- Amharic symptom phrases are normalized to English clinical terms before the ML model runs:
  - "የደረት ህመም" → "chest pain"
  - "ትኩሳት" → "fever"
  - "ራስ ምታት" → "headache"

### Example

Type: `ትኩሳት እና ራስ ምታት አለኝ`
(Translation: I have fever and headache)

The system processes this as "fever and headache" internally, runs the ML model, and returns the response in Amharic.

---

## 10. Security Features

### What happens behind the scenes to protect you

**Login security**
- After 5 failed login attempts, your account is locked for 15 minutes
- Passwords are hashed with Django's PBKDF2 algorithm — never stored in plain text

**Your health data**
- Before any external search (PubMed, DuckDuckGo), your symptom text is automatically scanned for personal information
- Names, phone numbers, email addresses, ID numbers, and dates are replaced with `[REDACTED]` before leaving the server
- Case records are automatically deleted after 30 days
- Audit logs are kept for 365 days

**Session security**
- Your login session uses JWT tokens that expire after 30 minutes
- The token is automatically refreshed in the background — you stay logged in for up to 7 days without re-entering your password
- Logging out immediately invalidates your token

**Data in transit**
- In production, all traffic uses HTTPS
- In development (localhost), HTTP is used for convenience

---

## Quick Reference — What to Do in Common Situations

| Situation | What to do |
|---|---|
| I feel unwell and want to check my symptoms | Go to **Symptom Check** → describe symptoms → get analysis |
| I want to chat with the AI | Go to **AI Assistant** → type your question |
| I need to find a hospital near me | Go to **Clinic Finder** → allow location or type your city |
| It's an emergency | Click the red **Emergency** button in the sidebar → call the number shown |
| I want to see my past conversations | Go to **AI Assistant** → click any session in the left panel |
| I want to update my medical history | Click your name in the sidebar → edit Profile |
| I am an admin and want to see system stats | Go to `http://127.0.0.1:8000/admin.html` |

---

## Important Disclaimer

> HealthAI is a **clinical decision support tool**, not a replacement for professional medical advice. All analysis results are informational only. Always consult a qualified healthcare professional before making any medical decision. In an emergency, call your local emergency number immediately.

---

*HealthAI — AI Healthcare Assistant | Academic Project 2026*
