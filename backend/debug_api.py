"""Debug chat-analyze 500 error."""
import requests, json, traceback

BASE = 'http://127.0.0.1:8000/api/v1'
s = requests.Session()

# Login as admin
r = s.post(f'{BASE}/auth/login/', json={'identifier': 'admin@example.com', 'password': 'Admin1234'}, timeout=10)
if r.status_code != 200:
    print(f"Login failed: {r.status_code} {r.text[:200]}")
    exit(1)
d = r.json()
tok = d['tokens']['access']
s.headers['Authorization'] = f'Bearer {tok}'

# Create session
r = s.post(f'{BASE}/chat/sessions/', json={'title': 'Debug'}, timeout=10)
if r.status_code != 201:
    print(f"Session create failed: {r.status_code} {r.text[:200]}")
    exit(1)
sid = r.json()['id']
print(f"Session ID: {sid}")

# Test chat-analyze with JSON
print("\n--- Chat Analyze (JSON) ---")
try:
    r = s.post(f'{BASE}/chat/sessions/{sid}/analyze/', 
               json={'symptom_text': 'I have been experiencing severe headaches and fever for three days', 'consent_given': True}, 
               timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test chat-analyze with FormData
print("\n--- Chat Analyze (FormData) ---")
try:
    r = s.post(f'{BASE}/chat/sessions/{sid}/analyze/', 
               data={'symptom_text': 'I have been experiencing severe headaches and fever for three days', 'consent_given': 'true'}, 
               timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test direct analyze for comparison
print("\n--- Direct Analyze (JSON) ---")
try:
    r = s.post(f'{BASE}/analyze/', 
               json={'symptom_text': 'I have been experiencing severe headaches and fever for three days', 'consent_given': True}, 
               timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test admin endpoints with admin user
print("\n--- Admin Config (GET) ---")
r = s.get(f'{BASE}/admin/config/', timeout=10)
print(f"Status: {r.status_code} {r.text[:200]}")

print("\n--- Admin Users ---")
r = s.get(f'{BASE}/admin/users/', timeout=10)
print(f"Status: {r.status_code} {r.text[:200]}")

print("\n--- Logout ---")
r = s.post(f'{BASE}/auth/logout/', json={'refresh': d['tokens']['refresh']}, timeout=10)
print(f"Status: {r.status_code} {r.text[:200]}")
