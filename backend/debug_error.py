"""Get the actual error from chat-analyze."""
import requests, re

BASE = 'http://127.0.0.1:8000/api/v1'
s = requests.Session()

r = s.post(f'{BASE}/auth/login/', json={'identifier': 'admin@example.com', 'password': 'Admin1234'}, timeout=10)
d = r.json()
s.headers['Authorization'] = f'Bearer {d["tokens"]["access"]}'

r = s.post(f'{BASE}/chat/sessions/', json={'title': 'Debug2'}, timeout=10)
sid = r.json()['id']

r = s.post(f'{BASE}/chat/sessions/{sid}/analyze/', 
           json={'symptom_text': 'I have been experiencing severe headaches and fever for three days', 'consent_given': True}, 
           timeout=30)

# Extract error from HTML
text = r.text
# Find the exception type and message
exc_match = re.search(r'<h1>([\w\s]+Error.*?)</h1>', text)
loc_match = re.search(r'at (/[^\s<]+)', text)
if exc_match:
    print(f"Exception: {exc_match.group(1)}")
if loc_match:
    print(f"Location: {loc_match.group(1)}")

# Find traceback lines
tb_lines = re.findall(r'<li class="frame[^"]*">.*?<code[^>]*>(.*?)</code>', text, re.DOTALL)
for line in tb_lines[:10]:
    clean = re.sub(r'<[^>]+>', '', line).strip()
    if clean:
        print(f"  {clean}")

# Also look for the specific error message
msg_match = re.search(r'<pre[^>]*>(.*?)</pre>', text, re.DOTALL)
if msg_match:
    msg = re.sub(r'<[^>]+>', '', msg_match.group(1)).strip()
    print(f"\nError detail:\n{msg[:500]}")
