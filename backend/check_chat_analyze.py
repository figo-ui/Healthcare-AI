import requests, re
BASE = 'http://127.0.0.1:8000/api/v1'
s = requests.Session()
r = s.post(f'{BASE}/auth/login/', json={'identifier':'admin@example.com','password':'Admin1234'}, timeout=10)
d = r.json()
s.headers['Authorization'] = f'Bearer {d["tokens"]["access"]}'

r2 = s.post(f'{BASE}/chat/sessions/', json={'title':'T'}, timeout=10)
sid = r2.json().get('id',1)

r3 = s.post(f'{BASE}/chat/sessions/{sid}/analyze/', json={'symptom_text':'I have been experiencing severe headaches and fever for three days','consent_given':True}, timeout=60)
print(f'chat-analyze: {r3.status_code}')
if r3.status_code != 200 and r3.status_code != 202:
    # Extract error from HTML
    text = r3.text
    exc_match = re.search(r'<h1>(.*?)</h1>', text)
    if exc_match:
        print(f"Exception: {exc_match.group(1)}")
    msg_match = re.search(r'<pre[^>]*>(.*?)</pre>', text, re.DOTALL)
    if msg_match:
        msg = re.sub(r'<[^>]+>', '', msg_match.group(1)).strip()
        print(f"Error: {msg[:300]}")
    else:
        print(r3.text[:300])
