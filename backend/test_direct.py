import requests
BASE = 'http://127.0.0.1:8000/api/v1'
s = requests.Session()
r = s.post(f'{BASE}/auth/login/', json={'identifier':'admin@example.com','password':'Admin1234'}, timeout=10)
d = r.json()
tok = d['tokens']['access']
s.headers['Authorization'] = f'Bearer {tok}'

print("Testing direct analyze with 120s timeout...")
r = s.post(f'{BASE}/analyze/', json={'symptom_text':'I have been experiencing severe headaches and fever for three days','consent_given':True}, timeout=120)
print(f'analyze-direct: {r.status_code}')
if r.status_code == 200:
    data = r.json()
    print(f'Conditions: {[c["condition"] for c in data.get("probable_conditions",[])]}')
else:
    print(r.text[:200])
