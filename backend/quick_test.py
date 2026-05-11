import requests
BASE = 'http://127.0.0.1:8000/api/v1'
s = requests.Session()
r = s.post(f'{BASE}/auth/login/', json={'identifier':'admin@example.com','password':'Admin1234'}, timeout=10)
d = r.json()
tok = d['tokens']['access']
s.headers['Authorization'] = f'Bearer {tok}'

r1 = s.post(f'{BASE}/analyze/', json={'symptom_text':'I have been experiencing severe headaches and fever for three days','consent_given':True}, timeout=30)
print(f'analyze-direct: {r1.status_code}')

r2 = s.post(f'{BASE}/chat/sessions/', json={'title':'T'}, timeout=10)
sid = r2.json().get('id',1)

r3 = s.post(f'{BASE}/chat/sessions/{sid}/analyze/', json={'symptom_text':'I have been experiencing severe headaches and fever for three days','consent_given':True}, timeout=30)
print(f'chat-analyze: {r3.status_code}')

r4 = s.get(f'{BASE}/admin/config/', timeout=10)
print(f'admin-config: {r4.status_code}')

r5 = s.post(f'{BASE}/auth/logout/', json={'refresh': d['tokens']['refresh']}, timeout=10)
print(f'logout: {r5.status_code}')
