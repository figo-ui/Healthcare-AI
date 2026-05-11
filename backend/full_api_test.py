"""Full API endpoint test."""
import requests, json, sys

BASE = 'http://127.0.0.1:8000/api/v1'
s = requests.Session()
results = []

# 1. Health
r = requests.get(f'{BASE}/health/', timeout=10)
results.append(('health', r.status_code, r.json().get('status','')))

# 2. Register a test user
r = s.post(f'{BASE}/auth/register/', json={
    'email': f'apitest{__import__("time").time()}@test.com',
    'password': 'TestPass123',
    'first_name': 'API',
    'last_name': 'Test'
}, timeout=10)
reg_data = r.json()
tok = reg_data.get('tokens',{}).get('access','')
ref = reg_data.get('tokens',{}).get('refresh','')
s.headers['Authorization'] = f'Bearer {tok}'
results.append(('register', r.status_code, 'ok' if tok else r.text[:80]))

# 3. Login
r = s.post(f'{BASE}/auth/login/', json={'identifier': reg_data['user']['email'], 'password': 'TestPass123'}, timeout=10)
login_data = r.json()
tok = login_data.get('tokens',{}).get('access','')
ref = login_data.get('tokens',{}).get('refresh','')
s.headers['Authorization'] = f'Bearer {tok}'
results.append(('login', r.status_code, 'ok' if tok else r.text[:80]))

# 4. Profile
r = s.get(f'{BASE}/profile/', timeout=10)
results.append(('profile', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 5. Profile update
r = s.patch(f'{BASE}/profile/', json={'first_name': 'Updated'}, timeout=10)
results.append(('profile-update', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 6. Chat sessions list
r = s.get(f'{BASE}/chat/sessions/', timeout=10)
results.append(('chat-sessions-list', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 7. Chat session create
r = s.post(f'{BASE}/chat/sessions/', json={'title': 'API Test Session'}, timeout=10)
sid = r.json().get('id', 1) if r.status_code == 201 else 1
results.append(('chat-session-create', r.status_code, 'ok' if r.status_code==201 else r.text[:80]))

# 8. Chat analyze (FormData - like frontend)
from collections import OrderedDict
form_data = {'symptom_text': 'I have been experiencing severe headaches and fever for three days', 'consent_given': 'true'}
r = s.post(f'{BASE}/chat/sessions/{sid}/analyze/', data=form_data, timeout=60)
results.append(('chat-analyze-form', r.status_code, 'ok' if r.status_code in [200,202] else r.text[:120]))

# 9. Chat analyze (JSON)
r = s.post(f'{BASE}/chat/sessions/{sid}/analyze/', json={'symptom_text': 'I have a persistent cough and chest pain for a week', 'consent_given': True}, timeout=60)
results.append(('chat-analyze-json', r.status_code, 'ok' if r.status_code in [200,202] else r.text[:120]))

# 10. Direct analyze (JSON)
r = s.post(f'{BASE}/analyze/', json={'symptom_text': 'I have been having stomach pain and nausea for two days now', 'consent_given': True}, timeout=60)
results.append(('analyze-direct-json', r.status_code, 'ok' if r.status_code in [200,202] else r.text[:120]))

# 11. Direct analyze (FormData)
r = s.post(f'{BASE}/analyze/', data={'symptom_text': 'I have been having stomach pain and nausea for two days now', 'consent_given': 'true'}, timeout=60)
results.append(('analyze-direct-form', r.status_code, 'ok' if r.status_code in [200,202] else r.text[:120]))

# 12. Chat messages
r = s.get(f'{BASE}/chat/sessions/{sid}/messages/', timeout=10)
results.append(('chat-messages', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 13. Chat history
r = s.get(f'{BASE}/chat/history/', timeout=10)
results.append(('chat-history', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 14. Chat export
r = s.get(f'{BASE}/chat/export/', timeout=10)
results.append(('chat-export', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 15. Quick prompts
r = s.get(f'{BASE}/quick-prompts/', timeout=10)
results.append(('quick-prompts', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 16. Location nearby
r = s.get(f'{BASE}/location/nearby/', params={'location_lat': 9.02, 'location_lng': 38.75}, timeout=30)
results.append(('location-nearby', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 17. Emergency contacts
r = s.get(f'{BASE}/location/emergency/', timeout=10)
results.append(('emergency-contacts', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 18. Location directions
r = s.get(f'{BASE}/location/directions/', params={'place_id': 'ChIJN1t_tDeuEmsRUsoyG83frY4'}, timeout=30)
results.append(('location-directions', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 19. Export profile
r = s.get(f'{BASE}/export/profile/', timeout=10)
results.append(('export-profile', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 20. Auth refresh
r = s.post(f'{BASE}/auth/refresh/', json={'refresh': ref}, timeout=10)
results.append(('auth-refresh', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 21. Social providers
r = requests.get(f'{BASE}/auth/social/providers/', timeout=10)
results.append(('social-providers', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 22. Verify email (POST - should return 400 since no token)
r = s.post(f'{BASE}/auth/verify-email/', json={}, timeout=10)
results.append(('verify-email-post', r.status_code, 'ok' if r.status_code==400 else r.text[:80]))

# 23. Verify email (GET - should return 400 since no token)
r = s.get(f'{BASE}/auth/verify-email/', timeout=10)
results.append(('verify-email-get', r.status_code, 'ok' if r.status_code==400 else r.text[:80]))

# 24. Password reset
r = s.post(f'{BASE}/auth/password-reset/', json={'email': reg_data['user']['email']}, timeout=10)
results.append(('password-reset', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 25. Resend verification
r = s.post(f'{BASE}/auth/resend-verification/', timeout=10)
results.append(('resend-verification', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 26. Admin config (GET)
r = s.get(f'{BASE}/admin/config/', timeout=10)
results.append(('admin-config-get', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 27. Admin users
r = s.get(f'{BASE}/admin/users/', timeout=10)
results.append(('admin-users', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 28. Admin analytics
r = s.get(f'{BASE}/admin/analytics/', timeout=10)
results.append(('admin-analytics', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 29. Admin audit log
r = s.get(f'{BASE}/admin/audit-log/', timeout=10)
results.append(('admin-audit-log', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 30. Admin model metrics
r = s.get(f'{BASE}/admin/model-metrics/', timeout=10)
results.append(('admin-model-metrics', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 31. Admin dialogue templates
r = s.get(f'{BASE}/admin/dialogue-templates/', timeout=10)
results.append(('admin-dialogue-templates', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 32. Admin facilities
r = s.get(f'{BASE}/admin/facilities/', timeout=10)
results.append(('admin-facilities', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# 33. Logout
r = s.post(f'{BASE}/auth/logout/', json={'refresh': ref}, timeout=10)
results.append(('logout', r.status_code, 'ok' if r.status_code==200 else r.text[:80]))

# Print results
print("\n=== FULL API TEST RESULTS ===")
passed = failed = 0
for name, code, status in results:
    ok = status == 'ok'
    symbol = 'PASS' if ok else 'FAIL'
    if ok: passed += 1
    else: failed += 1
    print(f'  [{symbol}] {name}: {code} {status}')

print(f'\nTotal: {passed} passed, {failed} failed out of {len(results)}')
