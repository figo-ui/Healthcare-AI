"""Check all API views for potential errors."""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ['WARMUP_ENABLED'] = 'false'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthcare_ai.settings')
sys.path.insert(0, '.')
import django; django.setup()

from django.conf import settings
from pathlib import Path

# 1. Check all model files exist
print("=== 1. Model Files ===")
model_paths = {
    'TEXT_MODEL': settings.TEXT_MODEL_PATH,
    'TFIDF_VECTORIZER': settings.TFIDF_VECTORIZER_PATH,
    'TEXT_LABELS': settings.TEXT_LABELS_PATH,
    'TRIAGE_CONSOLIDATION': settings.TRIAGE_LABEL_CONSOLIDATION_PATH,
    'IMAGE_TORCH': settings.IMAGE_TORCH_MODEL_PATH,
    'IMAGE_LABELS': settings.IMAGE_LABELS_PATH,
    'IMAGE_METADATA': settings.IMAGE_MODEL_METADATA_PATH,
    'CONDITION_MAP': settings.CONDITION_NAME_MAP_PATH,
    'DIALOGUE_MODEL': settings.DIALOGUE_INTENT_MODEL_PATH,
    'DIALOGUE_VECTORIZER': settings.DIALOGUE_INTENT_VECTORIZER_PATH,
    'DIALOGUE_CONSOLIDATION': settings.DIALOGUE_INTENT_CONSOLIDATION_PATH,
    'DIALOGUE_TEMPLATES': settings.DIALOGUE_RESPONSE_TEMPLATES_PATH,
}
missing = []
for name, p in model_paths.items():
    exists = Path(p).exists()
    status = "OK" if exists else "MISSING"
    if not exists: missing.append(name)
    print(f"  [{status}] {name}")
if missing:
    print(f"  MISSING: {missing}")

# 2. Check AdminConfigView GET handler references valid settings
print("\n=== 2. AdminConfigView Settings Check ===")
config_attrs = [
    'USE_LLM_TRIAGE', 'LLM_FALLBACK_TO_CLASSICAL', 'LLM_RAG_RESPONSE',
    'TRIAGE_LLM_BASE_MODEL', 'GITHUB_TOKEN', 'REDIS_URL',
    'IMAGE_INPUT_SIZE', 'CASE_RETENTION_DAYS', 'AUDIT_LOG_RETENTION_DAYS',
    'GEOAPIFY_API_KEY', 'GOOGLE_MAPS_API_KEY', 'DEBUG',
]
for attr in config_attrs:
    val = getattr(settings, attr, "NOT_DEFINED")
    if val == "NOT_DEFINED":
        print(f"  [MISSING] settings.{attr}")
    else:
        print(f"  [OK] settings.{attr}")

# 3. Try loading ML models
print("\n=== 3. ML Model Loading Test ===")
try:
    from guidance.services.text_model import _load_text_artifacts
    artifacts = _load_text_artifacts()
    if artifacts[0] is not None:
        print("  [OK] Text model loaded")
    else:
        print("  [FAIL] Text model returned None")
except Exception as e:
    print(f"  [ERROR] Text model: {e}")

try:
    from guidance.services.dialogue_style import _load_dialogue_artifacts
    dart = _load_dialogue_artifacts()
    if dart[0] is not None:
        print("  [OK] Dialogue model loaded")
    else:
        print("  [FAIL] Dialogue model returned None")
except Exception as e:
    print(f"  [ERROR] Dialogue model: {e}")

# 4. Check URL resolution
print("\n=== 4. URL Resolution Check ===")
from django.urls import reverse, resolve
from django.urls.exceptions import NoReverseMatch
test_urls = [
    ('health', {}),
    ('login', {}),
    ('register', {}),
    ('logout', {}),
    ('token-refresh', {}),
    ('profile', {}),
    ('chat-sessions', {}),
    ('analyze', {}),
    ('location-nearby', {}),
    ('location-emergency', {}),
    ('location-directions', {}),
    ('quick-prompts', {}),
    ('chat-history', {}),
    ('chat-export', {}),
    ('export-profile', {}),
    ('admin-users', {}),
    ('admin-analytics', {}),
    ('admin-audit-log', {}),
    ('admin-model-metrics', {}),
    ('admin-config', {}),
    ('admin-dialogue-templates', {}),
    ('admin-facilities', {}),
    ('verify-email', {}),
    ('resend-verification', {}),
    ('password-reset', {}),
    ('password-reset-confirm', {}),
    ('social-providers', {}),
]
from guidance import urls as guidance_urls
url_names = [p.name for p in guidance_urls.urlpatterns if hasattr(p, 'name')]
print(f"  Defined URL names: {len(url_names)}")
for name, _ in test_urls:
    if name in url_names:
        print(f"  [OK] {name}")
    else:
        print(f"  [MISSING] {name}")

# 5. Check view classes for parser issues
print("\n=== 5. View Parser Classes Check ===")
from guidance.views import (
    AnalyzeCaseView, ChatAnalyzeView, VerifyEmailView,
    AdminConfigView, LoginView, RegisterView,
    ProfileView, ChatSessionListView, ChatAnalyzeView as CA,
)
from rest_framework.parsers import JSONParser
views_to_check = {
    'AnalyzeCaseView': AnalyzeCaseView,
    'ChatAnalyzeView': ChatAnalyzeView,
    'VerifyEmailView': VerifyEmailView,
    'AdminConfigView': AdminConfigView,
    'LoginView': LoginView,
    'RegisterView': RegisterView,
}
for name, view_cls in views_to_check.items():
    parsers = view_cls.parser_classes
    has_json = any(p == JSONParser for p in parsers)
    print(f"  [{('OK' if has_json else 'WARN')}] {name}: {[p.__name__ for p in parsers]}")

print("\n=== Done ===")
