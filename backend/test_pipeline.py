"""Test run_case_analysis directly."""
import os, sys, traceback
os.environ['WARMUP_ENABLED'] = 'false'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthcare_ai.settings')
sys.path.insert(0, '.')
import django; django.setup()

from guidance.models import CaseSubmission
from guidance.services.pipeline import run_case_analysis

# Get the latest case
case = CaseSubmission.objects.order_by('-id').first()
print(f"Case ID: {case.id}, status: {case.status}")
print(f"Symptom: {case.symptom_text[:80]}")

try:
    result = run_case_analysis(case)
    print(f"SUCCESS: {list(result.keys())}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
