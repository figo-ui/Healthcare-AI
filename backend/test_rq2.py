"""Test RQ enqueue with a proper function."""
import os, sys
os.environ['WARMUP_ENABLED'] = 'false'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthcare_ai.settings')
sys.path.insert(0, '.')
import django; django.setup()

from guidance.services.async_inference import execute_case_analysis

print("Testing RQ enqueue with real function...")
try:
    import django_rq
    queue = django_rq.get_queue("analysis")
    job = queue.enqueue(execute_case_analysis, 1, job_timeout=600)
    print(f"  enqueue OK: job_id={job.id}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
