"""Test RQ connectivity directly."""
import os, sys
os.environ['WARMUP_ENABLED'] = 'false'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthcare_ai.settings')
sys.path.insert(0, '.')
import django; django.setup()

print("Testing RQ connectivity...")
try:
    import django_rq
    print("  django_rq imported OK")
    queue = django_rq.get_queue("analysis")
    print(f"  get_queue OK: {queue}")
    # Try a simple ping
    conn = django_rq.get_connection("analysis")
    print(f"  get_connection OK: {conn}")
    conn.ping()
    print("  Redis PING OK")
    # Try enqueue
    def dummy_job():
        return "hello"
    job = queue.enqueue(dummy_job)
    print(f"  enqueue OK: job_id={job.id}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
