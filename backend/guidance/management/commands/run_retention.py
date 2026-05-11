"""
Management command: run_retention
Purges expired PHI records according to configured retention windows.

Usage:
    python manage.py run_retention
    python manage.py run_retention --case-days 30 --audit-days 365 --dry-run
"""
import logging

from django.conf import settings
from django.core.management.base import BaseCommand

from guidance.services.retention import purge_expired_phi_records

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Purge expired PHI records (cases and audit logs) per retention policy."

    def add_arguments(self, parser):
        parser.add_argument(
            "--case-days",
            type=int,
            default=None,
            help="Override case retention window in days (default: CASE_RETENTION_DAYS from settings).",
        )
        parser.add_argument(
            "--audit-days",
            type=int,
            default=None,
            help="Override audit log retention window in days (default: AUDIT_LOG_RETENTION_DAYS from settings).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Print what would be deleted without actually deleting.",
        )

    def handle(self, *args, **options):
        case_days = options["case_days"] or getattr(settings, "CASE_RETENTION_DAYS", 30)
        audit_days = options["audit_days"] or getattr(settings, "AUDIT_LOG_RETENTION_DAYS", 365)
        dry_run = options["dry_run"]

        self.stdout.write(
            self.style.NOTICE(
                f"Retention policy: cases={case_days}d, audit_logs={audit_days}d"
                + (" [DRY RUN]" if dry_run else "")
            )
        )

        if dry_run:
            from django.utils import timezone
            from datetime import timedelta
            from guidance.models import AuditLog, CaseSubmission

            case_cutoff = timezone.now() - timedelta(days=case_days)
            audit_cutoff = timezone.now() - timedelta(days=audit_days)
            cases_count = CaseSubmission.objects.filter(created_at__lt=case_cutoff).count()
            audit_count = AuditLog.objects.filter(created_at__lt=audit_cutoff).count()
            self.stdout.write(
                self.style.WARNING(
                    f"[DRY RUN] Would delete {cases_count} case(s) and {audit_count} audit log(s)."
                )
            )
            return

        try:
            result = purge_expired_phi_records(
                case_retention_days=case_days,
                audit_retention_days=audit_days,
            )
            msg = (
                f"Retention complete: {result['cases_deleted']} case(s) deleted "
                f"({result['case_rows_deleted']} rows), "
                f"{result['audit_rows_deleted']} audit log row(s) deleted."
            )
            self.stdout.write(self.style.SUCCESS(msg))
            logger.info("run_retention: %s", msg)
        except Exception as exc:
            self.stderr.write(self.style.ERROR(f"Retention failed: {exc}"))
            logger.exception("run_retention failed: %s", exc)
            raise
