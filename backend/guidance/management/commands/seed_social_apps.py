"""
Seed SocialApp entries for django-allauth from environment variables.

Usage:
    python manage.py seed_social_apps

Environment variables (set in .env or system env):
    SOCIAL_GOOGLE_CLIENT_ID / SOCIAL_GOOGLE_SECRET
    SOCIAL_GITHUB_CLIENT_ID  / SOCIAL_GITHUB_SECRET
    SOCIAL_MICROSOFT_CLIENT_ID / SOCIAL_MICROSOFT_SECRET
    SOCIAL_FACEBOOK_CLIENT_ID  / SOCIAL_FACEBOOK_SECRET
    SOCIAL_APPLE_CLIENT_ID     / SOCIAL_APPLE_SECRET  (Apple also needs APPLE_TEAM_ID, APPLE_KEY_ID, APPLE_PRIVATE_KEY)

Only providers with both CLIENT_ID and SECRET set will be created/updated.
"""

import os

from allauth.socialaccount.models import SocialApp
from django.contrib.sites.models import Site
from django.core.management.base import BaseCommand


PROVIDERS = {
    "google": {
        "client_id_env": "SOCIAL_GOOGLE_CLIENT_ID",
        "secret_env": "SOCIAL_GOOGLE_SECRET",
    },
    "github": {
        "client_id_env": "SOCIAL_GITHUB_CLIENT_ID",
        "secret_env": "SOCIAL_GITHUB_SECRET",
    },
    "microsoft": {
        "client_id_env": "SOCIAL_MICROSOFT_CLIENT_ID",
        "secret_env": "SOCIAL_MICROSOFT_SECRET",
    },
    "facebook": {
        "client_id_env": "SOCIAL_FACEBOOK_CLIENT_ID",
        "secret_env": "SOCIAL_FACEBOOK_SECRET",
    },
    "apple": {
        "client_id_env": "SOCIAL_APPLE_CLIENT_ID",
        "secret_env": "SOCIAL_APPLE_SECRET",
    },
}


class Command(BaseCommand):
    help = "Create or update django-allauth SocialApp entries from environment variables."

    def add_arguments(self, parser):
        parser.add_argument(
            "--site-domain",
            default=os.getenv("SITE_DOMAIN", "localhost"),
            help="Domain for the Site object (default: localhost)",
        )

    def handle(self, *args, **options):
        site_domain = options["site_domain"]

        # Ensure a Site object exists
        site, _ = Site.objects.get_or_create(
            id=1,
            defaults={"domain": site_domain, "name": site_domain},
        )
        if site.domain != site_domain:
            site.domain = site_domain
            site.name = site_domain
            site.save()

        created_count = 0
        updated_count = 0
        skipped_count = 0

        for provider, env_vars in PROVIDERS.items():
            client_id = os.getenv(env_vars["client_id_env"], "").strip()
            secret = os.getenv(env_vars["secret_env"], "").strip()

            if not client_id or not secret:
                self.stdout.write(
                    self.style.WARNING(
                        f"  ⏭  {provider}: skipped (set {env_vars['client_id_env']} and {env_vars['secret_env']} to enable)"
                    )
                )
                skipped_count += 1
                continue

            app, created = SocialApp.objects.update_or_create(
                provider=provider,
                defaults={
                    "name": f"{provider.capitalize()} OAuth",
                    "client_id": client_id,
                    "secret": secret,
                },
            )
            app.sites.add(site)

            if created:
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f"  ✅ {provider}: created"))
            else:
                updated_count += 1
                self.stdout.write(self.style.SUCCESS(f"  🔄 {provider}: updated"))

        self.stdout.write(
            f"\nDone: {created_count} created, {updated_count} updated, {skipped_count} skipped."
        )
