from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.static import serve
import os

# Resolve frontend directory relative to this file
FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'UI')
)

urlpatterns = [
    # Django admin — must come before the catch-all
    path("admin/", admin.site.urls),
    # django-allauth (headless + social providers)
    path("_allauth/", include("allauth.urls")),
    # REST API
    path("api/v1/", include("guidance.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Serve plain HTML frontend — catch-all LAST so it never shadows API routes
urlpatterns += [
    path("", serve, {"document_root": FRONTEND_DIR, "path": "index.html"}),
    path("<path:path>", serve, {"document_root": FRONTEND_DIR}),
]
