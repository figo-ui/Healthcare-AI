from django.conf import settings
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError


class CookieJWTAuthentication(JWTAuthentication):
    """JWT authentication that reads tokens from both Authorization header and cookies.

    On AllowAny views (e.g. /analyze/, /analyze/<id>/), an invalid or missing
    token must NOT block the request — it should simply leave request.user as
    AnonymousUser.  Only views with IsAuthenticated will reject unauthenticated
    requests.

    Behaviour:
    - Valid token  → returns (user, token)  — request.user is set
    - No token     → returns None           — request.user stays AnonymousUser
    - Invalid/expired token → returns None  — request.user stays AnonymousUser
      (DRF will still enforce IsAuthenticated on protected views)
    """

    def authenticate(self, request):
        # ── 1. Try Authorization header first ────────────────────────────
        header = self.get_header(request)
        if header is not None:
            try:
                return super().authenticate(request)
            except (InvalidToken, TokenError):
                # Expired / malformed header token — treat as anonymous.
                # IsAuthenticated views will still reject the request.
                return None

        # ── 2. Fall back to HttpOnly cookie ──────────────────────────────
        raw_token = request.COOKIES.get(
            getattr(settings, "JWT_ACCESS_COOKIE_NAME", "healthcare_access")
        )
        if not raw_token:
            return None

        try:
            validated_token = self.get_validated_token(raw_token)
            return self.get_user(validated_token), validated_token
        except (InvalidToken, TokenError):
            return None
