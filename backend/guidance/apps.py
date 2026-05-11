import os
import threading
import logging

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


def _warmup():
    """Pre-load all ML models in a background thread so the first request is fast."""
    try:
        import os, tempfile
        # Ensure joblib uses a writable temp folder for memory-mapped arrays
        os.environ.setdefault("JOBLIB_TEMP_FOLDER", tempfile.gettempdir())

        try:
            from .services.text_model import _load_text_artifacts
            _load_text_artifacts()
            logger.info("[warmup] Text model loaded.")
        except Exception as exc:
            logger.warning("[warmup] Text model failed: %s", exc, exc_info=True)

        try:
            from .services.dialogue_style import _load_dialogue_artifacts
            _load_dialogue_artifacts()
            logger.info("[warmup] Dialogue model loaded.")
        except Exception as exc:
            logger.warning("[warmup] Dialogue model failed: %s", exc)

        try:
            from .services.pii_redaction import _presidio_engines
            _presidio_engines()
            logger.info("[warmup] Presidio loaded.")
        except Exception as exc:
            logger.warning("[warmup] Presidio failed: %s", exc)

        try:
            from .services.rag import _build_index
            _build_index()
            logger.info("[warmup] RAG index built.")
        except Exception as exc:
            logger.warning("[warmup] RAG index failed: %s", exc)

        try:
            from .services.image_model import predict_image_probabilities
            # Force torch model load by running a dummy prediction check
            from pathlib import Path
            dummy = Path(getattr(settings, "IMAGE_TORCH_MODEL_PATH", ""))
            if dummy.exists():
                logger.info("[warmup] Image model path found — will lazy-load on first image request.")
            else:
                logger.info("[warmup] No image model file — skipping image warmup.")
        except Exception as exc:
            logger.warning("[warmup] Image model check failed: %s", exc)
    except Exception as exc:
        logger.error("[warmup] Warmup crashed: %s", exc, exc_info=True)


class GuidanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "guidance"

    def ready(self):
        # Warmup runs in a daemon thread so it doesn't block server start.
        # Disable with: WARMUP_ENABLED=false
        if os.getenv("WARMUP_ENABLED", "false").lower() in ("1", "true", "yes", "on"):
            t = threading.Thread(target=_warmup, daemon=True)
            t.start()
