import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from django.conf import settings
from PIL import Image

from .preprocess import prepare_image_tensor

torch = None
models = None
DermCNN = None


def _get_keras_loader():
    try:
        from tensorflow.keras.models import load_model

        return load_model
    except Exception:  # pragma: no cover
        return None


def _get_torch_modules():
    global torch, models, DermCNN
    if torch is not None or DermCNN is not None or models is not None:
        return torch, models, DermCNN

    try:
        import torch as torch_module
    except Exception:  # pragma: no cover
        torch_module = None

    try:
        from torchvision import models as torchvision_models
    except Exception:  # pragma: no cover
        torchvision_models = None

    try:
        from .torch_cnn import DermCNN as dermcnn_class
    except Exception:  # pragma: no cover
        dermcnn_class = None

    torch = torch_module
    models = torchvision_models
    DermCNN = dermcnn_class
    return torch, models, DermCNN


def _load_metadata() -> Dict[str, object]:
    path = Path(getattr(settings, "IMAGE_MODEL_METADATA_PATH", ""))
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _estimate_quality(image_path: Path) -> float:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return 0.0

    width, height = image.size
    pixels = np.asarray(image, dtype=np.float32)
    edge_energy = float(np.std(np.diff(pixels, axis=0))) + float(np.std(np.diff(pixels, axis=1)))
    resolution_score = min(1.0, (width * height) / float(512 * 512))
    clarity_score = min(1.0, edge_energy / 45.0)
    return round(max(0.0, min(1.0, 0.55 * resolution_score + 0.45 * clarity_score)), 4)


def _build_torchvision_model(architecture: str, num_classes: int):
    torch_module, torchvision_models, _ = _get_torch_modules()
    if torchvision_models is None or torch_module is None:
        return None
    if architecture == "resnet50":
        model = torchvision_models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch_module.nn.Linear(in_features, num_classes)
        return model
    if architecture == "efficientnet_b3":
        model = torchvision_models.efficientnet_b3(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch_module.nn.Linear(in_features, num_classes)
        return model
    return None


@lru_cache(maxsize=1)
def _load_image_artifacts() -> Tuple[Optional[object], List[str]]:
    model_path = Path(settings.IMAGE_MODEL_PATH)
    torch_model_path = Path(settings.IMAGE_TORCH_MODEL_PATH)
    labels_path = Path(settings.IMAGE_LABELS_PATH)
    metadata = _load_metadata()
    labels: List[str] = []
    load_model = _get_keras_loader()
    torch_module, _, dermcnn_class = _get_torch_modules()

    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

    if load_model is not None and model_path.exists():
        model = load_model(model_path, compile=False)
        return {
            "backend": "keras",
            "model": model,
            "input_size": int(metadata.get("input_size", settings.IMAGE_INPUT_SIZE)),
            "metadata": metadata,
        }, labels

    if torch_module is not None and torch_model_path.exists():
        checkpoint = torch_module.load(torch_model_path, map_location="cpu")
        class_names = checkpoint.get("class_names", []) or labels
        if class_names:
            labels = [str(item) for item in class_names]
        architecture = str(checkpoint.get("architecture", "dermcnn"))
        num_classes = int(checkpoint.get("num_classes", len(labels) if labels else 7))
        input_size = int(checkpoint.get("input_size", metadata.get("input_size", settings.IMAGE_INPUT_SIZE)))
        normalization = checkpoint.get("normalization", metadata.get("normalization", {}))

        # ── New: EfficientNet + sklearn LogisticRegression backend ──────────
        if checkpoint.get("backend") in ("torchvision_sklearn", "sklearn_hog"):
            try:
                import joblib as _joblib
                pipeline_filename = checkpoint.get("sklearn_pipeline_path", "image_sklearn_pipeline.joblib")
                pipeline_path = torch_model_path.parent / pipeline_filename
                if not pipeline_path.exists():
                    return None, labels
                pipeline = _joblib.load(pipeline_path)
                # Build feature extractor
                try:
                    from torchvision import models as tv_models
                    import torch.nn as _nn
                    arch = checkpoint.get("architecture", "mobilenet_v2")
                    if arch == "mobilenet_v2":
                        backbone = tv_models.mobilenet_v2(weights=None)
                        feat_state = checkpoint.get("feature_extractor_state", {})
                        if feat_state:
                            bb_state = backbone.state_dict()
                            bb_state.update({k: v for k, v in feat_state.items() if k in bb_state})
                            backbone.load_state_dict(bb_state, strict=False)
                        extractor = _nn.Sequential(
                            backbone.features,
                            _nn.AdaptiveAvgPool2d(1),
                            _nn.Flatten(),
                        )
                    else:
                        weights = tv_models.EfficientNet_B3_Weights.DEFAULT
                        efn = tv_models.efficientnet_b3(weights=None)
                        feat_state = checkpoint.get("feature_extractor_state", {})
                        if feat_state:
                            efn_state = efn.state_dict()
                            efn_state.update({k: v for k, v in feat_state.items() if k in efn_state})
                            efn.load_state_dict(efn_state, strict=False)
                        extractor = _nn.Sequential(efn.features, efn.avgpool, _nn.Flatten())
                    extractor.eval()
                except Exception:
                    extractor = None
                return {
                    "backend": checkpoint.get("backend", "torchvision_sklearn"),
                    "extractor": extractor,
                    "pipeline": pipeline,
                    "input_size": input_size,
                    "architecture": architecture,
                    "num_classes": num_classes,
                    "normalization": normalization,
                    "metadata": checkpoint,
                }, labels
            except Exception:
                return None, labels

        if architecture in {"resnet50", "efficientnet_b3"}:
            model = _build_torchvision_model(architecture, max(1, num_classes))
            if model is None:
                return None, labels
        else:
            if dermcnn_class is None:
                return None, labels
            # Support both DermCNN (resnet50) and DermCNNv2 (compact CNN)
            if architecture == "dermcnn_v2":
                try:
                    from .torch_cnn import DermCNNv2
                    model = DermCNNv2(num_classes=max(1, num_classes))
                except Exception:
                    model = dermcnn_class(num_classes=max(1, num_classes))
            else:
                model = dermcnn_class(num_classes=max(1, num_classes))

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {
            "backend": "torch",
            "model": model,
            "input_size": input_size,
            "architecture": architecture,
            "num_classes": num_classes,
            "normalization": normalization,
            "metadata": checkpoint,
        }, labels

    return None, labels


def _apply_torch_normalization(tensor: np.ndarray, normalization: Dict[str, object]) -> np.ndarray:
    mean = normalization.get("mean") if isinstance(normalization, dict) else None
    std = normalization.get("std") if isinstance(normalization, dict) else None
    if not isinstance(mean, list) or not isinstance(std, list):
        return tensor
    array = tensor.copy()
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 3)
    return (array - mean_arr) / np.maximum(std_arr, 1e-6)


def predict_image_probabilities(image_path: Path, top_k: int = 5):
    model, labels = _load_image_artifacts()
    if model is None:
        return {
            "predictions": [],
            "confidence": 0.0,
            "model_version": "image-unavailable-v2",
            "quality_score": 0.0,
        }

    image_size = int(model.get("input_size", settings.IMAGE_INPUT_SIZE))
    tensor = prepare_image_tensor(image_path=image_path, image_size=image_size)
    if tensor is None:
        return {
            "predictions": [],
            "confidence": 0.0,
            "model_version": "image-invalid-v2",
            "quality_score": 0.0,
        }

    backend = str(model.get("backend", "unknown"))
    probs: np.ndarray
    if backend == "keras":
        probs = model["model"].predict(tensor * 255.0, verbose=0)[0]
    elif backend == "torchvision_sklearn":
        # Feature extraction + optional PCA + sklearn classifier
        torch_module, _, _ = _get_torch_modules()
        if torch_module is None or model.get("extractor") is None:
            return {
                "predictions": [],
                "confidence": 0.0,
                "model_version": "image-sklearn-torch-missing-v2",
                "quality_score": 0.0,
            }
        normalized = _apply_torch_normalization(tensor, model.get("normalization", {}))
        with torch_module.no_grad():
            x = torch_module.tensor(normalized, dtype=torch_module.float32).permute(0, 3, 1, 2).contiguous()
            feats = model["extractor"](x).cpu().numpy()
        pipeline = model["pipeline"]
        feats_sc = pipeline["scaler"].transform(feats)
        # Apply PCA if present in pipeline
        if "pca" in pipeline:
            feats_sc = pipeline["pca"].transform(feats_sc)
        probs = pipeline["classifier"].predict_proba(feats_sc)[0]
    elif backend == "sklearn_hog":
        # HOG + color histogram features + sklearn classifier (no torch needed)
        try:
            from skimage.feature import hog
            from skimage.color import rgb2hsv
        except ImportError:
            return {
                "predictions": [],
                "confidence": 0.0,
                "model_version": "image-hog-skimage-missing-v2",
                "quality_score": 0.0,
            }
        # tensor is (1, H, W, 3) float [0,1] — convert to uint8
        img_np = (tensor[0] * 255).astype(np.uint8)
        feats = []
        hog_feats = hog(
            img_np, orientations=9, pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), channel_axis=-1, feature_vector=True,
        )
        feats.append(hog_feats)
        for c in range(3):
            hist, _ = np.histogram(img_np[:, :, c], bins=16, range=(0, 256))
            feats.append(hist.astype(np.float32) / (img_np.shape[0] * img_np.shape[1]))
        hsv = (rgb2hsv(img_np.astype(np.float32) / 255.0) * 255).astype(np.uint8)
        for c in range(3):
            hist, _ = np.histogram(hsv[:, :, c], bins=16, range=(0, 256))
            feats.append(hist.astype(np.float32) / (img_np.shape[0] * img_np.shape[1]))
        feat_vec = np.concatenate(feats).reshape(1, -1)
        pipeline = model["pipeline"]
        feat_sc  = pipeline["scaler"].transform(feat_vec)
        probs    = pipeline["classifier"].predict_proba(feat_sc)[0]
    elif backend == "torch":
        torch_module, _, _ = _get_torch_modules()
        if torch_module is None:
            return {
                "predictions": [],
                "confidence": 0.0,
                "model_version": "image-torch-runtime-missing-v2",
                "quality_score": 0.0,
            }
        normalized = _apply_torch_normalization(tensor, model.get("normalization", {}))
        with torch_module.no_grad():
            x = torch_module.tensor(normalized, dtype=torch_module.float32).permute(0, 3, 1, 2).contiguous()
            logits = model["model"](x)
            probs = torch_module.softmax(logits, dim=1).cpu().numpy()[0]
    else:
        return {
            "predictions": [],
            "confidence": 0.0,
            "model_version": "image-model-backend-unknown-v2",
            "quality_score": 0.0,
        }

    if not labels:
        labels = [f"class_{idx}" for idx in range(len(probs))]

    predictions = [
        {"condition": labels[idx], "probability": round(float(prob), 4)}
        for idx, prob in enumerate(probs)
    ]
    predictions.sort(key=lambda item: item["probability"], reverse=True)
    predictions = predictions[:top_k]
    confidence = predictions[0]["probability"] if predictions else 0.0
    quality_score = _estimate_quality(image_path)

    architecture = str(model.get("architecture", model.get("metadata", {}).get("architecture", "generic")))
    num_classes = int(model.get("num_classes", len(labels)))
    limited_scope = architecture == "dermcnn" or num_classes <= 7
    return {
        "predictions": predictions,
        "confidence": round(confidence, 4),
        "model_version": f"image-{backend}-{architecture}-v2",
        "quality_score": quality_score,
        "scope": "dermatology_skin_only" if limited_scope else "multimodal_medical_image",
        "limited_scope": limited_scope,
    }
