"""SceneScout: local candidate-scene finder for a personal video library.

Prerequisites (Windows / NVIDIA recommended):
  * Python 3.10+
  * FFmpeg and FFprobe on PATH, or set FFMPEG / FFPROBE environment variables.
  * FreeSimpleGUI, NudeNet 3.4.2, ONNX Runtime (GPU build preferred).
  * Optional frame classifier: PyTorch + CUDA, Transformers, timm and/or moderators.
  * Pillow is optional but enables annotated "Show frame" previews.

Everything runs locally. The only network access is the one-time model download
performed by the model libraries (Hugging Face Hub) when a model is first loaded.

Single-file launch:
  python SceneScout.py
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import FreeSimpleGUI as sg


# ---------------------------------------------------------------------------
# Binaries, constants and application state

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".ts", ".m4v", ".avi", ".webm"}
WIN_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
ACCENT = "#58a6ff"
GREEN = "#238636"
RED = "#da3633"
GRAY = "#30363d"
LOG_BG = "#0d1117"
LOG_FG = "#c9d1d9"

RAW_MIN_SCORE = 0.10
HIGH_SCORE = 0.65
FINE_INTERVAL = 0.5
CHUNK_SECONDS = 600.0
CACHE_VERSION = 3

DEFAULT_CLASSIFIER = "TostAI/nsfw-image-detection-large"
CLASSIFIER_CHOICES = [
    "TostAI/nsfw-image-detection-large",
    "viddexa/nsfw-detection-2-mini",
    "Marqo/nsfw-image-detection-384",
]

# Verified from each model's config/README (2026):
#   TostAI  : id2label SAFE / QUESTIONABLE / UNSAFE, license CC BY-NC-SA 4.0.
#   viddexa : id2label safe / hentai / porn / sexy / drawing, license Apache-2.0.
#   Marqo   : label_names NSFW / SFW, license Apache-2.0.
CLASSIFIER_MODELS: dict[str, dict[str, Any]] = {
    "TostAI/nsfw-image-detection-large": {
        "backend": "transformers",
        "explicit": ("UNSAFE",),
        "suggestive": ("QUESTIONABLE",),
        "license": "CC BY-NC-SA 4.0",
        "size": "348.6 MB weights, 87.1M parameters",
        "download": "about 349 MB",
    },
    "viddexa/nsfw-detection-2-mini": {
        "backend": "moderators",
        "explicit": ("porn", "hentai"),
        "suggestive": ("sexy",),
        "license": "Apache-2.0",
        "size": "70.8 MB weights, 17.7M parameters",
        "download": "about 71 MB",
    },
    "Marqo/nsfw-image-detection-384": {
        "backend": "timm",
        "explicit": ("NSFW",),
        "suggestive": (),
        "license": "Apache-2.0",
        "size": "22.4 MB weights, 5.6M parameters",
        "download": "about 22 MB",
    },
}


def resolve_binary(name: str, env_var: str) -> Optional[str]:
    """Find a binary, giving explicit configuration precedence."""
    configured = os.environ.get(env_var, "").strip()
    candidates = [configured] if configured else []
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    suffixes = [".exe", ""] if os.name == "nt" else [""]
    for folder in (root, Path.cwd()):
        for suffix in suffixes:
            candidates.append(str(folder / (name + suffix)))
    for item in candidates:
        if item and Path(item).is_file():
            return item
    return shutil.which(name) or (shutil.which(name + ".exe") if os.name == "nt" else None)


def app_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "SceneScout" / "cache"


class AppState:
    def __init__(self) -> None:
        self.ffmpeg = resolve_binary("ffmpeg", "FFMPEG")
        self.ffprobe = resolve_binary("ffprobe", "FFPROBE")

        self.nudenet: Any = None
        self.nudenet_version = "not installed"
        self.nudenet_error = ""
        self.nudenet_models: list[dict[str, Any]] = []
        self.nudenet_models_by_display: dict[str, dict[str, Any]] = {}
        self.default_nudenet_display = ""
        self.nudenet_model_key = ""
        self.nudenet_model_name = ""
        self.nudenet_model_display = ""
        self.nudenet_resolution = 320
        self.nudenet_provider = "unavailable"

        self.classifier: Any = None
        self.classifier_model_id: str = "none"
        self.classifier_device = "unavailable"
        self.classifier_error = ""
        self.classifier_notes: set[str] = set()

        self.videos: list[dict[str, Any]] = []
        self.cancel = threading.Event()
        self.running = False
        self.job_kind = ""
        self.current_process: Optional[subprocess.Popen[str]] = None
        self.process_lock = threading.Lock()
        self.temp_root: Optional[Path] = None
        self.view_indices: list[int] = []
        self.cache_dir = app_cache_dir()


# ---------------------------------------------------------------------------
# Utilities, filesystem normalization and image dimensions

def run_command(state: AppState, args: list[str], timeout: Optional[float] = None) -> tuple[int, str, str]:
    """Run a cancellable subprocess. Never raises for a bad external tool."""
    try:
        with state.process_lock:
            if state.cancel.is_set():
                return -1, "", "Cancelled"
            state.current_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=WIN_FLAGS,
            )
            process = state.current_process
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return -1, stdout, "Timed out: " + stderr
        return process.returncode, stdout, stderr
    except Exception as exc:
        return -1, "", str(exc)
    finally:
        with state.process_lock:
            state.current_process = None


def normalize_file_list(result: Any) -> list[str]:
    """Normalize FreeSimpleGUI dialog results into clean filesystem strings."""
    if result is None or result == "":
        return []
    if isinstance(result, str):
        raw_items: list[Any] = [result]
    elif isinstance(result, (tuple, list)):
        raw_items = list(result)
    else:
        return []
    paths: list[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            continue
        for part in item.split(";"):
            text = part.strip().strip('"')
            if text:
                paths.append(text)
    return paths


def image_dimensions(path: str) -> Optional[tuple[int, int]]:
    """Return (width, height) using Pillow when available, else a tiny JPEG parser."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        pass
    try:
        with open(path, "rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None
            while True:
                prefix = handle.read(1)
                if not prefix:
                    return None
                if prefix != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if not marker:
                    return None
                code = marker[0]
                if code in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    length = int.from_bytes(handle.read(2), "big")
                    data = handle.read(max(0, length - 2))
                    if len(data) < 5:
                        return None
                    height = int.from_bytes(data[1:3], "big")
                    width = int.from_bytes(data[3:5], "big")
                    return width, height
                if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                    continue
                length = int.from_bytes(handle.read(2), "big")
                if length < 2:
                    return None
                handle.seek(length - 2, 1)
    except Exception:
        return None


def open_path(window: sg.Window, path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        sg.popup_error(f"Created {path}, but could not open it:\n{exc}", title="SceneScout")


def cleanup_temp(state: AppState) -> None:
    if state.temp_root:
        shutil.rmtree(state.temp_root, ignore_errors=True)
        state.temp_root = None


def cleanup_preview() -> None:
    folder = Path(tempfile.gettempdir()) / "scenescout_preview"
    shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# Time parsing and formatting

def probe_duration(state: AppState, path: str) -> Optional[float]:
    code, out, err = run_command(
        state,
        [state.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
    )
    try:
        return max(0.0, float(json.loads(out)["format"]["duration"])) if code == 0 else None
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def parse_time(value: str) -> Optional[float]:
    """Accept seconds, MM:SS, or HH:MM:SS, including fractional seconds."""
    text = value.strip()
    if not text or not re.fullmatch(r"\d+(?::\d+){0,2}(?:\.\d+)?", text):
        return None
    try:
        parts = [float(p) for p in text.split(":")]
        if len(parts) == 1:
            return parts[0]
        if any(p < 0 for p in parts) or parts[-1] >= 60 or (len(parts) == 3 and parts[-2] >= 60):
            return None
        return parts[-1] + parts[-2] * 60 + (parts[-3] * 3600 if len(parts) == 3 else 0)
    except ValueError:
        return None


def format_time(seconds: float, millis: bool = False) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    frac = seconds - int(seconds)
    if millis:
        tail = f"{secs:02d}.{int(round(frac * 1000)):03d}"
        if tail.endswith(".1000"):
            return format_time(round(seconds), True)
    else:
        tail = f"{secs:02d}"
    return f"{hours}:{minutes:02d}:{tail}" if hours else f"{minutes}:{tail}"


def log(window: sg.Window, message: str) -> None:
    window["-LOG-"].print(f"[{time.strftime('%H:%M:%S')}] {message}")


def setting_float(settings: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def setting_int(settings: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Subtitle hints

def subtitle_hints(video_path: str, keywords: list[str], duration: float) -> list[dict[str, Any]]:
    """Read a simple numbered/timestamped SRT file using only stdlib."""
    if not keywords:
        return []
    srt = Path(video_path).with_suffix(".srt")
    if not srt.exists():
        return []
    try:
        contents = srt.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    output: list[dict[str, Any]] = []
    time_re = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,.]\d+)\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d+)")
    for block in re.split(r"\r?\n\s*\r?\n", contents):
        match = time_re.search(block)
        if not match:
            continue
        text = time_re.sub("", block).lower()
        if not any(word.lower() in text for word in keywords):
            continue
        start = parse_time(match.group(1).replace(",", "."))
        end = parse_time(match.group(2).replace(",", "."))
        if start is not None and end is not None and start < end:
            output.append(
                {
                    "enabled": True,
                    "start": max(0, start),
                    "end": min(duration, end),
                    "conf": 0.0,
                    "source": "subtitle",
                    "labels": {},
                    "hits": 0,
                    "best_t": (max(0, start) + min(duration, end)) / 2,
                    "best_score": 0.0,
                    "nudenet_max": 0.0,
                    "classifier_max": 0.0,
                    "locked": False,
                    "manual": False,
                }
            )
    return output


# ---------------------------------------------------------------------------
# Frame sampling (one extraction feeds both detectors)

def extract_chunk(
    state: AppState,
    video: str,
    chunk_start: float,
    chunk_length: float,
    interval: float,
    width: int,
    folder: Path,
    use_cuda: bool,
) -> tuple[bool, str]:
    folder.mkdir(parents=True, exist_ok=True)
    output = str(folder / "frame_%06d.jpg")
    vf = f"fps=1/{interval:g},scale={width}:-2"
    base = [state.ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-ss", f"{chunk_start:.3f}", "-t", f"{chunk_length:.3f}"]
    if use_cuda:
        base += ["-hwaccel", "cuda"]
    command = base + ["-i", video, "-vf", vf, "-q:v", "3", output]
    code, _, error = run_command(state, command)
    if code == 0:
        return True, ""
    if use_cuda and not state.cancel.is_set():
        for image in folder.glob("*.jpg"):
            image.unlink(missing_ok=True)
        command = [
            state.ffmpeg,
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{chunk_start:.3f}",
            "-t",
            f"{chunk_length:.3f}",
            "-i",
            video,
            "-vf",
            vf,
            "-q:v",
            "3",
            output,
        ]
        code, _, retry_error = run_command(state, command)
        if code == 0:
            return True, "CUDA decode failed; used CPU for this chunk."
        return False, retry_error
    return False, error


# ---------------------------------------------------------------------------
# NudeNet detector

LABELS = {
    "-GENITALIA-": {"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "EXPOSED_FEMALE_GENITALIA", "EXPOSED_MALE_GENITALIA"},
    "-BREASTS-": {"FEMALE_BREAST_EXPOSED", "EXPOSED_FEMALE_BREAST"},
    "-MALEBREAST-": {"MALE_BREAST_EXPOSED", "EXPOSED_MALE_BREAST"},
    "-BUTTOCKS-": {"BUTTOCKS_EXPOSED", "FEMALE_BUTTOCKS_EXPOSED", "MALE_BUTTOCKS_EXPOSED", "EXPOSED_BUTTOCKS"},
    "-ANUS-": {"ANUS_EXPOSED", "EXPOSED_ANUS"},
    "-BELLY-": {"BELLY_EXPOSED", "EXPOSED_BELLY"},
    "-ARMPITS-": {"ARMPITS_EXPOSED", "ARMPIT_EXPOSED", "EXPOSED_ARMPITS"},
    "-FEET-": {"FEET_EXPOSED", "EXPOSED_FEET"},
}


def active_labels(values: dict[str, Any]) -> set[str]:
    """Build an exact, uppercase label set from checked boxes only."""
    chosen: set[str] = set()
    for key, labels in LABELS.items():
        if values.get(key):
            chosen.update(label.upper() for label in labels)
    return chosen


def normal_label(detection: dict[str, Any]) -> str:
    return str(detection.get("class", detection.get("label", detection.get("category", "")))).upper().replace("-", "_").replace(" ", "_")


def detection_score(detection: dict[str, Any]) -> float:
    try:
        return float(detection.get("score", detection.get("confidence", 0)))
    except (TypeError, ValueError):
        return 0.0


def normalize_box(detection: dict[str, Any]) -> list[float]:
    box = detection.get("box", detection.get("bbox", []))
    try:
        values = [float(v) for v in list(box)[:4]]
        return values if len(values) == 4 else []
    except (TypeError, ValueError):
        return []


def box_height(box: list[float]) -> float:
    return max(0.0, float(box[3])) if len(box) == 4 else 0.0


def get_detections(detector: Any, frame: str) -> list[dict[str, Any]]:
    """Support common NudeNet detector API variants without assuming its version."""
    result = detector.detect(frame)
    if isinstance(result, dict):
        result = result.get("detections", result.get("results", []))
    return result if isinstance(result, list) else []


def discover_nudenet_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = [
        {
            "name": "320n.onnx",
            "path": None,
            "resolution": 320,
            "display": "320n (default, 320 px)",
            "key": "320n.onnx@320",
        }
    ]
    seen = {models[0]["key"]}
    package_dir: Optional[Path] = None
    try:
        spec = importlib.util.find_spec("nudenet")
        if spec and spec.origin:
            package_dir = Path(spec.origin).parent
    except Exception:
        package_dir = None
    folders: list[Path] = []
    if package_dir:
        folders.append(package_dir)
    folders.append(Path(__file__).resolve().parent / "models")
    folders.append(Path.cwd() / "models")
    for folder in folders:
        if not folder.is_dir():
            continue
        for file in sorted(folder.glob("*.onnx")):
            try:
                resolved = file.resolve()
            except OSError:
                continue
            if resolved.name.lower() == "320n.onnx":
                continue
            stem = resolved.stem.lower()
            resolution = 640 if "640" in stem else 320 if "320" in stem else 320
            key = f"{resolved.name}@{resolution}"
            if key in seen:
                continue
            seen.add(key)
            models.append(
                {
                    "name": resolved.name,
                    "path": str(resolved),
                    "resolution": resolution,
                    "display": f"{resolved.stem} ({resolution} px)",
                    "key": key,
                }
            )
    return models


def default_nudenet_model_path() -> Optional[str]:
    try:
        spec = importlib.util.find_spec("nudenet")
        if spec and spec.origin:
            candidate = Path(spec.origin).parent / "320n.onnx"
            if candidate.is_file():
                return str(candidate)
    except Exception:
        pass
    return None


def ensure_nudenet(window: sg.Window, state: AppState, display: Optional[str]) -> bool:
    model = state.nudenet_models_by_display.get(display or "") or state.nudenet_models[0]
    if state.nudenet is not None and state.nudenet_model_key == model["key"]:
        return True
    try:
        import onnxruntime as ort  # type: ignore
        from nudenet import NudeDetector  # type: ignore

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in available:
            log(window, "NudeNet: CUDAExecutionProvider not available; using CPU.")
        detector = NudeDetector(
            model_path=model["path"],
            providers=providers,
            inference_resolution=model["resolution"],
        )
        try:
            current = detector.onnx_session.get_providers()
            if "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in current:
                target = model["path"] or default_nudenet_model_path()
                if target:
                    session = ort.InferenceSession(target, providers=providers)
                    detector.onnx_session = session
                    detector.input_name = session.get_inputs()[0].name
        except Exception as exc:
            log(window, f"NudeNet GPU session setup skipped: {exc}")
        used = detector.onnx_session.get_providers()
        state.nudenet = detector
        state.nudenet_model_key = model["key"]
        state.nudenet_model_name = model["name"]
        state.nudenet_model_display = model["display"]
        state.nudenet_resolution = model["resolution"]
        state.nudenet_provider = ", ".join(used)
        state.nudenet_error = ""
        log(window, f"NudeNet model: {model['name']} ({model['resolution']} px) | provider: {state.nudenet_provider}")
        return True
    except Exception as exc:
        state.nudenet = None
        state.nudenet_error = str(exc)
        log(window, f"NudeNet unavailable: {exc}")
        return False


# ---------------------------------------------------------------------------
# Whole-frame NSFW classifier

def classifier_prereqs(model_id: str) -> tuple[bool, str]:
    if importlib.util.find_spec("torch") is None:
        return False, "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    if importlib.util.find_spec("PIL") is None:
        return False, "pip install Pillow"
    backend = CLASSIFIER_MODELS[model_id]["backend"]
    if backend == "transformers":
        if importlib.util.find_spec("transformers") is None:
            return False, "pip install transformers"
    elif backend == "moderators":
        if importlib.util.find_spec("moderators") is None and importlib.util.find_spec("transformers") is None:
            return False, "pip install moderators transformers"
    elif backend == "timm":
        if importlib.util.find_spec("timm") is None:
            return False, "pip install timm"
    return True, ""


class FrameClassifier:
    """Loads one verified NSFW classifier and batches frames on CUDA when available."""

    def __init__(self, model_id: str, log_callback) -> None:
        self.model_id = model_id
        self.meta = CLASSIFIER_MODELS[model_id]
        self.log = log_callback
        self.torch: Any = None
        self.pil_image: Any = None
        self.device: Any = None
        self.model: Any = None
        self.processor: Any = None
        self.transform: Any = None
        self.labels: list[str] = []
        self.explicit_idx: list[int] = []
        self.suggestive_idx: list[int] = []
        self.backend_used = ""
        self.load()

    def load(self) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyTorch is not installed") from exc
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError("Pillow is not installed") from exc
        self.torch = torch
        self.pil_image = Image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            try:
                self.log(f"Classifier: CUDA device {torch.cuda.get_device_name(0)}")
            except Exception:
                self.log("Classifier: CUDA device in use")
        else:
            self.log("Classifier: CUDA unavailable; falling back to CPU (slow).")

        backend = self.meta["backend"]
        if backend == "transformers":
            self._load_transformers()
        elif backend == "moderators":
            self._load_moderators()
        elif backend == "timm":
            self._load_timm()
        else:
            raise RuntimeError(f"Unknown classifier backend: {backend}")

        self._map_labels()
        self.model.to(self.device)
        self.model.eval()
        self.log(f"Classifier model: {self.model_id} | backend: {self.backend_used} | device: {self.device}")

    def _load_transformers(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification  # type: ignore
        except Exception as exc:
            raise RuntimeError("transformers is not installed") from exc
        self.log(
            f"Loading classifier {self.model_id} "
            f"(first use may download {self.meta['download']} from Hugging Face)."
        )
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        config = self.model.config
        self.labels = [str(config.id2label[i]) for i in range(len(config.id2label))]
        self.backend_used = "transformers"

    def _load_moderators(self) -> None:
        try:
            from moderators import AutoModerator  # type: ignore

            self.log(
                f"Loading classifier {self.model_id} via moderators "
                f"(first use may download {self.meta['download']} from Hugging Face)."
            )
            moderator = AutoModerator.from_pretrained(self.model_id)
            pipe = getattr(moderator, "_pipe", None)
            if pipe is None:
                raise RuntimeError("moderators pipeline was not created")
            model = getattr(pipe, "model", None)
            processor = (
                getattr(pipe, "image_processor", None)
                or getattr(pipe, "feature_extractor", None)
                or getattr(pipe, "processor", None)
            )
            if model is None or processor is None:
                raise RuntimeError("moderators pipeline has no model/processor")
            config = getattr(model, "config", None)
            if config is None or not hasattr(config, "id2label"):
                raise RuntimeError("moderators model has no id2label")
            self.model = model
            self.processor = processor
            self.labels = [str(config.id2label[i]) for i in range(len(config.id2label))]
            self.backend_used = "moderators"
        except Exception as exc:
            self.log(f"moderators load failed ({exc}); falling back to transformers.")
            self._load_transformers()

    def _load_timm(self) -> None:
        try:
            import timm  # type: ignore
        except Exception as exc:
            raise RuntimeError("timm is not installed") from exc
        self.log(
            f"Loading classifier {self.model_id} via timm "
            f"(first use may download {self.meta['download']} from Hugging Face)."
        )
        self.model = timm.create_model(f"hf_hub:{self.model_id}", pretrained=True)
        self.labels = [str(x) for x in (self.model.pretrained_cfg.get("label_names") or [])]
        if not self.labels:
            raise RuntimeError("timm model has no label_names")
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.backend_used = "timm"

    def _map_labels(self) -> None:
        normalized = [label.strip().lower() for label in self.labels]
        explicit_names = {label.lower() for label in self.meta["explicit"]}
        suggestive_names = {label.lower() for label in self.meta["suggestive"]}
        self.explicit_idx = [i for i, label in enumerate(normalized) if label in explicit_names]
        self.suggestive_idx = [i for i, label in enumerate(normalized) if label in suggestive_names]
        if not self.explicit_idx:
            raise RuntimeError(
                f"Explicit labels {sorted(explicit_names)} not found in model labels {self.labels}"
            )

    def predict(self, paths: list[str], batch_size: int, cancel: threading.Event, log_callback) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = [
            {"explicit": 0.0, "suggestive": 0.0, "labels": {}} for _ in paths
        ]
        if self.model is None:
            return results
        size = max(1, int(batch_size))
        for start in range(0, len(paths), size):
            if cancel.is_set():
                break
            batch_paths = paths[start : start + size]
            try:
                images = []
                valid_indices: list[int] = []
                for offset, path in enumerate(batch_paths):
                    try:
                        with self.pil_image.open(path) as image:
                            images.append(image.convert("RGB"))
                        valid_indices.append(offset)
                    except Exception as exc:
                        log_callback(f"Classifier skipped {Path(path).name}: {exc}")
                if not images:
                    continue
                with self.torch.inference_mode():
                    if self.backend_used == "timm":
                        tensors = self.torch.stack([self.transform(image) for image in images]).to(self.device)
                        if self.device.type == "cuda":
                            with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                                logits = self.model(tensors)
                        else:
                            logits = self.model(tensors)
                    else:
                        inputs = self.processor(images=images, return_tensors="pt")
                        inputs = {key: value.to(self.device) for key, value in inputs.items()}
                        if self.device.type == "cuda":
                            with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                                logits = self.model(**inputs).logits
                        else:
                            logits = self.model(**inputs).logits
                    probabilities = self.torch.softmax(logits.float(), dim=-1).cpu()
                for row, offset in enumerate(valid_indices):
                    probs = probabilities[row]
                    label_probs = {
                        str(self.labels[index]): float(probs[index])
                        for index in range(len(self.labels))
                    }
                    explicit = sum(float(probs[index]) for index in self.explicit_idx)
                    suggestive = (
                        sum(float(probs[index]) for index in self.suggestive_idx)
                        if self.suggestive_idx
                        else 0.0
                    )
                    results[start + offset] = {
                        "explicit": explicit,
                        "suggestive": suggestive,
                        "labels": label_probs,
                    }
            except Exception as exc:
                log_callback(f"Classifier batch failed: {exc}")
        return results


def ensure_classifier(window: sg.Window, state: AppState, model_id: str, enabled: bool) -> bool:
    if not enabled:
        state.classifier = None
        state.classifier_model_id = "none"
        state.classifier_device = "disabled"
        return True
    if state.classifier is not None and state.classifier_model_id == model_id:
        return True
    try:
        classifier = FrameClassifier(model_id, lambda message: log(window, message))
        state.classifier = classifier
        state.classifier_model_id = model_id
        state.classifier_device = str(classifier.device)
        state.classifier_error = ""
        if model_id not in state.classifier_notes:
            state.classifier_notes.add(model_id)
            meta = CLASSIFIER_MODELS[model_id]
            log(window, f"Classifier license/size: {meta['license']} | {meta['size']} | {model_id}")
        return True
    except Exception as exc:
        state.classifier = None
        state.classifier_model_id = "none"
        state.classifier_device = "unavailable"
        state.classifier_error = str(exc)
        log(window, f"Frame classifier unavailable: {exc}")
        log(window, "Install classifier dependencies: pip install torch torchvision transformers timm moderators Pillow")
        return False


# ---------------------------------------------------------------------------
# Raw-results cache

def cache_path(key: str) -> Path:
    return app_cache_dir() / f"{key}.json"


def build_cache_key(
    path: str,
    interval: float,
    width: int,
    nudenet_model: str,
    classifier_model: str,
) -> Optional[str]:
    try:
        stat = Path(path).stat()
        material = (
            f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{float(interval):g}|{int(width)}|{nudenet_model}|{classifier_model}"
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()
    except OSError:
        return None


def cache_matches(
    meta: dict[str, Any],
    path: str,
    interval: float,
    width: int,
    nudenet_model: str,
    classifier_model: str,
) -> bool:
    try:
        stat = Path(path).stat()
        return (
            int(meta.get("version", -1)) == CACHE_VERSION
            and Path(str(meta.get("path", ""))).resolve() == Path(path).resolve()
            and int(meta.get("size", -1)) == stat.st_size
            and int(meta.get("mtime_ns", -1)) == stat.st_mtime_ns
            and abs(float(meta.get("interval", -1)) - float(interval)) < 1e-9
            and int(meta.get("width", -1)) == int(width)
            and str(meta.get("nudenet_model", "")) == nudenet_model
            and str(meta.get("classifier_model", "")) == classifier_model
        )
    except Exception:
        return False


def load_cache(key: Optional[str]) -> Optional[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if not key:
        return None
    target = cache_path(key)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        meta = data.get("meta")
        frames = data.get("frames")
        if not isinstance(meta, dict) or not isinstance(frames, list):
            return None
        return meta, frames
    except Exception:
        return None


def save_cache(key: Optional[str], meta: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    if not key:
        return
    try:
        folder = app_cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        target = cache_path(key)
        temporary = target.with_suffix(".tmp")
        payload = json.dumps({"meta": meta, "frames": frames}, ensure_ascii=False, separators=(",", ":"))
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Merging, hit filtering and shot-boundary snapping

def combine_sources(first: str, second: str) -> str:
    if first == second:
        return first
    if "manual" in (first, second):
        return "manual"
    if first == "subtitle":
        return second
    if second == "subtitle":
        return first
    if "both" in (first, second):
        return "both"
    if {first, second} == {"nudenet", "classifier"}:
        return "both"
    return first


def merge_candidates(items: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: item["start"])
    merged: list[dict[str, Any]] = []
    for original in ordered:
        item = dict(original)
        item["start"] = max(0.0, min(duration, float(item["start"])))
        item["end"] = max(0.0, min(duration, float(item["end"])))
        if item["start"] >= item["end"]:
            continue
        item.setdefault("enabled", True)
        item.setdefault("conf", 0.0)
        item.setdefault("source", "visual")
        item.setdefault("labels", {})
        item.setdefault("hits", 0)
        item.setdefault("best_t", (item["start"] + item["end"]) / 2)
        item.setdefault("best_score", item.get("conf", 0.0))
        item.setdefault("nudenet_max", 0.0)
        item.setdefault("classifier_max", 0.0)
        item.setdefault("locked", False)
        item.setdefault("manual", False)
        if merged and item["start"] <= merged[-1]["end"]:
            prior = merged[-1]
            prior["end"] = max(prior["end"], item["end"])
            prior["conf"] = max(float(prior["conf"]), float(item["conf"]))
            prior["enabled"] = bool(prior["enabled"]) or bool(item["enabled"])
            prior["source"] = combine_sources(str(prior["source"]), str(item["source"]))
            prior["hits"] = int(prior.get("hits", 0)) + int(item.get("hits", 0))
            prior["nudenet_max"] = max(float(prior.get("nudenet_max", 0.0)), float(item.get("nudenet_max", 0.0)))
            prior["classifier_max"] = max(float(prior.get("classifier_max", 0.0)), float(item.get("classifier_max", 0.0)))
            prior["locked"] = bool(prior.get("locked")) or bool(item.get("locked"))
            prior["manual"] = bool(prior.get("manual")) or bool(item.get("manual"))
            labels = dict(prior.get("labels") or {})
            for label, score in (item.get("labels") or {}).items():
                labels[label] = max(float(labels.get(label, 0.0)), float(score))
            prior["labels"] = labels
            if float(item.get("best_score", 0.0)) > float(prior.get("best_score", 0.0)):
                prior["best_score"] = float(item.get("best_score", 0.0))
                prior["best_t"] = float(item.get("best_t", (item["start"] + item["end"]) / 2))
        else:
            merged.append(item)
    return merged


def scene_boundaries(state: AppState, video: str) -> list[float]:
    command = [
        state.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-i",
        video,
        "-vf",
        "select='gt(scene,0.3)',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    code, _, error = run_command(state, command)
    if code != 0:
        return []
    return [float(x) for x in re.findall(r"pts_time:([0-9.]+)", error)]


def snap_ranges(items: list[dict[str, Any]], boundaries: list[float], duration: float) -> list[dict[str, Any]]:
    for item in items:
        start, end = item["start"], item["end"]
        before = [x for x in boundaries if start - 3 <= x <= start]
        after = [x for x in boundaries if end <= x <= end + 3]
        near_start = before[-1] if before else next((x for x in boundaries if start <= x <= start + 3), start)
        near_end = after[0] if after else next((x for x in reversed(boundaries) if end - 3 <= x <= end), end)
        if near_start <= start + 3:
            item["start"] = near_start
        if near_end >= end - 3:
            item["end"] = near_end
        item["start"] = max(0, item["start"])
        item["end"] = min(duration, item["end"])
    return merge_candidates(items, duration)


def merge_windows(windows: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for start, end in windows:
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end > start:
            cleaned.append((start, end))
    cleaned.sort()
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def inherited_enabled(candidate: dict[str, Any], old_results: list[dict[str, Any]]) -> bool:
    best: Optional[dict[str, Any]] = None
    best_overlap = 0.0
    for old in old_results:
        if old.get("locked"):
            continue
        overlap = min(candidate["end"], old["end"]) - max(candidate["start"], old["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best = old
    return bool(best["enabled"]) if best else True


def merge_with_locked(
    auto_results: list[dict[str, Any]],
    old_results: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    locked = [dict(item) for item in old_results if item.get("locked")]
    unlocked = [item for item in old_results if not item.get("locked")]
    if not locked:
        for candidate in auto_results:
            candidate["enabled"] = inherited_enabled(candidate, unlocked)
        return auto_results

    segments: list[dict[str, Any]] = []
    for candidate in auto_results:
        remaining = [(candidate["start"], candidate["end"])]
        for lock in locked:
            next_remaining: list[tuple[float, float]] = []
            for start, end in remaining:
                if end <= lock["start"] or start >= lock["end"]:
                    next_remaining.append((start, end))
                else:
                    if start < lock["start"]:
                        next_remaining.append((start, lock["start"]))
                    if end > lock["end"]:
                        next_remaining.append((lock["end"], end))
            remaining = next_remaining
        for start, end in remaining:
            if end - start > 0.05:
                segment = dict(candidate)
                segment["start"] = start
                segment["end"] = end
                segment["enabled"] = inherited_enabled(segment, unlocked)
                segments.append(segment)
    final = segments + [dict(item) for item in locked]
    final.sort(key=lambda item: (item["start"], item["end"]))
    return final


# ---------------------------------------------------------------------------
# Hit detection and candidate analysis

def frame_hit_info(
    frame: dict[str, Any],
    settings: dict[str, Any],
    labels: set[str],
    frame_height: float,
) -> Optional[dict[str, Any]]:
    threshold = setting_float(settings, "-THRESH-", 0.35)
    min_box = setting_float(settings, "-MINBOX-", 0.08)
    explicit_threshold = setting_float(settings, "-EXPLTHR-", 0.60)
    suggestive_threshold = setting_float(settings, "-SUGGTHR-", 0.70)
    include_suggestive = bool(settings.get("-SUGG-", False))

    label_scores: dict[str, float] = {}
    nudenet_max = 0.0
    for detection in frame.get("nudenet", []) or []:
        label = str(detection.get("label", "")).upper()
        if label not in labels:
            continue
        try:
            score = float(detection.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if score < threshold:
            continue
        box = detection.get("box") or []
        height = box_height(box)
        if frame_height > 0 and min_box > 0:
            limit = min_box * frame_height if height > 1.5 else min_box
            if height < limit:
                continue
        nudenet_max = max(nudenet_max, score)
        label_scores[label] = max(label_scores.get(label, 0.0), score)

    explicit = float(frame.get("explicit", 0.0) or 0.0)
    suggestive = float(frame.get("suggestive", 0.0) or 0.0)
    classifier_hit = explicit >= explicit_threshold or (
        include_suggestive and suggestive >= suggestive_threshold
    )
    if not label_scores and not classifier_hit:
        return None

    classifier_max = max(explicit, suggestive) if classifier_hit else 0.0
    if label_scores and classifier_hit:
        source = "both"
    elif label_scores:
        source = "nudenet"
    else:
        source = "classifier"
    return {
        "score": max(nudenet_max, classifier_max),
        "nudenet_max": nudenet_max,
        "classifier_max": classifier_max,
        "labels": label_scores,
        "source": source,
        "dt": float(frame.get("dt", 1.0)),
    }


def hit_frames_from_raw(
    frames: list[dict[str, Any]],
    settings: dict[str, Any],
    labels: set[str],
    pass_number: int,
    frame_height: float,
) -> dict[float, dict[str, Any]]:
    hits: dict[float, dict[str, Any]] = {}
    for frame in frames:
        try:
            if int(frame.get("pass", 1)) != pass_number:
                continue
            info = frame_hit_info(frame, settings, labels, frame_height)
            if not info:
                continue
            timestamp = round(float(frame.get("t", 0.0)), 3)
            existing = hits.get(timestamp)
            if existing is None:
                hits[timestamp] = info
                continue
            existing["score"] = max(float(existing["score"]), float(info["score"]))
            existing["nudenet_max"] = max(float(existing["nudenet_max"]), float(info["nudenet_max"]))
            existing["classifier_max"] = max(float(existing["classifier_max"]), float(info["classifier_max"]))
            existing["source"] = combine_sources(str(existing["source"]), str(info["source"]))
            for label, score in info["labels"].items():
                existing["labels"][label] = max(float(existing["labels"].get(label, 0.0)), float(score))
            existing["dt"] = min(float(existing.get("dt", info["dt"])), float(info["dt"]))
        except (TypeError, ValueError):
            continue
    return hits


def select_hit_frames(
    frames: list[dict[str, Any]],
    meta: dict[str, Any],
    settings: dict[str, Any],
    labels: set[str],
) -> dict[float, dict[str, Any]]:
    frame_height = float(meta.get("frame_height") or 0.0)
    interval = float(meta.get("interval") or 1.0)
    duration = float(meta.get("duration") or max([float(f.get("t", 0.0)) for f in frames] or [0.0]))
    pass1 = hit_frames_from_raw(frames, settings, labels, 1, frame_height)
    has_pass2 = any(int(frame.get("pass", 1)) == 2 for frame in frames)
    if not meta.get("refine") or not has_pass2:
        return pass1

    pass2 = hit_frames_from_raw(frames, settings, labels, 2, frame_height)
    min_hits = setting_int(settings, "-MINHITS-", 2)
    windows = merge_windows([(timestamp - interval, timestamp + interval) for timestamp in pass1], duration)
    selected: dict[float, dict[str, Any]] = {}
    for start, end in windows:
        fine = {t: value for t, value in pass2.items() if start - 1e-6 <= t <= end + 1e-6}
        coarse = {t: value for t, value in pass1.items() if start - 1e-6 <= t <= end + 1e-6}
        fine_strong = len(fine) >= min_hits or any(float(v["score"]) >= HIGH_SCORE for v in fine.values())
        if fine and fine_strong:
            selected.update(fine)
        elif coarse:
            selected.update(coarse)
        elif fine:
            selected.update(fine)
    for timestamp, value in pass1.items():
        if not any(start - 1e-6 <= timestamp <= end + 1e-6 for start, end in windows):
            selected[timestamp] = value
    return selected


def build_candidates_from_hits(
    hit_map: dict[float, dict[str, Any]],
    settings: dict[str, Any],
    duration: float,
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    merge_gap = setting_int(settings, "-GAP-", 12)
    before = setting_int(settings, "-BEFORE-", 3)
    after = setting_int(settings, "-AFTER-", 3)
    interval = float(meta.get("interval") or 1.0)

    timestamps = sorted(hit_map)
    clusters: list[list[float]] = []
    current: list[float] = []
    for timestamp in timestamps:
        if current and timestamp - current[-1] > merge_gap:
            clusters.append(current)
            current = []
        current.append(timestamp)
    if current:
        clusters.append(current)

    candidates: list[dict[str, Any]] = []
    for times in clusters:
        labels: dict[str, float] = {}
        sources: set[str] = set()
        best_t = times[0]
        best_score = 0.0
        nudenet_max = 0.0
        classifier_max = 0.0
        dt = interval
        for timestamp in times:
            info = hit_map[timestamp]
            for label, score in info["labels"].items():
                labels[label] = max(labels.get(label, 0.0), float(score))
            nudenet_max = max(nudenet_max, float(info["nudenet_max"]))
            classifier_max = max(classifier_max, float(info["classifier_max"]))
            sources.add(str(info["source"]))
            if float(info["score"]) > best_score:
                best_score = float(info["score"])
                best_t = timestamp
            dt = min(dt, float(info.get("dt", dt)))
        if "both" in sources or {"nudenet", "classifier"} <= sources:
            source = "both"
        elif sources:
            source = next(iter(sources))
        else:
            source = "nudenet"
        first, last = times[0], times[-1]
        start = max(0.0, first - before)
        end = min(duration, last + after + dt)
        if end <= start:
            continue
        candidates.append(
            {
                "enabled": True,
                "start": start,
                "end": end,
                "conf": best_score,
                "source": source,
                "labels": labels,
                "hits": len(times),
                "best_t": best_t,
                "best_score": best_score,
                "nudenet_max": nudenet_max,
                "classifier_max": classifier_max,
                "locked": False,
                "manual": False,
            }
        )
    return candidates, len(candidates)


def analyze_raw_to_candidates(
    frames: list[dict[str, Any]],
    meta: dict[str, Any],
    settings: dict[str, Any],
    path: str,
    duration: float,
    old_results: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = active_labels(settings)
    hit_map = select_hit_frames(frames, meta, settings, labels)
    visual, before_filter = build_candidates_from_hits(hit_map, settings, duration, meta)
    min_hits = setting_int(settings, "-MINHITS-", 2)
    filtered = [
        candidate
        for candidate in visual
        if int(candidate.get("hits", 0)) >= min_hits or float(candidate.get("conf", 0.0)) >= HIGH_SCORE
    ]

    keywords = (
        [word.strip() for word in re.split(r"[,;\n]+", str(settings.get("-KEYWORDS-", ""))) if word.strip()]
        if settings.get("-SRT-")
        else []
    )
    subtitles = subtitle_hints(path, keywords, duration) if keywords else []
    merged = merge_candidates(filtered + subtitles, duration)

    warnings: list[str] = []
    if settings.get("-SNAP-") and merged:
        boundaries = meta.get("boundaries") or []
        if boundaries:
            merged = snap_ranges(merged, boundaries, duration)
        else:
            warnings.append("No cached scene boundaries; used unsnapped ranges.")

    if old_results:
        merged = merge_with_locked(merged, old_results, duration)

    hits_nudenet = sum(1 for value in hit_map.values() if value["source"] in ("nudenet", "both"))
    hits_classifier = sum(1 for value in hit_map.values() if value["source"] in ("classifier", "both"))
    hits_both = sum(1 for value in hit_map.values() if value["source"] == "both")
    stats = {
        "frames_analyzed": int(meta.get("frames_analyzed", len(frames))),
        "hits": len(hit_map),
        "hits_nudenet": hits_nudenet,
        "hits_classifier": hits_classifier,
        "hits_both": hits_both,
        "before": before_filter,
        "after": len(merged),
        "warnings": warnings,
    }
    return merged, stats


# ---------------------------------------------------------------------------
# Background workers

def video_entry(path: str, duration: float) -> dict[str, Any]:
    return {
        "path": path,
        "duration": duration,
        "results": [],
        "stats": {},
        "cache_key": "",
        "meta": {},
    }


def set_running(window: sg.Window, state: AppState, running: bool, kind: str = "") -> None:
    state.running = running
    state.job_kind = kind
    window["-SCAN-"].update(disabled=running)
    window["-REANALYZE-"].update(disabled=running or not any(v.get("cache_key") for v in state.videos))
    window["-STOP-"].update(disabled=not running)
    window["-EXPORT-"].update(disabled=running)
    window["-EXPORTALL-"].update(disabled=running)
    if not running:
        window["-PROGRESS-"].update(0)
        window["-STATUS-"].update("")


def extract_frame_at(state: AppState, video: str, timestamp: float, folder: Path, width: int) -> Optional[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"frame_{int(timestamp * 1000):012d}.jpg"
    command = [
        state.ffmpeg,
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp):.3f}",
        "-i",
        video,
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        "2",
        str(target),
    ]
    code, _, error = run_command(state, command)
    if code == 0 and target.is_file():
        return target
    log_error = error.strip().splitlines()[-1] if error.strip() else "unknown ffmpeg error"
    return None


def _run_scan_impl(window: sg.Window, state: AppState, values: dict[str, Any]) -> None:
    started = time.time()
    state.cancel.clear()
    cleanup_temp(state)
    state.temp_root = Path(tempfile.mkdtemp(prefix="scenescout_"))
    paths = [entry["path"] for entry in state.videos]
    try:
        interval = max(0.1, float(values.get("-INTERVAL-", 1.0)))
    except (TypeError, ValueError):
        interval = 1.0
    try:
        width = max(160, int(values.get("-WIDTH-", 960)))
    except (TypeError, ValueError):
        width = 960
    use_cuda = bool(values.get("-CUDA-", True))
    use_classifier = bool(values.get("-CLS-", False))
    classifier_model = str(values.get("-CLSMODEL-", DEFAULT_CLASSIFIER))
    use_nudenet = ensure_nudenet(window, state, values.get("-NNMODEL-"))
    if not use_nudenet and not use_classifier:
        cleanup_temp(state)
        window.write_event_value("-ERROR-", "NudeNet is unavailable and the frame classifier is disabled.")
        return
    if use_classifier and not ensure_classifier(window, state, classifier_model, True):
        if not use_nudenet:
            cleanup_temp(state)
            window.write_event_value("-ERROR-", "Neither detector is available.")
            return
        log(window, "Continuing with NudeNet only.")
        use_classifier = False

    if not use_nudenet:
        log(window, "NudeNet unavailable; scanning with the whole-frame classifier only.")
    if not use_classifier:
        log(window, "Whole-frame classifier disabled; scanning with NudeNet only.")

    log(
        window,
        f"Scan started: {len(paths)} video(s), interval {interval:g}s, width {width}px, "
        f"CUDA decode {'on' if use_cuda else 'off'}, classifier {'on' if use_classifier else 'off'}.",
    )
    overall_frames = 0
    for video_index, path in enumerate(paths):
        if state.cancel.is_set():
            break
        duration = probe_duration(state, path)
        if duration is None:
            log(window, f"Skipping unreadable video: {path}")
            continue
        window["-STATUS-"].update(f"Scanning {Path(path).name} ({video_index + 1}/{len(paths)})")
        classifier_id = classifier_model if use_classifier else "none"
        nudenet_key = state.nudenet_model_key if use_nudenet else "none"
        key = build_cache_key(path, interval, width, nudenet_key, classifier_id)
        cached = load_cache(key)
        frames: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}
        used_cache = False
        if cached and cache_matches(cached[0], path, interval, width, nudenet_key, classifier_id):
            meta, frames = cached
            used_cache = True
            log(window, f"Using matching cache for {Path(path).name} ({len(frames)} raw frames).")
        else:
            if cached:
                log(window, f"Cache mismatch for {Path(path).name}; rescanning.")

            boundaries: list[float] = []
            if values.get("-SNAP-"):
                window["-STATUS-"].update(f"Detecting scene boundaries: {Path(path).name}")
                boundaries = scene_boundaries(state, path)
                log(window, f"Scene boundaries: {len(boundaries)}")

            frames = []
            frame_height = 0.0
            chunk_total = max(1, int((duration + CHUNK_SECONDS - 1) // CHUNK_SECONDS))
            for chunk_index, chunk_start in enumerate(
                [i * CHUNK_SECONDS for i in range(chunk_total)]
            ):
                if state.cancel.is_set():
                    break
                chunk_length = min(CHUNK_SECONDS, duration - chunk_start)
                if chunk_length <= 0:
                    break
                folder = state.temp_root / f"v{video_index}_p1_c{chunk_index}"
                ok, message = extract_chunk(
                    state, path, chunk_start, chunk_length, interval, width, folder, use_cuda
                )
                if message:
                    log(window, message)
                if not ok:
                    log(window, f"Chunk extraction failed at {format_time(chunk_start)}: {message}")
                    shutil.rmtree(folder, ignore_errors=True)
                    continue
                files = sorted(folder.glob("*.jpg"))
                if not files:
                    shutil.rmtree(folder, ignore_errors=True)
                    continue
                if frame_height <= 0:
                    dimensions = image_dimensions(str(files[0]))
                    if dimensions:
                        frame_height = float(dimensions[1])
                classifier_results: list[dict[str, Any]] = []
                if use_classifier and state.classifier is not None:
                    window["-STATUS-"].update(
                        f"Classifier: {Path(path).name} chunk {chunk_index + 1}/{chunk_total}"
                    )
                    classifier_results = state.classifier.predict(
                        [str(file) for file in files],
                        setting_int(values, "-BATCH-", 32),
                        state.cancel,
                        lambda message: log(window, message),
                    )
                for frame_index, file in enumerate(files):
                    if state.cancel.is_set():
                        break
                    timestamp = chunk_start + frame_index * interval
                    detections: list[dict[str, Any]] = []
                    if use_nudenet and state.nudenet is not None:
                        try:
                            detections = [
                                {
                                    "label": normal_label(detection),
                                    "score": detection_score(detection),
                                    "box": normalize_box(detection),
                                }
                                for detection in get_detections(state.nudenet, str(file))
                                if detection_score(detection) >= RAW_MIN_SCORE
                            ]
                        except Exception as exc:
                            log(window, f"NudeNet frame error at {format_time(timestamp)}: {exc}")
                    prediction = (
                        classifier_results[frame_index]
                        if frame_index < len(classifier_results)
                        else {"explicit": 0.0, "suggestive": 0.0, "labels": {}}
                    )
                    frames.append(
                        {
                            "t": round(timestamp, 3),
                            "pass": 1,
                            "dt": interval,
                            "nudenet": detections,
                            "explicit": float(prediction.get("explicit", 0.0)),
                            "suggestive": float(prediction.get("suggestive", 0.0)),
                            "classifier_labels": prediction.get("labels", {}),
                        }
                    )
                overall_frames += len(files)
                shutil.rmtree(folder, ignore_errors=True)
                progress = int(70 * (chunk_index + 1) / chunk_total)
                window["-PROGRESS-"].update(progress)
                window["-STATUS-"].update(
                    f"{Path(path).name}: pass 1 chunk {chunk_index + 1}/{chunk_total} | "
                    f"{overall_frames} frames"
                )

            if state.cancel.is_set():
                break

            refine = bool(values.get("-REFINE-", True)) and interval > FINE_INTERVAL + 1e-9
            if refine and not state.cancel.is_set():
                provisional_height = frame_height
                if provisional_height <= 0:
                    provisional_height = float(width) * 9 / 16
                labels = active_labels(values)
                pass1_hits = hit_frames_from_raw(frames, values, labels, 1, provisional_height)
                windows = merge_windows(
                    [(timestamp - interval, timestamp + interval) for timestamp in pass1_hits],
                    duration,
                )
                if windows:
                    log(window, f"Refine pass: {len(windows)} window(s) at {FINE_INTERVAL:g}s.")
                for window_index, (start, end) in enumerate(windows):
                    if state.cancel.is_set():
                        break
                    folder = state.temp_root / f"v{video_index}_p2_w{window_index}"
                    ok, message = extract_chunk(
                        state,
                        path,
                        start,
                        end - start,
                        FINE_INTERVAL,
                        width,
                        folder,
                        use_cuda,
                    )
                    if message:
                        log(window, message)
                    if not ok:
                        log(window, f"Refine extraction failed at {format_time(start)}: {message}")
                        shutil.rmtree(folder, ignore_errors=True)
                        continue
                    files = sorted(folder.glob("*.jpg"))
                    classifier_results = []
                    if use_classifier and state.classifier is not None and files:
                        classifier_results = state.classifier.predict(
                            [str(file) for file in files],
                            setting_int(values, "-BATCH-", 32),
                            state.cancel,
                            lambda message: log(window, message),
                        )
                    for frame_index, file in enumerate(files):
                        if state.cancel.is_set():
                            break
                        timestamp = start + frame_index * FINE_INTERVAL
                        detections = []
                        if use_nudenet and state.nudenet is not None:
                            try:
                                detections = [
                                    {
                                        "label": normal_label(detection),
                                        "score": detection_score(detection),
                                        "box": normalize_box(detection),
                                    }
                                    for detection in get_detections(state.nudenet, str(file))
                                    if detection_score(detection) >= RAW_MIN_SCORE
                                ]
                            except Exception as exc:
                                log(window, f"NudeNet frame error at {format_time(timestamp)}: {exc}")
                        prediction = (
                            classifier_results[frame_index]
                            if frame_index < len(classifier_results)
                            else {"explicit": 0.0, "suggestive": 0.0, "labels": {}}
                        )
                        frames.append(
                            {
                                "t": round(timestamp, 3),
                                "pass": 2,
                                "dt": FINE_INTERVAL,
                                "nudenet": detections,
                                "explicit": float(prediction.get("explicit", 0.0)),
                                "suggestive": float(prediction.get("suggestive", 0.0)),
                                "classifier_labels": prediction.get("labels", {}),
                            }
                        )
                    overall_frames += len(files)
                    shutil.rmtree(folder, ignore_errors=True)
                    progress = 70 + int(25 * (window_index + 1) / max(1, len(windows)))
                    window["-PROGRESS-"].update(progress)

            if state.cancel.is_set():
                break

            meta = {
                "version": CACHE_VERSION,
                "path": str(Path(path).resolve()),
                "size": Path(path).stat().st_size,
                "mtime_ns": Path(path).stat().st_mtime_ns,
                "interval": interval,
                "width": width,
                "duration": duration,
                "nudenet_model": nudenet_key,
                "classifier_model": classifier_id,
                "refine": bool(refine),
                "boundaries": boundaries,
                "frame_height": frame_height,
                "frames_analyzed": len(frames),
                "created": time.time(),
            }
            save_cache(key, meta, frames)

        if state.cancel.is_set():
            break
        old_results = next(
            (entry.get("results", []) for entry in state.videos if entry["path"] == path), []
        )
        results, stats = analyze_raw_to_candidates(
            frames, meta, values, path, duration, old_results
        )
        entry = next((item for item in state.videos if item["path"] == path), None)
        if entry is None:
            entry = video_entry(path, duration)
            state.videos.append(entry)
        entry["duration"] = duration
        entry["results"] = results
        entry["stats"] = stats
        entry["cache_key"] = key or ""
        entry["meta"] = meta
        for warning in stats.get("warnings", []):
            log(window, warning)
        log(
            window,
            f"{Path(path).name}: {stats['after']} candidate(s), {stats['hits']} hit frame(s) "
            f"(NudeNet {stats['hits_nudenet']}, classifier {stats['hits_classifier']}, both {stats['hits_both']})."
        )
        if used_cache:
            log(window, f"{Path(path).name}: reused cached raw detections; no video decoding.")
        window["-PROGRESS-"].update(100)
        window.write_event_value("-RESULTS-", {"path": path, "select_path": path})

    cleanup_temp(state)
    elapsed = time.time() - started
    log(window, f"Scan finished in {elapsed:.1f}s ({overall_frames} new frames).")
    window.write_event_value("-DONE-", {"cancelled": state.cancel.is_set()})


def run_scan(window: sg.Window, state: AppState, values: dict[str, Any]) -> None:
    try:
        _run_scan_impl(window, state, values)
    except Exception as exc:
        cleanup_temp(state)
        log(window, f"Scan failed: {exc}")
        window.write_event_value("-ERROR-", f"Scan failed: {exc}")
        window.write_event_value("-DONE-", {"cancelled": True})


def _run_reanalyze_impl(window: sg.Window, state: AppState, values: dict[str, Any]) -> None:
    state.cancel.clear()
    log(window, "Re-analyzing cached raw detections (video files will not be read).")
    window["-PROGRESS-"].update(0)
    updated = 0
    for index, entry in enumerate(state.videos):
        if state.cancel.is_set():
            break
        path = entry["path"]
        duration = float(entry.get("duration") or probe_duration(state, path) or 0.0)
        key = entry.get("cache_key") or ""
        cached = load_cache(key)
        if not cached:
            log(window, f"No cache for {Path(path).name}; run Scan first.")
            continue
        meta, frames = cached
        cached_interval = float(meta.get("interval", 1.0) or 1.0)
        cached_width = int(meta.get("width", 960) or 960)
        if not cache_matches(
            meta,
            path,
            cached_interval,
            cached_width,
            str(meta.get("nudenet_model", "")),
            str(meta.get("classifier_model", "")),
        ):
            log(window, f"Cache for {Path(path).name} is stale (video changed or incompatible); run Scan.")
            continue
        current_interval = float(values.get("-INTERVAL-", cached_interval) or cached_interval)
        current_width = int(values.get("-WIDTH-", cached_width) or cached_width)
        if abs(current_interval - cached_interval) > 1e-9 or current_width != cached_width:
            log(
                window,
                f"{Path(path).name}: using cached sampling ({cached_interval:g}s, {cached_width}px); "
                "change interval/width requires a new Scan.",
            )
        results, stats = analyze_raw_to_candidates(
            frames, meta, values, path, duration, entry.get("results", [])
        )
        entry["results"] = results
        entry["stats"] = stats
        entry["meta"] = meta
        updated += 1
        for warning in stats.get("warnings", []):
            log(window, warning)
        log(
            window,
            f"{Path(path).name}: re-analyzed {stats['frames_analyzed']} cached frame(s), "
            f"{stats['after']} candidate(s).",
        )
        window["-PROGRESS-"].update(int(100 * (index + 1) / max(1, len(state.videos))))
        window.write_event_value("-RESULTS-", {"path": path, "select_path": path})
    log(window, f"Cache re-analysis finished: {updated} video(s).")
    window.write_event_value("-DONE-", {"cancelled": state.cancel.is_set()})


def run_reanalyze(window: sg.Window, state: AppState, values: dict[str, Any]) -> None:
    try:
        _run_reanalyze_impl(window, state, values)
    except Exception as exc:
        log(window, f"Cache re-analysis failed: {exc}")
        window.write_event_value("-ERROR-", f"Cache re-analysis failed: {exc}")
        window.write_event_value("-DONE-", {"cancelled": True})


# ---------------------------------------------------------------------------
# Export and preview

def export_one(state: AppState, entry: dict[str, Any]) -> Optional[Path]:
    source = Path(entry["path"])
    folder = source.parent
    stem = source.stem + "_scenes"
    target = folder / f"{stem}.txt"
    counter = 1
    while target.exists():
        target = folder / f"{stem}_{counter}.txt"
        counter += 1
    lines = [f"# duration={float(entry['duration']):.3f}"]
    for item in entry.get("results", []):
        if item.get("enabled", True):
            lines.append(f"{format_time(float(item['start']), True)}-{format_time(float(item['end']), True)}")
    try:
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target
    except OSError as exc:
        sg.popup_error(f"Could not write {target}:\n{exc}", title="SceneScout")
        return None


def show_frame(window: sg.Window, state: AppState, entry: dict[str, Any], row: Optional[dict[str, Any]]) -> None:
    if not state.ffmpeg:
        sg.popup_error("FFmpeg was not found.", title="SceneScout")
        return
    timestamp = float(row["best_t"]) if row and row.get("best_t") is not None else 0.0
    folder = Path(tempfile.gettempdir()) / "scenescout_preview"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    frame = extract_frame_at(state, entry["path"], timestamp, folder, 1280)
    if not frame:
        sg.popup_error(f"Could not extract a frame at {format_time(timestamp)}.", title="SceneScout")
        return

    boxes: list[dict[str, Any]] = []
    try:
        cached = load_cache(entry.get("cache_key", ""))
        if cached:
            frames = cached[1]
            nearest = min(
                frames,
                key=lambda item: abs(float(item.get("t", 0.0)) - timestamp),
                default=None,
            )
            if nearest and abs(float(nearest.get("t", 0.0)) - timestamp) <= max(1.0, float(entry.get("meta", {}).get("interval", 1.0))):
                boxes = nearest.get("nudenet", [])
    except Exception:
        boxes = []

    if boxes:
        try:
            from PIL import Image, ImageDraw  # type: ignore

            with Image.open(frame) as image:
                canvas = image.convert("RGB")
                draw = ImageDraw.Draw(canvas)
                width, height = canvas.size
                cached_width = float(entry.get("meta", {}).get("width", 0) or 0)
                cached_height = float(entry.get("meta", {}).get("frame_height", 0) or 0)
                for detection in boxes:
                    box = detection.get("box") or []
                    if len(box) != 4:
                        continue
                    x, y, box_width, box_height_value = [float(v) for v in box]
                    if max(x, y, box_width, box_height_value) > 1.5:
                        scale_x = width / cached_width if cached_width > 0 else 1.0
                        scale_y = height / cached_height if cached_height > 0 else 1.0
                        left, top = x * scale_x, y * scale_y
                        right = (x + box_width) * scale_x
                        bottom = (y + box_height_value) * scale_y
                    else:
                        left, top = x * width, y * height
                        right, bottom = (x + box_width) * width, (y + box_height_value) * height
                    draw.rectangle([left, top, right, bottom], outline=ACCENT, width=4)
                    text = f"{detection.get('label', '')} {float(detection.get('score', 0.0)):.2f}"
                    draw.text((left + 5, max(0, top - 14)), text, fill="#ffffff")
                annotated = folder / "annotated.jpg"
                canvas.save(annotated, quality=92)
                frame = annotated
        except Exception:
            pass
    open_path(window, frame)
    log(window, f"Opened frame at {format_time(timestamp)} for {Path(entry['path']).name}.")


# ---------------------------------------------------------------------------
# GUI helpers

SOURCE_LABELS = {
    "nudenet": "NudeNet",
    "classifier": "Classifier",
    "both": "Both",
    "subtitle": "Subtitle",
    "manual": "Manual",
}


def active_entry(state: AppState) -> Optional[dict[str, Any]]:
    path = getattr(state, "active_path", "")
    for entry in state.videos:
        if entry["path"] == path:
            return entry
    return state.videos[0] if state.videos else None


def selected_result(state: AppState, values: dict[str, Any]) -> Optional[dict[str, Any]]:
    entry = active_entry(state)
    if entry is None:
        return None
    selection = values.get("-TABLE-") or []
    if not selection:
        return None
    row_index = int(selection[0])
    if 0 <= row_index < len(state.view_indices):
        result_index = state.view_indices[row_index]
        if 0 <= result_index < len(entry["results"]):
            return entry["results"][result_index]
    return None


def labels_text(labels: dict[str, Any], limit: int = 3) -> str:
    if not labels:
        return "-"
    ordered = sorted(labels.items(), key=lambda item: float(item[1]), reverse=True)[:limit]
    return "; ".join(f"{label} {float(score):.2f}" for label, score in ordered)


def refresh_videos(window: sg.Window, state: AppState, select_path: Optional[str] = None) -> None:
    paths = [entry["path"] for entry in state.videos]
    target = select_path or getattr(state, "active_path", "")
    if target not in paths and paths:
        target = paths[0]
    state.active_path = target if target in paths else ""
    selection = [paths.index(target)] if target in paths else []
    window["-VIDEOS-"].update(values=paths, set_to_index=selection)


def refresh_results(window: sg.Window, state: AppState, values: dict[str, Any]) -> None:
    entry = active_entry(state)
    if entry is None:
        state.view_indices = []
        window["-TABLE-"].update(values=[], select_rows=[])
        window["-COUNT-"].update("0 candidate(s)")
        return
    filter_text = str(values.get("-FILTERTEXT-", "") or "").strip().lower()
    source_filter = str(values.get("-SRCFILTER-", "All") or "All")
    try:
        min_conf = float(values.get("-MINCONF-", "0") or 0)
    except (TypeError, ValueError):
        min_conf = 0.0
    rows: list[list[str]] = []
    state.view_indices = []
    for index, item in enumerate(entry.get("results", [])):
        source = str(item.get("source", "nudenet"))
        if source_filter != "All" and source != source_filter:
            continue
        confidence = float(item.get("conf", 0.0) or 0.0)
        if confidence < min_conf:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", 0.0))
        label_text = labels_text(item.get("labels") or {})
        haystack = (
            f"{format_time(start)} {format_time(end)} {source} "
            f"{SOURCE_LABELS.get(source, source)} {label_text}"
        ).lower()
        if filter_text and filter_text not in haystack:
            continue
        rows.append(
            [
                "Yes" if item.get("enabled", True) else "No",
                format_time(start, True),
                format_time(end, True),
                format_time(max(0.0, end - start), True),
                f"{confidence:.2f}",
                SOURCE_LABELS.get(source, source),
                f"{float(item.get('nudenet_max', 0.0) or 0.0):.2f}",
                f"{float(item.get('classifier_max', 0.0) or 0.0):.2f}",
                str(int(item.get("hits", 0))),
                label_text,
            ]
        )
        state.view_indices.append(index)
    window["-TABLE-"].update(values=rows, select_rows=[])
    window["-COUNT-"].update(f"{len(rows)} shown / {len(entry.get('results', []))} candidate(s)")
    stats = entry.get("stats") or {}
    if stats:
        window["-STATUS-"].update(
            f"{Path(entry['path']).name}: {stats.get('frames_analyzed', 0)} cached frames | "
            f"hits {stats.get('hits', 0)} (N {stats.get('hits_nudenet', 0)}, "
            f"C {stats.get('hits_classifier', 0)}, both {stats.get('hits_both', 0)}) | "
            f"candidates {stats.get('after', 0)}"
        )


def add_paths(window: sg.Window, state: AppState, paths: list[str], values: Optional[dict[str, Any]] = None) -> int:
    if not state.ffprobe:
        sg.popup_error("FFprobe was not found. Install FFmpeg or set the FFPROBE environment variable.", title="SceneScout")
        return 0
    expanded: list[Path] = []
    skipped = 0
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(
                file for file in sorted(path.rglob("*")) if file.suffix.lower() in VIDEO_EXTENSIONS
            )
        elif path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
            expanded.append(path)
        else:
            skipped += 1
            log(window, f"Skipped (not a video or does not exist): {raw}")
    existing = {str(Path(entry["path"]).resolve()).lower() for entry in state.videos}
    added = 0
    duplicate_count = 0
    unreadable = 0
    for path in expanded:
        try:
            resolved = str(path.resolve())
        except OSError:
            skipped += 1
            continue
        key = resolved.lower()
        if key in existing:
            duplicate_count += 1
            continue
        duration = probe_duration(state, resolved)
        if duration is None:
            unreadable += 1
            log(window, f"Skipped unreadable or unsupported video: {resolved}")
            continue
        state.videos.append(video_entry(resolved, duration))
        existing.add(key)
        added += 1
        log(window, f"Added {path.name} ({format_time(duration)})")
    if duplicate_count:
        log(window, f"Skipped {duplicate_count} duplicate video(s).")
    if unreadable:
        log(window, f"Skipped {unreadable} unreadable video(s).")
    if skipped:
        log(window, f"Skipped {skipped} non-video or missing path(s).")
    if added:
        refresh_videos(window, state, select_path=state.videos[-1]["path"])
        refresh_results(window, state, values or {})
    window["-FILES-"].update("")
    return added


def update_availability(window: sg.Window, state: AppState) -> None:
    ffmpeg = state.ffmpeg or "NOT FOUND"
    ffprobe = state.ffprobe or "NOT FOUND"
    nudenet = state.nudenet_version
    if state.nudenet is not None:
        nudenet += f" | {state.nudenet_provider}"
    classifier = "ready" if state.classifier is not None else "off / not loaded"
    if state.classifier is not None:
        classifier = f"{state.classifier_model_id} | {state.classifier_device}"
    window["-STATUSBAR-"].update(
        f"FFmpeg: {ffmpeg} | FFprobe: {ffprobe} | NudeNet: {nudenet} | "
        f"Classifier: {classifier} | Cache: {state.cache_dir}"
    )


def make_layout(state: AppState) -> list[list[Any]]:
    file_row = [
        sg.Text("Videos", size=(8, 1)),
        sg.Input(key="-FILES-", expand_x=True, tooltip="Paste one or more file/folder paths"),
        sg.Button("Add", key="-ADD-", size=(7, 1)),
        sg.Button("Files...", key="-BROWSE-", size=(8, 1)),
        sg.Button("Folder...", key="-FOLDER-", size=(9, 1)),
        sg.Button("Clear", key="-CLEAR-", size=(7, 1)),
    ]
    video_row = [
        sg.Listbox(
            values=[],
            size=(100, 3),
            key="-VIDEOS-",
            enable_events=True,
            expand_x=True,
            horizontal_scroll=True,
        )
    ]
    sampling = sg.Frame(
        "Sampling",
        [
            [
                sg.Text("Interval (s)", size=(11, 1)),
                sg.Input("1.0", size=(6, 1), key="-INTERVAL-"),
                sg.Text("Frame width", size=(10, 1)),
                sg.Input("960", size=(6, 1), key="-WIDTH-"),
                sg.Checkbox("CUDA decode", default=True, key="-CUDA-"),
                sg.Checkbox("Snap to scenes", default=True, key="-SNAP-"),
                sg.Checkbox("Refine hits @0.5s", default=True, key="-REFINE-"),
            ]
        ],
        expand_x=True,
    )
    detection = sg.Frame(
        "Detection",
        [
            [
                sg.Text("NudeNet model", size=(12, 1)),
                sg.Combo(
                    list(state.nudenet_models_by_display.keys()),
                    default_value=state.default_nudenet_display,
                    size=(24, 1),
                    key="-NNMODEL-",
                    readonly=True,
                ),
                sg.Checkbox("Whole-frame classifier", default=False, key="-CLS-", enable_events=True),
                sg.Combo(
                    CLASSIFIER_CHOICES,
                    default_value=DEFAULT_CLASSIFIER,
                    size=(36, 1),
                    key="-CLSMODEL-",
                    readonly=True,
                    disabled=True,
                ),
                sg.Text("Batch", size=(5, 1)),
                sg.Input("32", size=(4, 1), key="-BATCH-"),
            ]
        ],
        expand_x=True,
    )
    review = sg.Frame(
        "Review rules",
        [
            [
                sg.Text("Box score", size=(9, 1)),
                sg.Slider(range=(0.10, 0.90), default_value=0.35, resolution=0.05, orientation="h", size=(12, 10), key="-THRESH-"),
                sg.Text("Min box", size=(7, 1)),
                sg.Input("0.08", size=(5, 1), key="-MINBOX-"),
                sg.Text("Min hits", size=(7, 1)),
                sg.Input("2", size=(4, 1), key="-MINHITS-"),
                sg.Text("Merge gap", size=(9, 1)),
                sg.Input("12", size=(4, 1), key="-GAP-"),
                sg.Text("Before", size=(6, 1)),
                sg.Input("3", size=(4, 1), key="-BEFORE-"),
                sg.Text("After", size=(5, 1)),
                sg.Input("3", size=(4, 1), key="-AFTER-"),
            ],
            [
                sg.Text("Explicit >=", size=(9, 1)),
                sg.Input("0.60", size=(5, 1), key="-EXPLTHR-"),
                sg.Checkbox("Suggestive", default=False, key="-SUGG-", enable_events=True),
                sg.Input("0.70", size=(5, 1), key="-SUGGTHR-", disabled=True),
                sg.Checkbox("Subtitle keywords", default=False, key="-SRT-"),
                sg.Input("sex,nude,naked,explicit", size=(32, 1), key="-KEYWORDS-"),
            ],
        ],
        expand_x=True,
    )
    labels = sg.Frame(
        "Labels (exact match only)",
        [
            [
                sg.Checkbox("Genitalia", default=True, key="-GENITALIA-"),
                sg.Checkbox("Breasts", default=True, key="-BREASTS-"),
                sg.Checkbox("Male breasts", default=False, key="-MALEBREAST-"),
                sg.Checkbox("Buttocks", default=True, key="-BUTTOCKS-"),
                sg.Checkbox("Anus", default=True, key="-ANUS-"),
                sg.Checkbox("Belly", default=False, key="-BELLY-"),
                sg.Checkbox("Armpits", default=False, key="-ARMPITS-"),
                sg.Checkbox("Feet", default=False, key="-FEET-"),
            ]
        ],
        expand_x=True,
    )
    action_row = [
        sg.Button("Scan", key="-SCAN-", size=(12, 1), button_color=(ACCENT, "#0d1117")),
        sg.Button("Stop", key="-STOP-", size=(10, 1), disabled=True),
        sg.Button("Re-analyze from cache", key="-REANALYZE-", size=(20, 1), disabled=True),
        sg.Button("Export selected", key="-EXPORT-", size=(15, 1)),
        sg.Button("Export all", key="-EXPORTALL-", size=(11, 1)),
    ]
    progress_row = [
        sg.ProgressBar(100, orientation="h", size=(50, 18), key="-PROGRESS-", expand_x=True, bar_color=(ACCENT, GRAY)),
        sg.Text("", size=(70, 1), key="-STATUS-"),
    ]
    table_head = [
        sg.Text("Review", font=("Segoe UI", 11, "bold")),
        sg.Text("", expand_x=True),
        sg.Text("", key="-COUNT-", size=(28, 1)),
    ]
    table = sg.Table(
        values=[],
        headings=["On", "Start", "End", "Length", "Conf", "Source", "NudeNet", "Classif", "Hits", "Labels"],
        key="-TABLE-",
        select_mode=sg.TABLE_SELECT_MODE_BROWSE,
        enable_events=True,
        expand_x=True,
        expand_y=True,
        auto_size_columns=False,
        col_widths=[5, 10, 10, 10, 7, 11, 42, 5],
        justification="left",
        num_rows=10,
        row_height=22,
        header_background_color=GRAY,
        header_text_color="#f0f6fc",
        alternating_row_color="#161b22",
    )
    filter_row = [
        sg.Text("Filter", size=(5, 1)),
        sg.Input(size=(22, 1), key="-FILTERTEXT-", enable_events=False),
        sg.Text("Source", size=(6, 1)),
        sg.Combo(
            ["All", "nudenet", "classifier", "both", "subtitle", "manual"],
            default_value="All",
            size=(12, 1),
            key="-SRCFILTER-",
            readonly=True,
        ),
        sg.Text("Min conf", size=(8, 1)),
        sg.Input("0.00", size=(6, 1), key="-MINCONF-"),
        sg.Button("Apply", key="-APPLYFILTER-", size=(7, 1)),
        sg.Button("Reset", key="-RESETFILTER-", size=(7, 1)),
    ]
    edit_row = [
        sg.Text("Start", size=(5, 1)),
        sg.Input("0:00.000", size=(11, 1), key="-START-"),
        sg.Text("End", size=(3, 1)),
        sg.Input("0:00.000", size=(11, 1), key="-END-"),
        sg.Button("Apply edit", key="-APPLYEDIT-", size=(10, 1)),
        sg.Button("Add manual", key="-ADDROW-", size=(11, 1)),
        sg.Button("Delete", key="-DELROW-", size=(8, 1)),
        sg.Button("Toggle enabled", key="-TOGGLE-", size=(13, 1)),
        sg.Button("Show frame", key="-SHOWFRAME-", size=(11, 1)),
    ]
    log_frame = sg.Frame(
        "Log",
        [[sg.Multiline("", size=(100, 5), key="-LOG-", autoscroll=True, write_only=True, expand_x=True, background_color=LOG_BG, text_color=LOG_FG, font=("Consolas", 9))]],
        expand_x=True,
        expand_y=False,
    )
    status_bar = [sg.Text("", key="-STATUSBAR-", size=(160, 1), font=("Consolas", 8))]
    return [
        file_row,
        video_row,
        [sampling],
        [detection],
        [review],
        [labels],
        action_row,
        progress_row,
        table_head,
        [table],
        filter_row,
        edit_row,
        [log_frame],
        status_bar,
    ]


# ---------------------------------------------------------------------------
# Main event loop

def main() -> None:
    sg.theme("DarkGrey13")
    sg.set_options(font=("Segoe UI", 10))
    state = AppState()
    try:
        state.nudenet_version = importlib.metadata.version("nudenet")
    except importlib.metadata.PackageNotFoundError:
        state.nudenet_version = "not installed"
    state.nudenet_models = discover_nudenet_models()
    state.nudenet_models_by_display = {model["display"]: model for model in state.nudenet_models}
    state.default_nudenet_display = state.nudenet_models[0]["display"]

    window = sg.Window(
        "SceneScout",
        make_layout(state),
        finalize=True,
        resizable=True,
        size=(1320, 920),
    )
    window["-FILES-"].bind("<Return>", "-ENTER-")
    window["-FILTERTEXT-"].bind("<Return>", "-ENTER-")
    state.active_path = ""
    log(window, "SceneScout ready. Add videos, choose detectors, then Scan.")
    log(window, "First classifier use downloads model weights once; later runs are fully local.")
    update_availability(window, state)
    set_running(window, state, False)

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break

        try:
            if event in ("-BROWSE-",):
                if state.running:
                    continue
                result = sg.popup_get_file(
                    "Select video files",
                    multiple_files=True,
                    file_types=(
                        ("Videos", "*.mp4;*.mkv;*.mov;*.ts;*.m4v;*.avi;*.webm"),
                        ("All files", "*.*"),
                    ),
                    no_window=True,
                )
                if result:
                    add_paths(window, state, normalize_file_list(result), values)
            elif event == "-FOLDER-":
                if state.running:
                    continue
                folder = sg.popup_get_folder("Select a folder of videos", no_window=True)
                if folder:
                    add_paths(window, state, [folder], values)
            elif event in ("-ADD-", "-FILES--ENTER-"):
                if state.running:
                    continue
                add_paths(window, state, normalize_file_list(values.get("-FILES-", "")), values)
            elif event == "-CLEAR-":
                if state.running:
                    sg.popup_error("Stop the current job before clearing videos.", title="SceneScout")
                    continue
                state.videos = []
                state.active_path = ""
                state.view_indices = []
                refresh_videos(window, state)
                refresh_results(window, state, values)
                log(window, "Cleared video list.")
            elif event == "-VIDEOS-":
                selection = values.get("-VIDEOS-") or []
                if selection:
                    state.active_path = selection[0]
                refresh_results(window, state, values)
            elif event == "-TABLE-":
                row = selected_result(state, values)
                if row:
                    window["-START-"].update(format_time(float(row["start"]), True))
                    window["-END-"].update(format_time(float(row["end"]), True))
            elif event == "-APPLYFILTER-":
                refresh_results(window, state, values)
            elif event == "-FILTERTEXT--ENTER-":
                refresh_results(window, state, values)
            elif event == "-RESETFILTER-":
                window["-FILTERTEXT-"].update("")
                window["-SRCFILTER-"].update("All")
                window["-MINCONF-"].update("0.00")
                filter_values = dict(values)
                filter_values.update({"-FILTERTEXT-": "", "-SRCFILTER-": "All", "-MINCONF-": "0.00"})
                refresh_results(window, state, filter_values)
            elif event == "-CLS-":
                window["-CLSMODEL-"].update(disabled=not bool(values.get("-CLS-")))
            elif event == "-SUGG-":
                window["-SUGGTHR-"].update(disabled=not bool(values.get("-SUGG-")))
            elif event == "-SCAN-":
                if state.running:
                    continue
                if not state.videos:
                    sg.popup_error("Add at least one video first.", title="SceneScout")
                    continue
                if not state.ffmpeg or not state.ffprobe:
                    sg.popup_error("FFmpeg and FFprobe are required.", title="SceneScout")
                    continue
                set_running(window, state, True, "scan")
                window["-PROGRESS-"].update(0)
                threading.Thread(target=run_scan, args=(window, state, values), daemon=True).start()
            elif event == "-REANALYZE-":
                if state.running:
                    continue
                if not any(entry.get("cache_key") for entry in state.videos):
                    sg.popup_error("No cached scan found. Run Scan first.", title="SceneScout")
                    continue
                set_running(window, state, True, "reanalyze")
                threading.Thread(target=run_reanalyze, args=(window, state, values), daemon=True).start()
            elif event == "-STOP-":
                state.cancel.set()
                with state.process_lock:
                    process = state.current_process
                if process and process.poll() is None:
                    try:
                        process.terminate()
                    except Exception:
                        pass
                log(window, "Stop requested; finishing current operation...")
            elif event == "-APPLYEDIT-":
                entry = active_entry(state)
                row = selected_result(state, values)
                if entry is None or row is None:
                    sg.popup_error("Select a candidate row first.", title="SceneScout")
                    continue
                start = parse_time(str(values.get("-START-", "")))
                end = parse_time(str(values.get("-END-", "")))
                if start is None or end is None or start >= end:
                    sg.popup_error("Enter valid start/end times (seconds, MM:SS, or HH:MM:SS).", title="SceneScout")
                    continue
                row["start"] = max(0.0, min(float(entry["duration"]), start))
                row["end"] = max(0.0, min(float(entry["duration"]), end))
                row["locked"] = True
                row["manual"] = False
                log(window, f"Edited range: {format_time(row['start'], True)}-{format_time(row['end'], True)} (locked).")
                refresh_results(window, state, values)
            elif event == "-ADDROW-":
                entry = active_entry(state)
                if entry is None:
                    sg.popup_error("Select a video first.", title="SceneScout")
                    continue
                start = parse_time(str(values.get("-START-", "")))
                end = parse_time(str(values.get("-END-", "")))
                if start is None or end is None or start >= end:
                    sg.popup_error("Enter valid start/end times.", title="SceneScout")
                    continue
                entry["results"].append(
                    {
                        "enabled": True,
                        "start": max(0.0, min(float(entry["duration"]), start)),
                        "end": max(0.0, min(float(entry["duration"]), end)),
                        "conf": 0.0,
                        "source": "manual",
                        "labels": {},
                        "hits": 0,
                        "best_t": (start + end) / 2,
                        "best_score": 0.0,
                        "nudenet_max": 0.0,
                        "classifier_max": 0.0,
                        "locked": True,
                        "manual": True,
                    }
                )
                log(window, "Added manual candidate (locked).")
                refresh_results(window, state, values)
            elif event == "-DELROW-":
                entry = active_entry(state)
                row = selected_result(state, values)
                if entry is None or row is None:
                    sg.popup_error("Select a candidate row first.", title="SceneScout")
                    continue
                try:
                    entry["results"].remove(row)
                except ValueError:
                    pass
                log(window, "Deleted candidate row.")
                refresh_results(window, state, values)
            elif event == "-TOGGLE-":
                entry = active_entry(state)
                row = selected_result(state, values)
                if entry is None or row is None:
                    sg.popup_error("Select a candidate row first.", title="SceneScout")
                    continue
                row["enabled"] = not bool(row.get("enabled", True))
                row["locked"] = True
                log(window, f"Candidate {'enabled' if row['enabled'] else 'disabled'} (locked).")
                refresh_results(window, state, values)
            elif event == "-SHOWFRAME-":
                entry = active_entry(state)
                row = selected_result(state, values)
                if entry is None:
                    sg.popup_error("Select a video first.", title="SceneScout")
                    continue
                show_frame(window, state, entry, row)
            elif event == "-EXPORT-":
                entry = active_entry(state)
                if entry is None:
                    sg.popup_error("Select a video first.", title="SceneScout")
                    continue
                target = export_one(state, entry)
                if target:
                    log(window, f"Exported {target}")
                    sg.popup(f"Exported:\n{target}", title="SceneScout")
            elif event == "-EXPORTALL-":
                if not state.videos:
                    sg.popup_error("Add at least one video first.", title="SceneScout")
                    continue
                written: list[str] = []
                for entry in state.videos:
                    target = export_one(state, entry)
                    if target:
                        written.append(str(target))
                        log(window, f"Exported {target}")
                if written:
                    sg.popup(f"Exported {len(written)} file(s):\n" + "\n".join(written), title="SceneScout")
            elif event == "-RESULTS-":
                payload = values.get("-RESULTS-") if isinstance(values.get("-RESULTS-"), dict) else {}
                refresh_videos(window, state, payload.get("select_path"))
                refresh_results(window, state, values)
                update_availability(window, state)
            elif event == "-DONE-":
                cancelled = bool(values.get("-DONE-", {}).get("cancelled")) if isinstance(values.get("-DONE-"), dict) else False
                set_running(window, state, False)
                refresh_videos(window, state)
                refresh_results(window, state, values)
                update_availability(window, state)
                log(window, "Job stopped." if cancelled else "Job complete.")
            elif event == "-ERROR-":
                set_running(window, state, False)
                message = str(values.get("-ERROR-", "Unknown error"))
                log(window, message)
                sg.popup_error(message, title="SceneScout")

        except Exception as exc:
            log(window, f"Event handler error: {exc}")

    state.cancel.set()
    with state.process_lock:
        process = state.current_process
    if process and process.poll() is None:
        try:
            process.terminate()
        except Exception:
            pass
    cleanup_temp(state)
    cleanup_preview()
    window.close()


if __name__ == "__main__":
    main()




