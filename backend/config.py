"""
Configuration constants for the Arabic Dialect Identification system.
"""
from pathlib import Path
from typing import List

# Audio processing constants (MUST match training pipeline exactly)
# Source: Training contract / inference_canonical.py
SAMPLE_RATE = 16000
HOP_LENGTH = 256  # Training contract: 256
N_MELS = 128
N_FFT = 2048  # Training contract: 2048
F_MIN = 20.0  # Training contract: 20.0
F_MAX = 7600.0  # Training contract: 7600.0

# Window durations in seconds
WINDOW_7S = 7.0
WINDOW_3S = 3.0

# Import labels from canonical source
from backend.labels_22 import DIALECT_LABELS

NUM_CLASSES = len(DIALECT_LABELS)

# Model directory
MODELS_DIR = Path(__file__).parent.parent / "models"

# Logging
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PREDICTION_LOG_FILE = LOGS_DIR / "predictions.jsonl"

# Supported model architectures
SUPPORTED_MODELS = [
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenet_v2",
    "efficientnet_b3",
    "scnn"
]

# Default model
DEFAULT_MODEL = "resnet18"

