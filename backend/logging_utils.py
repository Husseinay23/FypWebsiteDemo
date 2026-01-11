"""
Logging utilities for prediction tracking.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from backend.config import PREDICTION_LOG_FILE


def log_prediction(
    model_name: str,
    window_mode: str,
    dialect: str,
    confidence: float,
    duration_sec: float,
    request_id: str,
    client_ip: Optional[str] = None
):
    """
    Log a prediction to the log file.
    
    Args:
        model_name: Model used
        window_mode: Window mode used
        dialect: Predicted dialect
        confidence: Confidence score
        duration_sec: Audio duration
        request_id: Unique request ID
        client_ip: Client IP address (optional)
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "model_name": model_name,
        "window_mode": window_mode,
        "predicted_dialect": dialect,
        "confidence": confidence,
        "duration_sec": duration_sec,
        "client_ip": client_ip
    }
    
    # Append to JSONL file
    with open(PREDICTION_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')

