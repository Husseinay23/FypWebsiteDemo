"""
Pydantic schemas for API requests and responses.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime


class ModelInfo(BaseModel):
    """Model metadata."""
    name: str
    architecture: str
    num_classes: int


class TopKItem(BaseModel):
    """Top-k prediction item."""
    dialect: str
    prob: float


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    window_mode: str
    dialect: str
    confidence: float
    top_k: List[TopKItem]
    all_probs: Dict[str, float]
    duration_sec: float
    timestamp: str
    request_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"


class ModelsResponse(BaseModel):
    """Models list response."""
    models: List[ModelInfo]
    default_model: str

