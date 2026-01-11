"""
FastAPI backend for Arabic Dialect Identification.
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import uuid
import numpy as np
from typing import Optional
from datetime import datetime

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DEFAULT_MODEL, SUPPORTED_MODELS
from backend.models import ModelRegistry
from backend.preprocessing_canonical import preprocess_for_inference_canonical
from backend.audio_decode import check_ffmpeg
from backend.schemas import (
    HealthResponse, ModelsResponse, ModelInfo, PredictionResponse, TopKItem
)
from backend.logging_utils import log_prediction
from fastapi import Query

# Initialize FastAPI app
app = FastAPI(
    title="Arabic Dialect Identification API",
    description="API for identifying Arabic dialects from speech audio",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check FFmpeg availability at startup
if not check_ffmpeg():
    raise RuntimeError(
        "FFmpeg is not installed. Please install FFmpeg:\n"
        "  macOS: brew install ffmpeg\n"
        "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
        "  Windows: Download from https://ffmpeg.org/download.html"
    )

# Initialize model registry
print("Loading models...")
model_registry = ModelRegistry()
print(f"Loaded {len(model_registry.models)} models")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all available models."""
    models_info = model_registry.list_models()
    return ModelsResponse(
        models=[ModelInfo(**m) for m in models_info],
        default_model=DEFAULT_MODEL
    )


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(default=DEFAULT_MODEL),
    window_mode: str = Form(default="auto"),
    debug: bool = Query(default=False)
):
    """
    Predict dialect from audio file.
    
    Args:
        file: Audio file (WAV, MP3, WEBM)
        model_name: Model to use (default: resnet18)
        window_mode: Window mode (auto, 7s, 3s_center, 3s_5crop)
        debug: Enable debug mode (default: False)
    """
    # Validate inputs
    if model_name not in SUPPORTED_MODELS and model_name not in ["best", "Best (recommended)"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name. Supported: {SUPPORTED_MODELS}"
        )
    
    if window_mode not in ["auto", "7s", "3s_center", "3s_5crop"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid window_mode. Must be one of: auto, 7s, 3s_center, 3s_5crop"
        )
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    try:
        # Read audio file
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty"
            )
        
        # Preprocess audio using canonical preprocessing pipeline
        try:
            model_tensor, actual_window_mode, original_duration_sec, debug_info = preprocess_for_inference_canonical(
                file_content,
                window_mode,
                input_ext_or_mime=file.content_type
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Audio preprocessing error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Audio processing failed: {str(e)}"
            )
        
        # Get model
        model = model_registry.get_model(model_name)
        actual_model_name = model.name
        
        # Run prediction using the preprocessed tensor
        prediction = model.predict_from_tensor(model_tensor, actual_window_mode)
        
        # Get client IP
        client_ip = request.client.host if request.client else None
        
        # Log prediction
        log_prediction(
            model_name=actual_model_name,
            window_mode=actual_window_mode,
            dialect=prediction["dialect"],
            confidence=prediction["confidence"],
            duration_sec=original_duration_sec,
            request_id=request_id,
            client_ip=client_ip
        )
        
        # Build base response
        base_response = {
            "model_name": actual_model_name,
            "window_mode": actual_window_mode,
            "dialect": prediction["dialect"],
            "confidence": prediction["confidence"],
            "top_k": [TopKItem(**item) for item in prediction["top_k"]],
            "all_probs": prediction["all_probs"],
            "duration_sec": round(original_duration_sec, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
        
        # Add debug information if requested
        if debug:
            from backend.labels_22 import DIALECT_LABELS
            import numpy as np
            
            # Get top-5 indices and labels
            probs_array = np.array([prediction["all_probs"][label] for label in DIALECT_LABELS])
            top5_indices = np.argsort(probs_array)[::-1][:5]
            top5_labels = [DIALECT_LABELS[int(idx)] for idx in top5_indices]
            top5_probs = [float(probs_array[int(idx)]) for idx in top5_indices]
            
            debug_response = {
                **base_response,
                "debug": {
                    "input_mime_or_ext": file.content_type,
                    "sr_before_resample": None,  # Not available (using FFmpeg)
                    "sr_after_resample": debug_info['sr_after_resample'],
                    "waveform_len_before_7s": debug_info['waveform_len_before_7s'],
                    "waveform_len_after_7s": debug_info['waveform_len_after_7s'],
                    "waveform_duration_before": debug_info['waveform_duration_before'],
                    "waveform_duration_after": debug_info['waveform_duration_after'],
                    "mel_shape_before_crop": list(debug_info['mel_shape_before_crop']),
                    "mel_stats_before_norm": debug_info['mel_stats_before_norm'],
                    "mel_stats_after_norm": debug_info['mel_stats_after_norm'],
                    "tensor_shape_final": debug_info['tensor_shape_final'],
                    "top5_indices": [int(idx) for idx in top5_indices],
                    "top5_labels": top5_labels,
                    "top5_probs": top5_probs,
                    "label_list_used": DIALECT_LABELS,
                    "label_source": "backend/labels_22.py",
                    "preprocessing_source": "backend/preprocessing_canonical.py"
                }
            }
            return debug_response
        
        # Return normal response
        return PredictionResponse(**base_response)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        print(f"Error processing audio: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

