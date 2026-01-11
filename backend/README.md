# Backend API Documentation

## Overview

FastAPI backend for Arabic Dialect Identification. Provides REST API endpoints for model inference.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
# Development mode (auto-reload)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Or use the run script
./run.sh
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns server status.

### List Models

```bash
GET /models
```

Returns list of available models and metadata.

### Predict

```bash
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: Audio file (required)
- model_name: Model to use (optional, default: "resnet18")
- window_mode: Window mode (optional, default: "auto")
```

Returns prediction results with probabilities.

## Testing

```bash
# Test inference pipeline
python test_inference.py audio.wav --model resnet18 --window_mode auto

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/models
```

## Logging

Predictions are logged to `../logs/predictions.jsonl`

View logs:
```bash
python tools/summarize_logs.py
```

## Configuration

Edit `config.py` to modify:
- Audio processing parameters
- Model paths
- Dialect labels
- Default settings

