# Arabic Dialect Identification (ADI) Web Application

A full-stack web application for automatic 22-class Arabic dialect identification from short speech utterances. This is a production-quality system built for a final-year project.

## Overview

This system identifies Arabic dialects from speech audio using deep learning models. It supports 6 different CNN architectures (ResNet-18, ResNet-50, DenseNet-121, MobileNet-V2, EfficientNet-B3, and SCNN) and provides a modern web interface for recording or uploading audio files.

### Features

- **6 Pre-trained Models**: Support for multiple CNN architectures
- **Flexible Windowing**: 7s, 3s center, and 3s 5-crop modes with auto-detection
- **Modern UI**: React + TypeScript frontend with Tailwind CSS
- **Audio Recording**: Browser-based microphone recording
- **File Upload**: Support for WAV, MP3, and WEBM formats
- **Visualizations**: Waveform, probability charts, and full distribution tables
- **Production Ready**: FastAPI backend with logging and error handling

## Project Structure

```
.
├── backend/              # FastAPI backend
│   ├── main.py          # FastAPI application
│   ├── models.py        # Model loading and inference
│   ├── audio.py         # Audio preprocessing
│   ├── config.py        # Configuration constants
│   ├── schemas.py       # Pydantic models
│   ├── logging_utils.py # Logging utilities
│   ├── test_inference.py # Test script
│   └── tools/           # Utility scripts
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── lib/         # Utilities and API client
│   │   └── App.tsx      # Main app component
│   └── package.json
├── models/              # Trained model checkpoints
│   ├── resnet18/
│   ├── resnet50/
│   ├── densenet121/
│   ├── mobilenet_v2/
│   ├── efficientnet_b3/
│   └── scnn/
└── logs/                # Prediction logs
```

## Architecture

```
┌─────────┐
│ Browser │
└────┬────┘
     │ HTTP/HTTPS
     ▼
┌─────────────────┐
│  React Frontend │  (Vercel)
│  (TypeScript)   │
└────────┬────────┘
         │ REST API
         ▼
┌─────────────────┐
│  FastAPI       │  (Render/Server)
│  Backend       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Registry  │
│  (PyTorch)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model          │
│  Checkpoints    │
└─────────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn
- FFmpeg (for audio processing)

### Backend Setup

1. **Install Python dependencies:**

```bash
cd backend
pip install -r requirements.txt
```

2. **Verify models are in place:**

Ensure your trained models are in the `models/` directory with the structure:
```
models/
  {model_name}/
    {timestamp}/
      checkpoints/
        best_model.pt
```

3. **Run the backend:**

```bash
# Development mode
python -m backend.main

# Or using uvicorn directly
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

4. **Test the backend:**

```bash
# Test inference on an audio file
python backend/test_inference.py path/to/audio.wav --model resnet18

# Check API health
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models
```

### Frontend Setup

1. **Install dependencies:**

```bash
cd frontend
npm install
```

2. **Configure API URL (optional):**

Create a `.env` file in the `frontend/` directory:
```
VITE_API_URL=http://localhost:8000
```

3. **Run the development server:**

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

4. **Build for production:**

```bash
npm run build
```

The built files will be in `frontend/dist/`

## API Documentation

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

#### `GET /models`

List all available models.

**Response:**
```json
{
  "models": [
    {
      "name": "resnet18",
      "architecture": "resnet18",
      "num_classes": 22
    }
  ],
  "default_model": "resnet18"
}
```

#### `POST /predict`

Predict dialect from audio file.

**Request:**
- `file`: Audio file (multipart/form-data)
- `model_name`: Model to use (optional, default: "resnet18")
- `window_mode`: Window mode (optional, default: "auto")
  - Options: `"auto"`, `"7s"`, `"3s_center"`, `"3s_5crop"`

**Response:**
```json
{
  "model_name": "resnet18",
  "window_mode": "3s_5crop",
  "dialect": "Morocco",
  "confidence": 0.987,
  "top_k": [
    {"dialect": "Morocco", "prob": 0.987},
    {"dialect": "Algeria", "prob": 0.007},
    {"dialect": "Tunisia", "prob": 0.004}
  ],
  "all_probs": {
    "Bahrain": 0.001,
    "Morocco": 0.987,
    ...
  },
  "duration_sec": 3.2,
  "timestamp": "2025-01-15T10:30:00",
  "request_id": "uuid-here"
}
```

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@audio.wav" \
  -F "model_name=resnet18" \
  -F "window_mode=auto"
```

## Deployment

### Backend Deployment

#### Option 1: Docker

```bash
# Build image
docker build -t adi-backend -f backend/Dockerfile .

# Run container
docker run -p 8000:8000 adi-backend
```

#### Option 2: Render / Railway / Fly.io

1. Create a new web service
2. Set build command: `pip install -r backend/requirements.txt`
3. Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Ensure `models/` directory is included in deployment

#### Option 3: Vercel Serverless Functions

Create `api/predict.py` (Vercel Python runtime) or use Vercel's serverless functions with appropriate configuration.

### Frontend Deployment (Vercel)

1. **Connect your repository to Vercel**

2. **Configure build settings:**
   - Framework Preset: Vite
   - Build Command: `cd frontend && npm run build`
   - Output Directory: `frontend/dist`
   - Install Command: `cd frontend && npm install`

3. **Set environment variables:**
   - `VITE_API_URL`: Your backend API URL

4. **Deploy**

## Configuration

### Audio Processing

Default settings in `backend/config.py`:
- Sample Rate: 16 kHz
- Hop Length: 256
- N Mels: 128
- N FFT: 1024

### Dialect Labels

The system identifies 22 Arabic dialects:
- Gulf: Bahrain, Kuwait, Oman, Qatar, Saudi Arabia, UAE, Yemen
- Levant: Iraq, Jordan, Lebanon, Palestine, Syria
- Maghreb: Algeria, Libya, Mauritania, Morocco, Tunisia
- Other: Comoros, Djibouti, Egypt, Somalia, Sudan

## Logging

All predictions are logged to `logs/predictions.jsonl` in JSONL format.

### View Logs

```bash
# Summarize logs
python backend/tools/summarize_logs.py

# View raw logs
cat logs/predictions.jsonl | jq
```

## Development

### Running Tests

```bash
# Test backend inference
python backend/test_inference.py test_audio.wav

# Test API
curl http://localhost:8000/health
```

### Code Style

- Backend: Follow PEP 8, use type hints
- Frontend: Use TypeScript, follow React best practices

## Troubleshooting

### Backend Issues

1. **Model not found:**
   - Verify model checkpoints exist in `models/{model_name}/{timestamp}/checkpoints/best_model.pt`
   - Check that model architecture matches expected format

2. **Audio processing errors:**
   - Ensure FFmpeg is installed
   - Check audio file format is supported

3. **CUDA errors:**
   - Models will fall back to CPU if CUDA is unavailable
   - Check PyTorch installation

### Frontend Issues

1. **API connection errors:**
   - Verify `VITE_API_URL` is set correctly
   - Check CORS settings on backend
   - Ensure backend is running

2. **Audio recording not working:**
   - Check browser permissions for microphone
   - Use HTTPS in production (required for MediaRecorder)

## License

This project is part of a final-year project. All rights reserved.

## Acknowledgments

- PyTorch for deep learning framework
- FastAPI for the backend framework
- React and Vite for the frontend
- All contributors to the training data and models

