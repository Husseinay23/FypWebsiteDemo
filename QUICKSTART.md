# Quick Start Guide

## Prerequisites

- Python 3.10+
- Node.js 18+
- FFmpeg (for audio processing)

## Quick Setup (5 minutes)

### 1. Backend Setup

```bash
# Navigate to project root
cd "/Users/husseinay/Desktop/fyp website"

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Verify models are in place
ls models/*/202*/checkpoints/best_model.pt

# Run backend
python -m backend.main
# Or use the script:
./backend/run.sh
```

Backend will start at `http://localhost:8000`

### 2. Frontend Setup

```bash
# In a new terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will start at `http://localhost:3000`

### 3. Test the System

1. Open `http://localhost:3000` in your browser
2. Click "Record" tab and grant microphone permission
3. Record a few seconds of audio
4. Select a model (e.g., "ResNet-18")
5. Click "Analyze"
6. View the prediction results!

## Testing with Audio File

```bash
# Test backend directly
python backend/test_inference.py path/to/audio.wav --model resnet18

# Or use curl
curl -X POST "http://localhost:8000/predict" \
  -F "file=@audio.wav" \
  -F "model_name=resnet18" \
  -F "window_mode=auto"
```

## Common Issues

### Backend won't start

- **Issue**: `ModuleNotFoundError: No module named 'backend'`
  - **Fix**: Run from project root, not from backend directory
  - **Fix**: Ensure `backend/__init__.py` exists

- **Issue**: Model not found
  - **Fix**: Verify models are in `models/{model_name}/{timestamp}/checkpoints/best_model.pt`
  - **Fix**: Check model directory structure matches expected format

### Frontend won't connect

- **Issue**: CORS errors
  - **Fix**: Backend CORS is set to allow all origins in dev mode
  - **Fix**: Check backend is running on port 8000

- **Issue**: API URL not found
  - **Fix**: Create `frontend/.env` with `VITE_API_URL=http://localhost:8000`
  - **Fix**: Restart dev server after changing .env

### Audio recording not working

- **Issue**: Microphone permission denied
  - **Fix**: Grant browser permission for microphone
  - **Fix**: Use HTTPS in production (required for MediaRecorder)

- **Issue**: Recording stops immediately
  - **Fix**: Check browser console for errors
  - **Fix**: Try a different browser (Chrome/Firefox recommended)

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- View logs: `python backend/tools/summarize_logs.py`
- Deploy: See README.md deployment section

## Development Tips

- Backend auto-reloads on code changes (uvicorn --reload)
- Frontend hot-reloads on code changes (Vite HMR)
- Check browser console and terminal for errors
- Use `curl` or Postman to test API endpoints directly

