# Architecture Documentation

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         React Frontend (TypeScript)                  │  │
│  │  - Audio Recording (MediaRecorder API)              │  │
│  │  - File Upload                                        │  │
│  │  - UI Components (Tailwind CSS)                      │  │
│  │  - Visualizations (Recharts)                         │  │
│  └───────────────────┬──────────────────────────────────┘  │
└──────────────────────┼──────────────────────────────────────┘
                       │ HTTP/REST API
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Routes:                                          │  │
│  │  - GET  /health                                       │  │
│  │  - GET  /models                                       │  │
│  │  - POST /predict                                      │  │
│  └───────────────────┬──────────────────────────────────┘  │
│  ┌───────────────────┴──────────────────────────────────┐  │
│  │  Audio Processing Module                              │  │
│  │  - Load & resample audio                              │  │
│  │  - Window extraction (7s, 3s, 5-crop)                 │  │
│  │  - Mel-spectrogram computation                        │  │
│  └───────────────────┬──────────────────────────────────┘  │
│  ┌───────────────────┴──────────────────────────────────┐  │
│  │  Model Registry                                        │  │
│  │  - Load PyTorch models                                 │  │
│  │  - Inference pipeline                                  │  │
│  │  - Probability aggregation (5-crop)                   │  │
│  └───────────────────┬──────────────────────────────────┘  │
│  ┌───────────────────┴──────────────────────────────────┐  │
│  │  Logging Module                                        │  │
│  │  - Request logging (JSONL)                            │  │
│  │  - Prediction tracking                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Checkpoints (PyTorch)                    │
│  - resnet18/best_model.pt                                   │
│  - resnet50/best_model.pt                                   │
│  - densenet121/best_model.pt                                │
│  - mobilenet_v2/best_model.pt                               │
│  - efficientnet_b3/best_model.pt                            │
│  - scnn/best_model.pt                                       │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Prediction Pipeline

1. **Audio Input**
   - User records audio via browser (MediaRecorder) OR
   - User uploads audio file (WAV/MP3/WEBM)

2. **Frontend Processing**
   - Convert audio to Blob/File
   - Display waveform visualization
   - Send to backend via FormData

3. **Backend Audio Processing**
   ```
   Raw Audio → Load & Resample (16kHz) → Window Extraction → Mel-Spectrogram
   ```

4. **Model Inference**
   ```
   Mel-Spectrogram → Model Forward Pass → Logits → Softmax → Probabilities
   ```

5. **Response**
   - Top-1 prediction + confidence
   - Top-K predictions
   - Full probability distribution
   - Metadata (model, window mode, duration)

6. **Frontend Display**
   - Show prediction result
   - Display probability bar chart
   - Show full distribution table
   - Allow JSON download

## Component Architecture

### Backend Components

#### `config.py`
- Centralized configuration
- Audio processing constants
- Dialect labels
- Model paths

#### `audio.py`
- `load_audio()`: Load and resample audio
- `compute_mel_spectrogram()`: Generate mel-spectrograms
- `preprocess_audio()`: Window extraction and preprocessing
- `center_crop()`: Extract center window
- `get_5_crop_windows()`: Generate 5-crop windows

#### `models.py`
- `create_model()`: Instantiate PyTorch models
- `load_model_checkpoint()`: Load trained weights
- `DialectModel`: Wrapper class for inference
- `ModelRegistry`: Manage all loaded models

#### `main.py`
- FastAPI application
- API route handlers
- Request/response validation
- Error handling

### Frontend Components

#### `App.tsx`
- Main application component
- State management
- API integration
- Tab navigation

#### `AudioRecorder.tsx`
- MediaRecorder integration
- Recording controls
- Waveform visualization

#### `FileUploader.tsx`
- File selection
- Drag-and-drop support
- Waveform preview

#### `PredictionResult.tsx`
- Display prediction results
- Probability charts
- Distribution tables
- JSON download

#### `ProbabilityBarChart.tsx`
- Recharts integration
- Top-K visualization

## Audio Processing Pipeline

### Window Modes

#### 1. `7s` Mode
```
Input: Arbitrary length audio
Process:
  - If duration >= 7s: Center crop 7 seconds
  - If duration < 7s: Zero-pad to 7 seconds
Output: Single 7-second mel-spectrogram
```

#### 2. `3s_center` Mode
```
Input: Arbitrary length audio
Process:
  - Center crop 3 seconds (pad if necessary)
Output: Single 3-second mel-spectrogram
```

#### 3. `3s_5crop` Mode
```
Input: Arbitrary length audio
Process:
  - If duration >= 7s: 5 evenly spaced 3s windows
  - If 3s <= duration < 7s: Overlapping 3s windows
  - If duration < 3s: Pad and duplicate center window
Output: 5 mel-spectrograms → Average logits → Final probabilities
```

### Mel-Spectrogram Parameters

- **Sample Rate**: 16 kHz
- **Hop Length**: 256 samples
- **N Mels**: 128 filter banks
- **N FFT**: 1024
- **Normalization**: log(1 + x)

## Model Architecture Details

### Supported Architectures

1. **ResNet-18/50**: Modified for single-channel input
2. **DenseNet-121**: Modified first conv layer
3. **MobileNet-V2**: Lightweight architecture
4. **EfficientNet-B3**: Efficient scaling
5. **SCNN**: Custom Spectral CNN

All models:
- Input: (1, 128, time_frames) mel-spectrogram
- Output: 22-class logits
- Trained on Arabic dialect dataset

## Security Considerations

1. **CORS**: Configure allowed origins in production
2. **File Size Limits**: Implement max file size validation
3. **Rate Limiting**: Consider adding rate limits for production
4. **Input Validation**: All inputs validated via Pydantic
5. **Error Handling**: Sensitive errors not exposed to client

## Performance Optimizations

1. **Model Loading**: Models loaded once at startup
2. **GPU Support**: Automatic CUDA detection
3. **Batch Processing**: Future: support batch inference
4. **Caching**: Consider caching preprocessed audio
5. **Async Processing**: FastAPI async endpoints

## Deployment Architecture

### Development
```
Frontend (Vite Dev Server) → Backend (Uvicorn) → Local Models
```

### Production (Recommended)
```
Vercel (Frontend) → Render/Railway (Backend) → Model Storage
```

### Alternative: Docker
```
Docker Container (Backend + Models) → Exposed Port
```

## Monitoring & Logging

- **Request Logging**: All predictions logged to JSONL
- **Error Tracking**: FastAPI exception handlers
- **Health Checks**: `/health` endpoint for monitoring
- **Metrics**: Consider adding Prometheus metrics

## Future Enhancements

1. **Batch Processing**: Support multiple files
2. **Real-time Streaming**: WebSocket support for live audio
3. **Model Ensembling**: Combine predictions from multiple models
4. **Confidence Thresholds**: Reject low-confidence predictions
5. **User Accounts**: Track user predictions
6. **Analytics Dashboard**: View prediction statistics

