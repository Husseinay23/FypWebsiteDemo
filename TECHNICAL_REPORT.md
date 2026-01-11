# Technical Report: Arabic Dialect Identification Web System

## Fine-Grained Arabic Dialect Identification Using Deep Learning

**Date:** January 2025  
**Project:** Final Year Project (FYP)  
**System:** Web-based Arabic Dialect Identification (ADI) Demo

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Website Architecture](#website-architecture)
4. [Intended ML Pipeline Design](#intended-ml-pipeline-design)
5. [Inference Pipeline Analysis](#inference-pipeline-analysis)
6. [Error Diagnosis](#error-diagnosis)
7. [Identified Issues and Fixes](#identified-issues-and-fixes)
8. [Corrected Implementation](#corrected-implementation)
9. [Validation Strategy](#validation-strategy)
10. [Deployment Considerations](#deployment-considerations)
11. [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## Executive Summary

This technical report documents the architecture, issues, and fixes for a web-based Arabic Dialect Identification (ADI) system. The system identifies 22 Arabic dialects from short speech utterances (3-7 seconds) using deep learning CNN architectures (ResNet-18, ResNet-50, DenseNet-121, MobileNet-V2, EfficientNet-B3, SCNN).

**Key Findings:**

- **Critical Issue:** Parameter mismatch between training configs and inference code (`hop_length`: config says 256, code uses 160)
- **Critical Issue:** Label order potentially incorrect (placeholder in `labels_22.py`)
- **Issue:** Potential preprocessing pipeline inconsistencies
- **Issue:** Two preprocessing modules exist (`audio.py` and `audio_preprocessing.py`)

**Status:** System requires fixes to match training pipeline exactly. Predictions are incorrect/inconsistent due to preprocessing mismatches.

---

## System Overview

### Purpose

A production-grade web application for real-time Arabic dialect identification, designed as a front-end module for dialect-aware Automatic Speech Recognition (ASR) and spoken translation systems.

### Core Functionality

- **Input:** 3-7 second audio clips (recorded via browser or uploaded as files)
- **Processing:** Mel-spectrogram conversion → CNN inference
- **Output:** Predicted dialect (22 classes) + confidence scores

### Supported Models

- ResNet-18 (default)
- ResNet-50
- DenseNet-121
- MobileNet-V2
- EfficientNet-B3
- SCNN (Spectral CNN)

### Window Modes

- **7s:** Full 7-second window
- **3s_center:** Center 3-second crop
- **3s_5crop:** 5 evenly-spaced 3-second crops (probability averaging)
- **auto:** Automatically select based on audio duration

---

## Website Architecture

### Technology Stack

**Frontend:**

- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS
- MediaRecorder API (browser recording)

**Backend:**

- FastAPI (Python)
- PyTorch (model inference)
- librosa (audio processing)
- Uvicorn (ASGI server)

**Deployment:**

- Frontend: Vercel
- Backend: Render/Server (or Docker)

### Architecture Diagram

```
┌─────────────────┐
│   Web Browser   │
│  (React + TS)   │
└────────┬────────┘
         │ HTTPS/REST API
         ▼
┌─────────────────┐
│   FastAPI       │
│   Backend       │
│  (Python)       │
└────────┬────────┘
         │
         ├───► Audio Preprocessing
         │     (librosa)
         │
         ├───► Model Registry
         │     (PyTorch)
         │
         └───► Model Checkpoints
               (best_model.pt)
```

### Data Flow

1. **User Input:**

   - User records audio via MediaRecorder (WEBM/Opus)
   - OR uploads audio file (WAV/MP3/WEBM)

2. **Frontend Processing:**

   - Audio → Blob → File
   - FormData with `file`, `model_name`, `window_mode`
   - POST to `/predict` endpoint

3. **Backend Processing:**

   - Read audio bytes
   - Preprocess (resample → 7s window → mel-spectrogram)
   - Load model
   - Run inference
   - Return predictions

4. **Response:**

   - Predicted dialect + confidence
   - Top-K predictions
   - Full probability distribution
   - Metadata (model, window mode, duration)

5. **Frontend Display:**
   - Prediction result
   - Probability bar chart
   - Full distribution table

---

## Intended ML Pipeline Design

### Expected Audio Input Format

- **Sample Rate:** 16,000 Hz (16 kHz)
- **Channels:** Mono (single channel)
- **Duration:** 3-7 seconds (fixed-length 7s with 3s inference variants)
- **Format:** Float32 waveform in range [-1, 1]

### Training Configuration (from run_config.json)

Based on analysis of model checkpoint configs:

```json
{
  "sr": 16000,
  "hop_length": 256,
  "n_mels": 128,
  "n_fft": 1024,
  "window_seconds": ["7s", "3s"]
}
```

**⚠️ CRITICAL NOTE:** The codebase currently uses `hop_length=160`, but all `run_config.json` files specify `hop_length=256`. This is a **critical mismatch** that must be resolved by verifying the actual training code.

### Preprocessing Pipeline (Intended)

Based on training configs and code comments, the intended pipeline is:

1. **Audio Loading:**

   - Load audio file (any format: WAV/MP3/WEBM)
   - Resample to 16 kHz
   - Convert to mono
   - Float32, range [-1, 1]

2. **Window Standardization:**

   - Standardize to exactly 7 seconds
   - If longer: center-crop to 7s
   - If shorter: center-pad with zeros to 7s

3. **Mel-Spectrogram Conversion:**

   - Pre-emphasis filter (coef=0.97) **[NEEDS VERIFICATION]**
   - Trim silence (top_db=30) **[NEEDS VERIFICATION]**
   - Compute mel-spectrogram:
     - `n_fft=1024`
     - `hop_length=256` (from config) OR `160` (from code) **[MISMATCH]**
     - `win_length=400` (if specified)
     - `n_mels=128`
     - `fmin=50` (from code) **[NEEDS VERIFICATION]**
     - `fmax=7600` (from code) **[NEEDS VERIFICATION]**
     - `power=2.0`
   - Convert to dB scale: `power_to_db(S, ref=np.max)` **[NEEDS VERIFICATION]**

4. **Window Extraction (for 3s modes):**

   - Extract 3-second crops from 7s mel-spectrogram
   - Center crop: middle 3 seconds
   - 5-crop: 5 evenly-spaced 3-second windows

5. **Tensor Conversion:**
   - Shape: `(1, 1, 128, T)` for single prediction
   - Shape: `(5, 1, 128, T3)` for 5-crop

### Model Inference Flow

1. **Model Loading:**

   - Load checkpoint: `models/{architecture}/{timestamp}/checkpoints/best_model.pt`
   - Load state dict (handle DataParallel prefixes)
   - Set to eval mode

2. **Forward Pass:**

   - For single prediction: `model(tensor)` → logits `(1, num_classes)`
   - For 5-crop: `model(batch_tensor)` → logits `(5, num_classes)`

3. **Aggregation (5-crop only):**

   - Apply softmax to each crop: `(5, num_classes)`
   - Average probabilities: `mean(dim=0)` → `(num_classes,)`
   - **CRITICAL:** Average probabilities, NOT logits

4. **Output:**
   - Softmax to probabilities: `(num_classes,)`
   - Top-1: `argmax(probs)` → predicted class index
   - Map index to label: `DIALECT_LABELS[predicted_idx]`

### Output Format

```json
{
  "model_name": "resnet18",
  "window_mode": "3s_5crop",
  "dialect": "Morocco",
  "confidence": 0.987,
  "top_k": [
    {"dialect": "Morocco", "prob": 0.987},
    {"dialect": "Algeria", "prob": 0.007},
    ...
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

### Assumptions Made During Training (MUST Be Preserved)

1. **Audio format:** 16 kHz mono float32
2. **Window size:** Fixed 7-second base (with 3s crops)
3. **Mel parameters:** Must match exactly (hop_length, n_mels, n_fft, fmin, fmax)
4. **Preprocessing steps:** Pre-emphasis, trim, normalization must match
5. **Label order:** Class indices must match training label order
6. **Normalization:** dB scale (power_to_db), not log1p
7. **5-crop aggregation:** Average probabilities (not logits)

---

## Inference Pipeline Analysis

### Current Implementation (audio_preprocessing.py)

The current preprocessing pipeline (`backend/audio_preprocessing.py`) implements:

```python
def preprocess_for_inference(file_bytes, window_mode, sr=16000):
    # 1. Load audio to 16kHz mono
    y = load_audio_16k_mono(file_bytes, target_sr=sr)

    # 2. Standardize to 7 seconds
    y7 = make_7s_waveform(y, sr=sr, target_seconds=7.0)

    # 3. Convert to mel-spectrogram
    mel7 = waveform_to_mel_db(y7, sr=sr, n_mels=128)

    # 4. Extract window (7s, 3s_center, or 3s_5crop)
    # 5. Convert to tensor
    return tensor, window_mode, duration
```

**Mel-spectrogram conversion:**

- Pre-emphasis: `coef=0.97` ✅
- Trim silence: `top_db=30` ✅
- hop_length: `160` ❌ **MISMATCH** (config says 256)
- fmin: `50` ✅
- fmax: `7600` ✅
- Normalization: `power_to_db` ✅

### Model Inference (models.py)

```python
def predict_from_tensor(model_tensor, window_mode):
    if window_mode == '3s_5crop':
        logits_5 = model(model_tensor)  # (5, num_classes)
        probs_5 = torch.softmax(logits_5, dim=1)
        probs_avg = probs_5.mean(dim=0)  # ✅ Average probabilities
    else:
        logits = model(model_tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = np.argmax(probs_avg)
    dialect = DIALECT_LABELS[top_idx]  # ⚠️ Label order must match
    return prediction
```

---

## Error Diagnosis

### Critical Issues

#### 1. **hop_length Mismatch (CRITICAL)**

**Issue:**

- Training configs (`run_config.json`): `hop_length: 256`
- Inference code (`audio_preprocessing.py`): `hop_length=160`
- Config file (`config.py`): `HOP_LENGTH = 160`

**Impact:**

- Different time resolution in mel-spectrogram
- 7s audio: 256 → 437 frames, 160 → 700 frames
- Model expects specific frame count → wrong predictions

**Detection:**

- Compare `run_config.json` files with code
- Check mel-spectrogram shape: should match training

**Fix:**

- Verify actual training code
- If training uses 256: change code to 256
- If training uses 160: update config files
- Update all references (config.py, audio_preprocessing.py, crop functions)

#### 2. **Label Order Mismatch (CRITICAL)**

**Issue:**

- `labels_22.py` contains a placeholder with TODO comment
- Actual training label order unknown
- Wrong label order → correct index, wrong dialect name

**Impact:**

- Predictions appear correct (high confidence)
- But dialect names are wrong (e.g., index 5 = "Saudi Arabia" in training, but code says "UAE")

**Detection:**

- Test with known audio file
- Compare predicted label with ground truth
- Check label indices match training

**Fix:**

- Get exact label list from training code
- Copy to `labels_22.py`
- Verify all 22 labels match

#### 3. **Preprocessing Steps Verification Needed**

**Issue:**

- Code uses pre-emphasis and trim silence
- Training configs don't specify these
- Unknown if training used these steps

**Impact:**

- If training didn't use pre-emphasis: wrong feature extraction
- If training didn't use trim: different signal characteristics

**Detection:**

- Compare training preprocessing code
- Test with/without pre-emphasis
- Test with/without trim

**Fix:**

- Verify training preprocessing steps
- Match exactly: same steps, same parameters

### Common Issues (Systematic Analysis)

#### 4. **Sample Rate Mismatch**

**Issue:** Audio not resampled to 16 kHz

**Why it causes wrong results:**

- Different frequency resolution
- Model trained on 16 kHz expects specific frequency bins

**Detection:**

- Check loaded waveform sample rate
- Verify librosa resampling worked

**Fix:**

- Always resample to 16 kHz: `librosa.load(..., sr=16000)`

#### 5. **Stereo vs Mono**

**Issue:** Stereo audio not converted to mono

**Why it causes wrong results:**

- Model expects mono input
- Stereo channels may have different characteristics

**Detection:**

- Check waveform shape: should be 1D
- If 2D: stereo not converted

**Fix:**

- Always convert to mono: `librosa.load(..., mono=True)`

#### 6. **Normalization Mismatch**

**Issue:** Different normalization (log1p vs power_to_db)

**Why it causes wrong results:**

- Different value ranges
- Model trained on dB scale expects dB values

**Detection:**

- Check mel values: dB should be negative, log1p positive
- Compare with training preprocessing

**Fix:**

- Use same normalization: `power_to_db` if training used it

#### 7. **Mel Parameters Mismatch**

**Issue:** Different mel parameters (fmin, fmax, n_fft, win_length)

**Why it causes wrong results:**

- Different frequency coverage
- Different time resolution
- Model expects specific feature dimensions

**Detection:**

- Compare mel-spectrogram shape
- Check frequency range

**Fix:**

- Match exactly: same n_fft, hop_length, fmin, fmax, win_length

#### 8. **Window Duration Mismatch**

**Issue:** Wrong window duration (3s vs 7s)

**Why it causes wrong results:**

- Model expects specific input length
- Wrong length → wrong spatial dimensions after pooling

**Detection:**

- Check mel-spectrogram time dimension
- Verify crop functions

**Fix:**

- Match training: 7s base, 3s crops extracted correctly

#### 9. **Label Index Mismatch**

**Issue:** Class indices don't match training order

**Why it causes wrong results:**

- Model predicts index 5
- But index 5 is different dialect in code vs training

**Detection:**

- Test with known file
- Compare predicted index with training

**Fix:**

- Match label order exactly

#### 10. **Softmax Misuse**

**Issue:** Not applying softmax, or applying incorrectly

**Why it causes wrong results:**

- Model outputs logits, not probabilities
- Need softmax to get probabilities

**Detection:**

- Check output values: probabilities should sum to 1
- Logits don't sum to 1

**Fix:**

- Always apply softmax: `torch.softmax(logits, dim=1)`

#### 11. **Random Cropping at Inference**

**Issue:** Using random crops instead of deterministic

**Why it causes wrong results:**

- Different crops → different predictions
- Inconsistent results

**Detection:**

- Run same audio multiple times
- If results vary: random cropping

**Fix:**

- Use deterministic crops: center crop, fixed positions

#### 12. **Model.eval() Not Called**

**Issue:** Model in training mode (dropout active, batch norm using batch stats)

**Why it causes wrong results:**

- Dropout randomly zeros features
- Batch norm uses wrong statistics

**Detection:**

- Check `model.training`: should be False
- Compare with `model.eval()` vs `model.train()`

**Fix:**

- Always call `model.eval()` before inference
- Use `torch.inference_mode()` context

#### 13. **Frontend Encoding Issues**

**Issue:** Audio encoding/decoding errors

**Why it causes wrong results:**

- Corrupted audio
- Wrong sample rate after encoding

**Detection:**

- Check audio duration in backend
- Compare original vs received audio

**Fix:**

- Validate audio format
- Use appropriate codecs (WEBM/Opus for recording)

#### 14. **Audio Clipping / Silence Handling**

**Issue:** Clipping or excessive silence

**Why it causes wrong results:**

- Clipping distorts signal
- Silence after trim → wrong duration

**Detection:**

- Check audio levels
- Verify trim doesn't remove all audio

**Fix:**

- Handle clipping: normalize before clipping
- Handle silence: don't trim if all silent, or pad appropriately

---

## Identified Issues and Fixes

### Issue 1: hop_length Mismatch (CRITICAL)

**Current State:**

- Code: `hop_length=160`
- Configs: `hop_length=256`
- Impact: Wrong mel-spectrogram time dimension

**Fix:**

1. Verify training code to determine correct value
2. If 256: Update code to 256
3. If 160: Update config files to 160
4. Update `config.py`, `audio_preprocessing.py`, crop functions

**Action Required:** ⚠️ **VERIFY WITH TRAINING CODE**

### Issue 2: Label Order (CRITICAL)

**Current State:**

- `labels_22.py` has placeholder
- Actual order unknown
- Impact: Wrong dialect names for correct indices

**Fix:**

1. Get exact list from training: `adc/notebooks/lib/io_paths.py` → `get_22_country_labels()`
2. Copy exact list to `labels_22.py`
3. Verify indices match training

**Action Required:** ⚠️ **GET FROM TRAINING CODE**

### Issue 3: Preprocessing Steps Verification

**Current State:**

- Code uses pre-emphasis and trim
- Training configs don't specify
- Unknown if training used these

**Fix:**

1. Verify training preprocessing code
2. Match exactly: same steps, same parameters
3. Remove if not used, add if missing

**Action Required:** ⚠️ **VERIFY WITH TRAINING CODE**

### Issue 4: Duplicate Preprocessing Modules

**Current State:**

- Two modules: `audio.py` (old) and `audio_preprocessing.py` (new)
- `main.py` uses `audio_preprocessing.py`
- `test_inference.py` uses `audio.py`

**Fix:**

1. Standardize on `audio_preprocessing.py`
2. Update all imports
3. Deprecate or remove `audio.py`

### Issue 5: Config Consistency

**Current State:**

- `config.py` has `HOP_LENGTH = 160`
- But `run_config.json` files say 256
- Inconsistency

**Fix:**

1. Verify correct value
2. Update `config.py` to match training
3. Ensure consistency everywhere

---

## Corrected Implementation

### Recommended Fixes

#### Fix 1: Update hop_length to Match Training Configs

**Assumption:** Training used `hop_length=256` (from run_config.json)

```python
# backend/config.py
HOP_LENGTH = 256  # Changed from 160

# backend/audio_preprocessing.py
def waveform_to_mel_db(...):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,  # Changed from 160
        win_length=400,
        n_mels=n_mels,
        fmin=50,
        fmax=7600,
        power=2.0,
    )

def center_crop_mel_3s(mel7, sr=16000, hop_length=256):  # Changed from 160
    frames_per_sec = int(round(sr / hop_length))  # 16000 / 256 = 62.5
    crop_frames = int(round(3 * frames_per_sec))  # 187 frames for 3s
    ...

def five_crop_mel_3s(mel7, sr=16000, hop_length=256):  # Changed from 160
    ...
```

**⚠️ IMPORTANT:** This assumes training used 256. **VERIFY WITH TRAINING CODE FIRST.**

#### Fix 2: Update Label Order

**Action:** Copy exact label list from training code to `backend/labels_22.py`

```python
# backend/labels_22.py
# REPLACE PLACEHOLDER WITH EXACT LIST FROM TRAINING CODE

DIALECT_LABELS = [
    # Copy exact list from: adc/notebooks/lib/io_paths.py -> get_22_country_labels()
    # Must match training order exactly
]
```

**⚠️ CRITICAL:** Get exact list from training code. Do not guess.

#### Fix 3: Verify Preprocessing Steps

**Action:** Compare training preprocessing with inference preprocessing

1. Check training code for:

   - Pre-emphasis: yes/no? coef value?
   - Trim silence: yes/no? top_db value?
   - Normalization: power_to_db? log1p? other?

2. Update inference to match exactly

#### Fix 4: Clean Up Duplicate Modules

**Action:** Standardize on `audio_preprocessing.py`

1. Update `test_inference.py` to use `audio_preprocessing.py`
2. Remove or deprecate `audio.py`
3. Update all imports

---

## Validation Strategy

### Unit Tests

#### Test 1: Preprocessing Parameters

```python
def test_mel_parameters():
    # Verify mel parameters match training
    assert HOP_LENGTH == 256  # Or 160, verify first
    assert N_MELS == 128
    assert N_FFT == 1024
    assert FMIN == 50  # Verify
    assert FMAX == 7600  # Verify
```

#### Test 2: Mel-Spectrogram Shape

```python
def test_mel_shape():
    # 7s audio at 16kHz with hop_length=256
    # Expected frames: (7 * 16000) / 256 = 437.5 ≈ 438
    audio = np.zeros(7 * 16000, dtype=np.float32)
    mel = waveform_to_mel_db(audio, sr=16000, n_mels=128)
    assert mel.shape == (128, 438)  # Or (128, 700) if hop_length=160
```

#### Test 3: Label Order

```python
def test_label_order():
    # Verify labels match training
    # Compare with training code output
    assert len(DIALECT_LABELS) == 22
    # Check specific indices match training
```

### Integration Tests

#### Test 1: Known Sample

1. Use audio file with known dialect from ADC dataset
2. Run inference
3. Verify predicted dialect matches ground truth
4. Verify confidence > 0.5 (ideally > 0.8)

#### Test 2: Consistency

1. Run same audio multiple times
2. Verify predictions are identical (deterministic)
3. Verify confidences are identical

#### Test 3: Window Modes

1. Test all window modes (7s, 3s_center, 3s_5crop)
2. Verify predictions are reasonable
3. Verify 5-crop has higher confidence than single crop

### Sanity Checks

1. **Probability Sum:** All probabilities sum to 1.0
2. **Confidence Range:** Top prediction confidence in [0, 1]
3. **No NaN/Inf:** No NaN or Inf values in predictions
4. **Label Validity:** Predicted label is in DIALECT_LABELS
5. **Tensor Shapes:** Correct tensor shapes for model input

### Latency Benchmarks

- Audio loading: < 100ms
- Preprocessing: < 500ms
- Model inference: < 200ms (CPU), < 50ms (GPU)
- Total: < 1s (CPU), < 500ms (GPU)

### Failure Mode Testing

1. **Empty audio:** Should return error
2. **Very short audio (< 1s):** Should pad appropriately
3. **Very long audio (> 30s):** Should crop appropriately
4. **Silent audio:** Should handle gracefully
5. **Corrupted audio:** Should return error

---

## Deployment Considerations

### Backend Deployment

**Requirements:**

- Python 3.10+
- FFmpeg (for audio processing)
- Sufficient RAM for model loading (2-4 GB)
- GPU optional (CPU works, slower)

**Environment Variables:**

- `MODELS_DIR`: Path to models directory
- `LOGS_DIR`: Path to logs directory

**Docker:**

```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Deployment

**Requirements:**

- Node.js 18+
- HTTPS (required for MediaRecorder)

**Environment Variables:**

- `VITE_API_URL`: Backend API URL

### Model Storage

- Models stored in `models/{architecture}/{timestamp}/checkpoints/best_model.pt`
- Total size: ~500 MB - 2 GB (depending on models)
- Load at startup (lazy loading optional)

### Scalability

- Single server can handle moderate load
- For high load: use load balancer + multiple instances
- Consider model caching/GPU pooling for production

---

## Limitations and Future Improvements

### Current Limitations

1. **Fixed Models:** No retraining/fine-tuning via API
2. **Single Language:** Arabic dialects only
3. **Fixed Duration:** 3-7 seconds only
4. **No Batch Processing:** One audio at a time
5. **Limited Validation:** No confidence thresholding
6. **No Explanability:** No attention maps or feature visualization

### Future Improvements

1. **Confidence Thresholding:** Reject low-confidence predictions
2. **Batch Processing:** Process multiple audio files
3. **Model Ensemble:** Combine multiple models for better accuracy
4. **Explainability:** Add attention visualization, saliency maps
5. **Adaptive Duration:** Handle variable-length audio better
6. **Real-time Streaming:** Support streaming audio input
7. **Fine-tuning API:** Allow fine-tuning on new data
8. **Multi-language Support:** Extend to other languages
9. **Performance Optimization:** Model quantization, ONNX conversion
10. **Better Error Handling:** More detailed error messages

---

## Conclusion

This technical report documents the architecture, issues, and fixes for the Arabic Dialect Identification web system. The system requires fixes to match the training pipeline exactly, particularly:

1. **Critical:** Verify and fix `hop_length` mismatch (256 vs 160)
2. **Critical:** Update label order from training code
3. **Important:** Verify preprocessing steps match training
4. **Important:** Clean up duplicate preprocessing modules

Once these fixes are applied and validated, the system should produce correct and consistent predictions matching the training evaluation results.

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Author:** Senior Full-Stack ML Engineer Review
