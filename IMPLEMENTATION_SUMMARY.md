# Implementation Summary: Website Backend Alignment to Training Contract

## Overview

This document summarizes all changes made to align the website backend with the training contract exactly.

**Source of Truth:** Training contract (specified parameters)  
**Goal:** Remove all ambiguities and hard-align website inference to match training pipeline exactly.

---

## 1. Mismatch Table (Website vs Training Contract)

See `MISMATCH_TABLE.md` for complete mismatch analysis.

**Key Mismatches Fixed:**
1. ✅ `n_fft`: 1024 → 2048
2. ✅ `hop_length`: 160 → 256
3. ✅ `f_min`: 50 → 20.0
4. ✅ Removed pre-emphasis (was present, now removed)
5. ✅ Removed trim-silence (was present, now removed)
6. ✅ Window padding: center → right
7. ⚠️ Label format: spaces → underscores (template provided, needs actual list)
8. ✅ Audio decoding: Added FFmpeg-based decoding for WEBM/OPUS

---

## 2. Files Changed (Commit-Style Summary)

### Created Files:

1. **`backend/preprocessing_canonical.py`** (NEW)
   - Canonical preprocessing matching training contract exactly
   - NO pre-emphasis, NO trim-silence
   - Exact mel parameters: n_fft=2048, hop_length=256, f_min=20.0, f_max=7600.0
   - RIGHT-padding for short audio (not center-padding)
   - Returns debug_info dict for debug mode

2. **`backend/audio_decode.py`** (NEW)
   - FFmpeg-based audio decoding
   - Handles WEBM/OPUS, MP3, WAV robustly
   - `decode_audio_bytes_to_wav_pcm16()` function
   - `load_audio_from_bytes()` function
   - Startup check for FFmpeg availability

3. **`MISMATCH_TABLE.md`** (NEW)
   - Complete mismatch analysis
   - File:Line references for all mismatches

4. **`IMPLEMENTATION_SUMMARY.md`** (NEW - this file)
   - Summary of all changes

### Modified Files:

1. **`backend/config.py`**
   - Line 9: `HOP_LENGTH = 160` → `256`
   - Line 11: `N_FFT = 1024` → `2048`
   - Line 13: `FMIN = 50` → `F_MIN = 20.0`
   - Line 14: `FMAX = 7600` → `F_MAX = 7600.0`
   - Removed: `WIN_LENGTH` (not in training contract)

2. **`backend/labels_22.py`**
   - Updated to use underscores format (Saudi_Arabia, United_Arab_Emirates)
   - Added startup assertions as required
   - ⚠️ Still needs actual label list from training code (placeholder provided)

### Files to Update (Next Steps):

3. **`backend/models.py`** (REQUIRED)
   - Add model input adapter functions:
     - `adapt_input_for_efficientnet(x)` - 3-channel + resize to 300x300
     - `adapt_input_for_scnn(x)` - Fixed 700 frames
     - `adapt_input_for_default_cnn(x)` - 1-channel, variable time
   - Update `DialectModel.predict_from_tensor()` to use adapters

4. **`backend/main.py`** (REQUIRED)
   - Replace `audio_preprocessing.preprocess_for_inference` with `preprocessing_canonical.preprocess_for_inference_canonical`
   - Add `debug` query parameter support
   - Add debug response fields as specified in contract
   - Use `audio_decode.load_audio_from_bytes` with MIME type detection

5. **`test_api.py`** (REQUIRED - NEW)
   - Create validation script as specified
   - Test with known audio files
   - Assert deterministic outputs
   - Optional: top-1/top-3 accuracy computation

### Files to Deprecate/Remove:

6. **`backend/audio_preprocessing.py`** (DEPRECATE)
   - Contains old preprocessing with wrong parameters
   - Replace with `preprocessing_canonical.py`

7. **`backend/audio.py`** (DEPRECATE/REMOVE)
   - Old preprocessing module
   - No longer used

---

## 3. Code Changes Details

### A. Configuration (`backend/config.py`)

**Before:**
```python
HOP_LENGTH = 160  # CRITICAL: Training uses 160, not 256!
N_FFT = 1024
FMIN = 50
FMAX = 7600
WIN_LENGTH = 400
```

**After:**
```python
HOP_LENGTH = 256  # Training contract: 256
N_FFT = 2048  # Training contract: 2048
F_MIN = 20.0  # Training contract: 20.0
F_MAX = 7600.0  # Training contract: 7600.0
# WIN_LENGTH removed (not in training contract)
```

### B. Preprocessing (`backend/preprocessing_canonical.py`)

**Key Changes:**
- NO pre-emphasis (removed)
- NO trim-silence (removed)
- Mel parameters: n_fft=2048, hop_length=256, f_min=20.0, f_max=7600.0
- RIGHT-padding for short audio (pad_right only, not center-pad)
- Uses FFmpeg-based audio decoding
- Returns debug_info dict

**Function Signatures:**
```python
def preprocess_for_inference_canonical(
    file_bytes: bytes,
    window_mode: str,
    sr: int = 16000,
    input_ext_or_mime: Optional[str] = None
) -> Tuple[torch.Tensor, str, float, dict]:
    # Returns: (tensor, window_mode, duration, debug_info)
```

### C. Audio Decoding (`backend/audio_decode.py`)

**New Functions:**
- `check_ffmpeg()` - Check FFmpeg availability
- `decode_audio_bytes_to_wav_pcm16()` - FFmpeg-based decoding
- `load_audio_from_bytes()` - Load audio using FFmpeg

**Usage:**
```python
from backend.audio_decode import load_audio_from_bytes

waveform = load_audio_from_bytes(
    file_bytes,
    input_ext_or_mime="audio/webm",
    target_sr=16000
)
```

### D. Labels (`backend/labels_22.py`)

**Before:**
```python
DIALECT_LABELS = [
    "Bahrain", "Kuwait", ..., "Saudi Arabia", "UAE", ...
]
```

**After:**
```python
DIALECT_LABELS = [
    "Bahrain", "Kuwait", ..., "Saudi_Arabia", "United_Arab_Emirates", ...
]

# Startup assertions
assert len(DIALECT_LABELS) == 22
assert len(set(DIALECT_LABELS)) == 22
assert "Saudi_Arabia" in DIALECT_LABELS
assert "United_Arab_Emirates" in DIALECT_LABELS
```

**⚠️ NOTE:** Still needs actual label list from training code (placeholder provided).

---

## 4. Remaining Implementation Tasks

### Task 1: Update `backend/main.py`

**Changes Required:**

1. Import canonical preprocessing:
```python
from backend.preprocessing_canonical import preprocess_for_inference_canonical
```

2. Replace preprocessing call:
```python
# OLD:
model_tensor, actual_window_mode, original_duration_sec = preprocess_for_inference(
    file_content, window_mode
)

# NEW:
model_tensor, actual_window_mode, original_duration_sec, debug_info = preprocess_for_inference_canonical(
    file_content, window_mode, input_ext_or_mime=file.content_type
)
```

3. Add debug mode support:
```python
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(default=DEFAULT_MODEL),
    window_mode: str = Form(default="auto"),
    debug: bool = Query(default=False)  # NEW
):
    # ... preprocessing ...
    
    if debug:
        # Add debug fields to response
        response = {
            # ... existing fields ...
            "debug": {
                "input_mime_or_ext": file.content_type,
                "sr_after_resample": debug_info['sr_after_resample'],
                "waveform_len_before_7s": debug_info['waveform_len_before_7s'],
                "waveform_len_after_7s": debug_info['waveform_len_after_7s'],
                # ... all debug fields ...
            }
        }
    else:
        # Return normal response
        response = PredictionResponse(...)
```

### Task 2: Add Model Input Adapters (`backend/models.py`)

**Add adapter functions:**

```python
def adapt_input_for_efficientnet(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adapt input for EfficientNet-B3:
    - 3-channel input (repeat 1->3)
    - Resize to 300x300
    """
    # Repeat channel: (B, 1, n_mels, T) -> (B, 3, n_mels, T)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    
    # Resize to 300x300
    tensor = torch.nn.functional.interpolate(
        tensor, size=(300, 300), mode='bilinear', align_corners=False
    )
    
    return tensor


def adapt_input_for_scnn(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adapt input for SCNN:
    - Fixed time dimension = 700 frames
    - Pad or truncate along time axis
    """
    # tensor shape: (B, 1, 128, T)
    target_time = 700
    
    if tensor.shape[3] < target_time:
        # Pad right
        pad = target_time - tensor.shape[3]
        tensor = torch.nn.functional.pad(tensor, (0, pad), mode='constant', value=0)
    elif tensor.shape[3] > target_time:
        # Truncate right
        tensor = tensor[:, :, :, :target_time]
    
    return tensor


def adapt_input_for_default_cnn(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adapt input for default CNNs (ResNet, DenseNet, MobileNet):
    - 1-channel input (already correct)
    - Variable time axis allowed
    """
    return tensor
```

**Update `DialectModel.predict_from_tensor()`:**

```python
def predict_from_tensor(self, model_tensor, window_mode):
    # Apply model-specific adapter
    if self.architecture == "efficientnet_b3":
        model_tensor = adapt_input_for_efficientnet(model_tensor)
    elif self.architecture == "scnn":
        model_tensor = adapt_input_for_scnn(model_tensor)
    else:
        model_tensor = adapt_input_for_default_cnn(model_tensor)
    
    # ... rest of inference ...
```

### Task 3: Create `test_api.py`

**Create validation script:**

```python
import requests
import sys
from pathlib import Path

def test_api(audio_path: str, api_url: str = "http://localhost:8000"):
    """Test API with known audio file."""
    with open(audio_path, 'rb') as f:
        files = {'file': f}
        data = {
            'model_name': 'resnet18',
            'window_mode': '7s'
        }
        
        # Test 1: Normal request
        response1 = requests.post(f"{api_url}/predict", files=files, data=data)
        result1 = response1.json()
        
        # Test 2: Debug request
        files = {'file': open(audio_path, 'rb')}
        response2 = requests.post(
            f"{api_url}/predict?debug=true", files=files, data=data
        )
        result2 = response2.json()
        
        # Test 3: Deterministic (run twice)
        files = {'file': open(audio_path, 'rb')}
        response3 = requests.post(f"{api_url}/predict", files=files, data=data)
        result3 = response3.json()
        
        # Assert deterministic
        assert result1['dialect'] == result3['dialect']
        assert abs(result1['confidence'] - result3['confidence']) < 1e-6
        
        print("✓ Tests passed")
        print(f"Predicted: {result1['dialect']} (confidence: {result1['confidence']:.4f})")

if __name__ == "__main__":
    test_api(sys.argv[1])
```

---

## 5. Startup Health Checks

### FFmpeg Check

Add to `backend/main.py` startup:

```python
from backend.audio_decode import check_ffmpeg

# Check FFmpeg at startup
if not check_ffmpeg():
    raise RuntimeError(
        "FFmpeg is not installed. Please install FFmpeg:\n"
        "  macOS: brew install ffmpeg\n"
        "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
    )
```

### Label Assertions

Already added in `backend/labels_22.py`:
```python
assert len(DIALECT_LABELS) == 22
assert len(set(DIALECT_LABELS)) == 22
assert "Saudi_Arabia" in DIALECT_LABELS
assert "United_Arab_Emirates" in DIALECT_LABELS
```

---

## 6. Instructions to Run Locally

### Prerequisites

1. **Install FFmpeg:**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Install Python Dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Verify FFmpeg:**
   ```bash
   ffmpeg -version
   ```

### Run Backend

```bash
cd backend
python -m backend.main
# OR
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
# Test API
python test_api.py path/to/test_audio.wav

# Test preprocessing
python -m pytest backend/tests/ -v
```

---

## 7. Sanity Check Checklist

### What to Run:

1. **Start Backend:**
   ```bash
   cd backend
   python -m backend.main
   ```
   - ✅ Should start without errors
   - ✅ Should load all models
   - ✅ Should check FFmpeg availability

2. **Test Health Endpoint:**
   ```bash
   curl http://localhost:8000/health
   ```
   - ✅ Should return `{"status": "ok"}`

3. **Test Predict Endpoint (Debug Mode):**
   ```bash
   curl -X POST "http://localhost:8000/predict?debug=true" \
     -F "file=@test_audio.wav" \
     -F "model_name=resnet18" \
     -F "window_mode=7s"
   ```
   - ✅ Should return prediction + debug fields
   - ✅ Debug fields should include all specified fields

4. **Test Deterministic Output:**
   ```bash
   python test_api.py test_audio.wav
   ```
   - ✅ Same audio should produce identical predictions

5. **Test Known Audio File:**
   - Use audio file with known dialect from ADC dataset
   - ✅ Should predict correct dialect
   - ✅ Confidence > 0.5 (ideally > 0.8)

### What "Correct" Looks Like:

1. **Mel-Spectrogram Shapes:**
   - 7s audio: (128, 438) frames [with hop_length=256]
   - 3s audio: (128, 187) frames

2. **Predictions:**
   - Deterministic (same input → same output)
   - Reasonable confidence (>0.5 for correct predictions)
   - Labels use underscores (Saudi_Arabia, not "Saudi Arabia")

3. **Debug Mode:**
   - All debug fields present
   - Mel shapes match expected values
   - Statistics are reasonable

---

## 8. Critical Notes

### ⚠️ Label List Still Needs Training Code

**`backend/labels_22.py` currently has a placeholder.**

**Action Required:**
1. Get exact label list from training code (inference_canonical.py or training label mapping)
2. Replace placeholder in `backend/labels_22.py`
3. Verify order and spelling match training exactly

### ⚠️ Model Input Adapters Need Verification

**EfficientNet-B3 resize to 300x300 and SCNN fixed 700 frames need verification with training code.**

**Action Required:**
1. Verify EfficientNet-B3 input requirements (300x300 resize?)
2. Verify SCNN fixed time dimension (700 frames?)
3. Update adapter functions if needed

### ⚠️ Normalization Needs Verification

**Current implementation uses `ref=np.max` (per-spectrogram normalization).**

**Contract says "per-sample normalization exactly as done in inference_canonical.py".**

**Action Required:**
1. Check training code for normalization method
2. Update if per-sample normalization is needed

---

## 9. Summary

### Completed:
- ✅ Created mismatch table
- ✅ Updated config.py with correct parameters
- ✅ Created preprocessing_canonical.py (matching contract exactly)
- ✅ Created audio_decode.py (FFmpeg-based decoding)
- ✅ Updated labels_22.py (format + assertions, still needs actual list)
- ✅ Created implementation summary

### Remaining:
- ⚠️ Update main.py (use canonical preprocessing + debug mode)
- ⚠️ Add model input adapters to models.py
- ⚠️ Create test_api.py
- ⚠️ Get actual label list from training code
- ⚠️ Verify model input requirements (EfficientNet resize, SCNN frames)
- ⚠️ Verify normalization method (per-sample vs per-spectrogram)

---

**Status:** Core preprocessing implemented, remaining tasks require integration and verification.

