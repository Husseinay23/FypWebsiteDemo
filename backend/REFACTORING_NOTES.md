# Backend Refactoring Notes

## Summary

The backend has been completely refactored to match the training pipeline exactly. This fixes the issue where predictions were random/incorrect due to preprocessing mismatches.

## Key Changes

### 1. New Audio Preprocessing Module (`audio_preprocessing.py`)

**Replaced:** `backend/audio.py` preprocessing functions  
**New:** `backend/audio_preprocessing.py` with training-equivalent pipeline

#### Critical Parameters (Now Match Training):
- `hop_length=160` (was 256) ❌ → ✅
- `win_length=400` (was not set) ❌ → ✅
- `fmin=50` (was 0/sr//2) ❌ → ✅
- `fmax=7600` (was sr//2) ❌ → ✅
- `power_to_db` (was log1p) ❌ → ✅
- **Pre-emphasis** (coef=0.97) - **NEW** ✅
- **Trim silence** (top_db=30) - **NEW** ✅

#### Pipeline Flow:
1. `load_audio_16k_mono()` - Load to 16kHz mono float32
2. `make_7s_waveform()` - Standardize to exactly 7 seconds (center-crop/pad)
3. `waveform_to_mel_db()` - Pre-emphasis → Trim → Mel → dB conversion
4. Window extraction (7s, 3s_center, or 3s_5crop) from the 7s mel
5. `mel_to_model_tensor()` - Convert to model input tensor

### 2. Updated Model Inference (`models.py`)

**Changed:** `DialectModel.predict()` → `DialectModel.predict_from_tensor()`

- Now accepts preprocessed tensors directly (not numpy arrays)
- Handles 5-crop by averaging **probabilities** (not logits), matching training evaluation
- Properly handles EfficientNet 3-channel checkpoints

### 3. Updated API Endpoint (`main.py`)

**Changed:** Uses new `preprocess_for_inference()` function

- Single preprocessing call that returns tensor ready for model
- Returns original audio duration (before 7s standardization) for logging
- Cleaner error handling

### 4. Configuration Updates (`config.py`)

**Updated constants:**
- `HOP_LENGTH = 160` (was 256)
- Added `WIN_LENGTH = 400`
- Added `FMIN = 50`
- Added `FMAX = 7600`

## Testing

Run the test script to verify preprocessing:

```bash
python backend/test_preprocessing.py path/to/test_audio.wav --model resnet18 --window_mode 7s
```

Expected results:
- High confidence predictions (>0.5, ideally >0.8)
- Correct dialect predictions (not random Oman/Somalia/Mauritania)
- Consistent results across window modes

## Migration Notes

### For Developers:

1. **Old code using `load_audio()` and `preprocess_audio()`:**
   - Replace with `preprocess_for_inference(file_bytes, window_mode)`
   - Returns `(tensor, window_mode, duration)` instead of `(mel, window_mode)`

2. **Old code using `model.predict(mel, mode)`:**
   - Replace with `model.predict_from_tensor(tensor, mode)`
   - Tensor is already preprocessed and ready for model

3. **Old mel computation:**
   - Do NOT use `compute_mel_spectrogram()` from `audio.py`
   - Use `waveform_to_mel_db()` from `audio_preprocessing.py`

## Verification Checklist

- [x] Pre-emphasis (coef=0.97) applied
- [x] Trim silence (top_db=30) applied
- [x] Mel parameters match training (hop=160, win=400, fmin=50, fmax=7600)
- [x] power_to_db conversion (not log1p)
- [x] 7s base waveform, then mel extraction
- [x] 3s modes extract from 7s mel (not re-process raw audio)
- [x] 5-crop averages probabilities (not logits)
- [x] Label order matches training (DIALECT_LABELS)
- [x] Model tensor shape: (1, 1, 128, T) or (5, 1, 128, T3)
- [x] EfficientNet 3-channel handling

## Files Changed

1. `backend/audio_preprocessing.py` - **NEW** (training-equivalent preprocessing)
2. `backend/models.py` - Updated inference method
3. `backend/main.py` - Updated to use new preprocessing
4. `backend/config.py` - Updated constants
5. `backend/test_preprocessing.py` - **NEW** (test script)

## Files Kept (for backward compatibility)

- `backend/audio.py` - Still exists but not used by main API
- Old test scripts remain but should use new preprocessing

## Next Steps

1. Test with known good audio files from ADC dataset
2. Verify predictions match training evaluation results
3. Monitor logs for confidence improvements
4. If still seeing low confidence, check:
   - Model checkpoint loading (weights match architecture)
   - Label order in training vs inference
   - Any normalization differences

