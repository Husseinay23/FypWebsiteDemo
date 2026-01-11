# System Analysis Summary: Arabic Dialect Identification Web System

## Executive Summary

This document provides a comprehensive analysis of the Arabic Dialect Identification (ADI) web system, identifying all issues causing incorrect predictions and providing a roadmap for fixes.

**Status:** System has critical preprocessing mismatches that cause wrong predictions.

**Key Findings:**
1. **CRITICAL:** `hop_length` mismatch (config says 256, code uses 160)
2. **CRITICAL:** Label order is placeholder (actual order unknown)
3. **IMPORTANT:** Preprocessing steps need verification (pre-emphasis, trim)
4. **MINOR:** Duplicate preprocessing modules

---

## PART 1: System Understanding

### Intended System Design

**Audio Input Format:**
- Sample Rate: 16,000 Hz (16 kHz)
- Channels: Mono
- Duration: 3-7 seconds (standardized to 7s base)
- Format: Float32, range [-1, 1]

**Preprocessing Pipeline (from training configs):**
```
Raw Audio → Load (16kHz mono) → 7s Window (crop/pad) → Mel-Spectrogram → Model
```

**Mel-Spectrogram Parameters (from run_config.json):**
- Sample Rate: 16,000 Hz
- Hop Length: **256** (from config files) ⚠️ Code uses 160
- N Mels: 128
- N FFT: 1024 (from code)
- Win Length: 400 (from code)
- Fmin: 50 (from code, needs verification)
- Fmax: 7600 (from code, needs verification)
- Normalization: power_to_db (from code, needs verification)

**Preprocessing Steps (from code, needs verification):**
1. Pre-emphasis (coef=0.97) ⚠️ Needs verification
2. Trim silence (top_db=30) ⚠️ Needs verification
3. Mel-spectrogram computation
4. dB conversion (power_to_db)

**Model Inference:**
- Input: Mel-spectrogram tensor (1, 1, 128, T)
- Output: Logits (1, num_classes) or (5, num_classes) for 5-crop
- Softmax: Apply to get probabilities
- 5-crop: Average probabilities (not logits)
- Output: Top-1 dialect + confidence + full distribution

**Label Mapping:**
- 22 classes
- Index → Dialect name mapping
- ⚠️ **Current order is placeholder - must get from training code**

---

## PART 2: Full-Stack Architecture Design

### Current Architecture

```
Frontend (React + TS)
    ↓ HTTP POST (FormData)
Backend (FastAPI)
    ↓ Audio Preprocessing
    ↓ Model Inference
    ↓ Response (JSON)
Frontend (Display Results)
```

### Data Flow

1. **User Input:**
   - Browser: MediaRecorder → WEBM/Opus blob
   - OR: File upload (WAV/MP3/WEBM)

2. **Frontend:**
   - Convert blob → File
   - FormData: file, model_name, window_mode
   - POST to `/predict`

3. **Backend:**
   - Read audio bytes
   - Preprocess: `preprocess_for_inference()`
   - Load model: `ModelRegistry.get_model()`
   - Inference: `model.predict_from_tensor()`
   - Return: PredictionResponse

4. **Response:**
   - Dialect name, confidence, top-K, all probs

5. **Frontend:**
   - Display prediction, charts, tables

### API Design

**Endpoint:** `POST /predict`

**Request:**
- `file`: Audio file (multipart/form-data)
- `model_name`: Model to use (optional, default: resnet18)
- `window_mode`: Window mode (optional, default: auto)

**Response:**
```json
{
  "model_name": "resnet18",
  "window_mode": "3s_5crop",
  "dialect": "Morocco",
  "confidence": 0.987,
  "top_k": [...],
  "all_probs": {...},
  "duration_sec": 3.2,
  "timestamp": "...",
  "request_id": "..."
}
```

---

## PART 3: Error Diagnosis

### Issue 1: hop_length Mismatch (CRITICAL)

**Problem:**
- Training configs: `hop_length: 256`
- Inference code: `hop_length=160`
- Impact: Wrong mel-spectrogram time dimension

**Why It Causes Wrong Results:**
- Different time resolution
- Model expects specific frame count
- 7s audio: 256 → 438 frames, 160 → 700 frames (60% difference!)

**Detection:**
- Compare `run_config.json` files with code
- Check mel-spectrogram shape

**Fix:**
- Verify training code
- If training used 256: change code to 256
- If training used 160: update config files to 160
- Update all references

### Issue 2: Label Order Mismatch (CRITICAL)

**Problem:**
- `labels_22.py` has placeholder
- Actual order unknown
- Impact: Wrong dialect names

**Why It Causes Wrong Results:**
- Model predicts index 5
- But index 5 is different dialect in code vs training
- Predictions appear correct but labels are wrong

**Detection:**
- Test with known audio file
- Compare predicted label with ground truth

**Fix:**
- Get exact list from training code
- Copy to `labels_22.py`

### Issue 3: Preprocessing Steps Unknown (IMPORTANT)

**Problem:**
- Code uses pre-emphasis and trim
- Training configs don't specify
- Unknown if training used these

**Why It Causes Wrong Results:**
- If training didn't use pre-emphasis: wrong features
- If training didn't use trim: different signal

**Detection:**
- Compare training preprocessing code
- Test with/without pre-emphasis

**Fix:**
- Verify training preprocessing steps
- Match exactly

### Issue 4: Sample Rate Mismatch (Common)

**Problem:** Audio not resampled to 16 kHz

**Fix:** Always resample to 16 kHz

### Issue 5: Stereo vs Mono (Common)

**Problem:** Stereo audio not converted to mono

**Fix:** Always convert to mono

### Issue 6: Normalization Mismatch (Common)

**Problem:** Different normalization (log1p vs power_to_db)

**Fix:** Use same normalization as training

### Issue 7: Mel Parameters Mismatch (Common)

**Problem:** Different mel parameters

**Fix:** Match exactly (n_fft, hop_length, fmin, fmax)

### Issue 8: Window Duration Mismatch (Common)

**Problem:** Wrong window duration

**Fix:** Match training (7s base, 3s crops)

### Issue 9: Label Index Mismatch (Common)

**Problem:** Class indices don't match training order

**Fix:** Match label order exactly

### Issue 10: Softmax Misuse (Common)

**Problem:** Not applying softmax correctly

**Fix:** Always apply softmax to logits

### Issue 11: Random Cropping (Common)

**Problem:** Random crops instead of deterministic

**Fix:** Use deterministic crops

### Issue 12: Model.eval() Not Called (Common)

**Problem:** Model in training mode

**Fix:** Always call `model.eval()`

### Issue 13: Frontend Encoding Issues (Common)

**Problem:** Audio encoding/decoding errors

**Fix:** Validate audio format

### Issue 14: Audio Clipping / Silence (Common)

**Problem:** Clipping or excessive silence

**Fix:** Handle gracefully

---

## PART 4: Implementation Tasks

### Task 1: Fix hop_length

**Action:** Change to 256 (matching run_config.json)

**Files:**
- `backend/config.py`: Change `HOP_LENGTH = 256`
- `backend/audio_preprocessing.py`: Change `hop_length=256` in all functions

**⚠️ VERIFY WITH TRAINING CODE FIRST**

### Task 2: Fix Label Order

**Action:** Copy exact list from training code

**Files:**
- `backend/labels_22.py`: Replace placeholder

**⚠️ MUST GET FROM TRAINING CODE**

### Task 3: Verify Preprocessing Steps

**Action:** Compare with training code

**Steps:**
1. Check training preprocessing function
2. Verify pre-emphasis (yes/no? coef?)
3. Verify trim silence (yes/no? top_db?)
4. Update code to match

### Task 4: Clean Up Duplicate Modules

**Action:** Standardize on `audio_preprocessing.py`

**Files:**
- `backend/test_inference.py`: Update to use `audio_preprocessing.py`
- Optionally: Remove or deprecate `audio.py`

---

## PART 5: Validation & Testing

### Validation Checklist

1. **Preprocessing Tests:**
   - [ ] Mel parameters match training
   - [ ] Mel-spectrogram shapes match expected
   - [ ] Preprocessing steps match training

2. **Label Tests:**
   - [ ] Label order matches training
   - [ ] All 22 labels present
   - [ ] Indices match training

3. **Inference Tests:**
   - [ ] Model loads successfully
   - [ ] Tensor shapes are correct
   - [ ] Softmax applied correctly
   - [ ] 5-crop averages probabilities

4. **End-to-End Tests:**
   - [ ] Known audio file → correct prediction
   - [ ] Consistency: same audio → same prediction
   - [ ] All window modes work
   - [ ] All models work

5. **Sanity Checks:**
   - [ ] Probabilities sum to 1.0
   - [ ] Confidence in [0, 1]
   - [ ] No NaN/Inf values
   - [ ] Label is valid

### Testing Strategy

1. **Unit Tests:**
   - Test preprocessing functions
   - Test mel-spectrogram computation
   - Test tensor conversion

2. **Integration Tests:**
   - Test with known audio files
   - Test consistency
   - Test all window modes

3. **Sanity Tests:**
   - Test with ADC dataset samples
   - Compare with training evaluation results
   - Verify predictions match expected

### How to Verify Correctness Without Retraining

1. **Use Known Audio Files:**
   - Get audio files from ADC dataset with known dialect
   - Run inference
   - Verify predicted dialect matches ground truth

2. **Compare with Training Evaluation:**
   - Use same audio files as training evaluation
   - Compare predictions with training results
   - Should match exactly

3. **Consistency Tests:**
   - Run same audio multiple times
   - Should get identical predictions (deterministic)

4. **Shape Verification:**
   - Check mel-spectrogram shapes
   - Should match training (verify with training code)

5. **Label Verification:**
   - Compare label indices with training
   - Should match exactly

---

## PART 6: Technical Report

See `TECHNICAL_REPORT.md` for the complete technical report.

---

## Next Steps

### Immediate Actions

1. **VERIFY:** Check training code for `hop_length` (256 or 160?)
2. **COPY:** Get label list from training code
3. **VERIFY:** Check training preprocessing steps
4. **FIX:** Apply fixes based on verification
5. **TEST:** Test with known audio files
6. **VALIDATE:** Verify predictions match training

### Priority Order

1. **CRITICAL:** Label order (affects all predictions)
2. **CRITICAL:** hop_length (affects all predictions)
3. **IMPORTANT:** Preprocessing steps (affects all predictions)
4. **MINOR:** Clean up duplicate modules

### Status

- [ ] hop_length verified and fixed
- [ ] Label order copied from training code
- [ ] Preprocessing steps verified
- [ ] Code fixes applied
- [ ] Tested with known audio files
- [ ] Predictions validated

---

## Conclusion

The system has critical preprocessing mismatches that cause wrong predictions. The main issues are:

1. **hop_length mismatch** (256 vs 160) - CRITICAL
2. **Label order unknown** - CRITICAL
3. **Preprocessing steps need verification** - IMPORTANT

All issues must be fixed by verifying with the training code and matching exactly. Once fixed, predictions should match training evaluation results.

