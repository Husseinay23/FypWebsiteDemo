# Final Deliverables: Website Backend Alignment to Training Contract

## Executive Summary

This document provides the final deliverables for aligning the website backend with the training contract exactly.

**Status:** Core preprocessing implemented, remaining tasks documented.

---

## 1. Mismatch Table

**File:** `MISMATCH_TABLE.md`

Complete analysis of all mismatches between website code and training contract, with file paths and line references.

---

## 2. Commit-Style Summary of File Changes

### Created Files:

1. **`backend/preprocessing_canonical.py`** - NEW

   - Canonical preprocessing matching training contract exactly
   - NO pre-emphasis, NO trim-silence
   - Exact mel parameters (n_fft=2048, hop_length=256, f_min=20.0, f_max=7600.0)
   - RIGHT-padding for short audio
   - Returns debug_info for debug mode

2. **`backend/audio_decode.py`** - NEW

   - FFmpeg-based audio decoding
   - Handles WEBM/OPUS, MP3, WAV robustly
   - Startup FFmpeg check

3. **`MISMATCH_TABLE.md`** - NEW

   - Complete mismatch analysis

4. **`IMPLEMENTATION_SUMMARY.md`** - NEW

   - Detailed implementation summary

5. **`FINAL_DELIVERABLES.md`** - NEW (this file)
   - Final deliverables summary

### Modified Files:

1. **`backend/config.py`**

   - `HOP_LENGTH`: 160 → 256
   - `N_FFT`: 1024 → 2048
   - `FMIN`: 50 → `F_MIN`: 20.0
   - `FMAX`: 7600 → `F_MAX`: 7600.0
   - Removed `WIN_LENGTH`

2. **`backend/labels_22.py`**
   - Updated to use underscores format (Saudi_Arabia, United_Arab_Emirates)
   - Added startup assertions
   - ⚠️ Still needs actual label list from training code

### Files Requiring Updates (Next Steps):

3. **`backend/main.py`** - REQUIRED

   - Use `preprocessing_canonical.preprocess_for_inference_canonical`
   - Add debug mode support
   - Use `audio_decode.load_audio_from_bytes`

4. **`backend/models.py`** - REQUIRED

   - Add model input adapter functions
   - Update `predict_from_tensor()` to use adapters

5. **`test_api.py`** - REQUIRED (NEW)
   - Create validation script

---

## 3. Updated Backend Files

### A. `backend/config.py`

See file for complete changes. Key updates:

- Parameters match training contract exactly

### B. `backend/preprocessing_canonical.py`

Complete canonical preprocessing implementation matching training contract.

**Key Functions:**

- `preprocess_for_inference_canonical()` - Main preprocessing function
- `waveform_to_mel_db_canonical()` - Mel-spectrogram conversion
- `make_7s_waveform_right_pad()` - RIGHT-padding (not center-padding)
- `center_crop_mel_3s()` - Center 3s crop
- `five_crop_mel_3s()` - 5-crop extraction

### C. `backend/audio_decode.py`

FFmpeg-based audio decoding for robust WEBM/OPUS support.

**Key Functions:**

- `check_ffmpeg()` - Check FFmpeg availability
- `decode_audio_bytes_to_wav_pcm16()` - FFmpeg decoding
- `load_audio_from_bytes()` - Load audio using FFmpeg

### D. `backend/labels_22.py`

Updated label format with underscores and startup assertions.

**⚠️ ACTION REQUIRED:** Replace placeholder with actual label list from training code.

---

## 4. Instructions to Run Locally

### Prerequisites:

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

### Run Backend:

```bash
cd backend
python -m backend.main
# OR
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests:

```bash
# Test API (after creating test_api.py)
python test_api.py path/to/test_audio.wav

# Test preprocessing
python -m pytest backend/tests/ -v
```

---

## 5. Final Sanity-Check Checklist

### What to Run:

1. **Start Backend:**

   ```bash
   cd backend
   python -m backend.main
   ```

   - ✅ Should start without errors
   - ✅ Should load all models
   - ✅ Should check FFmpeg availability
   - ✅ Should validate label assertions

2. **Test Health Endpoint:**

   ```bash
   curl http://localhost:8000/health
   ```

   - ✅ Should return `{"status": "ok"}`

3. **Test Predict Endpoint (Normal Mode):**

   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -F "file=@test_audio.wav" \
     -F "model_name=resnet18" \
     -F "window_mode=7s"
   ```

   - ✅ Should return prediction (dialect, confidence, top_k, all_probs)

4. **Test Predict Endpoint (Debug Mode):**

   ```bash
   curl -X POST "http://localhost:8000/predict?debug=true" \
     -F "file=@test_audio.wav" \
     -F "model_name=resnet18" \
     -F "window_mode=7s"
   ```

   - ✅ Should return prediction + debug fields
   - ✅ Debug fields should include all specified fields

5. **Test Deterministic Output:**

   ```bash
   python test_api.py test_audio.wav
   ```

   - ✅ Same audio should produce identical predictions

6. **Test Known Audio File:**
   - Use audio file with known dialect from ADC dataset
   - ✅ Should predict correct dialect
   - ✅ Confidence > 0.5 (ideally > 0.8)

### What "Correct" Looks Like:

1. **Mel-Spectrogram Shapes:**

   - 7s audio: (128, 438) frames [with hop_length=256, n_fft=2048]
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

## 6. Critical Action Items

### ⚠️ HIGH PRIORITY:

1. **Get Actual Label List from Training Code**

   - Source: Training code (inference_canonical.py or training label mapping)
   - Action: Replace placeholder in `backend/labels_22.py`
   - Verify: Order and spelling match training exactly

2. **Update `backend/main.py`**

   - Use `preprocessing_canonical.preprocess_for_inference_canonical`
   - Add debug mode support
   - Use `audio_decode.load_audio_from_bytes`

3. **Add Model Input Adapters to `backend/models.py`**

   - `adapt_input_for_efficientnet()` - 3-channel + resize to 300x300
   - `adapt_input_for_scnn()` - Fixed 700 frames
   - `adapt_input_for_default_cnn()` - 1-channel, variable time
   - Update `predict_from_tensor()` to use adapters

4. **Create `test_api.py`**
   - Validation script as specified in contract
   - Test with known audio files
   - Assert deterministic outputs

### ⚠️ MEDIUM PRIORITY:

5. **Verify Model Input Requirements**

   - EfficientNet-B3: Verify 300x300 resize requirement
   - SCNN: Verify 700 frames fixed dimension requirement

6. **Verify Normalization Method**
   - Contract says "per-sample normalization"
   - Current: `ref=np.max` (per-spectrogram)
   - Action: Verify with training code

---

## 7. Summary of Completed Work

### ✅ Completed:

1. **Mismatch Analysis**

   - Complete mismatch table created
   - All mismatches identified with file:line references

2. **Configuration Updates**

   - `config.py` updated to match training contract
   - All parameters corrected (n_fft, hop_length, f_min, f_max)

3. **Canonical Preprocessing**

   - `preprocessing_canonical.py` created
   - Matches training contract exactly
   - NO pre-emphasis, NO trim-silence
   - RIGHT-padding (not center-padding)
   - Returns debug_info

4. **Audio Decoding**

   - `audio_decode.py` created
   - FFmpeg-based decoding for WEBM/OPUS
   - Startup FFmpeg check

5. **Label Format**

   - `labels_22.py` updated to use underscores
   - Startup assertions added
   - ⚠️ Still needs actual list from training

6. **Documentation**
   - Implementation summary created
   - Instructions provided
   - Sanity check checklist provided

### ⚠️ Remaining:

1. Update `main.py` (use canonical preprocessing + debug mode)
2. Add model input adapters to `models.py`
3. Create `test_api.py`
4. Get actual label list from training code
5. Verify model input requirements
6. Verify normalization method

---

## 8. Files Reference

### Documentation:

- `MISMATCH_TABLE.md` - Complete mismatch analysis
- `IMPLEMENTATION_SUMMARY.md` - Detailed implementation summary
- `FINAL_DELIVERABLES.md` - This file (final summary)

### Code (Created):

- `backend/preprocessing_canonical.py` - Canonical preprocessing
- `backend/audio_decode.py` - FFmpeg-based audio decoding

### Code (Modified):

- `backend/config.py` - Updated parameters
- `backend/labels_22.py` - Updated format + assertions

### Code (To Update):

- `backend/main.py` - Use canonical preprocessing + debug mode
- `backend/models.py` - Add model input adapters

### Code (To Create):

- `test_api.py` - Validation script

---

**Status:** Core preprocessing implemented and documented. Remaining tasks require integration and verification with training code.

**Next Steps:** Complete remaining tasks as specified in "Critical Action Items" section above.
