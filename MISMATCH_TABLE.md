# Mismatch Table: Website vs Training Contract

## Training Contract (Source of Truth)

Based on the provided training contract, these are the EXACT requirements:

| Parameter           | Training Contract                         | Current Website       | Status                | File:Line                                |
| ------------------- | ----------------------------------------- | --------------------- | --------------------- | ---------------------------------------- |
| **sample_rate**     | 16000                                     | 16000                 | ✅ MATCH              | config.py:8                              |
| **n_fft**           | 2048                                      | 1024                  | ❌ MISMATCH           | config.py:11, audio_preprocessing.py:132 |
| **hop_length**      | 256                                       | 160                   | ❌ MISMATCH           | config.py:9, audio_preprocessing.py:133  |
| **n_mels**          | 128                                       | 128                   | ✅ MATCH              | config.py:10                             |
| **f_min**           | 20.0                                      | 50                    | ❌ MISMATCH           | config.py:13, audio_preprocessing.py:136 |
| **f_max**           | 7600.0                                    | 7600                  | ✅ MATCH              | config.py:14, audio_preprocessing.py:137 |
| **feature_scale**   | AmplitudeToDB                             | power_to_db           | ✅ MATCH              | audio_preprocessing.py:142               |
| **normalization**   | per-sample (as in inference_canonical.py) | power_to_db(ref=max)  | ⚠️ NEEDS VERIFICATION | audio_preprocessing.py:142               |
| **pre-emphasis**    | NO                                        | YES (coef=0.97)       | ❌ MISMATCH           | audio_preprocessing.py:118               |
| **trim-silence**    | NO                                        | YES (top_db=30)       | ❌ MISMATCH           | audio_preprocessing.py:121               |
| **window_padding**  | RIGHT pad if < 7s                         | CENTER pad if < 7s    | ❌ MISMATCH           | audio_preprocessing.py:64-92             |
| **window_cropping** | CENTER crop if > 7s                       | CENTER crop if > 7s   | ✅ MATCH              | audio_preprocessing.py:87-89             |
| **label_format**    | Underscores (Saudi_Arabia)                | Spaces (Saudi Arabia) | ❌ MISMATCH           | labels_22.py:20-24                       |
| **label_order**     | Must match training exactly               | Placeholder TODO      | ❌ MISMATCH           | labels_22.py:9-24                        |

## Detailed Mismatches

### 1. Mel-Spectrogram Parameters

**File: `backend/config.py`**

- Line 9: `HOP_LENGTH = 160` → Should be `256`
- Line 11: `N_FFT = 1024` → Should be `2048`
- Line 13: `FMIN = 50` → Should be `20.0`

**File: `backend/audio_preprocessing.py`**

- Line 132: `n_fft=1024` → Should be `2048`
- Line 133: `hop_length=160` → Should be `256`
- Line 136: `fmin=50` → Should be `20.0`

### 2. Preprocessing Steps (MUST REMOVE)

**File: `backend/audio_preprocessing.py`**

- Lines 117-118: Pre-emphasis → **MUST DELETE**
- Lines 120-121: Trim silence → **MUST DELETE**
- Lines 123-126: Trim fallback handling → **MUST DELETE** (no trim, so no fallback needed)

### 3. Window Padding (WRONG DIRECTION)

**File: `backend/audio_preprocessing.py`**

- Lines 79-84: Center-padding → Should be RIGHT-padding
- Current: `pad_left = padding // 2; pad_right = padding - pad_left`
- Required: `pad_left = 0; pad_right = padding`

### 4. Label Format and Order

**File: `backend/labels_22.py`**

- Lines 20-24: Uses spaces (e.g., "Saudi Arabia") → Should use underscores (e.g., "Saudi_Arabia")
- Lines 9-24: Placeholder TODO → Must get exact list from training
- Missing assertions: Should have startup assertions for label validation

### 5. Audio Decoding

**File: `backend/audio_preprocessing.py` (load_audio_16k_mono)**

- Lines 16-61: Uses librosa.load() → May not handle WEBM/OPUS reliably
- Missing: FFmpeg-based decoding for WEBM/OPUS
- Required: Create `backend/audio_decode.py` with FFmpeg-based decoding

### 6. Normalization (NEEDS VERIFICATION)

**File: `backend/audio_preprocessing.py`**

- Line 142: `power_to_db(S, ref=np.max)` → Per-sample normalization
- Contract says "per-sample normalization exactly as done in inference_canonical.py"
- Current: Uses `ref=np.max` (per-spectrogram normalization)
- Status: ⚠️ NEEDS VERIFICATION - may need to change to per-sample

### 7. Model Input Adapters

**File: `backend/models.py`**

- EfficientNet-B3: Has 3-channel handling but may need resize to 300x300
- SCNN: May need fixed 700 frames (needs verification)
- Missing: Explicit adapter functions as specified in contract

### 8. Debug Mode

**File: `backend/main.py`**

- Missing: `/predict?debug=true` query parameter support
- Missing: Debug response fields as specified in contract

## Summary of Required Changes

### Critical (Must Fix):

1. ✅ Change `n_fft` from 1024 to 2048 (config.py, audio_preprocessing.py)
2. ✅ Change `hop_length` from 160 to 256 (config.py, audio_preprocessing.py)
3. ✅ Change `f_min` from 50 to 20.0 (config.py, audio_preprocessing.py)
4. ✅ Remove pre-emphasis (audio_preprocessing.py)
5. ✅ Remove trim-silence (audio_preprocessing.py)
6. ✅ Change padding from center to right (audio_preprocessing.py)
7. ✅ Fix label format to use underscores (labels_22.py)
8. ✅ Get exact label order from training (labels_22.py)

### Important (Should Fix):

9. ⚠️ Verify normalization (per-sample vs per-spectrogram)
10. ✅ Create audio_decode.py for WEBM/OPUS (FFmpeg-based)
11. ✅ Implement model input adapters (EfficientNet resize, SCNN fixed frames)
12. ✅ Add debug mode to /predict endpoint

### Files to Modify:

- `backend/config.py`
- `backend/audio_preprocessing.py` (or replace with canonical implementation)
- `backend/labels_22.py`
- `backend/main.py`
- `backend/models.py`

### Files to Create:

- `backend/audio_decode.py` (FFmpeg-based audio decoding)
- `test_api.py` (validation script)

### Files to Deprecate/Remove:

- `backend/audio.py` (old preprocessing, duplicate)
