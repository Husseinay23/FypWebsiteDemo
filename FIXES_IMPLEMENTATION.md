# Implementation Guide: Fixes for ADI Web System

## Critical Issues to Fix

### Issue 1: hop_length Mismatch (CRITICAL - REQUIRES VERIFICATION)

**Problem:**
- All `run_config.json` files specify `hop_length: 256`
- Code uses `hop_length=160` with comment "CRITICAL: training uses 160, not 256!"
- This is a direct contradiction

**Decision:**
- **Fix to match run_config.json (256)** since these are the actual training configs
- **BUT:** User must verify with training code before deploying

**Files to Update:**
1. `backend/config.py`: Change `HOP_LENGTH = 256`
2. `backend/audio_preprocessing.py`: Change `hop_length=256` in all functions
3. Update function signatures and docstrings

**Verification Required:**
1. Check actual training code for mel-spectrogram computation
2. Verify `hop_length` parameter used in training
3. If training used 160: revert change and update config files
4. If training used 256: keep fix, verify mel shapes match

### Issue 2: Label Order (CRITICAL - REQUIRES USER ACTION)

**Problem:**
- `backend/labels_22.py` has placeholder
- TODO comment says to copy from training code
- Wrong label order → correct index, wrong dialect name

**Action Required:**
1. Open training codebase
2. Find: `adc/notebooks/lib/io_paths.py` → `get_22_country_labels()`
3. Copy exact list (including exact spelling, capitalization)
4. Paste into `backend/labels_22.py`
5. Verify length is 22

**Files to Update:**
1. `backend/labels_22.py`: Replace placeholder with exact list

**Verification:**
1. Run: `python backend/debug_labels.py`
2. Compare indices with training notebook output
3. Test with known audio file

### Issue 3: Preprocessing Steps (REQUIRES VERIFICATION)

**Problem:**
- Code uses pre-emphasis (coef=0.97) and trim silence (top_db=30)
- Training configs don't specify these
- Unknown if training used these steps

**Action Required:**
1. Check training preprocessing code
2. Verify if pre-emphasis was used
3. Verify if trim silence was used
4. Match exactly

**If Training Used Pre-emphasis:**
- Keep current code ✅

**If Training Did NOT Use Pre-emphasis:**
- Remove pre-emphasis from `waveform_to_mel_db()`

**If Training Used Different Coef:**
- Update coef value to match training

**Same for Trim Silence:**
- Verify top_db value (30 or different)
- Remove if not used

### Issue 4: Clean Up Duplicate Modules

**Problem:**
- Two preprocessing modules: `audio.py` (old) and `audio_preprocessing.py` (new)
- `main.py` uses `audio_preprocessing.py` ✅
- `test_inference.py` uses `audio.py` ❌

**Action:**
1. Update `test_inference.py` to use `audio_preprocessing.py`
2. Optionally: Remove `audio.py` or mark as deprecated
3. Update any other files using `audio.py`

---

## Step-by-Step Implementation

### Step 1: Fix hop_length (After Verification)

**⚠️ VERIFY WITH TRAINING CODE FIRST**

If training used `hop_length=256` (from run_config.json):

```python
# backend/config.py
HOP_LENGTH = 256  # Changed from 160

# backend/audio_preprocessing.py
# Update waveform_to_mel_db:
hop_length=256,  # Changed from 160

# Update center_crop_mel_3s:
def center_crop_mel_3s(mel7, sr=16000, hop_length=256):  # Changed from 160
    frames_per_sec = int(round(sr / hop_length))  # 16000 / 256 = 62.5
    crop_frames = int(round(3 * frames_per_sec))  # 187 frames for 3s
    ...

# Update five_crop_mel_3s:
def five_crop_mel_3s(mel7, sr=16000, hop_length=256):  # Changed from 160
    ...
    frames_per_sec = int(round(sr / hop_length))  # 16000 / 256 = 62.5
    crop_frames = int(round(3 * frames_per_sec))  # 187 frames
    ...

# Update preprocess_for_inference:
mel_center = center_crop_mel_3s(mel7, sr=sr, hop_length=256)  # Changed from 160
crops = five_crop_mel_3s(mel7, sr=sr, hop_length=256)  # Changed from 160
```

**Expected Mel Shapes (if hop_length=256):**
- 7s audio: (128, 438) frames [700 samples/sec * 7s / 160 samples/frame = 437.5 ≈ 438]
- 3s audio: (128, 187) frames [62.5 frames/sec * 3s = 187.5 ≈ 187]

**⚠️ If training used 160:**
- Keep code as-is
- But update run_config.json files to 160
- Document the mismatch

### Step 2: Fix Label Order (User Action Required)

**User must:**
1. Open training codebase
2. Find label list function
3. Copy exact list
4. Paste into `backend/labels_22.py`

**Example (DO NOT USE - GET FROM TRAINING CODE):**
```python
# backend/labels_22.py
# REPLACE THIS WITH EXACT LIST FROM TRAINING CODE

DIALECT_LABELS = [
    "Bahrain", "Kuwait", "Oman", "Qatar", "Saudi Arabia", "UAE", "Yemen",
    "Iraq", "Jordan", "Lebanon", "Palestine", "Syria",
    "Algeria", "Libya", "Mauritania", "Morocco", "Tunisia",
    "Comoros", "Djibouti", "Egypt", "Somalia", "Sudan"
]
```

**Verification:**
```bash
python backend/debug_labels.py
# Compare output with training notebook idx_to_label
```

### Step 3: Verify Preprocessing Steps

**Check training code for:**

1. **Pre-emphasis:**
   - Search for: `preemphasis`, `pre-emphasis`
   - Check coef value (0.97 or different?)

2. **Trim Silence:**
   - Search for: `trim`, `silence`
   - Check top_db value (30 or different?)

3. **Mel-Spectrogram:**
   - Check: n_fft, hop_length, n_mels, fmin, fmax, win_length

4. **Normalization:**
   - Check: `power_to_db` or `log1p` or other?

**Update code to match exactly.**

### Step 4: Clean Up Duplicate Modules

**Update test_inference.py:**
```python
# OLD:
from backend.audio import load_audio, preprocess_audio

# NEW:
from backend.audio_preprocessing import preprocess_for_inference
from backend.models import ModelRegistry

# Update test function to use new API
```

**Optionally remove audio.py:**
- Check if any other files use it
- If not: remove or move to `_deprecated/`
- If yes: update those files first

---

## Testing After Fixes

### Test 1: Verify hop_length Fix

```python
# Test mel shape with hop_length=256
import numpy as np
from backend.audio_preprocessing import waveform_to_mel_db

# 7s audio at 16kHz
audio = np.zeros(7 * 16000, dtype=np.float32)
mel = waveform_to_mel_db(audio, sr=16000, n_mels=128)

# Expected: (128, 438) if hop_length=256
# Expected: (128, 700) if hop_length=160
print(f"Mel shape: {mel.shape}")
print(f"Expected: (128, 438) for hop_length=256")
print(f"Expected: (128, 700) for hop_length=160")
```

### Test 2: Verify Label Order

```bash
python backend/debug_labels.py
# Compare output indices with training notebook
```

### Test 3: End-to-End Test

```bash
# Use known audio file from ADC dataset
python backend/test_preprocessing.py path/to/test_audio.wav --model resnet18 --window_mode 7s

# Verify:
# 1. Predicted dialect matches ground truth
# 2. Confidence > 0.5 (ideally > 0.8)
# 3. Top-K predictions are reasonable
```

### Test 4: Consistency Test

```bash
# Run same audio multiple times
python backend/test_preprocessing.py test.wav --model resnet18 --window_mode 7s
python backend/test_preprocessing.py test.wav --model resnet18 --window_mode 7s
python backend/test_preprocessing.py test.wav --model resnet18 --window_mode 7s

# Verify: Predictions are identical (deterministic)
```

---

## Verification Checklist

After implementing fixes:

- [ ] hop_length matches training (verified with training code)
- [ ] Label order matches training (copied from training code)
- [ ] Preprocessing steps match training (verified with training code)
- [ ] Mel-spectrogram shapes match expected values
- [ ] Test with known audio file → correct prediction
- [ ] Test consistency → same predictions for same audio
- [ ] All models load successfully
- [ ] All window modes work correctly
- [ ] No errors in logs
- [ ] Confidences are reasonable (>0.5 for correct predictions)

---

## Deployment Notes

1. **Before deploying:** Verify all fixes with training code
2. **Document:** Any assumptions made (e.g., hop_length=256 if verified)
3. **Monitor:** Check prediction logs for anomalies
4. **Validate:** Test with multiple known audio files
5. **Rollback plan:** Keep old code as backup until verified

---

## Critical Notes

1. **Do NOT deploy without verifying training code**
2. **Label order is CRITICAL** - wrong order = wrong dialect names
3. **hop_length mismatch** - causes wrong predictions
4. **Preprocessing must match exactly** - small differences cause big errors
5. **Test thoroughly** - use known audio files from ADC dataset

