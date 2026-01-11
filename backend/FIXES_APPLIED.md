# Fixes Applied for Label Order and SCNN Issues

## Summary

Fixed two critical issues:
1. **Label order mismatch** - Created canonical labels module
2. **SCNN architecture/loading** - Added strict loading with debug output

## Changes Made

### 1. Canonical Label Module (`backend/labels_22.py`)

**Created:** Single source of truth for label order

- **Location:** `backend/labels_22.py`
- **Status:** ⚠️ **PLACEHOLDER** - Must be updated with exact order from training code
- **Action Required:** Copy exact list from `adc/notebooks/lib/io_paths.py` → `get_22_country_labels()`

**Updated all imports:**
- `backend/config.py` - Now imports from `labels_22.py`
- `backend/models.py` - Now imports from `labels_22.py`
- `backend/test_preprocessing.py` - Now imports from `labels_22.py`
- `backend/audio_preprocessing.py` - Removed unused import

### 2. SCNN Loading Improvements (`backend/models.py`)

**Added strict loading with debug output:**

- SCNN now loads with `strict=False` first to show missing/unexpected keys
- Then attempts `strict=True` to verify architecture match
- Prints warnings if architecture doesn't match checkpoint

**SCNN Architecture Notes:**
- Current definition: 4 conv layers (32→64→128→256 channels)
- Input shape: (batch, 1, 128, T)
- If predictions are wrong, verify against training code: `adc/models/scnn.py`

### 3. Debug Tools Created

**`backend/debug_labels.py`:**
- Prints label index → label name mapping
- Compare output with training notebook `idx_to_label`

**`backend/sanity_test_one_file.py`:**
- Complete inference test for single file
- Shows predicted index, label, confidence
- Prints full label reference for verification
- Usage: `python backend/sanity_test_one_file.py test.wav --model resnet18`

### 4. Documentation

**`backend/LABELS_SETUP.md`:**
- Step-by-step instructions for updating label order
- Verification checklist
- Why label order matters

## Next Steps (CRITICAL)

### Step 1: Update Label Order

1. Open training repo: `adc/notebooks/lib/io_paths.py`
2. Find `get_22_country_labels()` function
3. Copy the EXACT list it returns
4. Paste into `backend/labels_22.py` (replace the placeholder)
5. Run: `python backend/debug_labels.py`
6. Verify indices match training notebook output

### Step 2: Verify SCNN Architecture

1. Check training code: `adc/models/scnn.py` (or wherever SCNN is defined)
2. Compare with `backend/models.py` → `SpectralCNN` class
3. Ensure:
   - Same number of conv layers
   - Same channel progression
   - Same kernel sizes, padding, strides
   - Same normalization layers
   - Same final classifier dimensions
4. If different, update `SpectralCNN` to match exactly
5. Restart backend and check SCNN loading output for warnings

### Step 3: Test with Known File

1. Get a test file from ADC dataset where you know the dialect
2. Run: `python backend/sanity_test_one_file.py test_file.wav --model resnet18`
3. Verify:
   - Predicted index matches training notebook prediction
   - Predicted label is correct
   - Confidence is high (>0.5, ideally >0.8)

### Step 4: Monitor SCNN

After fixing label order, check SCNN predictions:
- Should NOT always predict Djibouti
- Should have reasonable confidence (>0.1, ideally >0.5)
- Should vary predictions based on input

If SCNN still misbehaves after label fix:
- Architecture mismatch (update `SpectralCNN` class)
- Checkpoint corruption (re-download model)
- Input shape issue (verify tensor is (batch, 1, 128, T))

## Files Changed

1. ✅ `backend/labels_22.py` - **NEW** (canonical label source)
2. ✅ `backend/config.py` - Updated to import from labels_22
3. ✅ `backend/models.py` - Updated imports, improved SCNN loading
4. ✅ `backend/debug_labels.py` - **NEW** (label verification tool)
5. ✅ `backend/sanity_test_one_file.py` - **NEW** (inference test)
6. ✅ `backend/LABELS_SETUP.md` - **NEW** (setup instructions)
7. ✅ `backend/FIXES_APPLIED.md` - **NEW** (this file)

## Verification Checklist

- [ ] Label order updated from training code
- [ ] `debug_labels.py` output matches training notebook
- [ ] SCNN architecture verified against training code
- [ ] SCNN loads with `strict=True` (no warnings)
- [ ] Sanity test predicts correct dialect for known file
- [ ] All models show reasonable predictions (not always same label)
- [ ] Confidences are reasonable (>0.1, ideally >0.5)

## Expected Results After Fixes

**Before (Current):**
- High confidence but wrong labels (label order mismatch)
- SCNN always predicts Djibouti with low confidence (~0.08)

**After (Expected):**
- Correct labels matching known test files
- SCNN shows varied, reasonable predictions
- All models have high confidence (>0.5) for correct predictions

