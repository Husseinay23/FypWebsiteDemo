# Critical Issues Requiring Immediate Attention

## ⚠️ CRITICAL: hop_length Mismatch

### Problem

**All `run_config.json` files specify:**
```json
{
  "hop_length": 256
}
```

**But code uses:**
```python
hop_length=160  # CRITICAL: training uses 160, not 256!
```

**This is a DIRECT CONTRADICTION.**

### Impact

- Wrong mel-spectrogram time dimension
- Model expects specific frame count → wrong predictions
- 7s audio: 256 → 438 frames, 160 → 700 frames (60% difference!)

### Decision Made

**Fix to match run_config.json (256)** since these are the actual training configs.

**BUT:** ⚠️ **MUST VERIFY WITH TRAINING CODE BEFORE DEPLOYING**

### Why This Decision

1. `run_config.json` files are generated during training
2. All models have the same config: `hop_length: 256`
3. Config files are the "source of truth" for training parameters
4. Code comment may be outdated or incorrect

### Verification Required

1. Check actual training code for mel-spectrogram computation
2. Verify `hop_length` parameter used in training
3. Compare with training preprocessing function
4. If training used 160: revert change and update config files
5. If training used 256: keep fix, verify mel shapes match

### Expected Behavior After Fix

- Mel-spectrogram shape for 7s audio: (128, 438) frames
- Mel-spectrogram shape for 3s audio: (128, 187) frames
- Predictions should match training evaluation results

---

## ⚠️ CRITICAL: Label Order Unknown

### Problem

`backend/labels_22.py` contains a placeholder:
```python
# TODO: Replace this with the EXACT list from training code
DIALECT_LABELS = [
    "Bahrain", "Kuwait", "Oman", ...
]
```

### Impact

- Wrong dialect names for correct indices
- Model predicts index 5 → code says "UAE" but training says "Saudi Arabia"
- Predictions appear correct (high confidence) but labels are wrong

### Fix Required

**User must copy exact label list from training code:**
1. Open training codebase
2. Find: `adc/notebooks/lib/io_paths.py` → `get_22_country_labels()`
3. Copy EXACT list (including exact spelling, capitalization, order)
4. Paste into `backend/labels_22.py`
5. Verify length is 22

### Verification

Run:
```bash
python backend/debug_labels.py
```

Compare output indices with training notebook `idx_to_label` mapping.

---

## ⚠️ IMPORTANT: Preprocessing Steps Unknown

### Problem

Code uses:
- Pre-emphasis (coef=0.97)
- Trim silence (top_db=30)

But training configs don't specify these. Unknown if training used them.

### Impact

- If training didn't use pre-emphasis: wrong feature extraction
- If training didn't use trim: different signal characteristics
- Mismatch → wrong predictions

### Fix Required

**Verify with training code:**
1. Check training preprocessing function
2. Verify if pre-emphasis was used
3. Verify if trim silence was used
4. Match exactly: same steps, same parameters

---

## Action Items

### Before Deploying

1. ✅ Verify `hop_length` with training code
2. ✅ Copy label list from training code
3. ✅ Verify preprocessing steps match training
4. ✅ Test with known audio files
5. ✅ Verify predictions match training evaluation

### Priority

1. **CRITICAL:** Label order (affects all predictions)
2. **CRITICAL:** hop_length (affects all predictions)
3. **IMPORTANT:** Preprocessing steps (affects all predictions)

### Status

- [ ] hop_length verified and fixed
- [ ] Label order copied from training code
- [ ] Preprocessing steps verified
- [ ] Tested with known audio files
- [ ] Predictions match training evaluation

