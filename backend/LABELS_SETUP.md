# Label Order Setup Instructions

## Critical: Label Order Must Match Training

The backend uses `backend/labels_22.py` as the **single source of truth** for label order.
This MUST match the exact order used during training.

## How to Get the Correct Label Order

1. **Open the training repository** (the one with `ADC_ResNet18.ipynb`)

2. **Navigate to:** `adc/notebooks/lib/io_paths.py`

3. **Find the function:** `get_22_country_labels()`

4. **Copy the EXACT list** it returns (including exact spelling, capitalization, spacing)

5. **Replace the list in:** `backend/labels_22.py`

## Example

If the training code has:

```python
def get_22_country_labels():
    return [
        "Lebanon",
        "Syria", 
        "Jordan",
        # ... etc (exact order from training)
    ]
```

Then `backend/labels_22.py` should have:

```python
DIALECT_LABELS = [
    "Lebanon",
    "Syria",
    "Jordan",
    # ... exact same order, exact same spelling
]
```

## Verification

After updating `labels_22.py`:

1. **Run the debug script:**
   ```bash
   python backend/debug_labels.py
   ```

2. **Compare output with training notebook:**
   - The indices printed should match `idx_to_label` from training
   - Index 0 should be the first label, index 21 should be the last

3. **Run sanity test:**
   ```bash
   python backend/sanity_test_one_file.py path/to/known_test_file.wav --model resnet18
   ```
   - The predicted index should match what the training notebook would predict
   - The predicted label should be correct for the known test file

## Current Status

⚠ **WARNING:** The current `labels_22.py` contains a placeholder list that is likely incorrect.

**You MUST replace it with the exact list from training code before the backend will work correctly.**

## Why This Matters

If the label order doesn't match:
- Predictions will have high confidence but wrong labels
- Index 0 might map to "Lebanon" in training but "Bahrain" in inference
- All predictions will be semantically incorrect

This is a **critical** fix that must be done before deployment.

