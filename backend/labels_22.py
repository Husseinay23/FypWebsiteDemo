"""
Canonical label order for 22 Arabic dialects matching training exactly.
Labels use underscores (e.g., "Saudi_Arabia", "United_Arab_Emirates").

CRITICAL: This list MUST match the training label order exactly.
The order and spelling must be identical to the training code.

Source: Training code (e.g., inference_canonical.py or training label mapping)
"""

# IMPORTANT: Replace this placeholder with the EXACT list from training code
# The training contract specifies labels use underscores, not spaces
# Example format: "Saudi_Arabia", "United_Arab_Emirates"

# TODO: Get the EXACT list from training code and replace this placeholder
# The list below is a TEMPLATE showing the expected format with underscores
# Replace with actual training labels in exact order

# SOURCE: Training label mapping (adc/notebooks/lib/io_paths.py:get_22_country_labels)
# Canonical order used for all ADC training and evaluation

DIALECT_LABELS = [
    "Bahrain",
    "Kuwait",
    "Oman",
    "Qatar",
    "Saudi_Arabia",
    "United_Arab_Emirates",
    "Yemen",
    "Iraq",
    "Jordan",
    "Lebanon",
    "Palestine",
    "Syria",
    "Algeria",
    "Libya",
    "Mauritania",
    "Morocco",
    "Tunisia",
    "Comoros",
    "Djibouti",
    "Egypt",
    "Somalia",
    "Sudan",
]


# Startup assertions (as required by contract)
assert len(DIALECT_LABELS) == 22, f"Expected 22 labels, got {len(DIALECT_LABELS)}"
assert len(set(DIALECT_LABELS)) == 22, f"Duplicate labels found in DIALECT_LABELS"
assert "Saudi_Arabia" in DIALECT_LABELS, "Saudi_Arabia not found in DIALECT_LABELS"
assert "United_Arab_Emirates" in DIALECT_LABELS, "United_Arab_Emirates not found in DIALECT_LABELS"
