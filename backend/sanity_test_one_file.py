"""
Sanity test script to verify preprocessing and label order.
Run with a known test file from ADC dataset.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.audio_preprocessing import (
    load_audio_16k_mono,
    make_7s_waveform,
    waveform_to_mel_db,
    center_crop_mel_3s,
    mel_to_model_tensor,
)
from backend.models import ModelRegistry
from backend.labels_22 import DIALECT_LABELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mel_to_tensor(mel_db: np.ndarray) -> torch.Tensor:
    """Convert mel to model tensor."""
    t = torch.from_numpy(mel_db).float()
    t = t.unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity test for one audio file")
    parser.add_argument("wav_path", help="Path to WAV file")
    parser.add_argument("--model", default="resnet18", help="Model name")
    parser.add_argument("--mode", default="7s", choices=["7s", "3s_center"], help="Window mode")
    args = parser.parse_args()

    print("=" * 60)
    print("SANITY TEST - Single File Inference")
    print("=" * 60)
    print(f"File: {args.wav_path}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print()

    # Load audio
    print("Step 1: Loading audio...")
    try:
        with open(args.wav_path, "rb") as f:
            audio_bytes = f.read()
        y16 = load_audio_16k_mono(audio_bytes)
        print(f"  ✓ Loaded: {len(y16)} samples ({len(y16)/16000:.2f}s)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        sys.exit(1)

    # Preprocess
    print(f"\nStep 2: Preprocessing ({args.mode})...")
    try:
        y7 = make_7s_waveform(y16)
        mel7 = waveform_to_mel_db(y7)
        print(f"  ✓ 7s mel shape: {mel7.shape}")

        if args.mode == "7s":
            mel_for_model = mel7
        else:
            mel_for_model = center_crop_mel_3s(mel7)
            print(f"  ✓ 3s center mel shape: {mel_for_model.shape}")

        x = mel_to_tensor(mel_for_model).to(device)
        print(f"  ✓ Model tensor shape: {x.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load model
    print(f"\nStep 3: Loading model ({args.model})...")
    try:
        registry = ModelRegistry()
        model = registry.get_model(args.model)
        print(f"  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Inference
    print(f"\nStep 4: Running inference...")
    try:
        model.model.eval()
        with torch.inference_mode():
            logits = model.model(x.to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        print(f"  ✓ Inference complete")
        print(f"  ✓ Logits shape: {logits.shape}")
        print(f"  ✓ Probabilities shape: {probs.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    pred_idx = int(np.argmax(probs))
    pred_label = DIALECT_LABELS[pred_idx]
    confidence = float(probs[pred_idx])
    
    print(f"Predicted Index: {pred_idx}")
    print(f"Predicted Label: {pred_label}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print()
    print("Top 5 Predictions:")
    top5_indices = np.argsort(probs)[-5:][::-1]
    for i, idx in enumerate(top5_indices, 1):
        print(f"  {i}. [{idx:2d}] {DIALECT_LABELS[idx]:20s}: {probs[idx]:.4f} ({probs[idx]*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Label Index Reference:")
    print("=" * 60)
    for idx, label in enumerate(DIALECT_LABELS):
        marker = " <-- PREDICTED" if idx == pred_idx else ""
        print(f"  {idx:2d} -> {label}{marker}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)
    print("1. Compare 'Predicted Index' with training notebook idx_to_label mapping")
    print("2. Verify the predicted label matches the known dialect for this test file")
    print("3. If index/label mismatch, update backend/labels_22.py with correct order")
    print("4. Confidence should be >0.5 for good predictions")
    
    if confidence < 0.5:
        print("\n⚠ WARNING: Low confidence - check preprocessing or model loading")

