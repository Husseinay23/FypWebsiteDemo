"""
Test script to verify preprocessing pipeline matches training.
Run this with a test audio file to check predictions are reasonable.
"""
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.audio_preprocessing import preprocess_for_inference
from backend.models import ModelRegistry
from backend.labels_22 import DIALECT_LABELS


def test_inference(audio_path: str, model_name: str = "resnet18", window_mode: str = "7s"):
    """
    Test inference on an audio file using the new preprocessing pipeline.
    
    Args:
        audio_path: Path to audio file
        model_name: Model to use
        window_mode: Window mode
    """
    print(f"Testing inference with {model_name} on {audio_path}")
    print(f"Window mode: {window_mode}")
    print("=" * 60)
    
    # Load audio file
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    
    print("Step 1: Preprocessing audio...")
    try:
        model_tensor, actual_window_mode, original_duration = preprocess_for_inference(
            audio_bytes,
            window_mode
        )
        print(f"  ✓ Preprocessing complete")
        print(f"  - Original duration: {original_duration:.2f}s")
        print(f"  - Actual window mode: {actual_window_mode}")
        print(f"  - Tensor shape: {model_tensor.shape}")
    except Exception as e:
        print(f"  ✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nStep 2: Loading model...")
    try:
        registry = ModelRegistry()
        model = registry.get_model(model_name)
        print(f"  ✓ Model loaded: {model_name}")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nStep 3: Running inference...")
    try:
        prediction = model.predict_from_tensor(model_tensor, actual_window_mode)
        print(f"  ✓ Inference complete")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Window Mode: {actual_window_mode}")
    print(f"Predicted Dialect: {prediction['dialect']}")
    print(f"Confidence: {prediction['confidence']:.4f} ({prediction['confidence']*100:.2f}%)")
    print("\nTop 5 Predictions:")
    for i, item in enumerate(prediction['top_k'], 1):
        print(f"  {i}. {item['dialect']:20s}: {item['prob']:.4f} ({item['prob']*100:.2f}%)")
    print("=" * 60)
    
    # Check if prediction is reasonable
    if prediction['confidence'] < 0.5:
        print("\n⚠ WARNING: Low confidence prediction. This might indicate:")
        print("  - Preprocessing mismatch with training")
        print("  - Audio quality issues")
        print("  - Model not properly loaded")
    else:
        print("\n✓ High confidence prediction - preprocessing appears correct!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test inference pipeline")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default="resnet18", help="Model name")
    parser.add_argument("--window_mode", default="7s", help="Window mode")
    args = parser.parse_args()
    
    test_inference(args.audio_path, args.model, args.window_mode)

