"""
Test script for inference pipeline.
"""
import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import ModelRegistry
from backend.audio import load_audio, preprocess_audio
from backend.config import SAMPLE_RATE

def test_inference(audio_path: str, model_name: str = "resnet18", window_mode: str = "auto"):
    """
    Test inference on an audio file.
    
    Args:
        audio_path: Path to audio file
        model_name: Model to use
        window_mode: Window mode
    """
    print(f"Testing inference with {model_name} on {audio_path}")
    
    # Load model registry
    print("Loading models...")
    registry = ModelRegistry()
    
    # Load audio
    print("Loading audio...")
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    
    waveform, sr = load_audio(audio_bytes, Path(audio_path).name)
    print(f"Audio loaded: {len(waveform)} samples at {sr} Hz ({len(waveform)/sr:.2f}s)")
    
    # Preprocess
    print(f"Preprocessing with window_mode={window_mode}...")
    mel_spec, actual_mode = preprocess_audio(waveform, sr, window_mode)
    print(f"Actual window mode: {actual_mode}")
    
    if isinstance(mel_spec, list):
        print(f"5-crop mode: {len(mel_spec)} mel spectrograms")
        print(f"Shape of each: {mel_spec[0].shape}")
    else:
        print(f"Mel spectrogram shape: {mel_spec.shape}")
    
    # Predict
    print("Running prediction...")
    model = registry.get_model(model_name)
    prediction = model.predict(mel_spec, actual_mode)
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Window Mode: {actual_mode}")
    print(f"Predicted Dialect: {prediction['dialect']}")
    print(f"Confidence: {prediction['confidence']:.4f} ({prediction['confidence']*100:.2f}%)")
    print("\nTop 5 Predictions:")
    for i, item in enumerate(prediction['top_k'], 1):
        print(f"  {i}. {item['dialect']}: {item['prob']:.4f} ({item['prob']*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default="resnet18", help="Model name")
    parser.add_argument("--window_mode", default="auto", help="Window mode")
    args = parser.parse_args()
    
    test_inference(args.audio_path, args.model, args.window_mode)

