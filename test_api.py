"""
Test script for API validation.
Tests deterministic outputs and prints predictions for known audio files.
"""
import sys
import requests
from pathlib import Path
from typing import Optional


def test_api(
    audio_path: str,
    api_url: str = "http://localhost:8000",
    model_name: str = "resnet18",
    window_mode: str = "7s"
):
    """
    Test API with audio file and assert deterministic outputs.
    
    Args:
        audio_path: Path to audio file
        api_url: API base URL
        model_name: Model to use
        window_mode: Window mode
    """
    print(f"Testing API: {api_url}")
    print(f"Audio file: {audio_path}")
    print(f"Model: {model_name}, Window mode: {window_mode}")
    print("=" * 60)
    
    # Read audio file
    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return
    
    # Test 1: First request
    print("\n[Test 1] Sending first request...")
    with open(audio_path, 'rb') as f:
        files = {'file': f}
        data = {
            'model_name': model_name,
            'window_mode': window_mode
        }
        response1 = requests.post(f"{api_url}/predict", files=files, data=data)
    
    if response1.status_code != 200:
        print(f"ERROR: Request failed with status {response1.status_code}")
        print(f"Response: {response1.text}")
        return
    
    result1 = response1.json()
    print(f"✓ First request successful")
    print(f"  Predicted: {result1['dialect']} (confidence: {result1['confidence']:.4f})")
    
    # Test 2: Second request (should be identical)
    print("\n[Test 2] Sending second request (deterministic check)...")
    with open(audio_path, 'rb') as f:
        files = {'file': f}
        data = {
            'model_name': model_name,
            'window_mode': window_mode
        }
        response2 = requests.post(f"{api_url}/predict", files=files, data=data)
    
    if response2.status_code != 200:
        print(f"ERROR: Second request failed with status {response2.status_code}")
        return
    
    result2 = response2.json()
    print(f"✓ Second request successful")
    print(f"  Predicted: {result2['dialect']} (confidence: {result2['confidence']:.4f})")
    
    # Assert deterministic outputs
    print("\n[Test 3] Asserting deterministic outputs...")
    assert result1['dialect'] == result2['dialect'], \
        f"Dialect mismatch: {result1['dialect']} != {result2['dialect']}"
    assert abs(result1['confidence'] - result2['confidence']) < 1e-6, \
        f"Confidence mismatch: {result1['confidence']} != {result2['confidence']}"
    assert result1['top_k'] == result2['top_k'], \
        f"Top-K mismatch"
    print("✓ Deterministic outputs confirmed (identical predictions)")
    
    # Print top-5 predictions
    print("\n" + "=" * 60)
    print("TOP-5 PREDICTIONS:")
    print("=" * 60)
    for i, item in enumerate(result1['top_k'], 1):
        print(f"  {i}. {item['dialect']:25s}: {item['prob']:.4f} ({item['prob']*100:.2f}%)")
    
    # Test 4: Debug mode
    print("\n[Test 4] Testing debug mode...")
    with open(audio_path, 'rb') as f:
        files = {'file': f}
        data = {
            'model_name': model_name,
            'window_mode': window_mode
        }
        response_debug = requests.post(
            f"{api_url}/predict?debug=true",
            files=files,
            data=data
        )
    
    if response_debug.status_code == 200:
        result_debug = response_debug.json()
        if 'debug' in result_debug:
            print("✓ Debug mode working")
            print(f"  Mel shape: {result_debug['debug'].get('mel_shape_before_crop', 'N/A')}")
            print(f"  Tensor shape: {result_debug['debug'].get('tensor_shape_final', 'N/A')}")
        else:
            print("⚠ Debug mode response missing 'debug' field")
    else:
        print(f"⚠ Debug mode request failed (status {response_debug.status_code})")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <audio_file> [api_url] [model_name] [window_mode]")
        print("Example: python test_api.py test_audio.wav http://localhost:8000 resnet18 7s")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "resnet18"
    window_mode = sys.argv[4] if len(sys.argv) > 4 else "7s"
    
    test_api(audio_path, api_url, model_name, window_mode)

