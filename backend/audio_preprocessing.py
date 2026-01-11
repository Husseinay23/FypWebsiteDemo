"""
Audio preprocessing pipeline that exactly matches the training pipeline.
This module ensures inference preprocessing is identical to training preprocessing.
"""
import numpy as np
import torch
import librosa
from typing import Tuple, List
import io
import tempfile
import os

from backend.config import SAMPLE_RATE


def load_audio_16k_mono(file_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio from bytes and convert to 16kHz mono float32 waveform.
    
    Args:
        file_bytes: Audio file bytes
        target_sr: Target sample rate (default 16000)
        
    Returns:
        Waveform as np.float32 array in [-1, 1], shape (T,)
    """
    filename_lower = ""  # We don't have filename in this context
    is_web_format = False
    
    # Try to detect format from magic bytes or use temp file approach
    temp_path = None
    try:
        # Create temporary file - librosa handles format detection
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        
        # Load with librosa - it will auto-detect format and resample to target_sr
        waveform, sr = librosa.load(
            temp_path,
            sr=target_sr,  # This automatically resamples to 16kHz
            mono=True,
            res_type='kaiser_best'
        )
        
        # Ensure float32 and in [-1, 1] range
        waveform = waveform.astype(np.float32)
        
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
    
    if len(waveform) == 0:
        raise ValueError("Loaded audio is empty")
    
    return waveform


def make_7s_waveform(y: np.ndarray, sr: int = 16000, target_seconds: float = 7.0) -> np.ndarray:
    """
    Standardize waveform to exactly 7 seconds by center-cropping or center-padding.
    
    Args:
        y: Input waveform (1D array)
        sr: Sample rate
        target_seconds: Target duration in seconds (default 7.0)
        
    Returns:
        Waveform of exactly target_length samples, float32
    """
    target_length = int(target_seconds * sr)  # 112000 samples for 7s at 16k
    current_length = len(y)
    
    if current_length < target_length:
        # Center-pad with zeros
        padding = target_length - current_length
        pad_left = padding // 2
        pad_right = padding - pad_left
        y_padded = np.pad(y, (pad_left, pad_right), mode='constant', constant_values=0.0)
        return y_padded.astype(np.float32)
    elif current_length > target_length:
        # Center-crop
        start = (current_length - target_length) // 2
        end = start + target_length
        return y[start:end].astype(np.float32)
    else:
        return y.astype(np.float32)


def waveform_to_mel_db(
    y: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Convert waveform to mel-spectrogram in dB scale, exactly matching training pipeline.
    
    Steps (in order):
    1. Pre-emphasis (coef=0.97)
    2. Trim silence (top_db=30)
    3. Mel-spectrogram with exact training parameters
    4. Convert to dB scale
    
    Args:
        y: Input waveform (1D array, float32)
        sr: Sample rate (default 16000)
        n_mels: Number of mel filter banks (default 128)
        
    Returns:
        Mel-spectrogram in dB, shape (n_mels, T_frames)
    """
    # Step 1: Pre-emphasis (exactly as in training)
    y = librosa.effects.preemphasis(y, coef=0.97)
    
    # Step 2: Trim silence (exactly as in training)
    y, _ = librosa.effects.trim(y, top_db=30)
    
    # Step 3: Handle edge case - if entire signal is silent after trim
    if y.size == 0:
        # Fallback: use zeros for 7 seconds
        y = np.zeros(int(sr * 7.0), dtype=np.float32)
    
    # Step 4: Compute mel-spectrogram with EXACT training parameters
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=160,  # CRITICAL: training uses 160, not 256!
        win_length=400,
        n_mels=n_mels,
        fmin=50,        # CRITICAL: training uses 50, not 0!
        fmax=7600,      # CRITICAL: training uses 7600, not sr/2!
        power=2.0,
    )
    
    # Step 5: Convert to dB scale (exactly as in training)
    mel_db = librosa.power_to_db(S, ref=np.max)
    
    return mel_db


def mel_to_model_tensor(mel_db: np.ndarray) -> torch.Tensor:
    """
    Convert mel-spectrogram to model input tensor.
    
    Args:
        mel_db: Mel-spectrogram in dB, shape (n_mels, T)
        
    Returns:
        Tensor with shape (1, 1, n_mels, T) for model input
    """
    tensor = torch.from_numpy(mel_db).float()
    
    # Ensure 2D: (n_mels, T)
    if tensor.ndim == 1:
        raise ValueError(f"Expected 2D mel, got 1D with shape {tensor.shape}")
    
    # Add batch and channel dimensions: (1, 1, n_mels, T)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # (1, n_mels, T)
    tensor = tensor.unsqueeze(0)      # (1, 1, n_mels, T)
    
    return tensor


def center_crop_mel_3s(mel7: np.ndarray, sr: int = 16000, hop_length: int = 160) -> np.ndarray:
    """
    Extract center 3-second crop from 7-second mel-spectrogram.
    
    Args:
        mel7: 7-second mel-spectrogram, shape (n_mels, T7)
        sr: Sample rate
        hop_length: Hop length used in mel computation
        
    Returns:
        3-second mel-spectrogram, shape (n_mels, T3)
    """
    frames_per_sec = int(round(sr / hop_length))  # 16000 / 160 = 100 frames/sec
    crop_frames = int(round(3 * frames_per_sec))  # 300 frames for 3 seconds
    
    T = mel7.shape[1]
    
    if T <= crop_frames:
        # Rare edge case: pad symmetrically
        pad = crop_frames - T
        left = pad // 2
        right = pad - left
        return np.pad(mel7, ((0, 0), (left, right)), mode="constant")
    
    # Center crop
    start = max(0, (T - crop_frames) // 2)
    end = start + crop_frames
    return mel7[:, start:end]


def five_crop_mel_3s(mel7: np.ndarray, sr: int = 16000, hop_length: int = 160) -> List[np.ndarray]:
    """
    Extract 5 evenly-spaced 3-second crops from 7-second mel-spectrogram.
    
    Args:
        mel7: 7-second mel-spectrogram, shape (n_mels, T7)
        sr: Sample rate
        hop_length: Hop length used in mel computation
        
    Returns:
        List of 5 mel-spectrograms, each shape (n_mels, T3)
    """
    frames_per_sec = int(round(sr / hop_length))
    crop_frames = int(round(3 * frames_per_sec))  # 300 frames
    
    T = mel7.shape[1]
    
    if T <= crop_frames:
        # Edge case: repeat center crop 5 times
        center = center_crop_mel_3s(mel7, sr=sr, hop_length=hop_length)
        return [center] * 5
    
    # Calculate 5 evenly-spaced positions
    positions = [
        0,  # left
        max(0, (T - crop_frames) // 4),  # left-center
        max(0, (T - crop_frames) // 2),  # center
        max(0, int(3 * (T - crop_frames) / 4)),  # right-center
        T - crop_frames,  # right
    ]
    
    crops = []
    for start in positions:
        end = start + crop_frames
        crops.append(mel7[:, start:end])
    
    return crops


def preprocess_for_inference(
    file_bytes: bytes,
    window_mode: str,
    sr: int = 16000
) -> Tuple[torch.Tensor, str, float]:
    """
    Complete preprocessing pipeline for inference, matching training exactly.
    
    Args:
        file_bytes: Raw audio file bytes
        window_mode: One of '7s', '3s_center', '3s_5crop', 'auto'
        sr: Sample rate (default 16000)
        
    Returns:
        Tuple of (model_tensor, actual_window_mode, original_duration_sec)
        - model_tensor: For 7s/3s_center: shape (1, 1, n_mels, T)
                       For 3s_5crop: shape (5, 1, n_mels, T3)
        - actual_window_mode: The window mode used
        - original_duration_sec: Original audio duration before 7s standardization
    """
    # Step 1: Load audio to 16kHz mono
    y = load_audio_16k_mono(file_bytes, target_sr=sr)
    original_duration = len(y) / sr
    
    # Step 2: Standardize to 7 seconds (center-crop or center-pad)
    y7 = make_7s_waveform(y, sr=sr, target_seconds=7.0)
    
    # Step 3: Convert to mel-spectrogram (with pre-emphasis, trim, etc.)
    mel7 = waveform_to_mel_db(y7, sr=sr, n_mels=128)
    
    # Step 4: Handle auto mode
    if window_mode == 'auto':
        if original_duration >= 7.0:
            window_mode = '7s'
        elif original_duration >= 3.0:
            window_mode = '3s_center'
        else:
            window_mode = '3s_center'  # Pad short audio
    
    # Step 5: Extract window according to mode
    if window_mode == '7s':
        mel_for_model = mel7
        tensor = mel_to_model_tensor(mel_for_model)
        return tensor, '7s', original_duration
        
    elif window_mode == '3s_center':
        mel_center = center_crop_mel_3s(mel7, sr=sr, hop_length=160)
        tensor = mel_to_model_tensor(mel_center)
        return tensor, '3s_center', original_duration
        
    elif window_mode == '3s_5crop':
        crops = five_crop_mel_3s(mel7, sr=sr, hop_length=160)
        # Convert each crop to tensor: each is (1, 1, n_mels, T3)
        crop_tensors = [mel_to_model_tensor(c) for c in crops]
        # Stack along batch dimension: (5, 1, n_mels, T3)
        # Each tensor is (1, 1, n_mels, T3), so squeeze first dim then stack
        stacked = torch.stack([t.squeeze(0) for t in crop_tensors], dim=0)
        # Result: (5, 1, n_mels, T3)
        return stacked, '3s_5crop', original_duration
        
    else:
        raise ValueError(f"Invalid window_mode: {window_mode}")

