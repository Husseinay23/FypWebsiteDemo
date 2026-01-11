"""
Canonical preprocessing pipeline matching training contract exactly.
This implements the exact preprocessing as specified in the training contract.
NO pre-emphasis, NO trim-silence, exact mel parameters, right-padding.
"""
import numpy as np
import torch
import librosa
from typing import Tuple, List, Optional
import io

from backend.config import SAMPLE_RATE, HOP_LENGTH, N_MELS, N_FFT, F_MIN, F_MAX
from backend.audio_decode import load_audio_from_bytes


def make_7s_waveform_right_pad(y: np.ndarray, sr: int = 16000, target_seconds: float = 7.0) -> np.ndarray:
    """
    Standardize waveform to exactly 7 seconds.
    - If shorter: RIGHT-pad with zeros
    - If longer: CENTER-crop to 7s
    
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
        # RIGHT-pad with zeros (training contract requirement)
        padding = target_length - current_length
        y_padded = np.pad(y, (0, padding), mode='constant', constant_values=0.0)
        return y_padded.astype(np.float32)
    elif current_length > target_length:
        # CENTER-crop (training contract requirement)
        start = (current_length - target_length) // 2
        end = start + target_length
        return y[start:end].astype(np.float32)
    else:
        return y.astype(np.float32)


def waveform_to_mel_db_canonical(
    y: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 256,
    f_min: float = 20.0,
    f_max: float = 7600.0
) -> np.ndarray:
    """
    Convert waveform to mel-spectrogram in dB scale, matching training contract exactly.
    
    Training contract:
    - NO pre-emphasis
    - NO trim-silence
    - n_fft = 2048
    - hop_length = 256
    - n_mels = 128
    - f_min = 20.0
    - f_max = 7600.0
    - AmplitudeToDB (power_to_db)
    - PER-SAMPLE NORMALIZATION (mean/std)
    """
    # 1. Compute mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
    )

    # 2. Convert to dB scale (AmplitudeToDB)
    mel_db = librosa.power_to_db(S, ref=np.max)

    # 3. PER-SAMPLE NORMALIZATION (CRITICAL)
    mean = mel_db.mean()
    std = mel_db.std() + 1e-9
    mel_db = (mel_db - mean) / std

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


def center_crop_mel_3s(mel7: np.ndarray, sr: int = 16000, hop_length: int = 256) -> np.ndarray:
    """
    Extract center 3-second crop from 7-second mel-spectrogram.
    
    Args:
        mel7: 7-second mel-spectrogram, shape (n_mels, T7)
        sr: Sample rate
        hop_length: Hop length used in mel computation
        
    Returns:
        3-second mel-spectrogram, shape (n_mels, T3)
    """
    frames_per_sec = sr / hop_length  # 16000 / 256 = 62.5 frames/sec
    crop_frames = int(round(3 * frames_per_sec))  # 187 frames for 3 seconds
    
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


def five_crop_mel_3s(mel7: np.ndarray, sr: int = 16000, hop_length: int = 256) -> List[np.ndarray]:
    """
    Extract 5 evenly-spaced 3-second crops from 7-second mel-spectrogram.
    
    Args:
        mel7: 7-second mel-spectrogram, shape (n_mels, T7)
        sr: Sample rate
        hop_length: Hop length used in mel computation
        
    Returns:
        List of 5 mel-spectrograms, each shape (n_mels, T3)
    """
    frames_per_sec = sr / hop_length
    crop_frames = int(round(3 * frames_per_sec))  # 187 frames for 3s
    
    T = mel7.shape[1]
    
    if T <= crop_frames:
        # Edge case: repeat center crop 5 times
        center = center_crop_mel_3s(mel7, sr=sr, hop_length=hop_length)
        return [center] * 5
    
    # Calculate 5 evenly-spaced positions (deterministic)
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


def preprocess_for_inference_canonical(
    file_bytes: bytes,
    window_mode: str,
    sr: int = 16000,
    input_ext_or_mime: Optional[str] = None
) -> Tuple[torch.Tensor, str, float, dict]:
    """
    Complete canonical preprocessing pipeline matching training contract exactly.
    
    Args:
        file_bytes: Raw audio file bytes
        window_mode: One of '7s', '3s_center', '3s_5crop', 'auto'
        sr: Sample rate (default 16000)
        input_ext_or_mime: Input extension or MIME type (for format detection)
        
    Returns:
        Tuple of (model_tensor, actual_window_mode, original_duration_sec, debug_info)
        - model_tensor: For 7s/3s_center: shape (1, 1, n_mels, T)
                       For 3s_5crop: shape (5, 1, n_mels, T3)
        - actual_window_mode: The window mode used
        - original_duration_sec: Original audio duration before 7s standardization
        - debug_info: Dictionary with preprocessing debug information
    """
    debug_info = {}
    
    # Step 1: Load audio to 16kHz mono using FFmpeg-based decoding
    y = load_audio_from_bytes(file_bytes, input_ext_or_mime, target_sr=sr)
    original_duration = len(y) / sr
    debug_info['sr_after_resample'] = sr
    debug_info['waveform_len_before_7s'] = len(y)
    debug_info['waveform_duration_before'] = original_duration
    
    # Step 2: Standardize to 7 seconds (RIGHT-pad if shorter, CENTER-crop if longer)
    y7 = make_7s_waveform_right_pad(y, sr=sr, target_seconds=7.0)
    debug_info['waveform_len_after_7s'] = len(y7)
    debug_info['waveform_duration_after'] = len(y7) / sr
    
    # Step 3: Convert to mel-spectrogram (NO pre-emphasis, NO trim-silence)
    mel7 = waveform_to_mel_db_canonical(
        y7,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        f_min=F_MIN,
        f_max=F_MAX
    )
    debug_info['mel_shape_before_crop'] = mel7.shape
    debug_info['mel_stats_before_norm'] = {
        'min': float(np.min(mel7)),
        'max': float(np.max(mel7)),
        'mean': float(np.mean(mel7)),
        'std': float(np.std(mel7))
    }
    
    # Normalization is done in mel conversion (power_to_db)
    debug_info['mel_stats_after_norm'] = debug_info['mel_stats_before_norm']
    
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
        debug_info['tensor_shape_final'] = list(tensor.shape)
        return tensor, '7s', original_duration, debug_info
        
    elif window_mode == '3s_center':
        mel_center = center_crop_mel_3s(mel7, sr=sr, hop_length=HOP_LENGTH)
        tensor = mel_to_model_tensor(mel_center)
        debug_info['tensor_shape_final'] = list(tensor.shape)
        return tensor, '3s_center', original_duration, debug_info
        
    elif window_mode == '3s_5crop':
        crops = five_crop_mel_3s(mel7, sr=sr, hop_length=HOP_LENGTH)
        # Convert each crop to tensor: each is (1, 1, n_mels, T3)
        crop_tensors = [mel_to_model_tensor(c) for c in crops]
        # Stack along batch dimension: (5, 1, n_mels, T3)
        stacked = torch.stack([t.squeeze(0) for t in crop_tensors], dim=0)
        debug_info['tensor_shape_final'] = list(stacked.shape)
        return stacked, '3s_5crop', original_duration, debug_info
        
    else:
        raise ValueError(f"Invalid window_mode: {window_mode}")

