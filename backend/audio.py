"""
Audio processing utilities for preprocessing audio files.
"""
import numpy as np
import torch
import torchaudio
import librosa
from typing import Tuple, List, Optional
import io
import tempfile
import os

from backend.config import (
    SAMPLE_RATE, HOP_LENGTH, N_MELS, N_FFT,
    WINDOW_7S, WINDOW_3S
)


def load_audio(file_content: bytes, filename: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes and resample to target sample rate.
    
    Args:
        file_content: Audio file bytes
        filename: Original filename (for format detection)
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    # Determine format from filename
    filename_lower = filename.lower() if filename else ""
    is_web_format = any(ext in filename_lower for ext in ['.webm', '.opus', '.ogg'])
    
    # For web formats, we need to use a temporary file since librosa/soundfile
    # may not handle BytesIO for these formats reliably
    if is_web_format or not filename:
        # Use temporary file for web formats
        temp_path = None
        try:
            # Create temporary file with appropriate extension
            suffix = '.webm' if '.webm' in filename_lower else ('.opus' if '.opus' in filename_lower else '.ogg')
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            # Load using librosa (which uses ffmpeg)
            waveform, sr = librosa.load(
                temp_path,
                sr=None,
                mono=True,
                res_type='kaiser_best'
            )
        except Exception as e:
            raise ValueError(f"Failed to load audio with librosa: {e}")
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    else:
        # For other formats (WAV, MP3), try multiple methods
        waveform = None
        sr = None
        
        # Method 1: Try torchaudio
        try:
            audio_io = io.BytesIO(file_content)
            audio_io.seek(0)  # Reset to beginning
            loaded_waveform, loaded_sr = torchaudio.load(audio_io)
            waveform = loaded_waveform.numpy()
            sr = loaded_sr
            
            # Convert to mono if stereo
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=0)
            else:
                waveform = waveform.flatten()
        except Exception:
            pass
        
        # Method 2: Try librosa with BytesIO
        if waveform is None:
            try:
                audio_io = io.BytesIO(file_content)
                audio_io.seek(0)  # Reset to beginning
                waveform, sr = librosa.load(
                    audio_io,
                    sr=None,
                    mono=True,
                    res_type='kaiser_best'
                )
            except Exception:
                pass
        
        # Method 3: Try librosa with temporary file
        if waveform is None:
            temp_path = None
            try:
                # Determine extension
                suffix = '.wav'
                if '.mp3' in filename_lower:
                    suffix = '.mp3'
                elif '.m4a' in filename_lower:
                    suffix = '.m4a'
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                waveform, sr = librosa.load(
                    temp_path,
                    sr=None,
                    mono=True,
                    res_type='kaiser_best'
                )
            except Exception as e:
                raise ValueError(f"Failed to load audio: {e}. Supported formats: WAV, MP3, WEBM, OGG, OPUS")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
    
    # Validate waveform
    if waveform is None or len(waveform) == 0:
        raise ValueError("Loaded audio is empty or could not be decoded")
    
    # Resample to target sample rate if needed
    if sr != SAMPLE_RATE:
        try:
            # Try with resampy (preferred, higher quality)
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
                res_type='kaiser_best'
            )
        except (ModuleNotFoundError, ImportError):
            # Fallback to scipy if resampy is not available
            try:
                from scipy import signal
                # Calculate number of samples for target sample rate
                num_samples = int(len(waveform) * SAMPLE_RATE / sr)
                waveform = signal.resample(waveform, num_samples)
            except ImportError:
                # Last resort: use simple linear interpolation
                import numpy as np
                old_indices = np.linspace(0, len(waveform) - 1, len(waveform))
                new_length = int(len(waveform) * SAMPLE_RATE / sr)
                new_indices = np.linspace(0, len(waveform) - 1, new_length)
                waveform = np.interp(new_indices, old_indices, waveform)
    
    return waveform, SAMPLE_RATE


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS
) -> np.ndarray:
    """
    Compute log-mel spectrogram from waveform.
    
    Args:
        waveform: Audio waveform (1D array)
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel filter banks
        
    Returns:
        Log-mel spectrogram (n_mels x time_frames)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=sr // 2
    )
    
    # Convert to log scale: log(1 + x)
    log_mel = np.log1p(mel_spec)
    
    return log_mel


def center_crop(waveform: np.ndarray, sr: int, duration: float) -> np.ndarray:
    """
    Center crop or zero-pad waveform to exact duration.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        duration: Target duration in seconds
        
    Returns:
        Cropped/padded waveform
    """
    target_samples = int(duration * sr)
    current_samples = len(waveform)
    
    if current_samples >= target_samples:
        # Center crop
        start = (current_samples - target_samples) // 2
        return waveform[start:start + target_samples]
    else:
        # Zero pad
        padding = target_samples - current_samples
        pad_left = padding // 2
        pad_right = padding - pad_left
        return np.pad(waveform, (pad_left, pad_right), mode='constant')


def get_5_crop_windows(waveform: np.ndarray, sr: int, duration: float = WINDOW_3S) -> List[np.ndarray]:
    """
    Generate 5 evenly spaced 3-second windows from audio.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        duration: Window duration in seconds (default 3s)
        
    Returns:
        List of 5 windowed waveforms
    """
    total_duration = len(waveform) / sr
    
    if total_duration >= 7.0:
        # 5 evenly spaced windows over 7 seconds
        window_samples = int(duration * sr)
        total_samples = int(7.0 * sr)
        
        # Calculate start positions for 5 windows
        step = (total_samples - window_samples) / 4
        windows = []
        
        for i in range(5):
            start = int(i * step)
            end = start + window_samples
            if end > len(waveform):
                # If audio is shorter than 7s, adjust
                end = len(waveform)
                start = max(0, end - window_samples)
            windows.append(waveform[start:end])
            
    elif total_duration >= duration:
        # Overlapping windows across available length
        window_samples = int(duration * sr)
        step = (len(waveform) - window_samples) / 4
        windows = []
        
        for i in range(5):
            start = int(i * step)
            end = start + window_samples
            if end > len(waveform):
                end = len(waveform)
                start = max(0, end - window_samples)
            windows.append(waveform[start:end])
    else:
        # Pad and use center window 5 times
        padded = center_crop(waveform, sr, duration)
        windows = [padded.copy() for _ in range(5)]
    
    # Ensure all windows are exactly the right length
    window_samples = int(duration * sr)
    windows = [
        center_crop(w, sr, duration) for w in windows
    ]
    
    return windows


def preprocess_audio(
    waveform: np.ndarray,
    sr: int,
    window_mode: str
) -> Tuple[np.ndarray, str]:
    """
    Preprocess audio according to window mode.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        window_mode: One of 'auto', '7s', '3s_center', '3s_5crop'
        
    Returns:
        Tuple of (preprocessed mel spectrogram, actual window mode used)
    """
    duration = len(waveform) / sr
    
    # Auto mode: choose based on duration
    if window_mode == 'auto':
        if duration >= 7.0:
            window_mode = '7s'
        elif duration >= 3.0:
            window_mode = '3s_center'
        else:
            window_mode = '3s_center'  # Pad short audio
    
    # Process according to window mode
    if window_mode == '7s':
        windowed = center_crop(waveform, sr, WINDOW_7S)
        mel = compute_mel_spectrogram(windowed, sr)
        return mel, '7s'
        
    elif window_mode == '3s_center':
        windowed = center_crop(waveform, sr, WINDOW_3S)
        mel = compute_mel_spectrogram(windowed, sr)
        return mel, '3s_center'
        
    elif window_mode == '3s_5crop':
        windows = get_5_crop_windows(waveform, sr, WINDOW_3S)
        # Return list of mels for averaging later
        mels = [compute_mel_spectrogram(w, sr) for w in windows]
        return mels, '3s_5crop'
    
    else:
        raise ValueError(f"Unknown window mode: {window_mode}")

