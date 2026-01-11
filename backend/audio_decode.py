"""
Audio decoding using FFmpeg for robust WEBM/OPUS/MP3/WAV support.
This ensures reliable decoding of MediaRecorder WEBM/Opus files.
"""
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional
import numpy as np
import librosa


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def decode_audio_bytes_to_wav_pcm16(
    file_bytes: bytes,
    input_ext_or_mime: Optional[str] = None
) -> bytes:
    """
    Decode audio bytes to WAV PCM 16-bit, 16kHz, mono using FFmpeg.
    
    Args:
        file_bytes: Raw audio file bytes
        input_ext_or_mime: Input extension or MIME type (e.g., 'webm', 'audio/webm')
        
    Returns:
        WAV PCM 16-bit, 16kHz, mono bytes
        
    Raises:
        RuntimeError: If FFmpeg is not installed
        ValueError: If decoding fails
    """
    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is not installed. Please install FFmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )
    
    # Determine input format from extension or MIME type
    input_format = None
    if input_ext_or_mime:
        if 'webm' in input_ext_or_mime.lower() or 'opus' in input_ext_or_mime.lower():
            input_format = 'webm'
        elif 'mp3' in input_ext_or_mime.lower():
            input_format = 'mp3'
        elif 'wav' in input_ext_or_mime.lower():
            input_format = 'wav'
    
    # Create temporary input file
    input_suffix = f'.{input_format}' if input_format else '.tmp'
    input_temp = None
    output_temp = None
    
    try:
        # Write input bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as f_in:
            f_in.write(file_bytes)
            input_temp = f_in.name
        
        # Create output temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f_out:
            output_temp = f_out.name
        
        # Run FFmpeg to convert to WAV PCM 16-bit, 16kHz, mono
        cmd = [
            'ffmpeg',
            '-i', input_temp,
            '-ar', '16000',  # Sample rate: 16kHz
            '-ac', '1',  # Channels: mono
            '-acodec', 'pcm_s16le',  # Codec: PCM 16-bit little-endian
            '-y',  # Overwrite output file
            output_temp
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            raise ValueError(f"FFmpeg decoding failed: {error_msg}")
        
        # Read decoded WAV bytes
        with open(output_temp, 'rb') as f:
            wav_bytes = f.read()
        
        return wav_bytes
        
    finally:
        # Clean up temp files
        for temp_file in [input_temp, output_temp]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass


def load_audio_from_bytes(
    file_bytes: bytes,
    input_ext_or_mime: Optional[str] = None,
    target_sr: int = 16000
) -> np.ndarray:
    """
    Load audio from bytes, using FFmpeg for robust decoding.
    
    This function:
    1. Decodes audio bytes to WAV PCM 16-bit using FFmpeg
    2. Loads WAV using librosa/soundfile
    3. Ensures 16kHz mono float32 output
    
    Args:
        file_bytes: Raw audio file bytes
        input_ext_or_mime: Input extension or MIME type (optional, for format detection)
        target_sr: Target sample rate (default 16000)
        
    Returns:
        Waveform as np.float32 array in [-1, 1], shape (T,)
    """
    # Decode to WAV using FFmpeg
    wav_bytes = decode_audio_bytes_to_wav_pcm16(file_bytes, input_ext_or_mime)
    
    # Load WAV using librosa (should handle it reliably now)
    try:
        # Use BytesIO for in-memory WAV
        import io
        waveform, sr = librosa.load(
            io.BytesIO(wav_bytes),
            sr=target_sr,
            mono=True,
            res_type='kaiser_best'
        )
    except Exception as e:
        # Fallback: write to temp file and load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(wav_bytes)
            temp_path = temp_file.name
        
        try:
            waveform, sr = librosa.load(
                temp_path,
                sr=target_sr,
                mono=True,
                res_type='kaiser_best'
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
    
    # Ensure float32 and in [-1, 1] range
    waveform = waveform.astype(np.float32)
    
    if len(waveform) == 0:
        raise ValueError("Decoded audio is empty")
    
    return waveform

