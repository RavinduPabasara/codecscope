"""Audio loading, rate matching, and synthetic test signals.

WAV files are read with the standard library so the core package needs only
numpy. Other formats (flac, mp3, ogg) go through ``soundfile`` if installed:
``pip install 'codecscope[audio]'``.

All signals in codecscope are mono float32 in [-1, 1].
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Tuple

import numpy as np

_SAMPLE_DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}


def load(path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file as (mono float32 samples, sample_rate)."""
    if Path(path).suffix.lower() == ".wav":
        try:
            return _load_wav(path)
        except wave.Error:
            pass  # compressed WAV (e.g. mu-law); fall through to soundfile
    return _load_soundfile(path)


def _load_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width not in _SAMPLE_DTYPES:
        raise ValueError(f"unsupported WAV sample width: {width * 8}-bit")
    data = np.frombuffer(raw, dtype=_SAMPLE_DTYPES[width]).astype(np.float32)
    if width == 1:  # 8-bit WAV is unsigned, centred on 128
        data = (data - 128.0) / 128.0
    else:
        data /= float(2 ** (width * 8 - 1))
    return to_mono(data.reshape(-1, channels) if channels > 1 else data), rate


def _load_soundfile(path: str) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "reading non-WAV audio needs soundfile: pip install 'codecscope[audio]'"
        ) from e
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    return to_mono(data), int(rate)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average any channel dimension away, returning contiguous float32."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1 if samples.shape[0] > samples.shape[1] else 0)
    return np.ascontiguousarray(samples, dtype=np.float32)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample to `target_rate`.

    Uses ``librosa`` when available (band-limited, the right thing for
    measurement). Falls back to linear interpolation, which is adequate for
    matching a codec's expected input rate but will add its own aliasing to
    reconstruction metrics — codecscope warns when this path is used on a
    quality run.
    """
    if source_rate == target_rate:
        return samples
    try:
        import librosa

        return np.ascontiguousarray(
            librosa.resample(samples, orig_sr=source_rate, target_sr=target_rate),
            dtype=np.float32,
        )
    except ImportError:
        return resample_linear(samples, source_rate, target_rate)


def resample_linear(
    samples: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """Dependency-free linear-interpolation resampler."""
    if source_rate == target_rate or samples.size == 0:
        return samples
    n_out = int(round(samples.size * target_rate / source_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_index = np.linspace(0, samples.size - 1, n_out, dtype=np.float64)
    out = np.interp(src_index, np.arange(samples.size), samples)
    return np.ascontiguousarray(out, dtype=np.float32)


def have_librosa() -> bool:
    """Whether the band-limited resampler is available."""
    try:
        import librosa  # noqa: F401

        return True
    except ImportError:
        return False


def sine(
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """A pure tone. Deterministic fixture for tests and smoke runs."""
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    return np.ascontiguousarray(
        amplitude * np.sin(2 * np.pi * frequency * t), np.float32
    )


def chirp(
    duration: float = 1.0,
    sample_rate: int = 16000,
    start: float = 100.0,
    end: float = 4000.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """A linear frequency sweep — exercises a codec across its band."""
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    phase = 2 * np.pi * (start * t + (end - start) * t**2 / (2 * max(duration, 1e-9)))
    return np.ascontiguousarray(amplitude * np.sin(phase), dtype=np.float32)
