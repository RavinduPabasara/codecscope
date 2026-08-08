"""Reconstruction quality metrics.

Only ``si_snr`` is always available — it is a few lines of numpy and applies
to any audio. ``stoi`` and ``pesq`` are speech intelligibility measures and
need optional dependencies:

    pip install 'codecscope[quality]'

PESQ is defined only for 8 kHz and 16 kHz speech. It will happily return a
number for music and that number is meaningless, so codecscope never computes
it unless you ask for it explicitly (``--speech``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

PESQ_RATES = (8000, 16000)


def align(reference: np.ndarray, estimate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate both signals to their common length.

    Codecs pad to a whole number of frames, so a reconstruction is usually a
    little longer than its input. Comparing without truncating silently
    scores that padding as error.
    """
    n = min(reference.size, estimate.size)
    return reference[:n].astype(np.float64), estimate[:n].astype(np.float64)


def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant signal-to-noise ratio, in dB.

    Scale invariance matters here because codecs are free to change overall
    gain; a plain SNR would punish that as if it were distortion.
    """
    ref, est = align(reference, estimate)
    if ref.size == 0:
        return float("nan")
    ref = ref - ref.mean()
    est = est - est.mean()
    ref_energy = float(ref @ ref)
    if ref_energy == 0.0:
        return float("nan")
    target = (float(est @ ref) / ref_energy) * ref
    noise = est - target
    target_energy, noise_energy = float(target @ target), float(noise @ noise)
    if noise_energy == 0.0:
        return float("inf")
    if target_energy == 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(target_energy / noise_energy))


def detect_delay(
    reference: np.ndarray, estimate: np.ndarray, max_lag: int = 2048
) -> int:
    """Samples the reconstruction lags the reference by, via cross-correlation.

    Codecs can shift their output in time. SI-SNR is computed without
    compensating for that, matching how codec papers report it, but a
    delay-driven score looks exactly like distortion unless you can see the
    delay. Reporting it separately keeps the metric comparable *and*
    interpretable.

    Positive means the reconstruction arrives late, negative that it arrives
    early — SNAC 24 kHz comes back about 85 samples early. Returns 0 when
    either signal is silent or too short to correlate.
    """
    ref, est = align(reference, estimate)
    n = ref.size
    if n < 2:
        return 0
    ref = ref - ref.mean()
    est = est - est.mean()
    if not ref.any() or not est.any():
        return 0
    correlation = np.correlate(est, ref, mode="full")
    lags = np.arange(-(n - 1), n)
    window = np.abs(lags) <= max_lag
    return int(lags[window][int(np.argmax(np.abs(correlation[window])))])


def stoi(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float:
    """Short-Time Objective Intelligibility (needs ``pystoi``)."""
    try:
        from pystoi import stoi as _stoi
    except ImportError as e:  # pragma: no cover
        raise ImportError("stoi needs pystoi: pip install 'codecscope[quality]'") from e
    ref, est = align(reference, estimate)
    return float(_stoi(ref, est, sample_rate, extended=False))


def pesq(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float:
    """Perceptual Evaluation of Speech Quality (needs ``pesq``).

    Raises if `sample_rate` is not 8 kHz or 16 kHz — the metric is undefined
    elsewhere and a resampled score is not comparable to published numbers.
    """
    try:
        from pesq import pesq as _pesq
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pesq needs the pesq package: pip install 'codecscope[quality]'"
        ) from e
    if sample_rate not in PESQ_RATES:
        raise ValueError(
            f"PESQ is defined only at {PESQ_RATES} Hz, got {sample_rate}; "
            "resample the pair first if you really want a score"
        )
    ref, est = align(reference, estimate)
    mode = "nb" if sample_rate == 8000 else "wb"
    return float(_pesq(sample_rate, ref, est, mode))


def available() -> dict:
    """Which optional quality metrics can actually be computed right now."""
    status = {}
    for name, module in (("stoi", "pystoi"), ("pesq", "pesq")):
        try:
            __import__(module)
            status[name] = True
        except ImportError:
            status[name] = False
    return status


def evaluate(
    reference: np.ndarray,
    estimate: Optional[np.ndarray],
    sample_rate: int,
    speech: bool = False,
) -> dict:
    """Compute every quality metric that is available and applicable.

    Returns a dict with keys si_snr / stoi / pesq, omitting any that could
    not be computed. Missing optional dependencies are skipped rather than
    raised, so a quality run degrades instead of failing.
    """
    if estimate is None:
        return {}
    scores = {
        "si_snr": si_snr(reference, estimate),
        "delay_samples": detect_delay(reference, estimate),
    }
    if speech:
        for name, fn in (("stoi", stoi), ("pesq", pesq)):
            try:
                scores[name] = fn(reference, estimate, sample_rate)
            except (ImportError, ValueError):
                pass
    return scores
