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
    early. Returns 0 when either signal is silent or too short to correlate.

    Known limitation: on strongly periodic material the correlation peak can
    land on a pitch period rather than the true offset. Measuring SNAC 24 kHz
    gives -85 samples on a sweep, -1 on speech, and 196 on a trumpet loop —
    a codec's delay does not actually vary that way. Treat this as a hint on
    broadband or transient-rich audio and as unreliable on sustained tones.
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


def shift(
    reference: np.ndarray, estimate: np.ndarray, lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide `estimate` back by `lag` samples and re-truncate both signals."""
    if lag > 0:
        return reference[: estimate.size - lag], estimate[lag:]
    if lag < 0:
        return reference[-lag:], estimate[: reference.size + lag]
    return reference, estimate


def si_snr_aligned(
    reference: np.ndarray, estimate: np.ndarray, lag: Optional[int] = None
) -> float:
    """SI-SNR after compensating for codec delay.

    Raw SI-SNR is what codec papers report, but it is not comparable across
    codecs with different latencies: an 8-sample offset costs DAC 16 kHz
    about 13 dB and DAC 24 kHz about 19 dB, which is enough to rank a 24 kbps
    codec below an 8 kbps one purely as an alignment artifact.

    Both numbers are reported. A large gap between them means the difference
    is timing, not fidelity — and the raw figure stays available so published
    comparisons remain reproducible.
    """
    if lag is None:
        lag = detect_delay(reference, estimate)
    ref, est = shift(*align(reference, estimate), lag)
    if ref.size == 0:
        return float("nan")
    return si_snr(ref, est)


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
    lag = detect_delay(reference, estimate)
    scores = {
        "si_snr": si_snr(reference, estimate),
        "si_snr_aligned": si_snr_aligned(reference, estimate, lag),
        "delay_samples": lag,
    }
    if speech:
        for name, fn in (("stoi", stoi), ("pesq", pesq)):
            try:
                scores[name] = fn(reference, estimate, sample_rate)
            except (ImportError, ValueError):
                pass
    return scores
