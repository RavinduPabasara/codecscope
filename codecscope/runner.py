"""Compare neural audio codecs over a signal or file."""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np

from . import adapters, audio as audio_mod, quality
from .metrics import Report, build_report

CodecSpec = Union[str, adapters.Adapter]


def _resolve(spec: CodecSpec) -> adapters.Adapter:
    return adapters.load(spec) if isinstance(spec, str) else spec


def analyze(
    samples: np.ndarray,
    sample_rate: int,
    codec: CodecSpec,
    source_bit_depth: int = 16,
    measure_quality: bool = True,
    speech: bool = False,
) -> Report:
    """Encode `samples` with one codec and compute metrics.

    The signal is resampled to the codec's operating rate first, and quality
    is measured against that resampled signal rather than the original — a
    codec should not be charged for the resampler's error.
    """
    adapter = _resolve(codec)
    codec_rate = adapter.sample_rate or sample_rate
    signal = audio_mod.resample(samples, sample_rate, codec_rate)

    output = adapter.encode(signal)

    # Rates are per second of what the codes actually describe, which is the
    # codec's frame-padded length rather than the signal we handed it.
    covered = output.n_samples
    if covered is None and output.reconstruction is not None:
        covered = int(output.reconstruction.size)
    if covered is None:
        covered = int(signal.size)
    duration = covered / codec_rate if codec_rate else 0.0

    scores = {}
    if measure_quality:
        scores = quality.evaluate(signal, output.reconstruction, codec_rate, speech)

    return build_report(
        name=adapter.name,
        codes=output.codes,
        codebook_size=output.codebook_size,
        duration=duration,
        source_sample_rate=sample_rate,
        codec_sample_rate=codec_rate,
        source_bit_depth=source_bit_depth,
        si_snr=scores.get("si_snr"),
        stoi=scores.get("stoi"),
        pesq=scores.get("pesq"),
        delay_samples=scores.get("delay_samples"),
    )


def compare(
    samples: np.ndarray,
    sample_rate: int,
    codecs: Sequence[CodecSpec],
    source_bit_depth: int = 16,
    measure_quality: bool = True,
    speech: bool = False,
) -> List[Report]:
    """Run `analyze` for several codecs, sorted by token rate (cheapest first)."""
    reports = [
        analyze(samples, sample_rate, c, source_bit_depth, measure_quality, speech)
        for c in codecs
    ]
    return sorted(reports, key=lambda r: r.token_rate)


def analyze_file(
    path: str,
    codec: CodecSpec,
    measure_quality: bool = True,
    speech: bool = False,
) -> Report:
    """Convenience wrapper: load an audio file, then `analyze` it."""
    samples, sample_rate = audio_mod.load(path)
    return analyze(samples, sample_rate, codec, 16, measure_quality, speech)


def compare_file(
    path: str,
    codecs: Sequence[CodecSpec],
    measure_quality: bool = True,
    speech: bool = False,
) -> List[Report]:
    """Convenience wrapper: load an audio file, then `compare` codecs over it."""
    samples, sample_rate = audio_mod.load(path)
    return compare(samples, sample_rate, codecs, 16, measure_quality, speech)


def optional_dependency_notes(
    reports: Sequence[Report], speech: bool = False
) -> List[str]:
    """Warnings worth printing alongside a run.

    Kept separate from the metrics so the numbers stay a pure function of
    the input and the caller decides how loudly to complain.
    """
    notes = []
    if speech:
        missing = [name for name, ok in quality.available().items() if not ok]
        if missing:
            notes.append(
                f"--speech asked for {', '.join(sorted(missing))} but the package(s) "
                "are not installed, so those columns were skipped "
                "(pip install 'codecscope[quality]')"
            )
    if speech and any(r.codec_sample_rate not in quality.PESQ_RATES for r in reports):
        notes.append(
            "PESQ is only defined at 8/16 kHz; it was skipped for codecs "
            "operating at other rates"
        )
    resampled = [r for r in reports if r.codec_sample_rate != r.source_sample_rate]
    if resampled and not audio_mod.have_librosa():
        notes.append(
            "librosa is not installed, so rate matching used linear interpolation; "
            "its aliasing is counted against the codec in quality scores "
            "(pip install 'codecscope[audio]')"
        )
    if any(r.is_multiscale for r in reports):
        notes.append(
            "multi-scale codec present: frame_rate is the finest codebook's rate, "
            "so token_rate != frame_rate * n_codebooks for those rows"
        )
    return notes
