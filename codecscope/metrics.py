"""Neural audio codec efficiency metrics.

Definitions:
- token_rate: discrete codes emitted per second of audio. This is the number
  that matters when a codec feeds a language model: a 12.5 Hz codec with 8
  codebooks costs 100 tokens per second of context, whatever its bitrate.
- frame_rate: frames per second of the *finest* codebook. For a flat codec
  (DAC, EnCodec) every codebook runs at this rate and
  token_rate == frame_rate * n_codebooks. Multi-scale codecs (SNAC) run
  their codebooks at different rates, so that identity does not hold — see
  `is_multiscale`. Reporting a single "frame rate" for those is the usual
  way published comparisons go wrong.
- bitrate: bits per second, token_rate * log2(codebook_size). Two codecs at
  the same bitrate can have very different token rates, which is why both
  are reported.
- codebook_utilization: fraction of available codebook entries actually
  emitted on this audio. Low utilization means entries you pay vocabulary
  for but never use — the audio analogue of a tokenizer's UNK rate, and a
  well-known RVQ training failure.
- compression_ratio: source PCM bitrate / codec bitrate.

Reconstruction quality (si_snr, stoi, pesq) is optional and only present
when the adapter returned a decoded waveform. See `codecscope.quality`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Report:
    """Efficiency and quality metrics for one codec over one audio signal.

    `code_counts` holds the number of codes emitted by each codebook, and
    `unique_counts` the number of distinct entries each one used. Both are
    per-codebook rather than aggregate so that multi-scale codecs, whose
    codebooks emit at different rates, stay representable.
    """

    codec: str
    codebook_size: int
    code_counts: Tuple[int, ...]
    unique_counts: Tuple[int, ...]
    duration: float
    source_sample_rate: int
    source_bit_depth: int
    codec_sample_rate: int
    si_snr: Optional[float] = None
    stoi: Optional[float] = None
    pesq: Optional[float] = None
    delay_samples: Optional[int] = None

    @property
    def n_codebooks(self) -> int:
        return len(self.code_counts)

    @property
    def token_count(self) -> int:
        return sum(self.code_counts)

    @property
    def is_multiscale(self) -> bool:
        return len(set(self.code_counts)) > 1

    @property
    def frame_rate(self) -> float:
        """Frames per second of the finest codebook."""
        if not self.code_counts or not self.duration:
            return 0.0
        return max(self.code_counts) / self.duration

    @property
    def token_rate(self) -> float:
        return self.token_count / self.duration if self.duration else 0.0

    @property
    def bits_per_code(self) -> float:
        return math.log2(self.codebook_size) if self.codebook_size > 1 else 0.0

    @property
    def bitrate(self) -> float:
        return self.token_rate * self.bits_per_code

    @property
    def codebook_utilization(self) -> float:
        capacity = self.n_codebooks * self.codebook_size
        return sum(self.unique_counts) / capacity if capacity else 0.0

    @property
    def source_bitrate(self) -> float:
        return float(self.source_sample_rate * self.source_bit_depth)

    @property
    def compression_ratio(self) -> float:
        return self.source_bitrate / self.bitrate if self.bitrate else 0.0

    def as_dict(self) -> dict:
        d = {
            "codec": self.codec,
            "codebook_size": self.codebook_size,
            "n_codebooks": self.n_codebooks,
            "frame_rate": round(self.frame_rate, 2),
            "token_rate": round(self.token_rate, 2),
            "bitrate": round(self.bitrate, 1),
            "codebook_utilization": round(self.codebook_utilization, 4),
            "compression_ratio": round(self.compression_ratio, 2),
            "multiscale": self.is_multiscale,
        }
        for key in ("si_snr", "stoi", "pesq"):
            value = getattr(self, key)
            if value is not None:
                d[key] = round(value, 4)
        if self.delay_samples:
            d["delay_samples"] = self.delay_samples
        return d


def build_report(
    name: str,
    codes: Sequence,
    codebook_size: int,
    duration: float,
    source_sample_rate: int,
    codec_sample_rate: int,
    source_bit_depth: int = 16,
    si_snr: Optional[float] = None,
    stoi: Optional[float] = None,
    pesq: Optional[float] = None,
    delay_samples: Optional[int] = None,
) -> Report:
    """Assemble a Report from a codec's emitted codes.

    `codes` is a sequence of 1-D integer arrays, one per codebook. A
    rectangular 2-D array works too and is treated as one row per codebook;
    ragged sequences (multi-scale codecs) are equally valid.
    """
    import numpy as np

    books = [np.asarray(book).reshape(-1) for book in codes]
    if not books:
        raise ValueError("codes must contain at least one codebook")
    return Report(
        codec=name,
        codebook_size=codebook_size,
        code_counts=tuple(int(b.size) for b in books),
        unique_counts=tuple(int(np.unique(b).size) for b in books),
        duration=duration,
        source_sample_rate=source_sample_rate,
        source_bit_depth=source_bit_depth,
        codec_sample_rate=codec_sample_rate,
        si_snr=si_snr,
        stoi=stoi,
        pesq=pesq,
        delay_samples=delay_samples,
    )
