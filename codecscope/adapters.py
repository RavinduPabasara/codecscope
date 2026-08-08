"""Codec adapters.

A codec spec is a string with an optional backend prefix:

- ``pcm:16``          — uniform PCM quantizer at N bits (no dependencies;
  compression ratio of exactly 1.0 against a 16-bit source by construction)
- ``dac:44khz``        — Descript Audio Codec (needs ``descript-audio-codec``)
- ``encodec:24khz@6``  — Meta EnCodec via transformers, bandwidth in kbps
- ``snac:24khz``       — multi-scale SNAC (needs ``snac``)
- ``hf:kyutai/mimi@8`` — any transformers codec, quantizer count after ``@``

A spec without a prefix defaults to the ``hf`` backend. Backends are imported
lazily, so the core package needs only numpy.

Variable-rate codecs (EnCodec, Mimi) take a configuration after ``@``. Their
libraries default to a setting that is rarely the one people quote — EnCodec
to its lowest bandwidth, Mimi to all 32 quantizers — so each adapter resolves
the value explicitly and records it in ``name``, and no report can leave the
configuration implicit.

Every adapter returns a `CodecOutput` whose `codes` is a list of 1-D arrays,
one per codebook. Flat codecs return equal-length arrays; multi-scale codecs
return ragged ones. Adapters do not resample — `codecscope.runner` matches
rates before calling them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class CodecOutput:
    """What a codec produced for one signal.

    `n_samples` is how many samples the emitted codes actually describe.
    Codecs pad their input up to a whole number of frames, so this is often
    longer than the signal handed in — SNAC pads 48000 samples to 49152.
    Rates must be computed against this, not against the original length,
    or every short clip reports an inflated bitrate. When an adapter leaves
    it None the runner falls back to the reconstruction length.
    """

    codes: List[np.ndarray]
    codebook_size: int
    reconstruction: Optional[np.ndarray] = None
    n_samples: Optional[int] = None


class Adapter(Protocol):
    name: str
    sample_rate: int

    def encode(self, audio: np.ndarray) -> CodecOutput:
        """Encode mono float32 audio already at `self.sample_rate`."""
        ...  # pragma: no cover


class PCMAdapter:
    """Uniform mid-tread PCM quantizer.

    The dependency-free baseline: at 16 bits against a 16-bit source it
    compresses by exactly 1.0 and emits one code per sample, which makes it
    the reference point every neural codec is measured against.
    """

    def __init__(self, spec: str = "16") -> None:
        bits = int(spec) if spec else 16
        if not 1 <= bits <= 24:
            raise ValueError(f"pcm bit depth must be 1..24, got {bits}")
        self.bits = bits
        self.name = f"pcm:{bits}"
        self.sample_rate = 0  # rate-agnostic: operates at whatever it is given

    def encode(self, audio: np.ndarray) -> CodecOutput:
        levels = 2**self.bits
        clipped = np.clip(audio.astype(np.float64), -1.0, 1.0)
        codes = np.rint((clipped + 1.0) * 0.5 * (levels - 1)).astype(np.int64)
        codes = np.clip(codes, 0, levels - 1)
        reconstruction = (codes / (levels - 1) * 2.0 - 1.0).astype(np.float32)
        return CodecOutput([codes], levels, reconstruction)


class DACAdapter:
    """Descript Audio Codec. Specs: ``dac:44khz``, ``dac:24khz``, ``dac:16khz``."""

    def __init__(self, spec: str = "44khz") -> None:
        try:
            import dac
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'dac' backend needs descript-audio-codec: "
                "pip install 'codecscope[dac]'"
            ) from e
        self._torch = torch
        self._model = dac.DAC.load(
            dac.utils.download(model_type=spec or "44khz")
        ).eval()
        self.name = f"dac:{spec or '44khz'}"
        self.sample_rate = int(self._model.sample_rate)

    def encode(self, audio: np.ndarray) -> CodecOutput:
        torch = self._torch
        x = torch.from_numpy(audio).reshape(1, 1, -1)
        with torch.no_grad():
            x = self._model.preprocess(x, self.sample_rate)
            z, codes, _, _, _ = self._model.encode(x)
            reconstruction = self._model.decode(z)
        books = [b.cpu().numpy() for b in codes[0]]  # (n_codebooks, T) -> per book
        return CodecOutput(
            books,
            int(self._model.codebook_size),
            reconstruction.reshape(-1).cpu().numpy().astype(np.float32),
        )


class EncodecAdapter:
    """Meta EnCodec through transformers.

    Specs: ``encodec:24khz``, ``encodec:48khz``, with an optional bandwidth
    in kbps after ``@`` — ``encodec:24khz@6``.

    EnCodec is a variable-bitrate RVQ: the bandwidth decides how many
    codebooks are kept, from 2 at 1.5 kbps up to 32 at 24 kbps. Transformers
    defaults to the *lowest* setting, so an unqualified ``encodec:24khz``
    would quietly enter a comparison at 1.5 kbps against codecs running
    flat out. codecscope therefore resolves the bandwidth explicitly and
    always names it in the report, so no table can misattribute a bitrate.
    """

    _REPOS = {"24khz": "facebook/encodec_24khz", "48khz": "facebook/encodec_48khz"}
    _DEFAULT_BANDWIDTH = 6.0

    def __init__(self, spec: str = "24khz") -> None:
        try:
            import torch
            from transformers import EncodecModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'encodec' backend needs transformers: "
                "pip install 'codecscope[encodec]'"
            ) from e
        self._torch = torch
        variant, _, bandwidth = (spec or "24khz").partition("@")
        variant = variant or "24khz"
        repo = self._REPOS.get(variant, variant)
        self._model = EncodecModel.from_pretrained(repo).eval()
        self.sample_rate = int(self._model.config.sampling_rate)
        self._channels = int(getattr(self._model.config, "audio_channels", 1))
        self._bandwidth = self._resolve_bandwidth(bandwidth)
        self.name = f"encodec:{variant}@{self._bandwidth:g}"

    def _resolve_bandwidth(self, requested: str) -> float:
        supported = list(self._model.config.target_bandwidths)
        if not requested:
            return (
                self._DEFAULT_BANDWIDTH
                if self._DEFAULT_BANDWIDTH in supported
                else supported[-1]
            )
        value = float(requested)
        if value not in supported:
            raise ValueError(
                f"encodec bandwidth {value} kbps is not supported by this model; "
                f"choose one of {supported}"
            )
        return value

    def encode(self, audio: np.ndarray) -> CodecOutput:
        torch = self._torch
        # EnCodec 48 kHz is a stereo model. codecscope works in mono, so the
        # signal is duplicated across channels rather than rejected — feeding
        # identical channels is the standard way to score a mono source on a
        # stereo codec, and the decoded channels are averaged back afterwards.
        x = torch.from_numpy(audio).reshape(1, 1, -1)
        if self._channels > 1:
            x = x.repeat(1, self._channels, 1)
        with torch.no_grad():
            encoded = self._model.encode(x, bandwidth=self._bandwidth)
            decoded = self._model.decode(encoded.audio_codes, encoded.audio_scales)[0]
        # audio_codes is (n_chunks, batch, n_codebooks, T). The 24 kHz model
        # emits a single chunk, but the 48 kHz one chunks at 1 s with 1%
        # overlap — reading only chunk 0 would silently discard most of the
        # stream. Every chunk is concatenated, so the count reflects what the
        # codec actually emits, overlap redundancy included.
        codes = torch.cat([chunk[0] for chunk in encoded.audio_codes], dim=-1)
        books = [b.cpu().numpy() for b in codes]
        reconstruction = decoded[0].cpu().numpy()
        if reconstruction.ndim > 1:
            reconstruction = reconstruction.mean(axis=0)
        return CodecOutput(
            books,
            int(self._model.config.codebook_size),
            np.ascontiguousarray(reconstruction.reshape(-1), dtype=np.float32),
        )


class SNACAdapter:
    """Multi-scale SNAC. Specs: ``snac:24khz``, ``snac:32khz``, ``snac:44khz``.

    SNAC's codebooks run at different frame rates, so `codes` is genuinely
    ragged here — this is the adapter that motivates `Report.is_multiscale`.
    """

    _REPOS = {
        "24khz": "hubertsiuzdak/snac_24khz",
        "32khz": "hubertsiuzdak/snac_32khz",
        "44khz": "hubertsiuzdak/snac_44khz",
    }

    def __init__(self, spec: str = "24khz") -> None:
        try:
            import torch
            from snac import SNAC
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'snac' backend needs snac: pip install 'codecscope[snac]'"
            ) from e
        self._torch = torch
        repo = self._REPOS.get(spec, spec or self._REPOS["24khz"])
        self._model = SNAC.from_pretrained(repo).eval()
        self.name = f"snac:{spec or '24khz'}"
        self.sample_rate = int(self._model.sampling_rate)

    def encode(self, audio: np.ndarray) -> CodecOutput:
        torch = self._torch
        x = torch.from_numpy(audio).reshape(1, 1, -1)
        with torch.no_grad():
            codes = self._model.encode(x)  # list of (1, T_i), T_i differ per scale
            reconstruction = self._model.decode(codes)
        books = [c.reshape(-1).cpu().numpy() for c in codes]
        return CodecOutput(
            books,
            int(self._model.codebook_size),
            reconstruction.reshape(-1).cpu().numpy().astype(np.float32),
        )


class HuggingFaceAdapter:
    """Any transformers audio codec exposing encode/decode (e.g. ``kyutai/mimi``).

    Specs: ``hf:kyutai/mimi``, with an optional quantizer count after ``@`` —
    ``hf:kyutai/mimi@8``.

    Like EnCodec, these models are variable-rate: Mimi ships with
    ``num_quantizers=32`` in its config but its headline figure of 1.1 kbps
    at 12.5 Hz is the *8*-codebook configuration. Left implicit, a run would
    silently report 4.4 kbps under the name "mimi" and look four times more
    expensive than the number everyone quotes. The resolved count is always
    appended to the report name.
    """

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'hf' backend needs transformers: pip install 'codecscope[hf]'"
            ) from e
        self._torch = torch
        repo, _, quantizers = model_name.partition("@")
        self._model = AutoModel.from_pretrained(repo).eval()
        config = self._model.config
        self.sample_rate = int(getattr(config, "sampling_rate", 24000))
        self._codebook_size = int(getattr(config, "codebook_size", 2048))
        self._quantizers = self._resolve_quantizers(quantizers, config)
        self.name = repo if self._quantizers is None else f"{repo}@{self._quantizers}"

    def _resolve_quantizers(self, requested: str, config) -> Optional[int]:
        available = getattr(config, "num_quantizers", None)
        if not requested:
            return int(available) if available is not None else None
        value = int(requested)
        if available is not None and not 1 <= value <= int(available):
            raise ValueError(
                f"{self.__class__.__name__}: quantizer count must be 1..{int(available)}, "
                f"got {value}"
            )
        return value

    def encode(self, audio: np.ndarray) -> CodecOutput:
        torch = self._torch
        x = torch.from_numpy(audio).reshape(1, 1, -1)
        kwargs = (
            {} if self._quantizers is None else {"num_quantizers": self._quantizers}
        )
        with torch.no_grad():
            try:
                encoded = self._model.encode(x, **kwargs)
            except TypeError:
                # Not every codec takes num_quantizers; fall back rather than
                # fail, but the name still records what was asked for.
                encoded = self._model.encode(x)
            codes = getattr(encoded, "audio_codes", encoded)
            decoded = self._model.decode(codes)
            reconstruction = getattr(decoded, "audio_values", decoded)
        codes = codes[0] if codes.ndim == 3 else codes[0][0]
        books = [b.reshape(-1).cpu().numpy() for b in codes]
        return CodecOutput(
            books,
            self._codebook_size,
            np.asarray(reconstruction.reshape(-1).cpu().numpy(), dtype=np.float32),
        )


_BACKENDS = {
    "pcm": PCMAdapter,
    "dac": DACAdapter,
    "encodec": EncodecAdapter,
    "snac": SNACAdapter,
    "hf": HuggingFaceAdapter,
}


def resolve_backend(spec: str):
    """Split a spec into (adapter class, remainder) without constructing it.

    Kept separate from `load` so spec routing can be tested without importing
    a backend or downloading weights.
    """
    backend, sep, rest = spec.partition(":")
    if not sep or backend not in _BACKENDS:
        backend, rest = "hf", spec
    return _BACKENDS[backend], rest


def load(spec: str) -> Adapter:
    """Load a codec from a spec string like 'snac:24khz'."""
    cls, rest = resolve_backend(spec)
    return cls(rest)
