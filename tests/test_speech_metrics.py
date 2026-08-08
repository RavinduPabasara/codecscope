"""Tests for the speech-only quality metrics and non-WAV audio loading.

These need the optional extras and skip cleanly without them:

    pip install 'codecscope[quality]'   # pystoi, pesq
    pip install 'codecscope[audio]'     # soundfile, librosa

Kept out of the integration marker because nothing here downloads a model —
they run in about a second once the extras are present.
"""

import numpy as np
import pytest

from codecscope import analyze, audio, quality
from codecscope.adapters import PCMAdapter

SAMPLE_RATE = 16000


@pytest.fixture(scope="module")
def speech_like():
    """A voiced-sounding signal in the speech band.

    STOI and PESQ are built for speech; scoring a pure tone or a full-band
    sweep with them produces numbers that do not mean anything.
    """
    rng = np.random.default_rng(0)
    t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    signal = (
        0.30 * np.sin(2 * np.pi * 180 * t) * (1 + 0.4 * np.sin(2 * np.pi * 3.1 * t))
        + 0.12 * np.sin(2 * np.pi * 820 * t)
        + 0.03 * rng.standard_normal(t.size)
    )
    return signal.astype(np.float32)


def _degrade(signal, bits):
    return PCMAdapter(str(bits)).encode(signal).reconstruction


class TestSTOI:
    def test_identical_signals_score_one(self, speech_like):
        pytest.importorskip("pystoi")
        assert quality.stoi(speech_like, speech_like, SAMPLE_RATE) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_falls_as_quantization_coarsens(self, speech_like):
        pytest.importorskip("pystoi")
        fine = quality.stoi(speech_like, _degrade(speech_like, 6), SAMPLE_RATE)
        coarse = quality.stoi(speech_like, _degrade(speech_like, 3), SAMPLE_RATE)
        assert 0.0 <= coarse < fine <= 1.0


class TestPESQ:
    def test_scores_near_the_ceiling_for_a_clean_copy(self, speech_like):
        pytest.importorskip("pesq")
        # Wideband PESQ tops out at 4.64.
        assert quality.pesq(speech_like, _degrade(speech_like, 16), SAMPLE_RATE) > 4.0

    def test_falls_as_quantization_coarsens(self, speech_like):
        pytest.importorskip("pesq")
        fine = quality.pesq(speech_like, _degrade(speech_like, 6), SAMPLE_RATE)
        coarse = quality.pesq(speech_like, _degrade(speech_like, 3), SAMPLE_RATE)
        assert coarse < fine

    def test_refuses_a_rate_it_is_not_defined_at(self, speech_like):
        pytest.importorskip("pesq")
        with pytest.raises(ValueError, match="defined only at"):
            quality.pesq(speech_like, speech_like, 44100)


class TestSpeechFlagEndToEnd:
    def test_analyze_reports_all_three_scores(self, speech_like):
        pytest.importorskip("pystoi")
        pytest.importorskip("pesq")
        report = analyze(speech_like, SAMPLE_RATE, "pcm:6", speech=True)
        assert report.si_snr is not None
        assert 0.0 <= report.stoi <= 1.0
        assert report.pesq > 1.0
        assert {"si_snr", "stoi", "pesq"} <= set(report.as_dict())

    def test_without_speech_flag_only_si_snr(self, speech_like):
        report = analyze(speech_like, SAMPLE_RATE, "pcm:6")
        assert report.si_snr is not None
        assert report.stoi is None and report.pesq is None

    def test_speech_metrics_skipped_at_unsupported_rate(self, speech_like):
        # 44.1 kHz: PESQ is undefined, so it must be absent rather than wrong.
        resampled = audio.resample_linear(speech_like, SAMPLE_RATE, 44100)
        report = analyze(resampled, 44100, "pcm:6", speech=True)
        assert report.pesq is None


class TestSoundfileLoading:
    def test_flac_roundtrip(self, tmp_path, speech_like):
        sf = pytest.importorskip("soundfile")
        path = tmp_path / "tone.flac"
        sf.write(str(path), speech_like, SAMPLE_RATE)
        loaded, rate = audio.load(str(path))
        assert rate == SAMPLE_RATE
        assert loaded.dtype == np.float32
        assert np.max(np.abs(loaded - speech_like)) < 1e-3

    def test_stereo_file_is_averaged_to_mono(self, tmp_path):
        sf = pytest.importorskip("soundfile")
        path = tmp_path / "stereo.flac"
        left = audio.sine(duration=0.5, sample_rate=SAMPLE_RATE, frequency=200)
        stereo = np.stack([left, -left], axis=1)
        sf.write(str(path), stereo, SAMPLE_RATE)
        loaded, _ = audio.load(str(path))
        assert loaded.ndim == 1
        assert np.max(np.abs(loaded)) < 1e-3  # opposite channels cancel


class TestLibrosaResampling:
    def test_band_limited_path_is_used_when_available(self):
        pytest.importorskip("librosa")
        assert audio.have_librosa()

    def test_resample_length_matches_linear_fallback(self):
        pytest.importorskip("librosa")
        signal = audio.chirp(duration=0.5, sample_rate=16000)
        band = audio.resample(signal, 16000, 24000)
        linear = audio.resample_linear(signal, 16000, 24000)
        assert abs(band.size - linear.size) <= 1

    def test_band_limited_resampling_beats_linear_on_aliasing(self):
        pytest.importorskip("librosa")
        # Downsampling a sweep is where linear interpolation aliases worst.
        signal = audio.chirp(duration=1.0, sample_rate=44100, start=200, end=18000)
        band = audio.resample(signal, 44100, 16000)
        linear = audio.resample_linear(signal, 44100, 16000)
        assert not np.allclose(band[: linear.size], linear, atol=1e-3)
