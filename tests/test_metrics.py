"""Tests for metric math, quality math, and the PCM baseline."""

import numpy as np
import pytest

from codecscope import analyze, compare
from codecscope.metrics import build_report
from codecscope import audio, quality


def _report(codes, codebook_size=1024, duration=1.0, sample_rate=16000, **kw):
    return build_report(
        "t", codes, codebook_size, duration, sample_rate, sample_rate, **kw
    )


class TestReportMath:
    def test_token_rate_and_frame_rate_flat_codec(self):
        # 4 codebooks x 50 frames over 1s -> 50 Hz frames, 200 tokens/s
        r = _report([np.zeros(50)] * 4)
        assert r.n_codebooks == 4
        assert r.token_count == 200
        assert r.frame_rate == 50.0
        assert r.token_rate == 200.0
        assert not r.is_multiscale

    def test_bitrate_from_codebook_size(self):
        # 200 tokens/s at 1024 entries = 10 bits each = 2000 bps
        r = _report([np.zeros(50)] * 4, codebook_size=1024)
        assert r.bits_per_code == 10.0
        assert r.bitrate == 2000.0

    def test_codebook_utilization(self):
        # 2 books, 1024 entries each; one uses 4 distinct, one uses 6
        books = [np.array([0, 1, 2, 3] * 4), np.array([0, 1, 2, 3, 4, 5] * 2)]
        r = _report(books, codebook_size=1024)
        assert r.unique_counts == (4, 6)
        assert r.codebook_utilization == pytest.approx(10 / 2048)

    def test_compression_ratio(self):
        # 16 kHz x 16-bit source = 256000 bps; codec at 2000 bps -> 128x
        r = _report([np.zeros(50)] * 4, codebook_size=1024)
        assert r.source_bitrate == 256000.0
        assert r.compression_ratio == pytest.approx(128.0)

    def test_multiscale_frame_rate_uses_finest_codebook(self):
        # SNAC-shaped: codebooks at different rates
        r = _report([np.zeros(12), np.zeros(24), np.zeros(48)])
        assert r.is_multiscale
        assert r.frame_rate == 48.0
        assert r.token_rate == 84.0
        # the identity that holds for flat codecs must NOT hold here
        assert r.token_rate != r.frame_rate * r.n_codebooks

    def test_zero_duration_no_division_error(self):
        r = _report([np.zeros(10)], duration=0.0)
        assert r.frame_rate == 0.0
        assert r.token_rate == 0.0
        assert r.compression_ratio == 0.0

    def test_as_dict_omits_absent_quality_scores(self):
        d = _report([np.zeros(10)]).as_dict()
        assert "si_snr" not in d and "stoi" not in d and "pesq" not in d

    def test_ragged_and_rectangular_inputs_agree(self):
        rect = _report(np.zeros((3, 20)))
        ragged = _report([np.zeros(20), np.zeros(20), np.zeros(20)])
        assert rect.code_counts == ragged.code_counts

    def test_empty_codes_rejected(self):
        with pytest.raises(ValueError):
            _report([])


class TestPCMBaseline:
    def test_compression_ratio_is_exactly_one(self):
        signal = audio.sine(duration=0.5, sample_rate=16000)
        r = analyze(signal, 16000, "pcm:16")
        assert r.compression_ratio == pytest.approx(1.0)
        assert r.codec == "pcm:16"

    def test_one_code_per_sample(self):
        signal = audio.sine(duration=0.25, sample_rate=8000)
        r = analyze(signal, 8000, "pcm:16")
        assert r.token_count == 2000
        assert r.frame_rate == pytest.approx(8000.0)

    def test_fewer_bits_compress_more_and_sound_worse(self):
        signal = audio.chirp(duration=0.5, sample_rate=16000)
        high = analyze(signal, 16000, "pcm:12")
        low = analyze(signal, 16000, "pcm:4")
        assert low.compression_ratio > high.compression_ratio
        assert low.si_snr < high.si_snr

    def test_bit_depth_bounds_validated(self):
        with pytest.raises(ValueError):
            analyze(audio.sine(0.1), 16000, "pcm:64")


class TestQualityMath:
    def test_si_snr_of_identical_signals_is_infinite(self):
        x = audio.sine(duration=0.1)
        assert quality.si_snr(x, x) == float("inf")

    def test_si_snr_is_scale_invariant(self):
        x = audio.sine(duration=0.1)
        noisy = x + 0.01 * audio.sine(duration=0.1, frequency=3000)
        assert quality.si_snr(x, noisy) == pytest.approx(
            quality.si_snr(x, noisy * 7.5), abs=1e-6
        )

    def test_si_snr_falls_as_noise_rises(self):
        rng = np.random.default_rng(0)
        x = audio.sine(duration=0.2)
        quiet = x + 0.01 * rng.standard_normal(x.size).astype(np.float32)
        loud = x + 0.20 * rng.standard_normal(x.size).astype(np.float32)
        assert quality.si_snr(x, quiet) > quality.si_snr(x, loud)

    def test_align_truncates_codec_padding(self):
        ref, est = quality.align(np.zeros(100), np.zeros(112))
        assert ref.size == est.size == 100

    def test_silent_reference_is_nan_not_crash(self):
        assert np.isnan(quality.si_snr(np.zeros(50), np.ones(50)))

    def test_evaluate_without_reconstruction_is_empty(self):
        assert quality.evaluate(audio.sine(0.1), None, 16000) == {}

    def test_pesq_rejects_unsupported_rate(self):
        x = audio.sine(duration=0.1, sample_rate=44100)
        with pytest.raises((ValueError, ImportError)):
            quality.pesq(x, x, 44100)


class TestDurationFromCoveredSamples:
    """Rates must count the samples the codes describe, not the input length.

    Codecs pad up to a whole frame. Dividing by the input length inflates
    every rate by the padding ratio — 2.4% for SNAC on a 2-second clip, which
    is the difference between matching a published bitrate and missing it.
    """

    class _PaddingCodec:
        """Emits one code per 100 samples, padding up to a whole frame."""

        name = "padder"
        sample_rate = 16000

        def encode(self, signal):
            from codecscope.adapters import CodecOutput

            n_frames = -(-signal.size // 100)  # ceil
            covered = n_frames * 100
            return CodecOutput(
                [np.zeros(n_frames, dtype=np.int64)],
                1024,
                reconstruction=np.zeros(covered, dtype=np.float32),
                n_samples=covered,
            )

    def test_rate_uses_padded_length(self):
        # 15950 samples -> 160 frames covering 16000 samples = exactly 1.0s
        signal = np.zeros(15950, dtype=np.float32)
        r = analyze(signal, 16000, self._PaddingCodec(), measure_quality=False)
        assert r.duration == pytest.approx(1.0)
        assert r.token_rate == pytest.approx(160.0)

    def test_input_length_would_have_inflated_the_rate(self):
        signal = np.zeros(15950, dtype=np.float32)
        r = analyze(signal, 16000, self._PaddingCodec(), measure_quality=False)
        naive = 160 / (15950 / 16000)
        assert naive > r.token_rate  # the bug this guards against

    def test_falls_back_to_reconstruction_length(self):
        from codecscope.adapters import CodecOutput

        class NoSampleCount:
            name = "nosamples"
            sample_rate = 16000

            def encode(self, signal):
                return CodecOutput(
                    [np.zeros(160, dtype=np.int64)],
                    1024,
                    reconstruction=np.zeros(16000, dtype=np.float32),
                )

        r = analyze(np.zeros(15950, np.float32), 16000, NoSampleCount(), False)
        assert r.duration == pytest.approx(1.0)


class TestDelayDetection:
    def test_zero_delay_for_identical_signals(self):
        x = audio.chirp(duration=0.2)
        assert quality.detect_delay(x, x) == 0

    @pytest.mark.parametrize("shift", [40, 128])
    def test_late_reconstruction_reports_positive_delay(self, shift):
        ref = audio.chirp(duration=0.3, sample_rate=16000)
        est = np.concatenate([np.zeros(shift, np.float32), ref])[: ref.size]
        assert quality.detect_delay(ref, est) == shift

    def test_early_reconstruction_reports_negative_delay(self):
        ref = audio.chirp(duration=0.3, sample_rate=16000)
        est = np.concatenate([ref[64:], np.zeros(64, np.float32)])
        assert quality.detect_delay(ref, est) == -64

    def test_silent_input_does_not_crash(self):
        assert quality.detect_delay(np.zeros(100), np.zeros(100)) == 0

    @pytest.mark.parametrize("shift", [0, 7, -13, 500])
    def test_matches_direct_correlation(self, shift):
        # The FFT path must agree exactly with textbook cross-correlation.
        rng = np.random.default_rng(0)
        ref = audio.chirp(duration=2.0, sample_rate=16000)
        ref = (ref + 0.01 * rng.standard_normal(ref.size)).astype(np.float32)
        if shift >= 0:
            est = np.concatenate([np.zeros(shift, np.float32), ref])[: ref.size]
        else:
            est = np.concatenate([ref[-shift:], np.zeros(-shift, np.float32)])

        centred_ref = ref - ref.mean()
        centred_est = est - est.mean()
        correlation = np.correlate(centred_est, centred_ref, mode="full")
        lags = np.arange(-(ref.size - 1), ref.size)
        inside = np.abs(lags) <= 2048
        expected = int(lags[inside][int(np.argmax(np.abs(correlation[inside])))])

        assert quality.detect_delay(ref, est) == expected == shift

    def test_cost_does_not_grow_with_track_length(self):
        """Guards against the O(N^2) correlation this replaced.

        Correlating directly cost ~184 s on 64 s of audio and would have
        taken ~26 minutes per codec on a 2-minute song, which made the
        library unusable on exactly the material it exists to measure.
        """
        import time

        def timed(seconds):
            ref = audio.chirp(duration=seconds, sample_rate=22050)
            est = np.concatenate([np.zeros(8, np.float32), ref])[: ref.size]
            start = time.perf_counter()
            assert quality.detect_delay(ref, est) == 8
            return time.perf_counter() - start

        short, long = timed(4.0), timed(64.0)
        # 16x the samples must not cost anything like 16x the time.
        assert long < max(0.5, short * 4)

    def test_window_selection_prefers_loud_material(self):
        # A recording that opens with silence must still align correctly.
        rng = np.random.default_rng(1)
        quiet = np.zeros(300_000, dtype=np.float32)
        loud = (0.5 * rng.standard_normal(300_000)).astype(np.float32)
        ref = np.concatenate([quiet, loud])
        est = np.concatenate([np.zeros(21, np.float32), ref])[: ref.size]
        assert quality.detect_delay(ref, est) == 21

    def test_delay_appears_in_report_dict_only_when_nonzero(self):
        r = _report([np.zeros(10)], delay_samples=0)
        assert "delay_samples" not in r.as_dict()
        assert (
            _report([np.zeros(10)], delay_samples=-85).as_dict()["delay_samples"] == -85
        )


class TestAlignedSISNR:
    """Delay compensation, without which SI-SNR is not comparable across codecs.

    An 8-sample offset costs DAC 16 kHz ~13 dB and DAC 24 kHz ~19 dB of
    apparent quality — enough to rank a 24 kbps codec below an 8 kbps one on
    real speech purely as an artifact of timing.
    """

    def test_recovers_quality_lost_to_a_pure_delay(self):
        ref = audio.chirp(duration=0.4, sample_rate=16000)
        delayed = np.concatenate([np.zeros(8, np.float32), ref])[: ref.size]
        raw = quality.si_snr(ref, delayed)
        aligned = quality.si_snr_aligned(ref, delayed)
        assert aligned > raw + 10

    def test_matches_raw_when_there_is_no_delay(self):
        ref = audio.chirp(duration=0.3, sample_rate=16000)
        est = ref + 0.02 * audio.sine(duration=0.3, sample_rate=16000, frequency=5000)
        assert quality.si_snr_aligned(ref, est) == pytest.approx(
            quality.si_snr(ref, est), abs=1e-9
        )

    def test_still_penalises_real_distortion(self):
        # Alignment must not launder quantization damage into a good score.
        from codecscope.adapters import PCMAdapter

        ref = audio.chirp(duration=0.3, sample_rate=16000)
        coarse = PCMAdapter("3").encode(ref).reconstruction
        assert quality.si_snr_aligned(ref, coarse) < 20.0

    def test_shift_helper_truncates_both_directions(self):
        ref, est = quality.shift(np.arange(10.0), np.arange(10.0), 3)
        assert ref.size == est.size == 7
        ref, est = quality.shift(np.arange(10.0), np.arange(10.0), -4)
        assert ref.size == est.size == 6

    def test_reported_alongside_raw(self):
        r = analyze(audio.chirp(duration=0.2, sample_rate=16000), 16000, "pcm:8")
        assert r.si_snr is not None and r.si_snr_aligned is not None
        assert "si_snr_aligned" in r.as_dict()


class TestCompare:
    def test_sorted_by_token_rate(self):
        signal = audio.sine(duration=0.2, sample_rate=16000)
        reports = compare(signal, 16000, ["pcm:8", "pcm:16"], measure_quality=False)
        # same token count; sort must be stable and total order well-defined
        assert [r.codec for r in reports] == ["pcm:8", "pcm:16"]

    def test_accepts_adapter_instance_not_just_spec(self):
        from codecscope.adapters import PCMAdapter

        signal = audio.sine(duration=0.1, sample_rate=16000)
        reports = compare(signal, 16000, [PCMAdapter("10")], measure_quality=False)
        assert reports[0].codec == "pcm:10"
