"""Real-weights checks for the SNAC backend.

Deselected by default (``-m "not integration"`` in pytest.ini). Run with:

    pytest -m integration

SNAC is the codec that exercises the multi-scale path: its three codebooks
run at different frame rates (``vq_strides == [4, 2, 1]``), so the flat-codec
identity ``token_rate == frame_rate * n_codebooks`` must NOT hold here.
"""

import pytest

from codecscope import audio, compare
from codecscope.adapters import SNACAdapter

pytestmark = pytest.mark.integration

pytest.importorskip("snac", reason="needs codecscope[snac]")

# SNAC 24 kHz, published: 3 codebooks of 4096 entries at ~0.98 kbps.
CODEBOOK_SIZE = 4096
N_CODEBOOKS = 3
PUBLISHED_BITRATE = 980.0


@pytest.fixture(scope="module")
def signal():
    return audio.chirp(duration=2.0, sample_rate=24000)


@pytest.fixture(scope="module")
def output(signal):
    return SNACAdapter("24khz").encode(signal)


class TestSNACRealWeights:
    def test_emits_three_codebooks_of_4096(self, output):
        assert len(output.codes) == N_CODEBOOKS
        assert output.codebook_size == CODEBOOK_SIZE

    def test_codebooks_run_at_different_rates(self, output):
        counts = [b.size for b in output.codes]
        # vq_strides [4, 2, 1] -> each scale is twice the previous
        assert counts == sorted(counts)
        assert counts[1] == 2 * counts[0]
        assert counts[2] == 2 * counts[1]

    def test_report_flags_multiscale(self, signal):
        report = compare(signal, 24000, ["snac:24khz"], measure_quality=False)[0]
        assert report.is_multiscale
        assert report.code_counts == (24, 48, 96)

    def test_flat_codec_identity_does_not_hold(self, signal):
        report = compare(signal, 24000, ["snac:24khz"], measure_quality=False)[0]
        # The whole reason Report stores per-codebook counts.
        assert report.token_rate != pytest.approx(
            report.frame_rate * report.n_codebooks
        )

    def test_bitrate_matches_published_figure(self, signal):
        # Only correct if duration comes from the padded length the codes
        # actually describe; using the input length inflates this by ~2.4%.
        report = compare(signal, 24000, ["snac:24khz"], measure_quality=False)[0]
        assert report.bitrate == pytest.approx(PUBLISHED_BITRATE, rel=0.01)

    def test_codec_pads_beyond_the_input_length(self, signal, output):
        assert output.reconstruction is not None
        assert output.reconstruction.size > signal.size

    def test_codec_delay_is_detected_and_reported(self, signal):
        report = compare(signal, 24000, ["snac:24khz"])[0]
        # SNAC returns its reconstruction early; the exact figure is a model
        # property, but a delay-free result would mean detection is broken.
        assert report.delay_samples is not None
        assert report.delay_samples != 0

    def test_codes_stay_inside_the_codebook(self, output):
        for book in output.codes:
            assert book.min() >= 0
            assert book.max() < CODEBOOK_SIZE
