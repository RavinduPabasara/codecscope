"""Real-weights checks for the Descript Audio Codec backend.

Deselected by default (``-m "not integration"`` in pytest.ini). Run with:

    pytest -m integration

`descript-audio-codec` pulls `descript-audiotools`, which pins
``protobuf<3.20`` and floats its torch requirement. Installing it into a
shared environment can downgrade torch and protobuf out from under other
packages, so prefer a dedicated virtualenv — see the README.
"""

import pytest

from codecscope import audio, compare

pytestmark = pytest.mark.integration

pytest.importorskip("dac", reason="needs codecscope[dac]")

# DAC 44 kHz / 8 kbps: 9 codebooks of 1024 entries at a 512-sample hop.
CODEBOOK_SIZE = 1024
N_CODEBOOKS = 9
HOP = 512
SAMPLE_RATE = 44100


@pytest.fixture(scope="module")
def signal():
    return audio.chirp(duration=2.0, sample_rate=SAMPLE_RATE)


@pytest.fixture(scope="module")
def report(signal):
    return compare(signal, SAMPLE_RATE, ["dac:44khz"])[0]


class TestDACRealWeights:
    def test_emits_nine_codebooks_of_1024(self, report):
        assert report.n_codebooks == N_CODEBOOKS
        assert report.codebook_size == CODEBOOK_SIZE

    def test_is_a_flat_codec(self, report):
        assert not report.is_multiscale
        assert len(set(report.code_counts)) == 1
        # the identity that fails for SNAC must hold here
        assert report.token_rate == pytest.approx(
            report.frame_rate * report.n_codebooks
        )

    def test_frame_rate_is_the_documented_hop(self, report):
        assert report.frame_rate == pytest.approx(SAMPLE_RATE / HOP, rel=0.01)

    def test_bitrate_is_the_exact_figure_behind_the_nominal_8kbps(self, report):
        # 9 books x 10 bits x 86.13 Hz = 7752 bps; "8 kbps" is the rounded label.
        assert report.bitrate == pytest.approx(7752, rel=0.01)

    def test_codec_pads_beyond_the_input(self, signal, report):
        assert report.duration > signal.size / SAMPLE_RATE

    def test_reconstruction_is_reasonable(self, report):
        # A working codec at ~8 kbps should comfortably beat 0 dB on a sweep.
        assert report.si_snr > 5.0
