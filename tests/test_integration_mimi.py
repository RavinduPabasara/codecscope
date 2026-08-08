"""Real-weights checks for Mimi through the generic transformers backend.

Deselected by default (``-m "not integration"`` in pytest.ini). Run with:

    pytest -m integration

Mimi is the case that justifies naming the quantizer count: its config says
``num_quantizers=32``, but the 1.1 kbps figure everyone quotes is the
8-codebook configuration. An unqualified run reports 4.4 kbps.
"""

import pytest

from codecscope import audio, compare
from codecscope.adapters import HuggingFaceAdapter

pytestmark = pytest.mark.integration

pytest.importorskip("transformers", reason="needs codecscope[hf]")

REPO = "kyutai/mimi"
SAMPLE_RATE = 24000
CODEBOOK_SIZE = 2048
FRAME_RATE = 12.5
BITS_PER_CODE = 11  # log2(2048)


@pytest.fixture(scope="module")
def signal():
    return audio.chirp(duration=2.0, sample_rate=SAMPLE_RATE)


class TestMimiRealWeights:
    @pytest.mark.parametrize("quantizers", [8, 16, 32])
    def test_quantizer_count_is_honoured(self, signal, quantizers):
        report = compare(
            signal, SAMPLE_RATE, [f"hf:{REPO}@{quantizers}"], measure_quality=False
        )[0]
        assert report.n_codebooks == quantizers

    def test_eight_codebooks_reproduce_the_published_1_1_kbps(self, signal):
        report = compare(signal, SAMPLE_RATE, [f"hf:{REPO}@8"], measure_quality=False)[
            0
        ]
        assert report.frame_rate == pytest.approx(FRAME_RATE, rel=0.01)
        assert report.token_rate == pytest.approx(100.0, rel=0.01)
        assert report.bitrate == pytest.approx(1100.0, rel=0.01)

    def test_bitrate_scales_linearly_with_quantizers(self, signal):
        eight, sixteen = compare(
            signal,
            SAMPLE_RATE,
            [f"hf:{REPO}@8", f"hf:{REPO}@16"],
            measure_quality=False,
        )
        assert sixteen.bitrate == pytest.approx(2 * eight.bitrate, rel=0.01)
        assert eight.frame_rate == pytest.approx(sixteen.frame_rate, rel=0.01)

    def test_quantizer_count_is_named_so_a_table_cannot_hide_it(self):
        assert HuggingFaceAdapter(f"{REPO}@8").name == f"{REPO}@8"
        # An unqualified spec must still say which configuration it ran.
        assert "@" in HuggingFaceAdapter(REPO).name

    def test_out_of_range_quantizer_count_is_rejected(self):
        with pytest.raises(ValueError, match="quantizer count"):
            HuggingFaceAdapter(f"{REPO}@99")

    def test_is_a_flat_codec(self, signal):
        report = compare(signal, SAMPLE_RATE, [f"hf:{REPO}@8"], measure_quality=False)[
            0
        ]
        assert not report.is_multiscale
        assert report.codebook_size == CODEBOOK_SIZE
        assert report.bits_per_code == BITS_PER_CODE
