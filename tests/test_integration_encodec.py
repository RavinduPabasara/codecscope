"""Real-weights checks for the EnCodec backend.

Deselected by default (``-m "not integration"`` in pytest.ini) because these
download ~300 MB and need torch. Run them explicitly:

    pytest -m integration

These assert against EnCodec's *published* configuration rather than against
whatever the adapter happens to produce, so they fail if a transformers
upgrade changes shapes, bandwidth handling, or the RVQ ladder.
"""

import pytest

from codecscope import audio, compare
from codecscope.adapters import EncodecAdapter

pytestmark = pytest.mark.integration

pytest.importorskip("transformers", reason="needs codecscope[hf]")

# EnCodec 24 kHz: 75 Hz frames, 1024-entry codebooks, and a documented
# bandwidth ladder where codebooks = bandwidth_kbps * 1000 / (75 * 10 bits).
BANDWIDTH_TO_CODEBOOKS = {1.5: 2, 3.0: 4, 6.0: 8, 12.0: 16, 24.0: 32}


@pytest.fixture(scope="module")
def signal():
    return audio.chirp(duration=2.0, sample_rate=24000)


class TestEncodecRealWeights:
    @pytest.mark.parametrize("bandwidth,expected_books", BANDWIDTH_TO_CODEBOOKS.items())
    def test_bandwidth_selects_documented_codebook_count(
        self, signal, bandwidth, expected_books
    ):
        adapter = EncodecAdapter(f"24khz@{bandwidth:g}")
        out = adapter.encode(signal)
        assert len(out.codes) == expected_books
        assert out.codebook_size == 1024

    def test_reported_bitrate_matches_the_requested_bandwidth(self, signal):
        # The strongest available check on the metric math: our bitrate is
        # derived from token rate and codebook size, EnCodec's label is not,
        # and the two must agree exactly.
        for bandwidth in BANDWIDTH_TO_CODEBOOKS:
            report = compare(
                signal, 24000, [f"encodec:24khz@{bandwidth:g}"], measure_quality=False
            )[0]
            assert report.bitrate == pytest.approx(bandwidth * 1000)

    def test_frame_rate_is_75hz(self, signal):
        report = compare(signal, 24000, ["encodec:24khz@6"], measure_quality=False)[0]
        assert report.frame_rate == pytest.approx(75.0, abs=0.5)
        assert not report.is_multiscale

    def test_quality_improves_with_bandwidth(self, signal):
        low, high = compare(signal, 24000, ["encodec:24khz@1.5", "encodec:24khz@24"])
        assert low.token_rate < high.token_rate
        assert low.si_snr < high.si_snr

    def test_bandwidth_is_named_so_a_table_cannot_hide_it(self):
        assert EncodecAdapter("24khz@6").name == "encodec:24khz@6"
        # An unqualified spec must not silently land on the lowest setting.
        assert EncodecAdapter("24khz").name != "encodec:24khz@1.5"

    def test_unsupported_bandwidth_is_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            EncodecAdapter("24khz@9")

    def test_reconstruction_length_matches_input(self, signal):
        out = EncodecAdapter("24khz@6").encode(signal)
        assert out.reconstruction is not None
        assert abs(out.reconstruction.size - signal.size) <= 320  # one frame hop


class TestEncodec48kHzStereoAndChunked:
    """The 48 kHz model is stereo *and* chunked, unlike the 24 kHz one.

    Both properties broke the adapter originally: mono input was rejected
    outright by the 2-channel encoder, and once that was fixed, only the
    first of several chunks was being read.
    """

    @pytest.fixture(scope="class")
    def signal(self):
        # Longer than the 1 s chunk length, so several chunks are produced.
        return audio.chirp(duration=3.0, sample_rate=48000)

    def test_mono_input_is_accepted_by_the_stereo_model(self, signal):
        out = EncodecAdapter("48khz@6").encode(signal)
        assert out.reconstruction is not None
        assert out.reconstruction.ndim == 1

    def test_all_chunks_are_read_not_just_the_first(self, signal):
        out = EncodecAdapter("48khz@6").encode(signal)
        # 1 s chunks at 150 Hz: reading only chunk 0 would give 150 codes.
        assert out.codes[0].size > 150
        assert out.codes[0].size == pytest.approx(600, abs=10)

    def test_bitrate_tracks_nominal_within_overlap_redundancy(self, signal):
        for bandwidth, nominal in ((6.0, 6000), (24.0, 24000)):
            report = compare(
                signal, 48000, [f"encodec:48khz@{bandwidth:g}"], measure_quality=False
            )[0]
            # 1% chunk overlap means slightly more codes are emitted than the
            # nominal figure implies; it must not be under, and not far over.
            assert nominal <= report.bitrate <= nominal * 1.03

    def test_quality_improves_with_bandwidth(self, signal):
        low, high = compare(signal, 48000, ["encodec:48khz@6", "encodec:48khz@24"])
        assert low.si_snr < high.si_snr
