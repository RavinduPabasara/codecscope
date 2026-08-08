"""Tests for adapter loading, audio I/O, rate matching, and the CLI."""

import json
import wave

import numpy as np
import pytest

from codecscope import audio
from codecscope.adapters import (
    DACAdapter,
    EncodecAdapter,
    HuggingFaceAdapter,
    PCMAdapter,
    SNACAdapter,
    load,
    resolve_backend,
)
from codecscope.cli import main
from codecscope.runner import analyze, optional_dependency_notes


class TestAdapterLoading:
    def test_pcm_spec(self):
        adapter = load("pcm:12")
        assert isinstance(adapter, PCMAdapter)
        assert adapter.bits == 12
        assert adapter.name == "pcm:12"

    def test_pcm_defaults_to_16_bits(self):
        assert load("pcm:").bits == 16

    def test_bare_name_routes_to_hf_backend(self):
        # Routing is checked without constructing the adapter, so the test
        # holds whether or not transformers is installed and never downloads.
        cls, rest = resolve_backend("kyutai/mimi")
        assert cls is HuggingFaceAdapter
        assert rest == "kyutai/mimi"

    def test_unknown_prefix_is_treated_as_hf_repo(self):
        cls, rest = resolve_backend("someorg/somecodec")
        assert cls is HuggingFaceAdapter
        assert rest == "someorg/somecodec"

    def test_known_prefixes_route_to_their_backends(self):
        assert resolve_backend("pcm:16")[0] is PCMAdapter
        assert resolve_backend("dac:44khz") == (DACAdapter, "44khz")
        assert resolve_backend("encodec:24khz") == (EncodecAdapter, "24khz")
        assert resolve_backend("snac:24khz") == (SNACAdapter, "24khz")


class TestPCMAdapterEncoding:
    def test_codes_are_within_codebook_range(self):
        out = PCMAdapter("8").encode(audio.chirp(duration=0.1, sample_rate=8000))
        codes = out.codes[0]
        assert out.codebook_size == 256
        assert codes.min() >= 0 and codes.max() <= 255

    def test_reconstruction_tracks_input(self):
        signal = audio.sine(duration=0.1, sample_rate=8000)
        out = PCMAdapter("16").encode(signal)
        assert out.reconstruction.shape == signal.shape
        assert np.max(np.abs(out.reconstruction - signal)) < 1e-3

    def test_clipping_is_bounded_not_wrapped(self):
        loud = np.array([-4.0, 4.0], dtype=np.float32)
        codes = PCMAdapter("8").encode(loud).codes[0]
        assert list(codes) == [0, 255]

    def test_single_codebook_so_never_multiscale(self):
        r = analyze(audio.sine(duration=0.1), 16000, "pcm:16")
        assert r.n_codebooks == 1
        assert not r.is_multiscale


class TestAudioIO:
    def test_wav_roundtrip(self, tmp_path):
        path = tmp_path / "tone.wav"
        original = audio.sine(duration=0.2, sample_rate=16000, amplitude=0.5)
        pcm = (original * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())

        loaded, rate = audio.load(str(path))
        assert rate == 16000
        assert loaded.dtype == np.float32
        assert np.max(np.abs(loaded - original)) < 1e-3

    def test_stereo_is_averaged_to_mono(self):
        stereo = np.stack([np.ones(100), -np.ones(100)], axis=1)
        assert np.allclose(audio.to_mono(stereo), 0.0)

    def test_resample_changes_length_proportionally(self):
        signal = audio.sine(duration=1.0, sample_rate=16000)
        assert audio.resample_linear(signal, 16000, 8000).size == 8000

    def test_resample_is_identity_at_same_rate(self):
        signal = audio.sine(duration=0.1, sample_rate=16000)
        assert audio.resample(signal, 16000, 16000) is signal

    def test_synthetic_signals_are_deterministic(self):
        assert np.array_equal(audio.chirp(0.1), audio.chirp(0.1))
        assert np.max(np.abs(audio.sine(0.1))) <= 0.5 + 1e-6


class TestFileWrappers:
    @pytest.fixture
    def wav(self, tmp_path):
        path = tmp_path / "tone.wav"
        signal = audio.chirp(duration=0.3, sample_rate=16000)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((signal * 32767).astype("<i2").tobytes())
        return str(path)

    def test_analyze_file_matches_analyze(self, wav):
        from codecscope import analyze_file

        samples, rate = audio.load(wav)
        assert (
            analyze_file(wav, "pcm:16").as_dict()
            == analyze(samples, rate, "pcm:16").as_dict()
        )

    def test_compare_file_returns_one_report_per_codec(self, wav):
        from codecscope import compare_file

        reports = compare_file(wav, ["pcm:16", "pcm:8"], measure_quality=False)
        assert [r.codec for r in reports] == ["pcm:16", "pcm:8"]

    def test_missing_file_raises(self):
        from codecscope import analyze_file

        with pytest.raises((FileNotFoundError, OSError)):
            analyze_file("does_not_exist.wav", "pcm:16")


class TestRunnerNotes:
    def test_multiscale_note_emitted(self):
        from codecscope.metrics import build_report

        ragged = build_report(
            "fake", [np.zeros(10), np.zeros(20)], 1024, 1.0, 16000, 16000
        )
        notes = optional_dependency_notes([ragged])
        assert any("multi-scale" in n for n in notes)

    def test_no_notes_for_plain_flat_run(self):
        r = analyze(audio.sine(duration=0.1), 16000, "pcm:16", measure_quality=False)
        assert optional_dependency_notes([r]) == []

    def test_speech_flag_reports_missing_quality_packages(self):
        from codecscope import quality

        r = analyze(audio.sine(duration=0.1), 16000, "pcm:16", measure_quality=False)
        notes = optional_dependency_notes([r], speech=True)
        missing = [n for n, ok in quality.available().items() if not ok]
        # Silently skipping a metric the user explicitly asked for is a bug.
        assert bool(missing) == any("not installed" in n for n in notes)


class TestCLI:
    def test_runs_on_synthetic_audio_without_a_file(self, capsys):
        assert main(["-c", "pcm:16", "-c", "pcm:8", "--duration", "0.2"]) == 0
        out = capsys.readouterr().out
        assert "Tokens/s" in out and "pcm:16" in out and "pcm:8" in out

    def test_writes_json_and_csv(self, tmp_path, capsys):
        json_path, csv_path = tmp_path / "r.json", tmp_path / "r.csv"
        main(
            [
                "-c",
                "pcm:16",
                "--duration",
                "0.2",
                "--json",
                str(json_path),
                "--csv",
                str(csv_path),
            ]
        )
        rows = json.loads(json_path.read_text())
        assert rows[0]["codec"] == "pcm:16"
        assert "token_rate" in rows[0]
        assert "codec" in csv_path.read_text().splitlines()[0]

    def test_no_quality_flag_omits_scores(self, tmp_path):
        json_path = tmp_path / "r.json"
        main(
            [
                "-c",
                "pcm:16",
                "--duration",
                "0.2",
                "--no-quality",
                "--json",
                str(json_path),
            ]
        )
        assert "si_snr" not in json.loads(json_path.read_text())[0]

    def test_booleans_render_as_flags_not_integers(self, capsys):
        main(["-c", "pcm:16", "--duration", "0.2"])
        out = capsys.readouterr().out
        assert "Multi" in out
        # a bool formatted with a width spec silently prints as 0/1
        assert "no" in out.split("\n")[2]

    def test_requires_at_least_one_codec(self):
        with pytest.raises(SystemExit):
            main(["--duration", "0.2"])
