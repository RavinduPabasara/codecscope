"""codecscope command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

from . import audio as audio_mod
from .metrics import Report
from .runner import compare, optional_dependency_notes

_COLUMNS = [
    ("codec", "Codec", 20),
    ("n_codebooks", "Books", 6),
    ("frame_rate", "Frame Hz", 9),
    ("token_rate", "Tokens/s", 9),
    ("bitrate", "Bitrate", 10),
    ("codebook_utilization", "Util", 7),
    ("compression_ratio", "Compress", 9),
    ("multiscale", "Multi", 6),
    ("delay_samples", "Delay", 6),
    ("si_snr", "SI-SNR", 8),
    ("stoi", "STOI", 7),
    ("pesq", "PESQ", 7),
]


def _format_table(reports: List[Report]) -> str:
    rows = [r.as_dict() for r in reports]
    cols = [c for c in _COLUMNS if any(c[0] in row for row in rows)]
    header = "  ".join(f"{title:<{w}}" for _, title, w in cols)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key, _, w in cols:
            value = row.get(key)
            if value is None:
                value = "-"
            elif isinstance(value, bool):
                # Without this a bool takes the integer format path and
                # prints as 0/1, which reads as a count rather than a flag.
                value = "yes" if value else "no"
            cells.append(f"{value:<{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codecscope",
        description="Compare how efficiently neural audio codecs tokenize your audio.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="audio file to analyze; omit to use a built-in synthetic chirp",
    )
    parser.add_argument(
        "-c",
        "--codec",
        action="append",
        required=True,
        dest="codecs",
        help="codec spec (repeatable): pcm:BITS, dac:RATE, encodec:RATE, "
        "snac:RATE, hf:REPO — bare names default to hf:",
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        help="also compute STOI and PESQ (speech-only metrics; "
        "meaningless on music)",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="skip reconstruction metrics (encode only, much faster)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="seconds of synthetic audio when no file is given (default 5)",
    )
    parser.add_argument(
        "--json", dest="json_path", help="also write results to a JSON file"
    )
    parser.add_argument(
        "--csv", dest="csv_path", help="also write results to a CSV file"
    )
    args = parser.parse_args(argv)

    if args.file:
        samples, sample_rate = audio_mod.load(args.file)
    else:
        sample_rate = 16000
        samples = audio_mod.chirp(duration=args.duration, sample_rate=sample_rate)

    reports = compare(
        samples,
        sample_rate,
        args.codecs,
        measure_quality=not args.no_quality,
        speech=args.speech,
    )

    print(_format_table(reports))
    for note in optional_dependency_notes(reports, speech=args.speech):
        print(f"note: {note}", file=sys.stderr)

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([r.as_dict() for r in reports], indent=2),
            encoding="utf-8",
        )
    if args.csv_path:
        rows = [r.as_dict() for r in reports]
        fieldnames = list({k: None for row in rows for k in row})
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
