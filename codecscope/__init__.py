"""codecscope — measure how efficiently neural audio codecs tokenize audio.

>>> import codecscope
>>> from codecscope import audio
>>> signal = audio.sine(duration=1.0, sample_rate=16000)
>>> report = codecscope.analyze(signal, 16000, "pcm:16")
>>> report.compression_ratio
1.0
"""

from .adapters import CodecOutput, load
from .metrics import Report
from .runner import analyze, analyze_file, compare, compare_file

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "analyze_file",
    "compare",
    "compare_file",
    "load",
    "CodecOutput",
    "Report",
    "__version__",
]
