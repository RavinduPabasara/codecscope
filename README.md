# codecscope

**Measure how efficiently neural audio codecs tokenize your audio.**

[![PyPI version](https://img.shields.io/pypi/v/codecscope.svg)](https://pypi.org/project/codecscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

When a neural codec feeds a language model, the number that decides your cost
is not bitrate — it's **tokens per second**. A 12.5 Hz codec with 8 codebooks
burns 100 tokens for every second of audio, and a 75 Hz codec with 4 codebooks
burns 300, even at a similar bitrate. Papers report these inconsistently, and
every comparison ships as a one-off eval script.

`codecscope` makes it one command: **token rate**, **bitrate**, **codebook
utilization**, and **reconstruction quality** across DAC, EnCodec, SNAC, Mimi,
and a raw-PCM baseline.

## Installation

```bash
pip install codecscope                 # core: numpy only
pip install 'codecscope[all]'          # + every codec and quality metric
pip install 'codecscope[dac,quality]'  # or pick what you need
```

## CLI

```bash
codecscope sample.wav \
    -c dac:44khz \
    -c encodec:24khz@6 \
    -c snac:24khz \
    -c hf:kyutai/mimi@8 \
    -c pcm:16 \
    --csv results.csv
```

Real output, every codec against real weights, over a 2-second sweep:

```
Codec             Books  Frame Hz  Tokens/s  Bitrate   Util     Compress  Multi  Delay  SI-SNR
-----------------------------------------------------------------------------------------------
snac:24khz        3      46.88     82.03     984.4     0.0099   716.8     yes    -85    -29.85
kyutai/mimi@8     8      12.5      100.0     1100.0    0.0094   641.45    no      -1    -10.54
encodec:24khz@6   8      75.0      600.0     6000.0    0.0833   117.6     no       -      5.56
dac:44khz         9      86.13     775.2     7752.0    0.1359   91.02     no       -     16.79
pcm:16            1      44100.0   44100.0   705600.0  0.4313   1.0       no       -     92.09
```

The bitrate column is derived independently — `token_rate × log2(codebook_size)` —
and lands on every codec's published figure:

| Codec | codecscope | Published |
|---|---|---|
| SNAC 24 kHz | 984.4 bps | 0.98 kbps |
| Mimi @8 | 1100.0 bps | 1.1 kbps |
| EnCodec 24 kHz @6 | 6000.0 bps | 6 kbps |
| DAC 44 kHz | 7752.0 bps | 8 kbps nominal (7752 is the exact math) |

Four independent agreements are the tightest available check that the metric
math is right.

Two things that table makes visible. SNAC has 3 codebooks at a 46.9 Hz finest
rate, yet 82 tokens/s rather than the 141 you would get by multiplying — the
multi-scale case, and the reason this tool exists. And the cheapest codec by
*bitrate* is not the cheapest by *token rate*: SNAC costs 82 tokens/s against
Mimi's 100, even though Mimi runs at a quarter of SNAC's frame rate. If a
language model is consuming these codes, that second column is your bill.

Run it with no file to sweep a synthetic chirp — useful for a quick smoke test
of a new backend:

```bash
codecscope -c pcm:16 -c pcm:8 -c pcm:4 --duration 3
```

## Python API

```python
import codecscope
from codecscope import audio

signal, rate = audio.load("sample.wav")

report = codecscope.analyze(signal, rate, "snac:24khz")
report.token_rate            # 207.0   codes/sec — what an LM actually pays
report.frame_rate            # 83.3    finest codebook's rate
report.bitrate               # 2070.0  bits/sec
report.codebook_utilization  # 0.88    fraction of entries actually emitted
report.compression_ratio     # 123.7   vs 16-bit PCM
report.si_snr                # 9.7     dB
report.is_multiscale         # True

reports = codecscope.compare(signal, rate, ["dac:44khz", "snac:24khz", "pcm:16"])
for r in reports:            # sorted by token rate, cheapest first
    print(r.as_dict())
```

## What the metrics mean

| Metric | Why it's here |
|---|---|
| `token_rate` | Codes per second. The real cost driver when a codec feeds an LM — and the number most papers bury. |
| `frame_rate` | Frames/sec of the **finest** codebook. |
| `bitrate` | `token_rate × log2(codebook_size)`. Two codecs at equal bitrate can differ 3× in token rate. |
| `codebook_utilization` | Fraction of codebook entries actually emitted. Low values mean vocabulary you pay for but never use — a known RVQ collapse signature, and the audio analogue of a tokenizer's UNK rate. |
| `compression_ratio` | Against the 16-bit PCM source. |
| `si_snr` | Scale-invariant SNR in dB. Always available; scale-invariant because a codec is free to change gain and shouldn't be punished for it. |
| `stoi` / `pesq` | Speech intelligibility. Opt-in via `--speech`. |

### Codec delay is reported, not silently corrected

SNAC returns its reconstruction about 85 samples early. SI-SNR is computed
**without** compensating for that, matching how codec papers report it —
silently re-aligning would flatter every codec with latency. But a
delay-driven score is indistinguishable from distortion unless you can see
the delay, so `delay_samples` is measured by cross-correlation and reported
alongside. A large delay with a poor SI-SNR means "check the alignment", not
"this codec is bad".

### Chunked codecs emit more codes than their nominal bitrate

EnCodec's 48 kHz model is stereo and chunks at 1 second with 1% overlap; its
24 kHz model does neither. Because the overlapping frames are genuinely
transmitted, `codecscope` counts them: `encodec:48khz@6` reports 6045 bps
against a nominal 6000, and 151.1 Hz against a true frame rate of 150. The
0.75% excess *is* the overlap redundancy, and it is a real cost that
per-frame arithmetic hides.

Mono input to a stereo codec is duplicated across channels rather than
rejected, and the decoded channels are averaged back before scoring.

### Rates count padded samples

Codecs pad their input up to a whole frame: SNAC turns 48,000 samples into
49,152. The codes describe the *padded* signal, so rates are computed against
that length. Dividing by the original input length instead inflates every
rate by the padding ratio — 2.4% on a 2-second SNAC clip, enough to miss the
published 0.98 kbps figure.

### Codebook utilization is a corpus property, not a codec property

Utilization counts the entries a codec actually emitted **on the audio you
gave it**. A short or narrowband clip will under-use any codebook — the 0.057
above is mostly a statement about a 2-second synthetic sweep, not about
EnCodec. Measure it over a representative corpus before drawing conclusions,
and compare codecs only on identical input.

### Variable-rate codecs must declare their configuration

Most neural codecs are variable-rate, and their libraries pick a default that
is rarely the configuration people quote:

- **EnCodec** keeps 2 to 32 codebooks depending on bandwidth, and transformers
  defaults to the **lowest** (1.5 kbps). An unqualified `encodec:24khz` would
  enter a comparison at 1.5 kbps against codecs running flat out.
- **Mimi** ships `num_quantizers=32` in its config, but its headline 1.1 kbps
  is the **8**-codebook setting. Unqualified, it reports 4.4 kbps — four times
  the number everyone cites.

`codecscope` resolves both explicitly and always prints the resolved value in
the codec name, so a table cannot misattribute a rate:

```bash
codecscope sample.wav -c encodec:24khz@6 -c encodec:24khz@24 -c hf:kyutai/mimi@8
```

### Multi-scale codecs

SNAC runs its codebooks at *different* frame rates. The identity most eval
scripts assume —

```
token_rate == frame_rate × n_codebooks
```

— is false for those codecs, and reporting a single "frame rate" for them is
the usual way published comparisons go wrong. `codecscope` stores per-codebook
counts, flags the row with `is_multiscale`, and prints a note when one is in
the table.

### PESQ is not a general audio metric

PESQ is defined only for 8 kHz and 16 kHz **speech**. It returns a confident
number for music that means nothing, so `codecscope` never computes it unless
you pass `--speech`, and refuses outright at other sample rates rather than
silently resampling into a score you can't compare to published figures.

## The PCM baseline

`pcm:N` is a plain uniform quantizer with no ML dependencies. At 16 bits
against a 16-bit source it compresses by exactly 1.0 and emits one code per
sample — the fixed reference every neural codec is measured against, and the
reason the core package installs with only numpy.

It's also a live correctness check: SI-SNR falls about 6.02 dB per bit
removed, the textbook quantization law.

```bash
$ codecscope -c pcm:16 -c pcm:8 -c pcm:4 -c pcm:2 --duration 3
pcm:16   ...  compress 1.0   SI-SNR 92.08
pcm:8    ...  compress 2.0   SI-SNR 44.05
pcm:4    ...  compress 4.0   SI-SNR 19.91
pcm:2    ...  compress 8.0   SI-SNR  6.31
```

## Codec specs

| Spec | Backend | Extra |
|---|---|---|
| `pcm:16` | uniform PCM quantizer | — (built in) |
| `dac:44khz`, `dac:24khz`, `dac:16khz` | Descript Audio Codec | `[dac]` |
| `encodec:24khz@6`, `encodec:48khz@12` | Meta EnCodec via transformers (bandwidth in kbps after `@`; default 6) | `[encodec]` |
| `snac:24khz`, `snac:32khz`, `snac:44khz` | multi-scale SNAC | `[snac]` |
| `hf:kyutai/mimi@8`, any repo id | transformers codecs (quantizer count after `@`) | `[hf]` |

A spec with no prefix is treated as a Hugging Face repo id.

### Installing the DAC extra

`descript-audio-codec` pulls `descript-audiotools`, which pins
`protobuf<3.20` and leaves its torch requirement unbounded. Installing it into
a shared environment can downgrade both protobuf and torch out from under
everything else — in testing it took torch from 2.10 to 2.7, which in turn
broke torchvision and every transformers codec. Give it its own virtualenv:

```bash
python -m venv .venv-dac && .venv-dac/bin/pip install 'codecscope[dac]'
```

## Notes on measurement

- Audio is resampled to each codec's operating rate before encoding, and
  quality is scored against that **resampled** signal — a codec shouldn't be
  charged for the resampler's error.
- Install `librosa` (`[audio]`) for band-limited resampling. Without it,
  rate matching falls back to linear interpolation and `codecscope` says so,
  because that aliasing lands in the quality scores.
- Reconstructions are truncated to the input length before scoring; codecs
  pad to whole frames, and scoring that padding as error is a silent bias.

## Tests

Two environments, because the DAC extra cannot share one (see above):

```bash
python -m venv .venv     && .venv/bin/pip     install -e '.[dev,hf,snac,quality,audio]'
python -m venv .venv-dac && .venv-dac/bin/pip install -e '.[dev,dac]'
```

```bash
pytest                  # 69 offline tests, no downloads, ~1s
pytest -m integration   # real-weights checks; 31 in .venv, 6 in .venv-dac
pytest -m ""            # everything available in the current environment
```

The integration suite asserts against each codec's *published* configuration
— EnCodec's 75 Hz frame rate and 2/4/8/16/32 codebook ladder, SNAC's three
4096-entry codebooks at 0.98 kbps with `vq_strides` [4, 2, 1], Mimi's 1.1 kbps
at 8 quantizers, DAC's 44100/512 hop — so it fails if an upstream release
changes shapes, rate handling, or frame math. Each file skips cleanly when its
backend is not installed.

## Related projects

- [tokscope](https://github.com/RavinduPabasara/tokscope) — the same question
  for text: how efficiently does a tokenizer handle your language?

## License

MIT © Ravindu Pabasara Karunarathna
