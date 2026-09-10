# Votrax SC-01 / SC-01-A Technical Overview

A comprehensive reference covering the original hardware, this emulation's architecture, and areas for improvement.

---

## Part 1: The Original Votrax SC-01A

### History

The Votrax SC-01A traces back to **Richard T. Gagnon**, who developed a formant-based speech synthesis approach at **Federal Screw Works** (a Michigan automotive parts company that diversified into electronics). The speech synthesis division became **Votrax** in 1974. The **SC-01** shipped in 1980, followed by the improved **SC-01A** in 1981. It was one of the first single-chip speech synthesizers that didn't require external ROM — all 64 phonemes were encoded in an on-die 512-byte ROM, making it cheap and easy to integrate.

The chip was not the beginning of the architecture but the end of it. Gagnon's design shipped for a decade before the SC-01, as rack and card-cage hardware — VS-4 (1972), VS-6 (1973–77), ML-1 (1978), and the potted VSK/VSL modules that the SC-01 is a single-chip reduction of. That generation had **five** cascaded formant resonators plus a nasal anti-resonator, 16 parameters per phoneme, a voltage-tunable vocal oscillator, and fricative noise injected at F2 so that F2–F5 resonated the fricatives too. Everything the chip lacks is something that was given up to fit 22 pins, which makes Gagnon's own description of the earlier hardware the best available specification for what an "enhanced" mode should be reaching for — see [ROADMAP.md](ROADMAP.md), Part 2.1, and the paper itself, which is held locally (References, below).

### Notable Uses

| Product | Type | Year |
|---|---|---|
| **Q\*bert** | Arcade (Gottlieb) | 1982 |
| **Gorf** | Arcade (Bally Midway) | 1981 |
| **Wizard of Wor** | Arcade (Bally Midway) | 1980 |
| **Type 'N Talk** | Serial speech module (Votrax) | 1982 |
| **Heathkit HERO** | Educational robot | 1982 |
| **Kurzweil Reading Machine** | OCR-to-speech for the blind | Early 1980s |
| **Apple II** | Via third-party cards (Sweet Talker, etc.) | 1981+ |

### Architecture

The SC-01A is a **22-pin CMOS** device containing:

- A **512-byte internal ROM** encoding parameters for 64 phonemes
- A **digital timing engine** that sequences and interpolates phoneme parameters
- An **excitation source** (9-level stepped glottal waveform + noise LFSR)
- A **13-stage analog pipeline** built from switched-capacitor filters
- An **inflection input** (2 bits) for pitch control

Everything runs from a single **720 kHz master clock** divided down internally.

### Specifications

| Parameter | Value |
|---|---|
| Master clock | 720 kHz |
| Analog sample rate (SCLOCK) | 40 kHz (÷18) |
| Chip update rate (CCLOCK) | 20 kHz (÷36) |
| Phonemes | 64 |
| Inflection levels | 4 (2-bit input) |
| ROM size | 512 bytes (64 entries × 2 words) |
| Parameters per phoneme | 12 fields |
| Formant filters | 4 (F1, F2, F3, F4) |
| Filter order | 3rd-order (standard), 2nd-order (noise shaper), 1st-order (FX, F2n) |
| Package | 22-pin DIP CMOS |
| Supply voltage | 5V |
| Output | Analog audio (pin 11) |

### The 64 Phonemes

The SC-01A encodes 64 phonemes, including vowels, consonants, pauses, and a stop code:

| Code | Name | Code | Name | Code | Name | Code | Name |
|---:|---|---:|---|---:|---|---:|---|
| 0x00 | EH3 | 0x10 | CH | 0x20 | A | 0x30 | AW2 |
| 0x01 | EH2 | 0x11 | SH | 0x21 | AY | 0x31 | UH2 |
| 0x02 | EH1 | 0x12 | Z | 0x22 | Y1 | 0x32 | UH1 |
| 0x03 | PA0 | 0x13 | AW1 | 0x23 | UH3 | 0x33 | UH |
| 0x04 | DT | 0x14 | NG | 0x24 | AH | 0x34 | O2 |
| 0x05 | A2 | 0x15 | AH1 | 0x25 | P | 0x35 | O1 |
| 0x06 | A1 | 0x16 | OO1 | 0x26 | O | 0x36 | IU |
| 0x07 | ZH | 0x17 | OO | 0x27 | I | 0x37 | U1 |
| 0x08 | AH2 | 0x18 | L | 0x28 | U | 0x38 | THV |
| 0x09 | I3 | 0x19 | K | 0x29 | Y | 0x39 | TH |
| 0x0A | I2 | 0x1A | J | 0x2A | T | 0x3A | ER |
| 0x0B | I1 | 0x1B | H | 0x2B | R | 0x3B | EH |
| 0x0C | M | 0x1C | G | 0x2C | E | 0x3C | E1 |
| 0x0D | N | 0x1D | F | 0x2D | W | 0x3D | AW |
| 0x0E | B | 0x1E | D | 0x2E | AE | 0x3E | PA1 |
| 0x0F | V | 0x1F | S | 0x2F | AE1 | 0x3F | STOP |

Vowels come in numbered variants (e.g., EH, EH1, EH2, EH3) that share identical formant targets but differ in closure delay, voice delay, and duration — encoding coarticulation context. Shorter variants are used between plosives, longer variants in open contexts.

The numbering runs shortest-first, consistently across every family — a fact
worth knowing because it identifies phonemes independently of any name table:

| Family | Phones, by ROM duration field |
|---|---|
| EH | EH3 = 19, EH2 = 23, EH1 = 38, EH = 58 |
| I | I3 = 18, I2 = 26, I1 = 38, I = 58 |
| UH | UH3 = 15, UH2 = 23, UH1 = 33, UH = 58 |
| A | A2 = 23, A1 = 33, A = 58 |

Multiplied by about 3.14, the ROM duration field reproduces the millisecond
figures on the datasheet's phoneme chart across all 64 entries (EH3 19 → 59 ms,
EH1 38 → 121 ms, AW 76 → 250 ms), which is a useful cross-check whenever a
table's ordering is in doubt.

### The two mask revisions

There are two production mask ROMs, and the difference between them is small,
sharp, and audible.

The 1980 **SC-01** and the later **SC-01-A** are identical in 52 of their 64
rows. In the other twelve, exactly one field moves — the voice amplitude,
`va` — and it moves in one direction:

| Code | Phone | SC-01 (1980) | SC-01-A |
|---:|---|---:|---:|
| 0x08 | AH2 | 15 | 9 |
| 0x13 | AW1 | 15 | 11 |
| 0x15 | AH1 | 15 | 9 |
| 0x23 | UH3 | 15 | 14 |
| 0x24 | AH | 15 | 9 |
| 0x2E | AE | 15 | 11 |
| 0x2F | AE1 | 15 | 11 |
| 0x30 | AW2 | 15 | 11 |
| 0x31 | UH2 | 15 | 14 |
| 0x32 | UH1 | 15 | 14 |
| 0x33 | UH | 15 | 14 |
| 0x3D | AW | 15 | 11 |

Every one of the twelve is an open vowel, and the original ran all of them at
full scale. The revision pulled them down. Nothing else changes: word0 is
untouched throughout, so the formant targets, durations, closure delays and
voice delays are bit-identical between the parts — only the loudness of the
open vowels differs.

The result is that the SC-01 is the louder, more strident voice, by roughly
+29% RMS and +48% peak on ordinary speech, with the vowels sitting much closer
to the top of the output range. That harder-edged vowel is a large part of what
people remember about the earliest Votrax units, and it is the reason this
project treats the SC-01 rather than the -A as the voice to aim at.

Both masks are compiled in — `MaskRevision::SC01` and `MaskRevision::SC01A` in
`csrc/rom_data.h`, mirrored in `py_emu/rom.py`, exposed as the `mask` argument
and property on the chip. There is no ROM file to find or load: a mask ROM is
immutable silicon, so the tables in the source *are* the chip.

```python
from pyvotrax.chip import VotraxSC01A, MaskRevision

chip = VotraxSC01A(mask=MaskRevision.SC01)   # the 1980 voice
chip.mask = MaskRevision.SC01A               # takes effect at the next phone
```

### ROM Parameter Encoding

Each of the 64 phonemes is stored as two words: a 12-bit **word0** and a 32-bit **word1**. From these, 12 parameters are extracted:

| Field | Bits | Source | Description |
|---|---|---|---|
| f1 | 4 | word1 slot 0 | Filter 1 (first formant) frequency |
| va | 4 | word1 slot 1 | Voiced amplitude |
| f2 | 4 | word1 slot 2 | Filter 2 frequency (used as 5-bit after commit) |
| fc | 4 | word1 slot 3 | Fricative/noise control |
| f2q | 4 | word1 slot 4 | Filter 2 Q factor |
| f3 | 4 | word1 slot 5 | Filter 3 frequency |
| fa | 4 | word1 slot 6 | Fricative amplitude |
| cld | 4 | word0+word1 | Closure delay (ticks before closure flag activates) |
| vd | 4 | word0+word1 | Voice delay (ticks before fa begins interpolating) |
| closure | 1 | word0 bit 4 | Closure flag |
| duration | 7 | word0 bits 5-11 | Phoneme duration (XOR inverted) |
| pause | 1 | derived | True for codes 0x03 (PA0) and 0x3E (PA1) |

**Bit-interleaving**: The 4-bit parameter extraction from word1 is unusual — bits are spaced 7 positions apart:

```
bit0 = (word1 >> slot) & 1
bit1 = (word1 >> (slot + 7)) & 1
bit2 = (word1 >> (slot + 14)) & 1
bit3 = (word1 >> (slot + 21)) & 1
```

The cld and vd fields are extracted from a combined (word0 << 4 | word1 >> 28) value with similar bit-interleaving (every other bit). The duration field has its bits reversed and XOR-inverted (`^ 0x7F`), corresponding to MAME's `bitswap(~val, ...)` pattern.

**Known die bugs**: The ROM contains at least one documented anomaly — certain phonemes have unexpected parameter values that appear to be manufacturing defects preserved in silicon.

### Digital Timing Engine

The timing engine runs at the 20 kHz CCLOCK rate:

**Duration counters** — A two-level counter system:
1. `phonetick` counts up from 0. When it reaches `(duration << 2) | 1`, it resets and increments `ticks`.
2. `ticks` counts from 0 to 16 (0x10). A phoneme is "done" when ticks reaches 16.
3. When `ticks` reaches the `cld` threshold, the closure flag activates.

**Update counter** — A modulo-48 (0x30) counter that generates two timing pulses:
- **tick_625** (~1250 Hz): fires when `(counter & 0xF) == 0` — drives amplitude interpolation
- **tick_208** (~417 Hz): fires when `counter == 0x28` — drives formant interpolation

**Interpolation** — An exponential-decay approach:
```
register = (register - (register >> 3) + (target << 1)) & 0xFF
```
This is equivalent to `reg = reg * 7/8 + target * 2`, an 8-bit fixed-point IIR smoother. Formant frequencies (f1, f2, f2q, f3, fc) interpolate at ~417 Hz; amplitudes (va, fa) interpolate at ~1250 Hz, gated by the cld and vd delay thresholds.

**Pitch counter** — An 8-bit counter that resets to 0 when it reaches:
```
target = (0xE0 ^ (inflection << 5) ^ (filt_f1 << 1)) + 2
```
This creates the F0 pitch period. The inflection input shifts pitch by 32 counts per level, and F1 coupling (`filt_f1 << 1`) makes the fundamental frequency track the first formant — a deliberate design choice that gives the SC-01A its characteristic sound.

Filter coefficients are committed when `(pitch & 0xF9) == 0x08`, which occurs near the start of each pitch cycle.

### Excitation Sources

**Glottal waveform** — The voice source is a 9-level stepped waveform derived from a transistor resistor ladder on the die:

```
Index:  0      1      2     3     4     5     6     7     8
Value:  0.0   -4/7   1.0   6/7   5/7   4/7   3/7   2/7   1/7
```

The pitch counter's upper bits (pitch >> 3) select the glottal index. With a typical pitch period of ~128 counts, indices 0-8 span the first 72 counts (~55% of the period), giving an open quotient of roughly 55%. The remaining counts produce silence (closed phase). The negative dip at index 1 creates a brief negative excursion before the main pulse — a crude approximation of glottal opening.

**Noise LFSR** — A 15-bit linear feedback shift register with XOR feedback from bits 13 and 14:
```
input = cur_noise AND (noise != 0x7FFF)
noise = (noise << 1) | input
cur_noise = NOT((noise >> 14) XOR (noise >> 13))
```
The noise is gated by pitch bit 6 (alternating on/off over ~64 CCLOCK ticks), creating a buzzy, periodically-modulated noise source used for fricatives.

### Analog Signal Path

The SC-01A implements a 13-stage analog pipeline using switched-capacitor filters:

```
Glottal ──*va/15──► F1 ──► F2v ──────────────────┐
                                                   ├──► F3 ──+noise──► F4 ──*closure──► FX ──*0.35──► OUT
Noise LFSR ──*fa/15──► Shaper ──┬──*fc/15──► F2n ─┘           ▲
                                 │                              │
                                 └──*(5+(15^fc))/20 ───────────┘
```

**Voice path**: Glottal source scaled by va/15, through F1 (first formant bandpass) and F2v (second formant bandpass, voiced).

**Noise path**: LFSR output scaled by fa/15 through a bandpass noise shaper, then split:
1. Through F2n (noise injection filter) scaled by fc/15, merged with the voice path at F3 input
2. Direct injection after F3, scaled by `(5 + (15 ^ fc)) / 20`

**Common path**: F3 (third formant), F4 (fixed fourth formant), closure attenuation, and FX (final lowpass).

The closure attenuator uses a 3-bit counter (0-7) that ramps up when closure is inactive and sound is present, providing smooth onset/offset:
```
attenuation = (7 XOR (closure_counter >> 2)) / 7
```

### Switched-Capacitor Filter Technology

The SC-01A is one of the earliest commercial applications of **switched-capacitor** filter technology, covered by US Patent 4,433,210. Instead of using physical resistors (which are expensive to fabricate precisely on CMOS dies), the chip uses capacitors switched at the clock rate to simulate resistors:

```
R_effective = 1 / (f_clock × C)
```

This allowed all filter components to be realized purely in CMOS, with the filter characteristics determined by **capacitor ratios** (which can be controlled precisely in fabrication) rather than absolute values. The die-measured capacitor values are specified in **µm²** (proportional to capacitance), and only their ratios matter for the filter math.

**Variable filters** use binary-weighted capacitor banks selected by the ROM parameters:
- F1: c3 = 2280 + bits_to_caps(f1, [2546, 4973, 9861, 19724])
- F2v: c2t varies with f2q, c3 varies with f2 (5-bit)
- F3: c3 = 8480 + bits_to_caps(f3, [2226, 4485, 9056, 18111])
- F4, noise shaper, FX: fixed capacitor values

Each standard formant filter implements a 3rd-order transfer function:

```
H(s) = (1 + k0·s) / (1 + k1·s + k2·s²)
```

where k0, k1, k2 are ratios of the die capacitor values.

### Sound Characteristics and Limitations

The SC-01A produces distinctly **robotic** speech, which was both its charm (in arcade games) and its limitation (for accessibility devices):

- **Only 4 pitch levels**: The 2-bit inflection input provides coarse pitch control. Combined with F0-F1 coupling, this makes natural-sounding prosody difficult.
- **F0-F1 coupling**: The fundamental frequency tracks the first formant via `filt_f1 << 1` in the pitch counter. This is physiologically backward (in human speech, F0 and F1 are largely independent) and makes certain vowels sound higher-pitched than intended.
- **Coarse formant resolution**: Each formant frequency is controlled by only 4 bits (16 values), quantizing the vowel space coarsely. F2 gets 5 bits (32 values) after the commit shift, but this is still far from continuous.
- **Weak plosives**: Stop consonants (P, B, T, D, K, G) rely on the closure mechanism and brief noise bursts, but lack the transient energy of natural stops.
- **Fixed F4**: The fourth formant has no variable parameters, using a fixed bandpass that approximates average vocal tract resonance.
- **3rd-order filters**: Real vocal tract resonances are well-modeled by 2nd-order poles, but the SC-01A's 3rd-order filters add a zero that creates subtle spectral coloring.

### Die Analysis

The SC-01A's internal workings were reverse-engineered by **Olivier Galibert** through die photography and analysis, published starting with **MAME 0.181** (2016). Die photographs and schematics are available at [og.kervella.org/sc01a](https://og.kervella.org/sc01a/).

This work revealed:
- The exact ROM contents and bit-interleaving scheme
- All capacitor values (in µm²) for the switched-capacitor filters
- The 9-level glottal waveform resistor ladder
- The noise LFSR feedback polynomial (15-bit with NXOR on bits 14/13)
- The interpolation algorithm and timing counter structure
- Several die bugs in the ROM data (see Part 3, "Intentionally-preserved MAME die bugs")

Prior to this die-level analysis, SC-01A emulation relied on external recordings and guesswork. The MAME implementation (votrax.cpp, `copyright-holders: Olivier Galibert`, BSD-3-Clause) is now the definitive reference for SC-01A behavior — including places where the observed chip diverges from what the schematic would predict (e.g. the FX lowpass cutoff).

---

## Part 2: The pyvotrax Emulation

### Architecture Overview

pyvotrax is a Python + C++ emulation of the SC-01A with the following file structure:

| File | Role |
|---|---|
| `pyvotrax/rom.py` | ROM data extraction (faithful port of MAME bitswap) |
| `pyvotrax/filters.py` | Bilinear z-transform filter construction |
| `pyvotrax/chip.py` | Core DSP: dual-rate 40/20 kHz engine |
| `pyvotrax/synth.py` | High-level phoneme sequencing, resampling, WAV output |
| `pyvotrax/tts.py` | CMU dict TTS with prosody |
| `pyvotrax/phonemes.py` | 64-entry phoneme table and name↔code lookup |
| `csrc/votrax_core.h` | C++ chip emulation (mirrors chip.py) |
| `csrc/filters.h` | C++ filter construction (mirrors filters.py) |
| `csrc/rom_data.h` | C++ ROM data (mirrors rom.py) |
| `csrc/bindings.cpp` | pybind11 bindings exposing VotraxSC01ACore to Python |

### Full Pipeline

```
Text ──► tts.py (CMU dict lookup, ARPAbet→Votrax mapping, prosody)
           │
           ▼
  [(phoneme_code, inflection), ...]
           │
           ▼
      synth.py (sequencing, phone_commit → generate until done)
           │
           ▼
       chip.py / votrax_core.h (DSP: ROM → interpolation → excitation → filters)
           │
           ▼
    40 kHz float64 samples
           │
           ▼
      synth.py (200 ms decay tail, rational resampling via scipy, RMS normalization)
           │
           ▼
    44.1 kHz int16 WAV file
```

### ROM Data Extraction (rom.py)

`rom.py` is a faithful port of MAME's ROM decoding. The raw ROM data is stored as 64 pairs of (word0, word1) integers. The extraction functions replicate the bit-interleaved parameter packing:

- `_extract_param(word1, slot)` — Extracts a 4-bit parameter by sampling bits at offsets 0, 7, 14, and 21 from the slot position, with MSB/LSB reordering.
- `_extract_clvd(word0, word1, slot)` — Extracts cld and vd from a combined word with every-other-bit interleaving.
- Duration extraction includes the `^ 0x7F` inversion matching MAME's `bitswap(~val, ...)`.

All 64 phonemes are decoded at import time into `ROM_DATA`, a list of `PhonemeParams` named tuples.

### Filter Construction (filters.py)

All filters use the **bilinear z-transform** with **frequency pre-warping** to map analog prototypes to discrete-time IIR filters. The approach matches MAME's votrax.cpp lines 849-986:

1. Compute analog prototype coefficients (k0, k1, k2) from capacitor ratios
2. Estimate the peak frequency: `fpeak = sqrt(|k0·k1 - k2|) / (2π·k2)`
3. Pre-warp: `zc = 2π·fpeak / tan(π·fpeak / SCLOCK)`
4. Apply bilinear transform to get discrete (a, b) coefficient arrays
5. Normalize so b[0] = 1

Five filter types are implemented:

| Filter | Order | Type | Transfer Function |
|---|---|---|---|
| `build_standard_filter` | 3rd | Bandpass (F1, F2v, F3, F4) | (1 + k0·s) / (1 + k1·s + k2·s²) |
| `build_noise_shaper_filter` | 2nd | Bandpass | k0·s / (1 + k1·s + k2·s²) |
| `build_lowpass_filter` | 1st | Lowpass (FX) | 1 / (1 + k·s) with 150/4000 fudge |
| `build_injection_filter` | 1st | Allpass-like (F2n) | (k0 + k2·s) / (k1 + k2·s) [pole-reflected] |
| `apply_filter` | — | IIR runner | Direct-form II transposed |

**F2n pole reflection**: The analog noise injection circuit has a transfer function `H(s) = (k0 + k2·s) / (k1 - k2·s)` with a right-half-plane pole (unstable). MAME neutralizes this by clamping. pyvotrax instead reflects the pole: `H_stable(s) = (k0 + k2·s) / (k1 + k2·s)`, which preserves the magnitude response while guaranteeing the discrete-time pole falls inside the unit circle.

**The 150/4000 fudge factor**: MAME's comment notes that the die-measured capacitor values for the final lowpass filter (FX) put the cutoff at ~150 Hz, but recordings show the actual cutoff is around 4 kHz. The filter code applies a `150/4000` scaling factor to compensate — the exact cause of this discrepancy is unknown.

### Core DSP (chip.py + votrax_core.h)

The core emulates the SC-01A's dual-rate architecture:

- **40 kHz (SCLOCK)**: `analog_calc()` runs every sample — computes the full filter cascade
- **20 kHz (CCLOCK)**: `chip_update()` runs every other sample — timing, interpolation, pitch, noise LFSR

The Python `VotraxSC01A` class delegates to the C++ `VotraxSC01ACore` when available:

```python
class VotraxSC01A:
    def __init__(self, use_native=True, enhanced=False):
        if use_native and _HAS_NATIVE:
            self._native = _NativeCore(enhanced)
        # Methods check self._native and delegate or run pure Python
```

Every method (reset, phone_commit, generate_one_sample, generate_samples, phone_done) follows this delegation pattern, with the pure Python implementation as a byte-identical fallback.

### Enhanced Mode (C++ only)

When `enhanced=True`, the C++ backend replaces the original 9-level stepped glottal waveform with a more realistic excitation source:

**KLGLOTT88 polynomial glottal pulse** — Based on the Klatt/Liljencrants model:
- **Opening phase** (0 to OQ/(1+SQ)): Smooth Hermite cubic rise: `3t² - 2t³`
- **Closing phase** (OQ/(1+SQ) to OQ): Quadratic fall: `1 - t²`
- **Closed phase** (OQ to 1.0): Zero output
- Default parameters: OQ (open quotient) = 0.55, SQ (speed quotient) = 2.0

**PolyBLEP anti-aliasing** — Applied at the glottal closure point (phase ≈ 0.55) to reduce aliasing from the discontinuity in the first derivative:
```cpp
glottal += polyblep(closure_phase, 1.0 / period) * 0.5;
```

**F0 jitter** (~1.5%) — Gaussian perturbation of the pitch period target:
```cpp
std::normal_distribution<double> jitter_dist(0.0, 0.015 * pitch_target);
int jittered = pitch_target + static_cast<int>(jitter_dist(m_rng));
```

**Amplitude shimmer** (~3%) — Gaussian scaling of the glottal output:
```cpp
std::normal_distribution<double> shimmer_dist(1.0, 0.03);
glottal *= shimmer_dist(m_rng);
```

These enhancements reduce the "stepped" quality of the original waveform while preserving the SC-01A's formant structure and timing characteristics.

### High-Level Synthesis (synth.py)

`VotraxSynthesizer` provides phoneme sequencing:

1. **Reset** the chip
2. For each (phoneme_code, inflection) pair:
   - Call `phone_commit(code, inflection)`
   - Generate samples until `phone_done` returns True
3. **200 ms decay tail** — After all phonemes, generate 8000 additional samples (200 ms at 40 kHz) to capture filter ring-down, mimicking how the real chip's filters continue resonating after the last phoneme
4. **Rational resampling** — Convert from 40 kHz to the target rate (default 44100 Hz) using `scipy.signal.resample_poly` with GCD-reduced up/down factors (44100/40000 → 441/400)
5. **RMS normalization** — Normalize to -12 dB below full scale using RMS rather than peak normalization, which preserves the natural relative levels between vowels and noisy consonants

### TTS Pipeline (tts.py)

`VotraxTTS` converts English text to speech:

**CMU dictionary lookup** — Words are looked up in the CMU Pronouncing Dictionary (134,000+ entries). Unknown words fall back to letter-by-letter spelling with pauses between letters.

**ARPAbet → Votrax mapping** — Each ARPAbet phoneme maps to one or more Votrax phonemes (e.g., `OY → ["O1", "Y"]`). The mapping table covers all 39 ARPAbet phonemes.

**Vowel variant selection** — Context-dependent rules select from the numbered vowel variants:
- Between two stops/affricates → shortest variant (e.g., EH3)
- Before a pause/word boundary → longest variant (e.g., EH)
- Word-final unstressed → shortest (vowel reduction)
- Primary stress → longest, secondary → medium, unstressed → shortest

**4-level inflection prosody**:

| ARPAbet Stress | Votrax Inflection | Effect |
|---|---|---|
| Primary (1) | 2 | Highest normal pitch |
| Secondary (2) | 1 | Moderate pitch |
| Unstressed (0) | 0 | Lowest pitch |
| Question-final | 3 | Rising pitch (highest) |

**Sentence-final contours**:
- Questions (?) — Last 3 non-pause phonemes set to inflection 3 (rising)
- Statements (.) — Last 3 non-pause phonemes set to inflection 0 (falling)
- Exclamations (!) — Same as statements

**Pre-pausal lengthening** — An extra PA0 pause is inserted before the final STOP, mimicking the natural lengthening speakers produce at phrase boundaries.

### C++ Backend Performance

The C++ backend (built via `python setup.py build_ext --inplace`) achieves a **601x speedup** over pure Python:

| | Pure Python | C++ Backend |
|---|---|---|
| 1 second of audio | ~2.3 s | ~3.8 ms |

Key optimizations:
- **Template-optimized filters** — `apply_filter<NA, NB>` and `shift_hist<N>` use compile-time template parameters, allowing the compiler to unroll loops and optimize for each filter order
- **Inline functions** — All filter building and application functions are `inline`
- **Contiguous memory** — Filter histories use stack-allocated C arrays instead of heap-allocated numpy arrays
- **No Python overhead** — The hot loop (generate_samples) runs entirely in C++ without GIL interaction

### Key Differences from MAME

| Aspect | MAME (votrax.cpp) | pyvotrax |
|---|---|---|
| F2n (noise injection) | Neutralizes unstable pole (clamp) | Reflects RHP pole to LHP (stable bilinear transform) |
| Enhanced mode | Not present | KLGLOTT88 glottal, PolyBLEP, jitter, shimmer |
| Post-phoneme decay | Not applicable (continuous emulation) | 200 ms tail after last phoneme |
| Normalization | Raw DAC output | RMS normalization to -12 dB |
| Output | Real-time audio stream | Offline WAV generation |
| Language | C++ (integrated in MAME framework) | Python + C++ pybind11 (standalone) |

---

## Part 3: Areas for Improvement

### Excitation Model

- **LF model**: The KLGLOTT88 polynomial is good but the Liljencrants-Fant (LF) model better captures the spectral tilt and return phase of real glottal pulses. Could implement the LF four-parameter model (Ee, Tp, Te, Ta).
- **Per-phoneme KLGLOTT88 parameters**: Currently OQ=0.55 and SQ=2.0 are fixed. Different phonemes would benefit from different open quotients (e.g., breathy vowels → higher OQ, creaky voice → lower OQ).
- **Aspiration noise mixing**: Real speech has aspiration noise mixed into the glottal source during the open phase, proportional to the glottal aperture. Currently noise is only injected through the separate fricative path.

### Formant Resolution

- **Only 16 values per formant**: The 4-bit ROM parameters quantize formant frequencies to 16 steps (32 for F2). Some phonemes land between ideal values.
- **Formant overrides**: Could allow per-phoneme formant frequency overrides (bypassing the ROM lookup) for fine-tuning vowel quality.
- **Higher formants**: F5 and above contribute to speaker identity and naturalness. Adding even a fixed F5 bandpass would help.
- **Independent bandwidth control**: Currently F2Q is the only bandwidth parameter. Real vocal tracts have independent bandwidth for each formant.

### Noise Model

- **LFSR limitations**: The 15-bit LFSR produces periodic noise (period 32767). Real turbulence noise has a different spectral shape.
- **Turbulence models**: Could implement Fant's turbulence noise model, where noise is generated at the constriction point and shaped by the downstream vocal tract.
- **Shaped noise**: Different fricatives (S vs. SH vs. F) need different noise spectral shapes. Currently all share the same LFSR source and shaping filter.

### Timing and Coarticulation

- **Fixed durations**: Phoneme durations are ROM-encoded constants. Real speech shows context-dependent duration variation (e.g., vowels shorten before voiceless stops).
- **No coarticulation model**: Formant transitions between phonemes rely entirely on the ~417 Hz exponential interpolation. Real coarticulation involves anticipatory and carryover effects with different time constants per formant.
- **Abrupt onsets**: The closure/release mechanism provides some smoothing, but plosive onsets still lack the gradual spectral transitions of natural speech.

### Prosody

- **Only 4 pitch levels**: The 2-bit inflection input gives only 4 possible pitch values. Natural speech uses continuous F0 contours.
- **No loudness variation**: All phonemes at a given inflection level produce similar amplitude. Natural speech varies loudness with stress and emphasis.
- **F0-F1 coupling**: The pitch counter's dependency on filt_f1 causes unnatural pitch-formant correlation. Decoupling these would require changing the pitch counter formula, diverging from hardware accuracy.

### Filters

- **The 150/4000 fudge factor — resolved (2026-04-21)**: MAME's own comment in `build_lowpass_filter()` is unambiguous: *"The caps values puts the cutoff at around 150Hz, put that's no good. Recordings shows we want it around 4K, so fuzz it."* The 150 Hz is what the schematic/die capacitor values imply; the 4 kHz is what actual recorded SC-01A output sounds like. Galibert intentionally matched the recordings, not the schematic. Independent confirmation: Gagnon's 1974 US patent 3,836,717 describes a fixed nasal-resonance filter with its pole at ~4 kHz at the end of the analog chain — so ~4 kHz is the authentic bandwidth by design. Our port inherits this fudge and should document it rather than "fix" it. A runtime knob that lets users dial between "as-schematic" (150 Hz, muffled) and "as-recorded" (4 kHz, authentic) is a legitimate sound-design feature.
- **Fixed F4**: The fourth formant is hardcoded. Making it variable (even with 2-3 bits) would better model individual speaker characteristics.
- **3rd-order limit**: The standard filters implement 3rd-order transfer functions. While adequate for formant modeling, 4th-order Butterworth or Chebyshev filters would provide sharper resonances matching real vocal tract resonances.

### Sound-design parameters the SC-01A datasheet explicitly endorses

Research pass (2026-04-21) against the 1980 Votrax SC-01 datasheet surfaces one big omission in our current parameter surface: the master clock is not a user control in pyvotrax, but Votrax documents it as the primary sound-design knob.

- **Variable master clock** — Datasheet §"NOTE" under Signal Description: *"Varying clock frequency varies voice and sound effects. As clock frequency decreases, audio frequency decreases and phoneme timing lengthens."* Figure 6 shows a 50 kΩ pot + 6.8 kΩ + 120 pF on MCRC for manual control; Figure 7 shows DAC current injection for software control. Clock formula is ≈ 1.25 / RC. In our emulator `MASTER_CLOCK` is a hardcoded constant in `pyvotrax/constants.py`; making it runtime-variable (and propagating to SCLOCK/CCLOCK derivations + filter coefficients) is the single highest-impact music-production feature we can ship.
- **Per-phoneme amplitude** — Output voltage swing is 0.18–0.26 × Vp peak-peak, with the AH phoneme as reference. Current code normalizes output; exposing a per-phoneme gain envelope that matches authentic phoneme-to-phoneme amplitude differences is a low-effort authenticity improvement.
- **DC bias in output** — Datasheet confirms AO is DC-biased ("applied to an audio output device"). Our optional `dc_block` flag is directly justified by this and should default to on for WAV export (currently off).
- **Mid-phoneme inflection changes** — Datasheet: I1/I2 "Instantaneously sets pitch level of voiced phonemes." The chip supports changing inflection in the middle of a phoneme. Our `phone_commit(code, inflection)` only latches at phoneme boundary — an additional method to write inflection mid-phoneme would unlock pitch-bend-style effects.
- **Duration is master-clock-dependent** — The 47–250 ms phoneme durations in the ROM table are at nominal 720 kHz clock. Slower clock → longer phonemes (confirmed in datasheet). If master clock becomes variable, duration-mul and pitch-shift become linked knobs by design.

### Intentionally-preserved MAME die bugs

MAME's `votrax.cpp` includes comments flagging two hardware bugs that are **preserved in emulation** rather than corrected:

```
// Formant update.  Die bug there: fc should be updated, not va.
interpolate(m_cur_fc,  m_rom_fc);
// Non-formant update. Same bug there, va should be updated, not fc.
interpolate(m_cur_va, m_rom_va);
```

The fc/va variables are swapped between the "formant update" and "non-formant update" blocks. The real SC-01A chip has this bug; MAME reproduces it faithfully; our C++ port inherits it. A `strict_authenticity` vs `bugs_fixed` toggle would let music users choose between the real chip's voice and a mathematically-correct reference voice.

---

## Part 4: The Standalone C/C++ Front End

The goal for this project is a Votrax speech synthesizer that is C/C++ all the
way down: text in, samples out, no Python, no dictionary file, no emulator to
host it. That is what an NVDA add-on wants — a small native library it can
drive over a handful of C entry points.

That path is now open end to end. `ttv_translate` turns English into phone
codes, `vx_write` and `vx_render` turn phone codes into samples, and the whole
thing builds into a single library with no data files beside it. What follows
is what the tables are, where they came from, and how the matcher over them
works.

### Where the tables came from

A third-party NVDA driver (`sc01.dll`, 2026) wrapped MAME's chip simulation and
shipped a text-to-phoneme engine alongside it. The chip half was of no use to
us — it *is* the emulator we are trying not to depend on — but the front-end
half was pure data, and that data is now source in this repository. The
binaries themselves have been removed; nothing at build or run time refers to
them.

Four tables came out of it, plus the mask ROM dumps covered in Part 1:

| Table | Size | What it is |
|---|---:|---|
| `NRL_RULES` | 355 rules, 27 groups | English spelling → ARPABET |
| `ARPABET_TO_SC01` | 81 entries | ARPABET → SC-01 phones, context sensitive |
| `NRL_EXCEPTIONS` | 17 pairs | Respellings the rules get wrong |
| `ABBREVIATIONS` | 3 pairs | Dr, Mr, Mrs |
| `CARDINALS` / `ORDINALS` | 28 each | Number names |
| `ASCII_NAMES` | 128 | Spoken name of every ASCII code |

Prosody and the phone scheduler are ours, not recovered — see below.

The rules are the Naval Research Laboratory letter-to-sound set (Elovitz et
al., NRL Report 7948, 1976) in the arrangement popularised by John A. Wasser's
public-domain `english.c` (1985) — a US Government work in a public-domain
arrangement, which is why it can simply live here as source.

### Stage 1 — spelling to ARPABET

The matcher walks the text left to right. At each position it takes the rule
group for the character under the cursor, tries each rule in order, and the
first one whose `match` is present and whose left and right contexts both hold
wins; its output is appended and the cursor advances by the length of `match`.
Rules are ordered most-specific-first within a group, ending in an
unconditional single-letter fallback, so matching never fails.

Context characters:

| Char | Means |
|---|---|
| `#` | one or more vowels |
| `:` | zero or more consonants |
| `^` | one consonant |
| `+` | a front vowel (E, I or Y) |
| `%` | a suffix — E, ER, ES, ED, ING or ELY (right context only) |
| `.` | a voiced consonant (B D V G J L M N R W Z) |
| space | a word boundary |

Anything else matches itself. Left contexts are written in reverse reading
order, so the character nearest the cursor comes last: `{" :", "ANY", "", …}`
reads as "a word boundary, then any run of consonants, immediately before ANY".

### Stage 2 — ARPABET to SC-01 phones

This is the more interesting table, and the part with no equivalent anywhere
else in the tree. It is not a dictionary lookup; it is tuned to what the SC-01
can actually say.

The chip has no diphthongs, so the map spells them out as glides:

| ARPABET | SC-01 phones |
|---|---|
| `EY` | `A AY` |
| `OW` | `O1 U1` |
| `AY` | `AH E1` |
| `AW` | `AH O1` |
| `OY` | `O1 E1` |
| `UW` | `IU U` |

Affricates become stop plus fricative — `CH` → `T CH`, `JH` → `D J` — which is
what they are articulatorily, and which the chip's own interpolator then smears
into something convincing.

Most of the table, though, is coarticulation around liquids, and this is the
craft in it. A vowel before /l/ gets an offglide; a vowel after /l/ gets an
`UH3` onglide; a vowel before /r/ gets an `I3` or `EH3` offglide:

| Context | Mapping |
|---|---|
| `AE` plain | `AE` |
| `AE` after `L` | `UH3 AE` |
| `AE` before `R` | `AE1 EH3` |
| `AE` after `L`, before `R` | `UH3 AE EH3` |
| `AO` plain | `AW` |
| `AO` before `R` | `O` |
| `AO` before `ER` | `AW O2` |
| `L` after `IY`/`EY`/`AY`/`OY` | `I3 L` |
| `L` after `AE`/`AO`/`OW` | `UH3 L` |
| `ER` after `IY` | `I3 ER` |
| `ER` after `R` | `UH3 R` |

The SC-01 interpolates between phoneme targets rather than jumping, so an extra
transitional phone is the only lever available for shaping a transition. The
table is a catalogue of where that lever is worth pulling.

Compare this with `pyvotrax/tts.py`, which maps ARPABET to Votrax through a
flat dictionary with no context at all, and picks a single phone where this
table picks two. Several of the flat mappings also disagree outright — `IY` →
`I1` against `E` here, `UH` → `UH` against `OO`, `EY` → `E1` against `A AY`.
Porting this table is the single largest available improvement to
intelligibility, and it costs nothing at runtime.

### Numbers, symbols and spelling

`CARDINALS` and `ORDINALS` are indexed 0–19, then 20, 30 … 90 at indices 20–27,
so any number under a thousand is two lookups. `ASCII_NAMES` is indexed by
character code and covers all 128, including the control codes (`0` → "null",
`13` → "carriage return") — which is exactly what a screen reader needs when it
is asked to read a character rather than a word. The letter names for spell
mode are the same table at `'A'`…`'Z'`.

`ABBREVIATIONS` and `NRL_EXCEPTIONS` both run before the rules, as whole-word
replacements on space-padded text. The abbreviations go first, so " DR " becomes
" DOCTOR " and is then read by the ordinary rules. (" PHD " was recognised too,
but had no expansion — it was spelled out letter by letter, which is still the
right answer.) The exceptions are short and entirely about one vowel: `SEARCH` →
`SURCH`, `HEARD` → `HURD`, `BEAR` → `BAIR`, `PEAR` → `PAIR` and their
inflections. The `EA` rules cannot see far enough ahead to get these right, so
the front end cheats, respelling the word into something the rules do handle.

### The pieces, and where they live

| File | What it is |
|---|---|
| `csrc/rom_data.h` | Both mask ROMs, decoded |
| `csrc/filters.h`, `csrc/votrax_core.h` | The chip: switched-capacitor filters and the analog signal path |
| `csrc/ttv_tables.h` | The letter-to-sound and phone-mapping tables |
| `csrc/ttv.h` | The matcher that walks them — text to phone codes |
| `csrc/votrax_capi.h`, `csrc/votrax_capi.cpp` | The flat `extern "C"` layer |
| `csrc/bindings.cpp` | The pybind11 layer, for the workbench and the tests |
| `nvda-addon/` | The add-on: a ctypes shim and a packaging script |

Everything but the two `.cpp` files is header-only, so the synthesizer can be
dropped into another project by adding `csrc/` to the include path.

### The C API

`csrc/votrax_capi.h` is the surface an add-on binds to. It is deliberately
small and deliberately flat — no C++ ABI, no structs passed by value, nothing
ctypes cannot express:

```c
vx_chip *vx_create(int mask, unsigned int clock_hz);
void     vx_destroy(vx_chip *);
void     vx_reset(vx_chip *);
void     vx_set_clock(vx_chip *, unsigned int hz);
double   vx_sample_rate(vx_chip *);
void     vx_set_mask(vx_chip *, int mask);
void     vx_inflection(vx_chip *, unsigned char level);
int      vx_get_inflection(vx_chip *);
void     vx_set_speed(vx_chip *, double speed);
int      vx_phone_samples(vx_chip *, unsigned char phone);
void     vx_write(vx_chip *, unsigned char phone);
int      vx_ready(vx_chip *);
int      vx_speak(vx_chip *, const unsigned char *phones, int count);
int      vx_pending(vx_chip *);
void     vx_cancel(vx_chip *);
int      vx_render(vx_chip *, int16_t *buffer, int count);
int      ttv_translate(const char *text, unsigned char *out, int capacity);
int      ttv_translate_flat(const char *text, unsigned char *out, int capacity);
int      ttv_spell(const char *text, unsigned char *out, int capacity);
const char *vx_phone_name(int code);
int      vx_phone_by_name(const char *name);
```

Note what is *not* in it: there is no ROM argument, and no path to anything.
Both masks are compiled in, and so is the whole English front end, so the
add-on ships one library file and nothing beside it.

The driving loop is the shape any Votrax front end has always used — write a
phone when the chip asks for one, render audio in between:

```c
unsigned char phones[512];
int n = ttv_translate("Hello.", phones, 512);
int i = 0;
while (i < n) {
    if (vx_ready(chip)) vx_write(chip, phones[i++]);
    vx_render(chip, block, 512);      /* push block to the audio device */
}
```

Building it, on Windows with MSVC:

```
cl /std:c++17 /EHsc /O2 /LD /Icsrc /Fe:votraxsc01.dll csrc\votrax_capi.cpp
```

and on anything else:

```
c++ -std=c++17 -O2 -shared -fPIC -Icsrc -o libvotraxsc01.so csrc/votrax_capi.cpp
```

### Known rough edges in the front end

Two gaps between the recovered tables had to be closed in `ttv.h`, and they are
worth knowing about because they are the kind of thing that goes unnoticed:

- **`h` and `j`.** The letter-to-sound rules write /h/ and /dʒ/ as lone
  lower-case letters, but the phone map keys them `HH` and `JH`. A tokenizer
  that does not know this drops both sounds silently — "hello" comes out
  without its H. `canonical_symbol` promotes a lone consonant to its
  two-letter form when that is the key that exists.
- **`NG` versus `NX`.** The rules emit `NG` for the velar nasal; the map spells
  the same sound `NX`. Without the alias, "young" ends in /n/ + /g/.

One more is a judgement call rather than a gap. The rules spell a word boundary
as a space, but real text ends words with commas and full stops. Matching a
literal space loses every rule that needs to see the end of a word, so "hello,"
would not get the same vowel as "hello". `ttv.h` treats any non-alphanumeric
character as a boundary, which is what the rules plainly intend.

Beyond those, the front end has the limits the 1976 ruleset has always had: it
reads single letters as words (so "S C" comes out as sounds, not letter names —
callers wanting letter names should use `ttv_spell`), and it handles numbers up
to three digits as words and longer runs digit by digit.

### Prosody

The SC-01's pitch input is two bits: four levels, nothing between. Measured on
our own core at the datasheet clock they are 78, 89, 104 and 125 Hz — a range
of about a fifth, in four steps, and that is the entire pitch budget.

It is not much. But flat delivery is the single most fatiguing thing about
early synthesizers, and four levels are enough for the one contour that matters
most: **declination**. English statements drift downward across a clause and
drop at the end; questions do the opposite and rise. Listeners use that fall to
hear where a sentence ends, which is exactly why flat output makes running text
feel like it never stops.

So the front end assigns a level per phone from its position within its own
sentence, and the sentence's final punctuation chooses the shape:

| Sentence | Contour | Levels |
|---|---|---|
| Statement | falls to the floor | 2 → 1 → 0 |
| Question | climbs over the last third | 1 → 2 → 3 |
| Exclamation | starts high, keeps some energy | 3 → 2 → 1 |

Contours are per sentence, not per utterance — `ttv_translate` splits on `.`,
`?` and `!` followed by whitespace, so a paragraph gets a fresh contour for
each sentence rather than one long slide. "One. Two? Three!" comes out as
`2,1,0` then `1,1,1,3,3,3` then `3,2,1`.

Pauses are skipped when measuring position — they are silent, so counting them
would let a comma-heavy sentence spend its contour on gaps — but they inherit
the level of whatever preceded them, so the stream never has a discontinuity.

**Phones and levels travel together in one byte**: the phone in bits 0-5, the
level in bits 6-7. A byte with its top bits clear is therefore just a plain
phone code, which is what lets packed and unpacked streams be used
interchangeably wherever a phone is accepted.

The caller's own pitch setting shifts the whole contour rather than replacing
it. `vx_inflection` sets a base, and the scheduler computes
`base + packed − VX_NEUTRAL_INFLECTION`, clamped to 0–3. Leaving the base at
its default of 1 reproduces the contour exactly; moving it transposes. The
consequence of having only four levels is that at base 0 or 3 the contour
clips flat against the end of the range — the hardware's limit, not a bug, and
the reason a caller mapping a 0–100 pitch setting onto this should favour the
middle.

What is deliberately *not* modelled: stress and syllables. The letter-to-sound
rules do not emit either, so there is nothing to build a word-level accent
model on. The clause-level contour is the part four levels can genuinely
express, and attempting more without stress information would be guesswork.

### The NVDA add-on

`nvda-addon/` builds the whole thing into an add-on that is **169 KB**:
one Python file and one DLL per architecture. The previous Python-based add-on
bundled numpy, scipy and a pronunciation dictionary to do the same job, at
roughly 40 MB.

```
nvda-addon/
  manifest.ini
  addon/synthDrivers/votraxsc01.py     the shim
  package.py                           builds both DLLs, zips the add-on
```

NVDA 2026 is 64-bit only, so `votraxsc01-x64.dll` is the one that gets loaded.
`votraxsc01-x86.dll` ships alongside it for NVDA 2025 and earlier, which ran as
32-bit processes; the driver chooses between them from the bitness of the
process it finds itself in rather than from any NVDA version check, so one
add-on serves both without knowing which it is running under.

`package.py` builds both and refuses to produce an add-on without the x64
library, warning rather than failing if the x86 one is missing — a missing
library for the host architecture would otherwise show up only as NVDA
silently not listing the synthesizer.

The driver itself owns no synthesis logic at all. It turns speech sequences
into work items on a queue; one thread owns the chip and turns those into
audio, so nothing needs locking. Cancellation is an epoch counter: `cancel()`
bumps it and stale items are dropped when the thread reaches them. Control
items carry no epoch on purpose — a settings change must survive a cancel.

NVDA's settings map onto the chip like this:

| NVDA setting | Chip control |
|---|---|
| Voice | mask revision — SC-01 or SC-01-A |
| Rate | `vx_set_speed`, exponential over half to double |
| Rate + "authentic rate" | `vx_set_clock` instead — pitch rises with speed |
| Pitch | inflection base, quantised to the four real levels |

The pitch setting is quantised on the way in *and* snapped on the way out, so
the number NVDA announces always corresponds to a level you can actually hear.
Without the snap the settings ring moves through numbers that produce no
audible change, which reads as the control being broken.

Because the driver needs nothing from NVDA but a handful of modules, it is
tested here rather than inside a screen reader: `tests/test_nvda_driver.py`
stubs those modules, loads the real driver against the real DLL, and exercises
speaking, indexing, spelling, cancellation and every settings mapping. Those
tests skip cleanly when the DLL has not been built.

### What is still to do

The synthesizer is complete: text in, samples out, in C++, with a working
add-on around it. What is left is refinement rather than construction.

- **Stress.** The NRL rules do not emit it, so neither prosody nor duration can
  use it. A stress model would need either a dictionary — which is the
  dependency this design exists to avoid — or a rules-based guess at syllable
  structure.
- **Volume.** The chip has no volume control; the driver would have to scale
  samples on the way out, which is the one piece of DSP that would sit outside
  the emulation.
- **The older driver.** `nvda-addon/` still holds the Python-based synth and
  its bundled wheels. It is superseded but not removed.

### Rate control without a time-stretcher

Pitch and tempo come from two different places on this chip, and that is the
whole trick.

Pitch is the glottal oscillator, clocked straight off the master clock. Tempo
is the duration counter, a separate thing that does nothing but decide when the
chip asks for the next phone. The obvious way to speak faster — raise the
master clock — moves both, which is the 1980 hardware's single knob and turns
the voice into a chipmunk. Faithful, but not what a screen-reader user wants at
400 words a minute.

The alternative is to leave the clock alone and simply not wait. Write the next
phone before the chip has finished the current one, and the current one is cut
short. The formant interpolators carry on from wherever they had reached toward
the new targets, so the result still sounds like speech rather than like a
chopped-up recording — and the glottal oscillator never learns that anything
happened.

**The natural length of a phone is exact**, which is what makes this practical:

```
samples = 32 × (4 × duration + 1)
```

A phone runs 16 ticks, a tick is `4 × duration + 1` chip updates, and a chip
update is two samples. Verified against all 64 phones on both masks. The
`duration` field lives in word0, which is identical between the two mask
revisions, so this does not vary with the voice.

That closed form matters. The reference NVDA driver measured every phone
empirically at startup — reset the chip, write the phone, render until it asks
for the next, 64 times per voice — because it had no way to know the number in
advance. We do, so there is no measurement pass at all.

To speak at `speed`, hold each phone for `natural / speed` samples and commit
the next one there. That is the entire implementation; the scheduler in
`csrc/votrax_capi.cpp` is about fifteen lines. Speeds below 1.0 extend rather
than truncate — the chip finishes the phone and sustains it, holding the final
steady state until the next one is due.

Measured over "The quick brown fox jumps over the lazy dog", 45 phones:

| Speed | Duration | Tempo | Pitch |
|---|---|---|---|
| 0.5 | 8.39 s | 0.500× | 84 Hz |
| 1.0 | 4.20 s | 1.000× | 84 Hz |
| 1.5 | 2.80 s | 1.500× | 84 Hz |
| 2.0 | 2.10 s | 2.000× | 84 Hz |
| 3.0 | 1.40 s | 3.000× | 84 Hz |
| clock ×2 | 2.10 s | 2.000× | **168 Hz** |

The timings are exact to within a sample. The last row is the contrast: the
same doubling of tempo, bought by doubling the clock, and the pitch doubles
with it.

The two are independent and compose. Set a clock for the voice you want — the
datasheet explicitly endorses varying it, and it is a legitimate sound-design
control — then set a speed for the tempo you want, and the tempo will not
disturb the voice you chose.

The honest limitation: truncation is not free at extreme speeds. The formant
interpolators need time to reach their targets, so past roughly 3× the targets
are never approached and the speech thins out into something mushier. That is
inherent to the technique rather than a defect in it, and it is what fast
Votrax speech has always sounded like. Nothing about it is a resampling
artefact — the audio is the chip's own output at every speed.

Through the C API:

```c
vx_set_speed(chip, 2.0);            /* tempo only, pitch untouched */
vx_speak(chip, phones, n);
while (vx_pending(chip))
    vx_render(chip, block, 512);    /* the scheduler does the rest */
```

The chip's 2-bit inflection input is the other control worth surfacing — four
levels, nothing between, which maps onto a screen reader's pitch setting only
if the setting is quantised to those four steps.

## References

### Primary sources

- **MAME `votrax.cpp`** — Olivier Galibert (BSD-3-Clause). The definitive reference implementation. https://github.com/mamedev/mame/blob/master/src/devices/sound/votrax.cpp
- **Die schematics** — https://og.kervella.org/sc01a — Annotated die photographs and circuit analysis
- **Votrax SC-01 Phoneme Speech Synthesizer Data Sheet (1980)** — Federal Screw Works / Votrax. Full 22-pin pinout, timing table, phoneme chart with durations in ms, electrical characteristics, variable-clock application notes. http://www.bitsavers.org/pdf/federalScrewWorks/Votrax_SC-01_Phoneme_Speech_Synthesizer_Data_Sheet_1980.pdf
- **US Patent 3,836,717** — R.T. Gagnon, "Speech synthesizer," filed 1971, issued 1974. Original claims for the formant-cascade architecture; F1 0–1000 Hz, F2 500–3000 Hz, F3 1000–4000 Hz, fixed nasal 4000 Hz, 70 ms smoothing LPF, 30–150 ms phoneme duration. https://patents.google.com/patent/US3836717A/en
- **US Patent 3,908,085** — R.T. Gagnon, 1975. Improvement patent covering the binary-weighted duty-cycle serialization and 16-parameter 4-bit ROM encoding with phoneme-timer ramp (slope = one of the 16 parameters). https://patents.google.com/patent/US3908085A/en
- **US Patent 4,433,210** — "Switched-capacitor filter" — Covers the SC-01A's filter technology
- **SC-01 internal mask ROM dumps** — the 512-byte on-die phoneme ROMs, **held at `reference/roms/`**. SC-01 (1980 mask): CRC32 `528d1c57`, SHA1 `268b5884dce04e49e2376df3e2dc82e852b708c1`. SC-01-A: CRC32 `fc416227`, SHA1 `1d6da90b1807a01b5e186ef08476119a862b5e6d`. Both are compiled into `src/votrax_rom.c` and transcribed again into `py_emu/rom.py`; `tools/verify_rom.py` checks those two, and Galibert's `rom.cc`, against the dumps row by row, and `tests/test_masks.py::TestAgainstTheDumps` runs that check plus a parameter-level decode on every test run. See `reference/roms/README.md` for the row format — the ROM is content-addressed, and rows are not stored in phone order.

### Letter-to-sound references

- **Elovitz, Johnson, McHugh & Shore, "Automatic Translation of English Text to Phonetics by Means of Letter-to-Sound Rules"** — NRL Report 7948, Naval Research Laboratory, 1976. The ruleset in `csrc/ttv_tables.h`. A US Government work.
- **John A. Wasser, `english.c`** (1985) — the public-domain C arrangement of the NRL rules that circulated with Votrax-era hardware, and the shape the tables here follow (four fields per rule, one group per letter, most-specific-first).

### TTS pipeline references

- **CMU Pronouncing Dictionary** — Carnegie Mellon University. Used by `pyvotrax/tts.py`.
- **NRL Report 7948** — Elovitz, Johnson, McHugh, Shore (1976), "Automatic Translation of English Text to Phonetics by Means of Letter-to-Sound Rules." 329 public-domain rules; candidate replacement for CMU dict to shrink the NVDA addon bundle. https://apps.dtic.mil/sti/pdfs/ADA021929.pdf — extracted rule set: https://github.com/Lord-Nightmare/NRL_TextToPhonemes

### Secondary references (for deeper enhancement work)

- **Gagnon (1978)**, "VOTRAX Real Time Hardware for Phoneme Synthesis of Speech," Proc. ICASSP 1978, pp. 175–178. **Held locally**, at `C:\GIT\speech synthesis\papers\analyzed\votrax-real-time-hardware-for-phoneme-synthesis-of-speech.pdf` — an earlier draft of this file recorded it as paywalled on IEEE Xplore, which is no longer the situation we are in. Four pages, and the primary source for the pre-chip architecture: the block diagram (input buffer → 16-parameter ROM lookup → articulation generators → 10-pole, 2-zero vocal tract), the vocal-tract diagram (F1–F5 cascade with the nasal notch feeding F1, vocal oscillator injected at F1, fricative noise at F2), the full 64-phoneme command set with Votrax's own gloss for each, and the design rationale for treating stops as a gate on the tract transfer function with articulation continuing through the closure.
- **Klatt (1980)**, "Software for a cascade/parallel formant synthesizer," JASA 67(3):971–995. Foundation for independent formant-bandwidth control.
- **Klatt & Klatt (1990)**, "Analysis, synthesis, and perception of voice quality variations among female and male talkers," JASA 87(2):820–857. Source of KLGLOTT88 (used in `py_emu` enhanced mode).
