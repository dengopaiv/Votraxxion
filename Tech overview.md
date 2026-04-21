# Votrax SC-01A Technical Overview

A comprehensive reference covering the original hardware, this emulation's architecture, and areas for improvement.

---

## Part 1: The Original Votrax SC-01A

### History

The Votrax SC-01A traces back to **Richard T. Gagnon**, who developed a formant-based speech synthesis approach at **Federal Screw Works** (a Michigan automotive parts company that diversified into electronics). The speech synthesis division became **Votrax** in 1974. The **SC-01** shipped in 1980, followed by the improved **SC-01A** in 1981. It was one of the first single-chip speech synthesizers that didn't require external ROM — all 64 phonemes were encoded in an on-die 512-byte ROM, making it cheap and easy to integrate.

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

## References

### Primary sources

- **MAME `votrax.cpp`** — Olivier Galibert (BSD-3-Clause). The definitive reference implementation. https://github.com/mamedev/mame/blob/master/src/devices/sound/votrax.cpp
- **Die schematics** — https://og.kervella.org/sc01a — Annotated die photographs and circuit analysis
- **Votrax SC-01 Phoneme Speech Synthesizer Data Sheet (1980)** — Federal Screw Works / Votrax. Full 22-pin pinout, timing table, phoneme chart with durations in ms, electrical characteristics, variable-clock application notes. http://www.bitsavers.org/pdf/federalScrewWorks/Votrax_SC-01_Phoneme_Speech_Synthesizer_Data_Sheet_1980.pdf
- **US Patent 3,836,717** — R.T. Gagnon, "Speech synthesizer," filed 1971, issued 1974. Original claims for the formant-cascade architecture; F1 0–1000 Hz, F2 500–3000 Hz, F3 1000–4000 Hz, fixed nasal 4000 Hz, 70 ms smoothing LPF, 30–150 ms phoneme duration. https://patents.google.com/patent/US3836717A/en
- **US Patent 3,908,085** — R.T. Gagnon, 1975. Improvement patent covering the binary-weighted duty-cycle serialization and 16-parameter 4-bit ROM encoding with phoneme-timer ramp (slope = one of the 16 parameters). https://patents.google.com/patent/US3908085A/en
- **US Patent 4,433,210** — "Switched-capacitor filter" — Covers the SC-01A's filter technology

### TTS pipeline references

- **CMU Pronouncing Dictionary** — Carnegie Mellon University. Used by `pyvotrax/tts.py`.
- **NRL Report 7948** — Elovitz, Johnson, McHugh, Shore (1976), "Automatic Translation of English Text to Phonetics by Means of Letter-to-Sound Rules." 329 public-domain rules; candidate replacement for CMU dict to shrink the NVDA addon bundle. https://apps.dtic.mil/sti/pdfs/ADA021929.pdf — extracted rule set: https://github.com/Lord-Nightmare/NRL_TextToPhonemes

### Secondary references (for deeper enhancement work)

- **Gagnon (1978)**, "VOTRAX Real Time Hardware for Phoneme Synthesis of Speech," Proc. ICASSP 1978, pp. 175–178. IEEE Xplore, paywalled.
- **Klatt (1980)**, "Software for a cascade/parallel formant synthesizer," JASA 67(3):971–995. Foundation for independent formant-bandwidth control.
- **Klatt & Klatt (1990)**, "Analysis, synthesis, and perception of voice quality variations among female and male talkers," JASA 87(2):820–857. Source of KLGLOTT88 (used in `py_emu` enhanced mode).
