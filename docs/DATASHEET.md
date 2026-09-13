# The SC-01 data sheet, checked and implemented

`reference/SC-01_Data_Sheet_v1_text.pdf` is Votrax's own data sheet for the chip
(Troy, MI; copyright 1980), a scan with a text layer. Pages 1–9 are the sheet
proper, and page 10 is a later sheet of SC-01A application circuits. Everything
else in this repository comes from MAME's reading of the die, from Galibert's
gate-level simulator, or from the dumped mask ROMs. The data sheet is the one
source independent of all three, and it was added on 2026-09-13.

This document goes through it one statement at a time. For each it records what
the sheet says, what was done with it, how that was checked, and where the check
now lives. The tests are in `tests/test_datasheet.py` and, for the command-line
options, `tests/test_cli.py`.

A second data sheet is used for the amplifier in Figure 8: Texas Instruments'
LM386 (SNAS545D). It is not redistributed; §10 lists what was taken from it.

| # | Data sheet | Status |
|---|---|---|
| 1 | Table 1: codes, symbols, durations | **Confirmed**, every phone; a small unexplained offset recorded |
| 2 | Table 2: production categories | **Confirmed** in the ROM for six of seven categories; nasals are not marked there |
| 3 | Table 1 notes: T before CH, D before J | **Confirmed**, the front end never emits either bare |
| 4 | Table 3: duration range 47–250 ms | **Confirmed** |
| 5 | I1/I2 set pitch "instantaneously" | **Implemented**: it was taking effect at the next phone |
| 6 | Clock ≈ 1.25 / RC | **Implemented** as `vx_clock_from_rc` |
| 7 | Figure 6/8 voice knob | **Implemented** as `vx_clock_from_knob` |
| 8 | Figures 6/7: vary the clock while speaking | **Implemented**: a clock change used to reset the chip |
| 9 | Lower clock: lower pitch, longer phonemes | **Confirmed** |
| 10 | Figure 8 audio circuit, with TI's LM386 data sheet | **Implemented** as an optional output stage with its volume control; the circuit clips the 1980 mask's open vowels |
| 11 | External timing (ignoring A/R) | Already present: that is what `vx_set_speed` does |
| — | Pin-out, electrical limits, bus timing | Not applicable to an emulation |

---

## 1. Table 1 — the phoneme chart

**The sheet:** 64 phonemes, in code order, each with a symbol, a duration in ms
and an example word. The durations are "dependent on master clock frequency,
720 kHz".

**Symbols.** `vx_phone_name(0..63)` returns the chart's symbols in the chart's
order. That includes 0x05 = A2 and 0x06 = A1, which `src/ttv_tables.c` already
defended against a transposed table on ROM evidence. The chart settles it.

**Durations.** A phone's length here is exact and closed-form,
`samples = 32 × (4 × duration + 1)`, with `duration` the ROM's 7-bit field
(`docs/tech-overview.md`, Part 4). At 720 kHz that is 40 000 samples a second.
All 64 lengths were computed this way and set beside the chart (Appendix A):

- every phone is within 7% and 6 ms of the chart
- the mean is +3.1%, the worst long is +6.2% (I3 and D, 58.4 ms against 55),
  and the worst short is −2.4% (AH and AW, 244 ms against 250)
- the ROM uses only 12 distinct duration values, and the chart gives each of
  them one consistent millisecond figure, so it was computed from the same
  field

**Known:** the chart and the ROM agree on which phones are long and which are
short, and on the order of every family (EH3 < EH2 < EH1 < EH, and so on).

**Not explained:** the offset. It is not a scale factor, since the short phones
run long here and the longest run short, so no single clock error accounts for
it. The chart may be rounded from measurements of a real part, or computed by a
formula slightly different from the die's. Nothing available decides which. The
die-derived formula is kept, because it is what the chip's counters actually
do, and the difference is below what a listener could attribute to one phone.

**Test:** `TestPhonemeChart`.

## 2. Table 2 — production categories

**The sheet:** the 64 symbols grouped as voiced, "voiced" fricative, "voiced"
stop, fricative stop, fricative, nasal and no sound.

**Checked** against the decoded ROM of both masks, grouping each phone's voice
amplitude `va`, fricative amplitude `fa` and closure flag by its Table 2
category. Six categories fall out exactly:

| Category | ROM signature | Members |
|---|---|---|
| voiced | `va ≥ 8`, `fa = 0`, no closure | 40 vowels, liquids and glides |
| voiced fricative | `va = 1`, `fa > 0`, no closure | Z ZH J V THV |
| voiced stop | `va > 0`, `fa = 0`, closure | B D G |
| fricative stop | `va = 0`, `fa > 0`, closure | T DT K P |
| fricative | `va = 0`, `fa > 0`, no closure | S SH CH TH F H |
| no sound | `va = 0`, `fa = 0` | PA0 PA1 STOP |

**Nasals** (M, N, NG) have the voiced signature and nothing else marking them.
Their F1 is low (1–2), but so is that of E, Y and Y1. The SC-01 has no nasal
anti-resonator, so a nasal is a voiced sound with low formants, and the category
is the report's phonetics rather than a difference in the silicon. Recorded, not
forced.

The "voiced" fricatives and stops carry quotation marks in the sheet itself. The
ROM shows why: a voiced fricative has only `va = 1` of voicing under its noise,
and a voiced stop is a closure over full voicing.

**Test:** `TestCategoriesInTheRom`, on both masks.

## 3. T before CH, D before J

**The sheet**, under Table 1: "'T' must precede 'CH' to produce CH sound. 'D'
must precede 'J' to produce J sound."

**Checked:** the second-stage map (`TTV_ARPABET`, Geczy's transcription of the
NRL rules) has exactly two rows that emit the CH and J phones, `CH → T CH` and
`JH → D J`. All 81 rows were scanned, and no other row emits either phone. Words
full of affricates were then run through both `ttv_translate` and `ttv_spell`,
and every CH was found preceded by T and every J by D.

Phoneme input (the GUI's phoneme mode, `votrax-say --phones`) still accepts a
bare CH, as the real chip did. The rule is about what sounds right, not what is
allowed.

**Test:** `TestAffricateRule`.

## 4. Table 3 — duration range

**The sheet:** "Time of Phoneme Duration", 47 minimum, 107 typical, 250 maximum,
at 720 kHz.

**Checked:** ours run from 48.8 ms (the duration-15 phones) to 244 ms (AH, AW).
The same offset as §1 applies. **Test:**
`TestPhonemeChart.test_duration_range_matches_table_3`.

## 5. The inflection pins — implemented

**The sheet**, Signal Description: "Inflection Level Setting (I1, I2):
Instantaneously sets pitch level of voiced phonemes." MAME agrees:
`inflection_w` updates the level at once, and the pitch counter compares
against it on every tick (`reference/mame_votrax.cpp`).

**Before:** `vx_inflection` stored the level, and the core only picked it up
when the next phone was committed. A change in the middle of AH (244 ms) was not
heard until AH ended.

**Now:** `vx_inflection` writes the core's level immediately
(`vx_core_set_inflection`), so the pitch counter uses it at its next comparison.
A phone the scheduler committed keeps its sentence-contour step: the chip
remembers the step as `contour`, so a base change mid-phone lands at base +
step, as it would at the next commit.

**Checked** by measuring pitch by autocorrelation on AH:

| | Measured | Documented in `votrax.h` |
|---|---:|---:|
| level 0 | 78.1 Hz | 78 |
| level 1 | 89.3 Hz | 89 |
| level 2 | 104.4 Hz | 104 |
| level 3 | 125.0 Hz | 125 |
| level 0, switched to 3 mid-phone | 78.1 → 125.0 Hz, measured from 10 ms after the change | — |
| contour step +1, base raised 1 → 2 mid-phone | 104.4 → 125.0 Hz | — |

**Cost:** none to existing output. Every golden entry sets the level before
speaking, so none moved. `datasheet.inflection_mid_phone` was added to the
fingerprint. **Test:** `TestInflectionPins`.

## 6. The clock relation — implemented

**The sheet**, Electrical Characteristics, note \*\*\*: "Frequency of Master
Clock ≃ 1.25 / RC", with typical values of 6.5 k and 300 pF, and 720 kHz named
as the typical clock.

**Implemented:** `vx_clock_from_rc(ohms, farads)` returns `1.25 / (R × C)` in Hz,
or 0 for a non-positive part.

**Recorded, not corrected:** the sheet's own typical parts give 641 kHz, 11%
short of its typical clock. The sheet writes ≃, and a real RC oscillator depends
on the CMOS input thresholds as well. So the relation is offered for the
figures that use it, and the default clock stays 720 kHz, not 641.

**Test:** `TestMasterClock.test_rc_relation` and the test after it;
`votrax-say --rc OHMS,FARADS`.

## 7. The voice knob — implemented

**The sheet:** Figure 6, "Variable Voice by Potentiometer Control". The page-6
drawing has no values. The page-10 redrawing, and Figure 8 on the same page, give
**6.8 k fixed + 50 k audio-taper pot, 120 pF**, with MCRC and MCX tied together.

**Implemented:** `vx_clock_from_knob(position)`, with position 0–1 along the
track:

| Position | Resistance | Clock |
|---:|---:|---:|
| 0 (pot shorted) | 6.8 k | 1.53 MHz |
| 0.5 | 11.3 k | 0.92 MHz |
| 0.6 | 14.3 k | 0.73 MHz (728 kHz) — the data sheet clock |
| 1 (full track) | 56.8 k | 183 kHz |

**Inferred, and said so in `votrax.h`:** the taper law. The sheet says "audio
taper" and gives no curve. The conventional audio law is used, 10% of the track
at half rotation (`fraction = (10^(2p) − 1) / 99`). With it, the data sheet's
720 kHz falls at 0.6 of the travel, which is plausible for a knob whose middle
is meant to be the normal voice.

**Test:** `TestMasterClock.test_knob_ends_and_middle` and
`test_audio_taper_is_gentle_at_first`; `votrax-say --knob P`.

## 8. Changing the clock while speaking — implemented

**The sheet:** Figures 6 and 7 move the clock with a knob, or with a DAC
injecting current into the RC node. The note beside them says varying the clock
"varies voice and sound effects". A chip whose clock is being turned does not
stop and reset.

**Before:** `vx_set_clock` re-initialised the core: filters and interpolators
cleared, the phone replaced by STOP, and the level zeroed until the next phone.
Its comment said callers change the clock between utterances.

**Now:** `vx_core_set_clock` sets the three clock figures and rebuilds the
filters from the current register values without clearing any state. The phone,
its progress, the queue and the hold counter all carry on. The hold counter
needs no rescaling, because a phone is a fixed number of samples at any clock.

**Found while doing it:** most of the filter coefficients do not depend on the
clock at all. Every standard and injection section's corner frequency scales
with the chip clock, and the sample rate scales with it too, so the corner over
the sample rate — all a digital filter sees — is fixed.
`test_voiced_sound_is_the_same_samples_at_every_clock` renders AH L O1 M at
360 kHz, 720 kHz and 1.44 MHz, and the three sample streams are bit-identical.
A clock change moves only the rate those samples play at.

The exception is the noise shaper, whose coefficient has the clock in a
numerator (`src/votrax_filters.c`, following MAME). S AH differs by up to 11 000
counts between 360 kHz and 720 kHz. So fricatives change colour with the clock
as well as speed, and the test pins that too.

**What it means for callers:** a clock change is still a sample-rate change, so a
fixed-rate audio device must be reopened, as the NVDA driver already does for
"authentic rate". Smooth DAC-style clock sweeps into a fixed-rate device would
need a resampler, which is **not implemented**.

**Cost:** one golden entry, `live_switch`. Its third hash, taken after a clock
change on a chip that has been talking, now includes the carried-over state.
Before this change that hash was byte-identical to the first, taken at a
different clock on a fresh chip — the invariance above, visible in the old
fingerprint. **Test:** `TestMasterClock.test_a_live_change_keeps_the_chip_talking`.

## 9. Lower clock, lower pitch, longer phonemes

**The sheet:** "As clock frequency decreases, audio frequency decreases and
phoneme timing lengthens."

**Confirmed:** a phone's sample count does not change with the clock, and the
sample rate is the clock over 18, so halving the clock doubles every duration and
halves every frequency. **Test:** `test_lower_clock_lengthens_phonemes`.

## 10. The Figure 8 audio circuit, with the LM386 — implemented

**The sheet:** Figure 8 on page 10, "Typical Application", is the only complete
audio path Votrax drew. It was read at 300 dpi (`pdftoppm -r 300`, cropped):

```
AO (22) ─┤1 µF├─ R1 4.7 k ─A─┬─ C1 0.05 µF ─ ground
                              └─ 10 k audio-taper volume pot
                                    A ─ Rt ─ wiper W ─ Rb ─ ground,  Rt + Rb = 10 k
                              W ─┬─ 1.2 k ─ ground
                                 ├─ C2 0.05 µF ─ pin 2 (ground)
                                 └─ LM386N-1 pin 3, pins 1 and 8 open
                                       out ─┤330 µF├─ 8 Ω speaker          Vp = 12 V
AF (21) ─ 4.7 k ─ ground
```

The 1 µF line appears to cross the 6.8 k clock resistor's lead. The scan draws a
hop there, not a dot, so the two are not connected, and the clock network is fed
from Vp as Figure 6 shows. The note beside the figure gives the supply: "5 V
supply to be raised last and lowered before or at same time as Vp (12 V)".

### The amplifier: TI's LM386 data sheet

The first version of this stage treated the LM386 as an ideal gain block and
modelled the volume at full only. Texas Instruments' LM386 data sheet (SNAS545D,
revised August 2023) supplies what that left out. It is TI's copyrighted
document, so it is **not in the repository**; TI publishes it at
[ti.com/lit/ds/symlink/lm386.pdf](https://www.ti.com/lit/ds/symlink/lm386.pdf).
What was taken from it:

| LM386 data sheet | Value | Used for |
|---|---|---|
| 6.5 Electrical Characteristics, R<sub>IN</sub> | 50 k | loads the wiper, in parallel with the 1.2 k |
| 6.5, A<sub>V</sub>, pins 1 and 8 open; §9.2.1.2.1 | 20 (26 dB), from the internal 1.35 k | amplifier gain |
| 6.5, BW, pins 1 and 8 open; Figure 6-4 | 300 kHz | checked: −0.02 dB at 20 kHz, so not modelled |
| Figure 6-3, Output Voltage vs Supply Voltage | 6.6 V peak to peak, 12 V, 8 Ω | the clipping level, ±3.3 V |
| Figure 6-6, THD vs Power Out | about 0.2%, then vertical | clipping is hard, so a hard limit |
| 6.3 Recommended Operating Conditions | 4–12 V (LM386N-1) | Figure 8's 12 V is the top of the range |

Figure 6-3 was read from a 200 dpi render of page 6. **Read off a graph, so
approximate:** the 8 Ω curve at 12 V sits just above the 6 V grid line and has
flattened. Figure 6-7 looks as if it disagrees, putting the 10% distortion point
at 12 V near 0.36 W, which would be 4.8 V peak to peak into 8 Ω. But Figure 6-7
does not state its load, and 0.36 W is 3.6 V peak to peak into 4 Ω — exactly
Figure 6-3's 4 Ω curve at 12 V. So Figure 6-7 is read as a 4 Ω plot, and 6.6 V
into 8 Ω is kept. **Inferred.**

### Levels: the SC-01 sheet again

The clipping level only means something once chip units are volts. The SC-01
sheet's audio-output section, and its table on page 8, give AO's swing on AH as
0.18–0.26 × Vp peak to peak. Its middle, 0.22 × 12 V = 2.64 V, is set against AH
on the 1980 mask at the neutral level, 1.383 chip units peak to peak, which
gives **1.909 V per chip unit**. Both masks share the scale, because the mask
changes the ROM and not the output circuit.

**Inferred:** that the range is centred on its middle. Measured, AH on the SC-01
mask runs 1.31–1.55 units from level 0 to level 3, a spread of 18%. The sheet's
range is 44% wide, more than pitch alone explains, so it probably also covers
part-to-part variation.

### The model

`vx_set_output(chip, VX_OUTPUT_FIGURE8)` and `vx_set_output_volume(chip, P)`,
with P from 0 to 1 along the pot's track. The taper law is the conventional audio
law used for the voice knob (§7):

| Stage | From | Model |
|---|---|---|
| input coupling | 1 µF into R1 + Rt + (Rb ∥ 1.2 k ∥ 50 k) | high-pass, 27.7 Hz at full volume |
| network | R1, C1, Rt, Rb ∥ 1.2 k ∥ 50 k, C2 | second order, below |
| LM386 | gain 20, ±3.3 V | multiply, hard limit |
| output coupling | 330 µF into 8 Ω | high-pass, 60.3 Hz |

The network by nodal analysis at A and W, with G1 = 1/R1 and
Gw = 1/Rb + 1/1.2 k + 1/50 k:

```
H(s) = G1 / ( G1 + Gw + Rt·G1·Gw  +  s·((1 + Rt·Gw)·C1 + (1 + Rt·G1)·C2)  +  s²·Rt·C1·C2 )
```

At full volume Rt = 0, C1 and C2 act as one 0.1 µF, and H is a single pole at
**1855 Hz** with a gain of **0.1825**. It is made bilinear, prewarped at that
corner. The corners are fixed in hertz and are rebuilt when the clock or the
volume moves: the loudspeaker circuit did not change when the voice knob turned.

The output is scaled back by 1.909 V per unit, the gain of 20 and the
full-volume gain of 0.1825. An unclipped signal at full volume therefore keeps
the chip's own level, and turning the volume down makes it quieter, as the pot
did.

**Deliberately not modelled:**

- the LM386's 0.2% distortion below clipping
- its 300 kHz bandwidth
- the speaker's own response
- the 0.1 µF from the LM386 output to ground
- AO's source resistance (the sheet gives 90 Ω at most, beside 4.7 k)

### The finding: Votrax's reference circuit clips

At full volume, clipping at ±3.3 V out corresponds to ±0.474 chip units of
low-frequency drive, 0.947 units peak to peak. AH on the 1980 mask is 1.37 units
peak to peak, and on the SC-01A mask 0.82. Measured through the model on a
sustained AH:

| Volume | SC-01 AH, p-p | SC-01A AH, p-p | SC-01 / SC-01A |
|---:|---:|---:|---:|
| chip, no stage | 1.367 | 0.820 | 1.667 |
| 1.00 | 0.987 | 0.758 | 1.302 |
| 0.98 | | | 1.419 |
| 0.96 | | | 1.533 |
| 0.94 | | | 1.629 |
| 0.92 | | | 1.667 |
| 0.90 | 0.656 | 0.393 | 1.667 |
| 0.80 | 0.462 | 0.277 | 1.667 |
| 0.50 | 0.210 | 0.126 | 1.667 |

Wherever the LM386 is linear, the ratio between the masks is the chip's own
1.667. It falls only where the louder mask clips. So on the board Votrax
published, the 1980 part's open vowels flatten against the amplifier from about
0.93 of the volume track upward. The SC-01A, the chip Figure 8 actually draws,
fits at full volume.

That is consistent with the revision's quieter open vowels
(`docs/tech-overview.md`, Part 1), but consistent is all it is. Nothing says
that was the reason for the change.

### Checked

On "She sells sea shells by the sea shore." (SC-01 mask) and on sustained AH:

| Measure | Result | Test bound |
|---|---:|---|
| (6–12 kHz ÷ 300–1000 Hz), Figure 8 at volume 0.8, over the same for the chip | 0.0075 | below 0.1 |
| the same at a 1.44 MHz clock | 0.0058 | below 0.1 |
| the same at volume 1.0, where it clips | 0.26 | none — clipping makes harmonics |
| below 20 Hz, as a share of the Figure 8 total | 0.000018 | below 0.01 |
| SC-01A AH peak at full volume | 0.455 | below the 0.474 limit |
| SC-01 AH peak at full volume | 0.495 | 0.95–1.15 × the limit |
| SC-01 / SC-01A AH at volume 0.8 | 1.667 | the chip's ratio, ±3% |
| volume 0 | silence | exactly zero |

A cancel clears the stage's state.

**Test:** `TestFigure8OutputStage`; `votrax-say --output-stage figure8 --volume P`.

## 11. External timing

**The sheet:** "If external phoneme timing is desired, phoneme requests can be
ignored. However, best speech is realized with internal timing."

**Already present:** `vx_set_speed` is this. The scheduler writes the next phone
after `natural / speed` samples, whether or not the chip has asked for it.
`vx_write` with `vx_ready` is the internal-timing handshake. Nothing changed.

## Not applicable

The following matter to a circuit board, not to an emulation, and are not
modelled:

- the pin-out and package dimensions
- the input logic levels
- the setup, hold and strobe times of Table 3
- power, and the AO line's 90 Ω protection resistor

The AO swing (0.18–0.26 × Vp peak-to-peak on AH) is a statement about absolute
volts, which a sample stream does not have.

---

## Appendix A — Table 1 against the phone lengths

`ROM duration` is the 7-bit field, identical on both masks. `Samples` is
`32 × (4 × duration + 1)`, from `vx_phone_samples`. `Ours` is that at 40 000
samples a second.

| Code | Symbol | Table 2 category | ROM duration | Samples | Ours (ms) | Sheet (ms) | Difference |
|---:|---|---|---:|---:|---:|---:|---:|
| 00 | EH3 | voiced | 19 | 2464 | 61.6 | 59 | +4.4% |
| 01 | EH2 | voiced | 23 | 2976 | 74.4 | 71 | +4.8% |
| 02 | EH1 | voiced | 38 | 4896 | 122.4 | 121 | +1.2% |
| 03 | PA0 | no sound | 15 | 1952 | 48.8 | 47 | +3.8% |
| 04 | DT | fricative stop | 15 | 1952 | 48.8 | 47 | +3.8% |
| 05 | A2 | voiced | 23 | 2976 | 74.4 | 71 | +4.8% |
| 06 | A1 | voiced | 33 | 4256 | 106.4 | 103 | +3.3% |
| 07 | ZH | voiced fricative | 29 | 3744 | 93.6 | 90 | +4.0% |
| 08 | AH2 | voiced | 23 | 2976 | 74.4 | 71 | +4.8% |
| 09 | I3 | voiced | 18 | 2336 | 58.4 | 55 | +6.2% |
| 0A | I2 | voiced | 26 | 3360 | 84.0 | 80 | +5.0% |
| 0B | I1 | voiced | 38 | 4896 | 122.4 | 121 | +1.2% |
| 0C | M | nasal | 33 | 4256 | 106.4 | 103 | +3.3% |
| 0D | N | nasal | 26 | 3360 | 84.0 | 80 | +5.0% |
| 0E | B | voiced stop | 23 | 2976 | 74.4 | 71 | +4.8% |
| 0F | V | voiced fricative | 23 | 2976 | 74.4 | 71 | +4.8% |
| 10 | CH | fricative | 23 | 2976 | 74.4 | 71 | +4.8% |
| 11 | SH | fricative | 38 | 4896 | 122.4 | 121 | +1.2% |
| 12 | Z | voiced fricative | 23 | 2976 | 74.4 | 71 | +4.8% |
| 13 | AW1 | voiced | 46 | 5920 | 148.0 | 146 | +1.4% |
| 14 | NG | nasal | 38 | 4896 | 122.4 | 121 | +1.2% |
| 15 | AH1 | voiced | 46 | 5920 | 148.0 | 146 | +1.4% |
| 16 | OO1 | voiced | 33 | 4256 | 106.4 | 103 | +3.3% |
| 17 | OO | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 18 | L | voiced | 33 | 4256 | 106.4 | 103 | +3.3% |
| 19 | K | fricative stop | 26 | 3360 | 84.0 | 80 | +5.0% |
| 1A | J | voiced fricative | 15 | 1952 | 48.8 | 47 | +3.8% |
| 1B | H | fricative | 23 | 2976 | 74.4 | 71 | +4.8% |
| 1C | G | voiced stop | 23 | 2976 | 74.4 | 71 | +4.8% |
| 1D | F | fricative | 33 | 4256 | 106.4 | 103 | +3.3% |
| 1E | D | voiced stop | 18 | 2336 | 58.4 | 55 | +6.2% |
| 1F | S | fricative | 29 | 3744 | 93.6 | 90 | +4.0% |
| 20 | A | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 21 | AY | voiced | 21 | 2720 | 68.0 | 65 | +4.6% |
| 22 | Y1 | voiced | 26 | 3360 | 84.0 | 80 | +5.0% |
| 23 | UH3 | voiced | 15 | 1952 | 48.8 | 47 | +3.8% |
| 24 | AH | voiced | 76 | 9760 | 244.0 | 250 | -2.4% |
| 25 | P | fricative stop | 33 | 4256 | 106.4 | 103 | +3.3% |
| 26 | O | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 27 | I | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 28 | U | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 29 | Y | voiced | 33 | 4256 | 106.4 | 103 | +3.3% |
| 2A | T | fricative stop | 23 | 2976 | 74.4 | 71 | +4.8% |
| 2B | R | voiced | 29 | 3744 | 93.6 | 90 | +4.0% |
| 2C | E | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 2D | W | voiced | 26 | 3360 | 84.0 | 80 | +5.0% |
| 2E | AE | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 2F | AE1 | voiced | 33 | 4256 | 106.4 | 103 | +3.3% |
| 30 | AW2 | voiced | 29 | 3744 | 93.6 | 90 | +4.0% |
| 31 | UH2 | voiced | 23 | 2976 | 74.4 | 71 | +4.8% |
| 32 | UH1 | voiced | 33 | 4256 | 106.4 | 103 | +3.3% |
| 33 | UH | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 34 | O2 | voiced | 26 | 3360 | 84.0 | 80 | +5.0% |
| 35 | O1 | voiced | 38 | 4896 | 122.4 | 121 | +1.2% |
| 36 | IU | voiced | 19 | 2464 | 61.6 | 59 | +4.4% |
| 37 | U1 | voiced | 29 | 3744 | 93.6 | 90 | +4.0% |
| 38 | THV | voiced fricative | 26 | 3360 | 84.0 | 80 | +5.0% |
| 39 | TH | fricative | 23 | 2976 | 74.4 | 71 | +4.8% |
| 3A | ER | voiced | 46 | 5920 | 148.0 | 146 | +1.4% |
| 3B | EH | voiced | 58 | 7456 | 186.4 | 185 | +0.8% |
| 3C | E1 | voiced | 38 | 4896 | 122.4 | 121 | +1.2% |
| 3D | AW | voiced | 76 | 9760 | 244.0 | 250 | -2.4% |
| 3E | PA1 | no sound | 58 | 7456 | 186.4 | 185 | +0.8% |
| 3F | STOP | no sound | 15 | 1952 | 48.8 | 47 | +3.8% |
