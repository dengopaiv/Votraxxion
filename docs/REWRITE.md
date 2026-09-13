# The C rewrite

## Why

The synthesizer works and is already native: `csrc/votrax_capi.cpp` compiles to
a 195 KB DLL that the NVDA add-on drives over ctypes, with both mask ROMs and
the English front end inside it. The rewrite is not about capability. It is
about what the library *is*.

Today it is C++ -- header-only classes, `std::string`, `std::vector`,
`std::deque`, templates, `new`. That costs three things that matter for a
screen reader:

1. **One toolchain.** MSVC-only in practice. Plain C builds under MinGW, clang,
   or gcc for a Linux screen reader, with no C++ ABI questions.
2. **Allocation on the speech path.** `std::deque` allocates when phones are
   queued and `ttv::translate` builds `std::string`s and `std::vector`s per
   utterance. It is fast enough today, but a synthesizer feeding an audio
   callback should not be taking the allocator lock at all.
3. **Size.** Much of the library is template instantiation and exception
   tables for a program that never throws.

A fourth reason was in the first draft of this plan and was wrong, so it is
recorded here rather than quietly dropped: *"the DLL links MSVCP140.dll, and on
a machine without the Visual C++ redistributable the add-on does not load."*
That is true of a `/MD` build and not of this one. `cl` defaults to `/MT`, which
is what `nvda-addon/package.py` gets, and `dumpbin /dependents` on the C++ DLL
built that way shows `KERNEL32.dll` and nothing else. The C++ build was never at
risk of the failure the argument described. The measurements are under step 8.

C also matches what the thing is. The chip is a state machine and seven IIR
filters over a fixed set of arrays. None of that wanted a class.

## The rule

**Byte-identical output, verified, or it is a bug.** A DSP rewrite that
"sounds the same" is a rewrite nobody can trust. `tools/goldens.py` captures a
fingerprint of everything the library can be asked to do — every one of the 64
phones on both masks, whole utterances at five speeds and three clocks, the
front end on text covering numbers and abbreviations and all three sentence
contours, cancel and reset, the ready handshake — and diffs two libraries
entry by entry. The C++ baseline is captured before the first line of C is
written. The C library must match it exactly.

The one licensed exception: `ttv_*` output is byte-identical too, but if the C
front end ever needs to differ, that is a decision to write down here, not a
diff to accept quietly.

## The shape

`src/`, C11, one translation unit per concern, each with a header:

| File | From | Notes |
|---|---|---|
| `votrax_rom.{h,c}` | `csrc/rom_data.h` | Both mask tables, decoded once at load. `enum class` → `enum`, `std::array` → array. Mechanical. |
| `votrax_filters.{h,c}` | `csrc/filters.h` | Already C but for `<cmath>` and two templates. The templates become fixed-size loops. Mechanical. |
| `votrax_core.{h,c}` | `csrc/votrax_core.h` | The class becomes a struct plus functions taking `vx_core *`. All state is already plain scalars and arrays. Mechanical, but this is the file where a typo is a different voice, so it goes one method at a time against the goldens. |
| `ttv_tables.{h,c}` | `csrc/ttv_tables.h` | `const char *` tables. Mechanical. |
| `ttv.{h,c}` | `csrc/ttv.h` | **The real work.** `std::string`/`std::vector` throughout. Becomes fixed-capacity buffers with explicit bounds, which is also where the current code's silent truncation behaviour has to be reproduced exactly. |
| `votrax.{h,c}` | `csrc/votrax_capi.{h,cpp}` | The public API, unchanged in signature. `std::deque` becomes a ring buffer. |

`csrc/` keeps only `bindings.cpp`, which includes the C headers through
`extern "C"` and stops being a second copy of anything.

## The order

Bottom-up, so that each step is verifiable before the next depends on it.

- [x] **0. Baseline.** Build the C++ DLL, capture goldens. *(done — 42 entries; 60/64 phone hashes unique, exactly 12 phones differ between masks, matching the 12 documented ROM deltas.)*
- [x] **1. Cleanup.** Remove build output and the superseded Python add-on, move reference material out of the root. *(done — 560 MB to 2.1 MB.)*
- [x] **2. Documentation.** [ARCHITECTURE.md](ARCHITECTURE.md), this file.
- [x] **3. `votrax_rom.c`** — tables and decode, checked against the C++ decode for all 64 phones on both masks.
- [x] **4. `votrax_filters.c`** — coefficient builders, checked coefficient-by-coefficient against the C++ ones at three clocks.
- [x] **5. `votrax_core.c`** — the chip. Checked by rendering every phone and diffing samples.
- [x] **6. `ttv_tables.c` + `ttv.c`** — the front end. Checked by translating a word list far larger than the goldens' corpus and diffing phone streams.
- [x] **7. `votrax.c`** — API and scheduler. Full golden comparison passes.
- [x] **8. Wire it up.** `nvda-addon/package.py` builds C; `bindings.cpp` includes the C headers; `csrc/` loses its duplicates.
- [x] **9. Regression.** Goldens identical, 500 existing tests green, a test that keeps the goldens honest from here on.

## Step log

Kept short on purpose: what changed, and what proved it.

**0. Baseline** — Built `csrc/votrax_capi.cpp` with MSVC 19 (VS 18 Community):
195 KB. Captured 42 golden entries in 1.75 s.

Two things surfaced while making the harness discriminate:

- A freshly reset chip is *silent for its first ~1200 samples* while the
  formant interpolators ramp from zero. The first version of the live-switch
  check hashed 1024 samples and so compared silence to silence, passing
  regardless. Windows are 4096 samples now.
- Changing the master clock does not change the sample values, except through
  the noise path. `sclock` and `cclock` scale together and the bilinear
  transform normalizes them out; only `build_noise_shaper_filter` has `cclock`
  in a numerator, so only the fricatives move. The corpus keeps all three
  clocks because that asymmetry is exactly the kind of thing a rewrite breaks.

**1. Cleanup** — 560 MB to 2.1 MB. The old Python add-on carried 299 MB of
vendored numpy/scipy/CMUdict wheels to do what the native one does in 169 KB.
Galibert's gate-level simulator and MAME's `votrax.cpp` moved to `reference/`
with a README saying what each is evidence for, rather than being deleted:
`rom.cc` is where the ROM tables came from and that claim needs a source.

**2. Documentation** — This file and `ARCHITECTURE.md`. The architecture doc
records the two things that are easy to get wrong (cancellation leaving a tail,
and the two incompatible meanings of "faster") because both were learned the
expensive way and neither is visible in the code.

**3. `votrax_rom.c`** -- Tables and bit extraction. The raw words were lifted
out of the C++ header by script rather than retyped, which removes
transcription error as a possibility. Checked by decoding all 64 phones on both
masks in C and in C++ and diffing all twelve fields: **128 rows x 12 fields
identical**. Twelve rows differ between the masks, which is an independent
check that the delta table was applied rather than merely copied.

**4. `votrax_filters.c`** -- The four builders and the two templates. The
templates (`apply_filter<NA,NB>`, `shift_hist<N>`) became functions taking
lengths; the call sites pass constants, so the compiler unrolls them the same.
Checked by building every coefficient set at three clocks across the full range
of each variable filter's control bits -- **19 261 coefficients, `memcmp` on
the raw doubles, zero mismatches**.

Writing the header forced an explanation of something the goldens had already
shown: most of these builders are clock-invariant, because `sclock` and
`cclock` scale together and the bilinear transform normalizes them out. The
exception is the noise shaper, whose `k1` has `cclock` in the numerator where
every other term has it in a denominator. That is MAME's, and it is the entire
reason a clock change is audible in fricatives and nowhere else.

**5. `votrax_core.c`** -- The chip. The class became a struct plus sixteen
functions, every field keeping its name. Checked by rendering all 64 phones on
both masks, plus the non-native articulation-rate and selective-closure
branches: **771 872 samples compared, zero mismatches, worst absolute
difference 0**.

Two things needed care rather than transcription:

- `rom` is a *copy* of the phoneme parameters, not a pointer into the table. It
  has to stay a copy: `phone_commit_override` writes arbitrary parameters into
  it, and a pointer would have made that a write to the ROM.
- The `(true || m_filt_fa)` in the noise LFSR is MAME's, and reads like a
  condition someone disabled and forgot to remove. It is preserved verbatim,
  short-circuit and all, with a comment. Deleting the dead half changes
  nothing, but it is the kind of tidying that leaves a later reader thinking
  the condition was always `filt_fa` and "restoring" a bug that was not there.

**6. `ttv_tables.c` + `ttv.c`** -- The front end, and the only part that was a
rewrite rather than a transcription. The tables were converted by script (they
were already valid C but for `<cstddef>`); the matcher was rewritten from
`std::string`/`std::vector` onto four fixed-capacity buffers held in one
context struct on the caller's stack, about 32 KB, so nothing allocates and two
threads can translate at once.

The classifiers now test ASCII ranges directly where the C++ called
`isalpha()`. In the C locale that is the same classification, and it closes a
latent hazard: under a locale where `isalpha()` is true for a byte above 127,
`NRL_RULES[1 + (c - 'A')]` indexes past the end of a 27-entry array.

Checked on a corpus of **7 195 lines** -- every distinct word in the
repository's own English prose, stems crossed with all six suffix forms the `%`
context matches, the abbreviation and respelling tables, numbers of every
shape, all three sentence terminators, every printable ASCII character alone,
and 2 000 random strings -- diffing all three entry points: **21 585
comparisons, zero mismatches**.

**7. `votrax.c`** -- The API and the scheduler. `std::deque<unsigned char>`
became a 1024-entry ring buffer. One deliberate behaviour change: `vx_speak`
now drops phones past the queue's capacity and reports how many are queued,
where the C++ would have grown until memory ran out. The header says so.

Naming collision worth noting: the exported C API is `ttv_translate`, and the
front end's own entry point had the same name, which C has no namespaces to
separate. The internal ones became `ttv_text_to_phones`, `_flat`, and
`ttv_spell_text`.

**Full golden comparison against the C++ baseline: all 42 entries identical.**

**8. Wire-up** -- `nvda-addon/package.py` compiles the six C sources with
`/std:c11 /W4` and no `/EHsc`; both architectures build with no warnings.
`csrc/` is now `bindings.cpp` alone, wrapping the C core for Python; the seven
C++ headers it duplicated are gone.

Two MSVC command-line quirks cost time and are now commented in the packager,
because both only appear when a path contains a space and this repository's
does: `/Fo` must name a directory when there are several sources, a directory
name ends in a backslash, and a trailing backslash inside quotes escapes the
closing quote and eats the rest of the command line. The fix is to compile from
inside `src/` and name the object directory relatively.

Measured, x64:

| Build | C++ | C |
|---|---|---|
| `/MT` (what the add-on ships) | 195 KB | 153 KB |
| `/MD` | 80.5 KB | 53.5 KB |
| `/MD` dependencies | MSVCP140, VCRUNTIME140, VCRUNTIME140_1, 5 x api-ms-win-crt | VCRUNTIME140, 4 x api-ms-win-crt |

The `.nvda-addon` went from 169 KB to 142 KB.

**9. Regression** -- The C++ baseline fingerprint is committed as
`tests/data/golden.json`, and `tests/test_c_core.py` captures the shipped DLL
and diffs it entry by entry, so a future change to the DSP has to be a
deliberate re-blessing rather than silent drift.

That test is only worth having if it fails when it should, so it was checked
against a mutation. The first attempt was useless and instructive: scaling the
output by 0.3500000000001 instead of 0.35 changed nothing the test could see,
because a relative change of 3e-13 on a sample near 16 000 disappears in the
rounding to int16. At 0.3501 -- 0.03%, still far below audibility -- it failed
on every audio entry: both phone tables, all thirty scheduler combinations,
inflection and the live switch. So the fingerprint's floor is the int16 grid,
which is the right floor: it catches everything that can reach a listener.

515 tests pass (500 existing, 15 new).

A second new file, `tests/test_c_api.py`, covers what the differential test
could not: the C++ had no limits, so truncation, queue overflow, clamping and
NULL handling are all new behaviour with nothing to diff against. 37 tests,
552 in total.

---

## The licensed exception, used once: pauses (2026-09-10)

The rule at the top of this file allows exactly one kind of divergence — a
deliberate change to `ttv_*` output, written down here rather than accepted as
a quiet diff. This is that entry.

**What was wrong.** Every punctuation mark in the language came out as the
short pause. `NRL_RULES_PUNCT` mapped `,` `.` `?` and `!` each to a space,
following NRL Report 7948, which was a letter-to-sound algorithm with no
opinions about timing. Stage two then turned that space into **PA0** — 1952
samples, 49 ms. Meanwhile the ARPABET map two tables down had rows saying `,`
becomes **PA1** (7456 samples, 186 ms) and `.` becomes **PA1 PA1** (372 ms).
Those rows were unreachable: nothing upstream could ever hand them a comma.

So a full stop was worth 49 ms — about a fifth of what the table intended and
a seventh of what the reference implementation produces. Sentences ran into
each other, and clause boundaries were audible only as a catch of breath. It
reads as words being dropped, which is how it was reported.

**How it was found.** Not by reading the tables. Tamas Geczy's
`votraxsc01-1.0.2` add-on ships MAME's device wrapped in a `vx_*` C API and its
own `ttv_translate`, so both front ends were driven over the same corpus and
their phone histograms diffed. Ours never emitted PA1 once; the reference
emitted it sixty times. Phone by phone on the same sentence:

```
'Hello, world.'   before   H EH L UH3 O1 U1 PA0 PA0 W UH3 ER L D PA0
                  after    H EH L UH3 O1 U1 PA1 PA0 W UH3 ER L D PA1 PA1
                  ref      H EH L UH3 O1 U1 PA1 PA0 W UH3 ER L D PA1 PA1
```

**The change.** `,` now emits `,`, and `.` `?` `!` all emit `.`, so stage two
sees them and the map rows that were always there finally fire. Pitch is
unaffected: the sentence contour reads the terminator from the original text
and never consulted this table.

The hyphen was left alone. There is a `-` → PA1 row in the map, and the
reference does pause on a standalone dash, but a hyphen is far more often
inside a word than standing as one, and 186 ms in the middle of "well-known"
is worse than the pause being missed.

**What it cost.** Four entries of `tests/data/golden.json` — `translate`,
`translate_flat`, `inflection` and all thirty `speak_*` combinations — every
one of them downstream of the front end. No DSP entry moved, which is the
check that this stayed where it was meant to. The fingerprint was re-blessed
against the corrected output, deliberately, which is what this section exists
to record.

---

## The licensed exception, used again: letter names and numbers (2026-09-13)

A second deliberate change to `ttv_*` output, in two parts. Both came out of
studying Tamas Geczy's `votraxsc01-nvda` repository (v1.0.2, `d520be9`, cloned
to `reference repositories/votraxsc01-nvda`), which had fixed three letter
names and reads numbers with Wasser's full `saynum.c`.

### Letter names

`TTV_ASCII_NAMES` is the name table as it circulated with Wasser's
`spellword.c`, and some of its entries are not names. Geczy's changelog named
O, U and S. Every letter was then spelled through our DLL and read back as
phone names (`ttv_spell` + `vx_phone_name`), which found four more:

| Letter | Was (ARPABET → phones) | Heard as | Now |
|---|---|---|---|
| H | `EYtCH` → A AY T T CH | a doubled T | `EYCH` → A AY T CH |
| O | `AA` → AH | "ah" | `OW` → O1 U1 |
| Q | `kw` → K W | "kw" | `kyUW` → K Y1 IU U |
| S | `EHz` → EH Z | "ezz", voiced | `EHs` → EH S |
| U | `AHw` → UH W | "uh-w" | `yUW` → Y1 IU U |
| W | `dAHblyUWw` → … U W | "double-you-w" | `dAHblyUW` |
| Y | `wAYIY` → W AH E1 E | "why-ee" | `wAY` → W AH E1 |

The H is a stage-two effect worth knowing: the ARPABET map already spells `CH`
as T CH, so writing `tCH` in the table doubles the stop. Upper- and lower-case
rows are fixed together. This matters more than its size suggests, because
NVDA spells a letter on every character-review keystroke.

### Numbers

Before, a run of up to three digits was read as a number and anything longer
digit by digit, on the theory that long runs are identifiers. That read
"1984" as "one nine eight four" and "1,000" as "one, zero zero zero" — the
comma a pause — and "3.14" as "three", a full-stop pause, "fourteen".

Now `ttv.c` has a number reader in the shape of Wasser's `saynum.c` (1985,
public domain): scales to billions, "and" before a remainder under a hundred,
1100–1999 in hundreds, and ordinals with TH on the final word. It departs from
Wasser in four places, each for a screen reader:

- **Identifiers stay digit by digit** — a leading zero ("007") or more than
  twelve digits. Twelve is the most `unsigned long long` arithmetic needs no
  special care for, and a longer run is not a quantity anyone says aloud.
- **Thousands separators join** — a run of one to three digits followed by
  `,ddd` groups is one number. "1,2,3" and "12,34" stay lists.
- **Any ST/ND/RD/TH suffix makes an ordinal.** Wasser matched the suffix
  against the last digit, so "11th" came out "eleven T H".
- **Decimals and versions.** Each `.digit` is "point" and the digits one by
  one, repeated, so "1.2.3" is "one point two point three" and the full stop
  never reaches the rules as a sentence pause.

Dollar amounts use the currency words that were already in the tables, unused:
"$4.20" is "four dollars and twenty cents", "$1" singular, "$1.5" "one point
five dollars". `TTV_DOLLARS` was `dAAlAArz` ("dollahrs") and is now `dAAlERz`,
matching `TTV_DOLLAR`.

### What it cost

`tests/data/golden.json` was re-blessed. Before the corpus was extended, the
diff was checked entry by entry: `translate` and `translate_flat` moved only
for "Dr. Smith has 3 cats and 1024 reasons." (1024 is now "one thousand and
twenty four"); `spell` moved for its three sentences, all of which contain O,
S, U, H or Y; and the thirty `speak_*` hashes, downstream of `translate`. No
`phones_mask*` entry moved — the DSP is untouched. The corpus then gained
"Pay $4.20 by the 21st: version 1.2.3, 1,000,000 or 007." so the new reader is
fingerprinted too.

`tests/test_ttv_words.py` pins every case above by phone name, so a regression
says what is heard rather than that a hash moved.

---

## Data sheet behaviour (2026-09-13)

Not a front-end change, but golden output moved, so it is recorded here. Two
API behaviours were brought into line with Votrax's 1980 data sheet
(`docs/DATASHEET.md`, §5 and §8):

- `vx_inflection` now changes the pitch of the phone already sounding, as the
  I1/I2 pins do, instead of waiting for the next phone. No golden entry
  moved, because every one sets the level before speaking.
- `vx_set_clock` now retunes a talking chip instead of re-initialising it.
  One entry moved: the third hash of `live_switch`, taken after a clock change
  on a chip that had been talking. Before, that hash equalled the first — a
  fresh chip at a different clock — which is the voiced-sound clock
  invariance the data sheet work then measured directly.

New, and fingerprinted under `datasheet`: `vx_clock_from_rc`,
`vx_clock_from_knob`, the mid-phone inflection change, and both output stages.
Every other entry is byte-identical. `tools/verify_gui.py` still
reports the GUI executable and the library agreeing on all 21 cases.
