# Roadmap — disassembling the Votrax engines

Written 2026-09-10, revised the same day. A plan and an inventory. Phase 1 is
built; the rest is not. Where it says "verified", that means verified in the
session that wrote this file, and the check is named so it can be re-run.

This repository is one thing and stays one thing: an emulator of the Votrax
SC-01, in C, with an NVDA add-on and a workbench on top. It passes 558 tests as
of 2026-09-10. What this document adds is *context* rather than scope — because
"Votrax" is not one engine, and knowing which of the three you are looking at
decides what belongs here and what does not. There are three unrelated
sound-generation architectures spread over fifteen years, plus the firmware
front ends that fed them. Each is its own synthesizer, with its own library and
its own add-on; the SC-01 is the one that lives here.

The alternative — a single library with a chip selector, an add-on offering
three generations of Votrax in one voice list — was the shape of the first
draft of this plan and is not what gets built. See Part 0.

---

## Part 0 — The question that had to be answered first

> *Do we have materials from two different Votrax models here, or one?*

**One model in this repository, in two mask revisions. A second model in a
sibling repository, emulated only at high level. A third architecture — the
one that came before both — documented on this machine but never implemented
anywhere.**

The confusion is worth spelling out, because "two ROMs" and "two models" look
the same from a distance and are not:

- `src/votrax_rom.c` carries **two** ROM tables. They are the **SC-01** (1980)
  and **SC-01-A** (1981) mask ROMs of the **same chip**. Two masks, one model.
  They differ in twelve of sixty-four rows, in one field, in one direction —
  verified below, not taken on faith.
- The genuinely different chip, the **SC-02 / SSI-263**, is not in this
  repository at all. Its material lives in `C:\GIT\speech synthesis\braillenspeak`,
  and what is there is a *high-level* emulation: SC-02 phoneme codes remapped
  onto SC-01 phonemes and voiced by this repository's DSP. No SC-02 silicon is
  being simulated by anyone, here or upstream.
- The **pre-chip Votrax** (VS-4 through VS-6, ML-1, VSK/VSL) is a third
  architecture with more formants and more parameters than either chip. We hold
  its designer's own conference paper. No emulation of it exists anywhere.

So: three engines, of which one is finished, one is faked, and one has never
been attempted.

### What that means for how this gets built

Three engines does not mean one library with three modes. The rule is **one
engine, one library, one add-on.** Somebody installing an NVDA add-on for the
SC-01 gets the SC-01 and nothing else — not a menu of chips, not the SC-02's
tables riding along unused, not a 1970s rack synthesizer they never asked for.
It reads the same in the other direction: an SC-02 add-on ships the SC-02.

The place they stack is the Workbench GUI, and only there. Choosing which
Votrax to render a sample with is exactly what a sound-design tool is for, so
the GUI offers whichever engines are present as a choice, while every
screen-reader deliverable stays one chip wide.

That settles the shape of everything below. Nothing merges into `src/`: the
SC-02 does not become a third mask revision, the pre-chip engine does not
become an "enhanced mode", and Phase 5 is not a switchboard inside one library.
Each engine is built in its own tree against its own reference material, and
produces its own library and its own add-on. What they share is method —
the same verification discipline, the same golden-diff tooling — not a binary.

---

## Part 1 — Inventory of what is on this machine

### 1.1 Silicon: the SC-01 mask ROMs (verified this session)

Two authentic 512-byte on-die mask ROM dumps, found inside a third-party NVDA
add-on at `C:\GIT\speech synthesis\synthesizers for NVDA\formant\votraxsc01-1.0.2.nvda-addon.zip`:

| File | Size | CRC32 | SHA-1 |
|---|---|---|---|
| `sc01.bin` | 512 | `528d1c57` | `268b5884dce04e49e2376df3e2dc82e852b708c1` |
| `sc01a.bin` | 512 | `fc416227` | `1d6da90b1807a01b5e186ef08476119a862b5e6d` |

Both match the `ROM_LOAD` lines in `reference/mame_votrax.cpp` exactly, which
makes them the standard MAME dumps rather than someone's re-derivation.

What was checked against them, byte for byte:

- **Row addressing.** The ROM is content-addressed, not index-addressed: each
  row is a little-endian 64-bit word whose bits 56–61 hold the phone number,
  and the physical order is *not* phone order (the first rows are phones 03,
  3E, 3F, 3D, 3C, 3B…). Bits 44–55 are zero in every row of both masks. This
  is why MAME scans all 64 rows looking for a match instead of indexing.
- **Our tables are the ROM.** Decoding both dumps by phone number reproduces
  `RAW_ROM_W0[64]` and `RAW_ROM_W1[64]` in `src/votrax_rom.c` with **zero**
  mismatches, and reproduces Galibert's independent die transcription in
  `reference/gate-sim/rom.cc` with zero mismatches. Three transcriptions, one
  answer.
- **The mask difference is exactly what the docs claim.** word0 is identical
  in all 64 rows. word1 differs in exactly twelve: 08, 13, 15, 23, 24, 2E, 2F,
  30, 31, 32, 33, 3D — precisely the twelve entries of `SC01_W1_DELTAS`, with
  the same values. All twelve are open vowels; all twelve are a `va` change.

The script that establishes this is worth keeping rather than leaving in a
scratch directory — see Phase 1.

### 1.2 Silicon: the SC-02 / SSI-263

Held at `C:\GIT\speech synthesis\braillenspeak`:

- `bnspeak/ssi263_codec.py` — all 64 SSI-263 mnemonics, transcribed from the
  Silicon Systems 1986 Data Book (SSI 263A datasheet p. 1-88, User's Guide
  phoneme chart p. 1-97), plus a hand-built SSI-263 → SC-01 identity map that
  **corrects** MAME's `ssi263hle.cpp` at 0x10, 0x33 and 0x3E (MAME maps real
  phonemes, including V, onto PA0 silence).
- `bnspeak/ssi263_emulator.py` — a register model: phoneme + duration,
  12-bit inflection, rate, articulation, amplitude, filter frequency, and the
  datasheet timing `frame = 4096·(16−rate)/2 µs`,
  `phoneme = frame·(4−duration)`. It drives this repository's SC-01 DSP as its
  backend.
- The primary documents are on disk: `C:\GIT\speech synthesis\papers\silicon systems\`
  holds the 1985 and 1986 Silicon Systems Data Books, complete, with the 1986
  book also split into 20 PDFs of 15 pages each.

This is a competent HLE and it is honest about being one. It is not the SC-02 —
and it is a *different synthesizer* from the one this repository builds, so it
stays in its own tree. See Phase 3.

### 1.3 Firmware: the engines that fed the chips

- `C:\GIT\speech synthesis\roms\UK_V2.01_4.04_ROM.BIN` — **not Votrax.** An
  earlier draft of this file made this ROM the centrepiece of Phase 2 on the
  strength of its shape alone: 64 KB, a prefix-compressed word → phoneme
  exception dictionary in the low regions, tables around 0xE000, code from
  0x9000 up, version stamps "2.01" and "4.04". That shape is what *any*
  rule-plus-dictionary text-to-speech front end looks like, and it is not
  evidence of whose. The ROM is most likely a **Dolphin Apollo 2** — a UK
  hardware synthesizer with no Votrax silicon in it, and therefore somebody
  else's project. Nothing inside the file settles it either way: it carries no
  copyright banner, no product name, no ASCII identity string at all.
  Recorded here only so the mistake is not made twice.
- Not yet on disk, but dumped and freely available: the **Votrax Type 'N Talk**
  firmware (MC6802 + 4K ROM + SC-01A) as MAME's `votraxtnt`, and the **Personal
  Speech System** (Z80 + two 8K EPROMs + SC-01). These are Votrax's *own*
  text-to-phoneme engines — the thing our `src/ttv.c` reimplements from the NRL
  report rather than from Votrax's code.

### 1.4 Binaries

- `sc01-x64.dll` / `sc01-x86.dll` from the 1.0.2 add-on. Their RTTI strings
  (`.?AVvotrax_sc01_device@@`, `.?AVvotrax_sc01a_device@@`, and a
  `timer_alloc<votrax_sc01_device>` lambda) show them to be MAME's C++ device
  lifted whole and wrapped in a `vx_*` C API, loading the two `.bin` dumps at
  runtime with a CRC check. Same lineage as our core, different construction.
- `pyvotrax/_votrax_core.*.pyd`, `nvda-addon/addon/synthDrivers/votraxsc01-*.dll`
  — our own builds, source in this repository. Not disassembly targets.

### 1.5 Papers

- `papers/analyzed/votrax-real-time-hardware-for-phoneme-synthesis-of-speech.pdf`
  — **Gagnon, ICASSP 1978.** `docs/tech-overview.md` lists this as "IEEE Xplore,
  paywalled". We have it. It is the primary description of the pre-chip engine
  and it is summarized in Part 2 below.
- `papers/silicon systems/` — the SSI-263A datasheet, via the data books.
- The SC-01 datasheet and Gagnon's patents 3,836,717 / 3,908,085 are already
  cited in `docs/tech-overview.md`.

---

## Part 2 — What the hardware actually did

Three generations, three different ways of making the sound. The repository has
only ever described the middle one.

### 2.1 The discrete era, 1970–1980 (VS-4 … VS-6, ML-1, VSK/VSL)

Richard T. Gagnon designed it in his basement in 1970 and licensed it to
Federal Screw Works, where it became the Vocal Division, then Votrax
International in 1980. VS-4 (1972) was three stacked boards, three formants,
passive C/L/C pi filters. VS-6 (1973–77) moved to active op-amp filters in a
card cage with EPROM parameter storage. ML-1 (1978) was rack-mount with 128
phonemes, digital amplitude and pitch and a FIFO. VSK/VSL (1978–80) shrank it
to a potted module — and the SC-01 is, in Votrax's own description, "very
similar to VSL except all on one chip".

From Gagnon's 1978 paper, the sound path of that generation:

```
8-bit phoneme command (6 bits phoneme, 2 bits inflection)
        -> input buffer / FIFO
        -> phoneme parameter ROM look-up: 16 parameters
             2 -> timing control (duration, transition rates)
            14 -> articulation generators
        -> articulation generators: 1st- and 2nd-order filtering and delay
           of the 14 parameters, rise-times sped up and slowed down with
           the speech rate control
        -> 6 parameters to the sources: 6 vocal + 4 fricative
        -> 10-pole, 2-zero vocal tract:
             F1 -> F2 -> F3 -> F4 -> F5 cascade,
             a NASAL NOTCH resonator feeding F1 (conjugate zeros,
             emulating the velum before and after nasal closures),
             vocal oscillator injected at F1,
             fricative noise injected at F2 (so F2..F5 resonate the
             fricatives as well as the voiced sound)
        -> audio out, 100 Hz - 5 kHz
```

Details that matter and that the SC-01 does *not* have:

- **Five formants and a nasal anti-resonator.** The SC-01 has F1, F2 (split
  voiced/noise), F3, a fixed F4 and a final FX lowpass — four resonators and no
  notch. `py_emu`'s experimental "nasal anti-resonator" mode is, unknowingly,
  a step back toward the VS-6.
- **A voltage-tunable vocal oscillator**, its frequency set by the *filtered*
  inflection signal plus a front-panel pitch control, "rich in all the
  harmonics from its fundamental up to over 5 kHz". Not the SC-01's 9-level
  stepped ladder.
- **Voiced fricatives done properly.** Fricative energy is modulated by the
  vocal oscillator whenever voicing and frication coincide, which is how /z/
  /v/ /zh/ /thv/ are produced rather than approximated.
- **Stops as a gate on the tract transfer function**, with articulation
  parameters continuing to move *during* the closure — Gagnon is explicit that
  this is what distinguishes B/D/G from P/T/K perceptually.
- 16 parameters per phoneme against the SC-01's 12; 128 phonemes on ML-1
  against 64.

### 2.2 The SC-01 / SC-01-A, 1980–1988

Already documented in `docs/tech-overview.md`, Part 1, at die level: 22-pin
CMOS, 720 kHz master clock, 512-byte content-addressed internal ROM, 64
phonemes, 2-bit inflection, a 9-level glottal ladder and a 15-bit noise LFSR
through seven switched-capacitor filters whose coefficients are rebuilt from
interpolating parameter registers at CCLOCK = master/36, output at
SCLOCK = master/18. Nothing in this roadmap changes that description; it has
been checked against the dumps and it holds.

The one thing to record here is the *lineage*: the SC-01 is a single-chip
reduction of VSL, which is a reduction of VS-6. Every difference between
Part 2.1 and Part 2.2 is something Votrax gave up to fit 22 pins — which makes
the 1978 paper a specification for what the chip is missing, and therefore the
best possible guide for any "enhanced" mode.

### 2.3 The SC-02 / SSI-263, 1983–1995

Same analog formant core, "somewhat more dynamic", fabricated by Silicon
Systems as SSI-263P and sold under both names (later SSI-263AP fixed bugs;
also Semtech SSI 263-2-2, Artic 263, TDK 78A263A). Different package, different
pinout, and — the part that matters — **a register interface instead of a
6-bit phoneme port**: duration, a 12-bit inflection (against the SC-01's four
levels), rate, articulation, amplitude and a filter-frequency byte.

Two facts frame all SC-02 work below:

1. **No low-level emulation of it exists anywhere.** MAME's `ssi263hle.cpp` is
   explicitly a placeholder, extracted from the Thayer's Quest driver, that
   remaps SC-02 phonemes onto an SC-01 and voices them there. That is exactly
   what `braillenspeak` does, and neither is the chip.
2. **The raw material for a real one exists.** visual6502 hosts a 17265 × 14313
   SSI-263P die shot, stitched from 203 images. That is the same starting point
   Galibert had for the SC-01 in 2016, and the SC-01 result is sitting in
   `reference/gate-sim/` as a worked example of where that road ends.

---

## Part 3 — Where the code stands

| Piece | State | Evidence |
|---|---|---|
| SC-01 DSP in C | Complete | `src/`, 558 tests pass 2026-09-10 |
| Both mask ROMs | Complete and verified against the dumps | Part 1.1 |
| English front end in C | Complete (NRL rules, not Votrax's own) | `src/ttv.c`, `src/ttv_tables.c` |
| NVDA add-on | Complete, 142 KB, both architectures | `nvda-addon/` |
| Native GUI | Complete, one ~200 KB exe, both architectures | `gui-native/`, `tools/verify_gui*.py` |
| Workbench GUI | Complete | `pyvotrax/`, `packaging/` |
| `py_emu` second transcription | Complete, plus experimental modes | `py_emu/` |
| SC-02 | HLE only, in a sibling repository | `braillenspeak/` |
| Pre-chip VS-6/ML-1 | Nothing | — |
| Votrax's own firmware engines | Nothing | — |

The C rewrite the earlier plan called for (`docs/REWRITE.md`) is **done**. The
"rewrite it in C" item in the goal is satisfied for the SC-01; it is the SC-02
and the firmware engines that are still un-rewritten, and they are un-*written*,
not un-rewritten — there is no C to port, only silicon and 6502/6802/Z80 code.

---

## Part 4 — The plan

Five phases. Each has a deliverable and a gate; the gate is the thing that
decides whether the phase actually worked. Phases 1 and 2 are cheap and settle
facts; 3 is the large one; 4 and 5 depend on 3.

### Phase 1 — Consolidate the evidence — **done, 2026-09-10**

The ROM dumps lived inside a third-party add-on zip, and the script that
verifies our tables against them lived in a temp directory. Both are here now.

1. **`reference/roms/`** — `sc01.bin` and `sc01a.bin`, with a README recording
   the hashes, the row format, the content-addressing, where the files came
   from and why nothing builds against them.
2. **`tools/verify_rom.py`** — decodes both dumps and cross-checks
   `src/votrax_rom.c`, `py_emu/rom.py` and `reference/gate-sim/rom.cc` row by
   row, then the twelve-row SC-01 delta, then that the delta touches `va` and
   nothing else. Stdlib only, reads the checked-in source text rather than
   anything built or imported, so it runs on a clean tree.
3. **`tests/test_masks.py::TestAgainstTheDumps`** — runs that cross-check on
   every test run, and additionally decodes both dumps and compares all twelve
   parameters of all 64 phones against what the built extension returns. The
   test file's own `EXPECTED_VA` constant is now checked against the silicon
   too, so a fourth transcription cannot drift in unnoticed.
4. **`docs/tech-overview.md`** — the Gagnon 1978 paper is recorded as held
   locally rather than paywalled, with what it contains; Part 1's History
   gained the pre-chip lineage; and the stale claim that the ROM data lives in
   `csrc/rom_data.h` is corrected to `src/votrax_rom.c` with the verification
   path spelled out.

**Gate — met.** 558 tests pass (552 before, plus six). The checker was
confirmed to fail as well as pass: a single flipped bit in `sc01a.bin` is
caught by the hash check, and with the expected hash patched to match the
corrupted file, by the row comparison — naming phone `0x3E` as the first
divergence.

Verbatim from `python tools/verify_rom.py`:

```
sc01a.bin  512 bytes  crc32 fc416227  sha1 1d6da90b1807a01b5e186ef08476119a862b5e6d
sc01.bin  512 bytes  crc32 528d1c57  sha1 268b5884dce04e49e2376df3e2dc82e852b708c1
content-addressed: physical row order starts 03, 3E, 3F, 3D, 3C, 3B ...
src/votrax_rom.c             64/64 rows match sc01a.bin
py_emu/rom.py                64/64 rows match sc01a.bin
reference/gate-sim/rom.cc    64/64 rows match sc01a.bin
src/votrax_rom.c             12/12 SC-01 deltas match sc01.bin
py_emu/rom.py                12/12 SC-01 deltas match sc01.bin
mask delta: 12 phones, va only, 08, 13, 15, 23, 24, 2E, 2F, 30, 31, 32, 33, 3D
```

### Phase 2 — Votrax's own front end

Our text-to-phoneme front end, `src/ttv.c`, is a reimplementation of NRL Report
7948 — the same rules Votrax used, but not Votrax's code. What the Type 'N Talk
actually shipped is a 4 K ROM that has been dumped for years and that nobody
here has read. That is the front end of *this* synthesizer, so it is in scope
where a foreign device's firmware is not.

Ghidra 12.0 at `C:\GIT\environment\ghidra\ghidraRun.bat`; `aRomAT` beside it for
first-pass ROM structure; `JAVA_HOME` is set, so Ghidra will start.

1. **Type 'N Talk** — fetch `votraxtnt` from MAME and disassemble the 4 K
   MC6802 ROM. Small, well-documented hardware (kevtris has the board), and it
   drives an SC-01A over exactly the interface `src/votrax.c` implements.
2. **The rules and the exception list.** Find the letter-to-sound engine and
   whatever dictionary sits beside it. The question worth answering is where it
   diverges from NRL 7948 — every divergence is a place `src/ttv.c` could be
   speaking differently from the hardware it claims to reproduce.
3. **The phone stream.** Recover how the firmware paces phones and drives the
   inflection pins, and diff that against our scheduler.
4. If the Personal Speech System dumps (Z80, two 8 K EPROMs) surface, the same
   treatment — it is the same front end with more room.

**Deliverable:** `docs/firmware/` — one document per ROM: identification,
memory map, the tables found, and the extracted data as JSON.
**Gate:** a diff of Votrax's rules against `src/ttv_tables.c`, with every
divergence either adopted or written down as a deliberate difference. "No
divergences found" is a legitimate outcome and still needs the diff to show it.

### Phase 3 — The SC-02, for real — *its own synthesizer, elsewhere*

This is the frontier: nobody has a low-level SC-02. It is also not this
repository's chip. The work belongs in `braillenspeak`'s tree (or a tree of its
own), producing an SC-02 library and an SC-02 add-on that ship without a byte
of SC-01 in them. What comes back here is method, and eventually the golden
tooling of Phase 5 — never a merge into `src/`.

Three tiers, in order of increasing cost, and it is legitimate to stop after
any one of them.

**Tier A — datasheet-exact register model (weeks).** Promote the existing
SSI-263 work to a first-class C engine in its own tree: the real register map,
the real timing equations, the corrected phoneme table, its own 64-entry
parameter table read from the datasheet's phoneme chart. It borrows the SC-01
filter bank as a *stand-in* while nothing better exists, and says so — this is
an SC-02 front end voiced on SC-01 silicon, which is what every existing
emulation is, done better. Borrowing a filter bank is not the same as living in
the SC-01's library, and the deliverable is still SC-02-only.

**Tier B — architectural model (months).** Reconstruct the SC-02's own filter
bank from the datasheet plus the SC-01 die as a structural prior: the 12-bit
inflection, the articulation control, the filter-frequency register, and
whatever the parameter interpolation turns out to be. Calibrate against real
SSI-263 recordings — Apple II Mockingboard and Phasor cards with genuine
SSI-263s still exist and are still sold.

**Tier C — die-level (open-ended).** The visual6502 die shot, Galibert's
method, `reference/gate-sim/` as the template. This is the only path to a
*correct* SC-02 and it is a research project, not a sprint. Start it only after
Tier A ships, and treat any progress as a bonus.

**Gate for Tier A:** every one of the 64 SSI-263 phonemes renders, the timing
equations reproduce the datasheet's frame and phoneme durations to the
microsecond, and A/B against real BNS audio is at least no worse than the
current HLE by ear.
**Gate for Tier B:** the model reproduces measured formant tracks from real
SSI-263 recordings, not just plausible-sounding output.

### Phase 4 — The pre-chip engine (VS-6 / ML-1) — *its own synthesizer too*

Nobody has emulated this either, and unlike the SC-02 there is no silicon to
photograph — but there is a full architectural description from the designer,
two patents, and our own SC-01 filter code as a starting point. It is a third
synthesizer with a third deliverable: a rack-Votrax library and, if anyone
wants to hear a screen reader speak in it, a rack-Votrax add-on. It is
emphatically *not* an "enhanced mode" bolted onto the SC-01, which is what it
would become if it were built in this tree.

Build it from the paper: five cascaded formants, a nasal notch resonator, a tunable
oscillator with real harmonic content, fricative injection at F2, articulation
generators as first- and second-order smoothers over 14 parameters, and stops
as a gate with articulation continuing through the closure.

The honest framing is that this cannot be *verified* the way the SC-01 was —
there is no dump to match. It is a reconstruction from a specification, and it
should be labelled that way in every file it touches. What makes it worth doing
anyway: `py_emu`'s enhanced modes are already groping toward this architecture
without knowing it, and there are surviving recordings of VS-6 and ML-1 units
to check the result against by ear.

**Gate:** the five-formant reconstruction, driven by SC-01 phoneme parameters
upsampled to 16, is recognisably the same *voice family* as an SC-01 while
being audibly less buzzy — and the differences point in the direction the 1978
paper predicts (nasals, voiced fricatives, stop bursts).

### Phase 5 — Shared method, separate deliverables

1. Extend `tools/goldens.py` to fingerprint any engine, not just the SC-01, so
   that two implementations of the *same* engine can be diffed
   sample-for-sample — including the third-party `sc01-x64.dll`, which is
   MAME's device wrapped in a C API and therefore an independent SC-01 to check
   ourselves against. That comparison is cheap and worth doing before any
   Ghidra work on it: if the audio matches, the disassembly tells us nothing we
   do not already know.
2. **One API shape, not one library.** Every engine exposes the same
   `vx_*`-style entry points — create, speak, render, cancel, rate, pitch — so
   a driver written against one is a driver written against all of them. They
   stay separate binaries: an add-on links exactly one, and a GUI links as many
   as it finds.
3. **One add-on per engine.** `nvda-addon/` keeps shipping the SC-01 and only
   the SC-01. An SC-02 add-on is built the same way from its own tree, out of
   its own library, and appears in NVDA's synthesizer list as its own entry
   rather than as a voice hidden inside somebody else's driver.
4. **The Workbench is where they stack.** The GUI is the one place a person
   *wants* the choice — rendering the same phrase through the SC-01, the SC-02
   and the rack Votrax to compare is the whole point of a sound-design tool.
   It loads whichever engine libraries are present and offers the ones it
   found, degrading quietly to just the SC-01 when it is the only one built.

   `gui-native/` is not that, and the two should not be confused. It is a
   single-engine native front end — the SC-01 and nothing else, statically
   linked, no Python — built to the same shape as the native GUIs in the
   sibling SAM and STSPEECH projects. Each engine gets one of those; the
   stacking GUI is a separate thing that sits above all of them.

---

## Part 5 — Tooling

Everything needed is installed; see `C:\GIT\environment\TOOLCHAIN.md`.

| Job | Tool |
|---|---|
| Firmware disassembly | Ghidra 12.0, `C:\GIT\environment\ghidra\ghidraRun.bat` |
| First-pass ROM structure | `C:\GIT\environment\aRomAT\aRomAt.exe` |
| Windows DLLs | Ghidra; `dumpbin` from the MSVC 2026 developer prompt |
| Building | MSVC 14.51 (default), WinLibs GCC 16.2 for the MinGW path |
| Audio comparison | `tools/goldens.py`, extended per Phase 5 |

---

## Part 6 — Open questions

1. Where does the multi-engine Workbench live once there is more than one
   engine to stack? It is in this repository today, on top of the SC-01 it was
   written for. The moment an SC-02 library exists, the GUI is the one
   component that legitimately spans engines while every other deliverable
   stays one chip wide — so it either stays here and loads foreign libraries,
   or moves out to sit above all of them. Worth deciding before Phase 3 Tier A
   ships, not after.
2. Do we hold, or can we obtain, a real SSI-263 recording set for calibration?
   Without it Tier B of Phase 3 cannot be gated honestly.
3. Is the SC-02's parameter interpolation the same mechanism as the SC-01's, or
   did the register interface replace it? The datasheet's "articulation" field
   suggests the former, exposed as a control. Tier A will make this testable.
4. Was the SC-02's filter bank actually the same silicon as the SC-01's, or a
   redesign? Every secondary source says "same analog core"; none of them cite
   a die. Tier C is the only real answer.
5. ~~Does the third-party `sc01-x64.dll` produce bit-identical audio to ours?~~
   **Answered 2026-09-10: no, and the differences are worth having written
   down.** `tools/compare_reference.py` drives both through the same phones.
   Ten of 64 match exactly (the silent ones), ten within one LSB; every vowel
   is 9% louder here (`fx_fudge`); the sibilants SH, CH, ZH and J are about
   **five times** louder, because MAME neutralizes its F2-noise injection
   filter as an admitted stopgap and this engine runs a stable version of it;
   and `vx_ready` fires a constant 5 samples earlier here. No disassembly was
   needed — the binary exposes a `vx_*` C API and takes the ROM as a buffer,
   so it can simply be driven. See `docs/tech-overview.md`, Part 2.

   The same tool found something the audio diff would have missed: our front
   end never emitted PA1, so every comma and full stop was a 49 ms PA0 instead
   of 186 ms and 372 ms. Fixed; recorded in `docs/REWRITE.md`.

---

## Part 7 — Licensing and provenance

- MAME code and tables (`reference/mame_votrax.cpp`, `reference/gate-sim/`, the
  SSI-263 HLE remap) are BSD-3-Clause: Olivier Galibert, Ryan Holtz. Attribution
  stays with anything derived from them.
- The mask ROM dumps are the contents of a commercial IC from a company that
  stopped producing them in the late 1980s. They are archived openly in MAME and
  everywhere else; they are kept here as reference material with provenance
  recorded, not as a redistributable product.
- The Type 'N Talk and Personal Speech System firmware ROMs are Votrax's own
  code, dumped and archived in MAME. Disassembling them to compare against our
  front end is reference work; lifting their tables into shipped code is a
  separate question that has to be settled before, not after.
- Gagnon's patents (3,836,717; 3,908,085) and US 4,433,210 have long expired.
  NRL Report 7948 is a US Government work.

---

## Sources consulted for Part 2

- Gagnon, R.T., "VOTRAX Real Time Hardware for Phoneme Synthesis of Speech",
  Proc. ICASSP 1978, pp. 175–178 — held locally at
  `papers/analyzed/votrax-real-time-hardware-for-phoneme-synthesis-of-speech.pdf`
- [Votrax — Wikipedia](https://en.wikipedia.org/wiki/Votrax) (model line and dates)
- [SC-01A Speech Synthesizer and Related ICs — redcedar.com](http://redcedar.com/sc01.htm)
  (SC-01 vs SC-01-A vs SC-02, Gevaryahu's decap notes)
- [SC-01A die analysis — og.kervella.org/sc01a](https://og.kervella.org/sc01a/)
  (schematics, layer SVGs, 50 MB top-metal die photo)
- [Silicon Systems SSI-263P die shots — visual6502.org](http://www.visual6502.org/images/pages/Silicon_Systems_SSI_263P_die_shots.html)
- [Votrax SC-02 / SSI-263A data sheet, 1985 — bitsavers](https://www.bitsavers.org/pdf/federalScrewWorks/Votrax_SC-02_SSI-263A_Phoneme_Speech_Synthesizer_Data_Sheet_1985.pdf)
  (also held locally in the Silicon Systems data books)
- [MAME PR #11915 — placeholder SSI-263A HLE device](https://github.com/mamedev/mame/pull/11915)
- [MAME `votrax.cpp`](https://github.com/mamedev/mame/blob/master/src/devices/sound/votrax.cpp)
- [The Votrax Type 'N Talk — kevtris.org](http://www.kevtris.org/Projects/votraxtnt/index.html)
- [Votrax SC01-A in VHDL — github.com/shufps](https://github.com/shufps/votrax-sc01a-vhdl)
  (an FPGA recreation from Galibert's simulation; a fourth transcription to
  cross-check against)
