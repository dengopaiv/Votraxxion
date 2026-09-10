#!/usr/bin/env python3
"""Differential test: the built GUI executable against the library itself.

    python tools/verify_gui.py [x64|x86]

The GUI links the synthesizer's C sources straight into its executable.
That is the whole point of it -- there is no DLL beside it and no data
file to find -- but it also means the shipped binary contains a *second*
build of the engine, and nothing so far has checked that the second build
says the same thing as the first.

So: drive `votrax_native.exe --selftest`, which runs the same VoiceFrom,
BuildPhones, RenderPhones and MakeWav the buttons run, and drive the DLL
the NVDA add-on ships through ctypes with the same settings. Compare the
WAV bytes. A match is a statement about what ships; a harness that merely
shared the sources could pass while the executable was broken.

What it covers beyond the samples: the WAV header the GUI writes (the
sample rate moves with the clock, so a constant there would be wrong at
every clock but one), the phone stream the front end produces, the
phoneme-string parser, and the queue-topping loop that a long utterance
goes through.

Exit status is 0 when every case matches, 1 otherwise. Skips with 0 if
either binary is missing -- build them with gui-native\\build.cmd and
nvda-addon\\package.py.
"""
import ctypes
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import goldens  # noqa: E402

BUILD = os.path.join(ROOT, 'gui-native', 'build')
ADDON = os.path.join(ROOT, 'nvda-addon', 'addon', 'synthDrivers')

VX_QUEUE_CAPACITY = 1024
VX_NEUTRAL_INFLECTION = 1
PHONE_BUF = 4096

#: mask, clock kHz, speed %, inflection, phoneme mode, text.
#:
#: The first block is the six presets the GUI's combo offers, which is
#: how they get covered: choosing a preset only writes these four control
#: values, so passing the values is passing the preset.
CASES = [
    # The presets, in the order gui-native/votrax_native.cpp lists them.
    (0,  720, 100, 1, 0, 'Hello, my name is Votrax S C zero one.'),
    (1,  720, 100, 1, 0, 'Hello, my name is Votrax S C zero one.'),
    (0,  720, 300, 1, 0, 'Hello, my name is Votrax S C zero one.'),
    (0, 1080, 100, 1, 0, 'Hello, my name is Votrax S C zero one.'),
    (0,  360, 100, 0, 0, 'Hello, my name is Votrax S C zero one.'),
    (1,  950, 100, 2, 0, 'Hello, my name is Votrax S C zero one.'),

    # Each control at the ends of its range, one at a time.
    (0,  200, 100, 1, 0, 'The slowest clock the box allows.'),
    (0, 2000, 100, 1, 0, 'The fastest clock the box allows.'),
    (0,  720,  10, 1, 0, 'Speed at ten percent.'),
    (0,  720, 1000, 1, 0, 'Speed at a thousand percent.'),
    (0,  720, 100, 0, 0, 'Inflection at the floor.'),
    (0,  720, 100, 3, 0, 'Inflection at the ceiling.'),

    # Both masks on the vowels that differ between them.
    (0,  720, 100, 1, 0, 'The cat sat on the mat, ah, aw, uh.'),
    (1,  720, 100, 1, 0, 'The cat sat on the mat, ah, aw, uh.'),

    # Phoneme mode: bare names, mixed case, and per-phone pitch.
    (0,  720, 100, 1, 1, 'H AH1 L OO PA1 W ER L D'),
    (0,  720, 100, 1, 1, 'h ah1 l oo pa1'),
    (0,  720, 100, 1, 1, 'AY:3 M PA0 AY:0 M'),
    (1,  360, 100, 2, 1, 'AY AE M A R O B AH T STOP'),

    # The front end's own shapes: contours, numbers, abbreviations.
    (0,  720, 100, 1, 0, 'What? Really! Yes, indeed.'),
    (0,  720, 100, 1, 0, 'Dr. Smith paid $42.50 on Jan. 3rd.'),
    (0,  720, 100, 1, 0, 'The quick brown fox jumps over the lazy dog.'),

    # Long enough to run the queue past VX_QUEUE_CAPACITY and make the
    # top-up loop in RenderPhones do its job.
    (0,  720, 100, 1, 0, ('Speech synthesis by rule is the production of '
                          'spoken language from written text. ') * 8),
]


def exe_path(arch):
    return os.path.join(BUILD, 'votrax_native-%s.exe' % arch)


def dll_path(arch):
    return os.path.join(ADDON, 'votraxNative-%s.dll' % arch)


def phones_from_text(lib, text):
    """What the GUI's TextToPhones does, through the same entry point."""
    buf = (ctypes.c_ubyte * PHONE_BUF)()
    n = lib.ttv_translate(text.encode('cp1252', 'replace'), buf, PHONE_BUF)
    if n > PHONE_BUF:
        buf = (ctypes.c_ubyte * n)()
        n = lib.ttv_translate(text.encode('cp1252', 'replace'), buf, n)
    return bytes(buf[:n])


def phones_from_names(lib, text):
    """What the GUI's ParsePhones does: NAME[:LEVEL], case-insensitively."""
    out = bytearray()
    for token in text.replace(',', ' ').split():
        name, level = token.upper(), VX_NEUTRAL_INFLECTION
        if len(name) >= 3 and name[-2] == ':' and name[-1] in '0123':
            name, level = name[:-2], int(name[-1])
        code = lib.vx_phone_by_name(name.encode('ascii'))
        if code < 0:
            raise ValueError('not a phoneme name: %r' % token)
        out.append((code & 0x3F) | (level << 6))
    return bytes(out)


def render_reference(lib, mask, clock_khz, speed_pct, inflection, phones):
    """RenderPhones, in Python, against the DLL."""
    chip = lib.vx_create(mask, clock_khz * 1000)
    if not chip:
        raise RuntimeError('vx_create returned NULL')
    try:
        lib.vx_set_speed(chip, speed_pct / 100.0)
        lib.vx_inflection(chip, inflection)

        rate = int(lib.vx_sample_rate(chip) + 0.5)
        block, limit = 1024, rate * 600
        pcm, queued, total = bytearray(), 0, 0

        while True:
            done = queued >= len(phones) and lib.vx_pending(chip) == 0
            if queued < len(phones):
                room = VX_QUEUE_CAPACITY - lib.vx_pending(chip)
                if room > 0:
                    take = min(room, len(phones) - queued)
                    chunk = (ctypes.c_ubyte * take)(*phones[queued:queued + take])
                    lib.vx_speak(chip, chunk, take)
                    queued += take
                    done = False
            if done or total >= limit:
                break
            pcm += goldens.render(lib, chip, block)
            total += block

        pcm += goldens.render(lib, chip, rate // 4)
        return bytes(pcm), rate
    finally:
        lib.vx_destroy(chip)


def make_wav(pcm, rate):
    """The header MakeWav writes, byte for byte."""
    return (b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVEfmt '
            + struct.pack('<IHHIIHH', 16, 1, 1, rate, rate * 2, 2, 16)
            + b'data' + struct.pack('<I', len(pcm)) + pcm)


def python_arch():
    return 'x64' if ctypes.sizeof(ctypes.c_void_p) == 8 else 'x86'


def run_case(exe, case, out):
    """One --selftest run. Returns the WAV bytes, or None if it failed."""
    mask, clock, speed, infl, mode, text = case
    rc = subprocess.call([exe, '--selftest', str(mask), str(clock), str(speed),
                          str(infl), str(mode), out, text])
    if rc != 0:
        return rc
    with open(out, 'rb') as f:
        return f.read()


def report(label, got, want, rate):
    """True if they match; otherwise say where they part company."""
    if got == want:
        if rate:
            print('  ok   %s  %d Hz, %.2f s'
                  % (label, rate, ((len(want) - 44) / 2) / rate))
        else:
            print('  ok   %s  %d bytes' % (label, len(want)))
        return True
    if len(got) != len(want):
        print('  FAIL %s -- %d bytes, expected %d'
              % (label, len(got), len(want)))
    else:
        first = next(n for n in range(len(got)) if got[n] != want[n])
        where = ('the WAV header, byte %d' % first if first < 44
                 else 'sample %d of %d' % ((first - 44) // 2,
                                           (len(want) - 44) // 2))
        print('  FAIL %s -- diverges at %s' % (label, where))
    return False


def check_against_library(arch):
    """The exe of this interpreter's bitness, against the DLL itself."""
    exe, dll = exe_path(arch), dll_path(arch)
    if not os.path.isfile(exe):
        print('  %s: not built, skipping' % os.path.basename(exe))
        return None
    if not os.path.isfile(dll):
        print('  %s: not built, skipping' % os.path.basename(dll))
        return None

    lib = goldens.bind(dll)
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'gui.wav')
        for i, case in enumerate(CASES):
            mask, clock, speed, infl, mode, text = case
            label = '%s case %2d  mask=%d clock=%d speed=%d infl=%d %s' % (
                arch, i, mask, clock, speed, infl,
                'phonemes' if mode else 'text')

            got = run_case(exe, case, out)
            if isinstance(got, int):
                print('  FAIL %s -- --selftest exited %d' % (label, got))
                failures += 1
                continue

            phones = (phones_from_names(lib, text) if mode
                      else phones_from_text(lib, text))
            pcm, rate = render_reference(lib, mask, clock, speed, infl, phones)
            if not report(label, got, make_wav(pcm, rate), rate):
                failures += 1

    return failures


def check_against_exe(arch, reference_arch):
    """The other bitness, against the exe that was just checked.

    ctypes can only load the DLL that matches the interpreter, so the x86
    executable cannot be driven against an x86 library from a 64-bit
    Python. Rather than skip it, compare it with the executable that was
    checked against the library: if the two builds agree byte for byte
    and one of them agrees with the library, so does the other. The chain
    is only as good as its first link, which is why this runs second and
    reports what it is comparing against.
    """
    exe, reference = exe_path(arch), exe_path(reference_arch)
    if not os.path.isfile(exe):
        print('  %s: not built, skipping' % os.path.basename(exe))
        return None
    if not os.path.isfile(reference):
        print('  %s: no %s executable to compare against, skipping'
              % (os.path.basename(exe), reference_arch))
        return None

    print('  (no %s Python here, so: against %s, which is checked above)'
          % (arch, os.path.basename(reference)))
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        mine = os.path.join(tmp, 'mine.wav')
        theirs = os.path.join(tmp, 'theirs.wav')
        for i, case in enumerate(CASES):
            mask, clock, speed, infl, mode, text = case
            label = '%s case %2d  mask=%d clock=%d speed=%d infl=%d %s' % (
                arch, i, mask, clock, speed, infl,
                'phonemes' if mode else 'text')

            got = run_case(exe, case, mine)
            if isinstance(got, int):
                print('  FAIL %s -- --selftest exited %d' % (label, got))
                failures += 1
                continue
            want = run_case(reference, case, theirs)
            if isinstance(want, int):
                print('  FAIL %s -- the %s reference exited %d'
                      % (label, reference_arch, want))
                failures += 1
                continue

            rate = struct.unpack_from('<I', want, 24)[0] if len(want) >= 28 else 0
            if not report(label, got, want, rate):
                failures += 1

    return failures


def main(argv):
    native = python_arch()
    arches = [argv[0]] if argv else [native] + [a for a in ('x64', 'x86')
                                                if a != native]
    total, ran = 0, 0

    for arch in arches:
        print('%s:' % arch)
        if arch == native:
            result = check_against_library(arch)
        else:
            result = check_against_exe(arch, native)
        if result is None:
            continue
        ran += 1
        total += result

    if ran == 0:
        print('\nNothing to check. Build the GUI with gui-native\\build.cmd '
              'and the library with nvda-addon\\package.py.')
        return 0
    if total:
        print('\n%d case(s) failed.' % total)
        return 1
    print('\nOK: the GUI executable and the library agree on every case.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
