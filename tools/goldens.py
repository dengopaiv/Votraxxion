#!/usr/bin/env python3
"""Capture a fingerprint of a Votrax library's entire observable output.

The point of this file is the C rewrite.  A rewrite of a DSP core is only
believable if you can show it produces the same samples as what it replaced,
and "sounds the same" is not a check -- a one-bit difference in a filter
coefficient is inaudible on one phone and a different voice on a paragraph.
So: run a fixed corpus through the library, hash every sample, and diff the
hashes.  Identical hashes across two independent implementations is proof;
anything else is a bug with a known first divergent case.

    python tools/goldens.py capture path/to/votrax.dll out.json
    python tools/goldens.py compare a.json b.json

The corpus deliberately covers what unit tests are bad at: every phone on both
masks, the scheduler's rate arithmetic at speeds that do and do not divide
evenly, clock changes that move the sample rate, and the front end on text
that exercises the number reader, the abbreviation table and all three
sentence contours.
"""

import ctypes
import hashlib
import json
import sys

BASE_CLOCK = 720000
MASKS = (0, 1)          # VX_MASK_SC01A, VX_MASK_SC01

#: Text chosen to reach the parts of the front end that ordinary prose misses.
CORPUS = (
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Is this a question? Yes! It is.",
    "Dr. Smith has 3 cats and 1024 reasons.",
    "Section 7, paragraph 12.",
    "young thing, judge the hedge",
    "Pay $4.20 by the 21st: version 1.2.3, 1,000,000 or 007.",
)

#: Speeds either side of 1.0, including ones whose sample arithmetic does not
#: divide evenly -- that is where an off-by-one in the hold counter shows up.
SPEEDS = (0.5, 1.0, 1.3, 2.0, 3.7)

#: Clocks the datasheet endorses varying to.  Each moves the sample rate, so
#: every filter coefficient in the chip is rebuilt.
CLOCKS = (BASE_CLOCK, 500000, 1000000)


def bind(path):
    """Load the library with every argtype declared -- see votrax_capi.h."""
    lib = ctypes.CDLL(path)
    p = ctypes.c_void_p
    u8p = ctypes.POINTER(ctypes.c_ubyte)
    i16p = ctypes.POINTER(ctypes.c_int16)
    sigs = {
        "vx_create": ([ctypes.c_int, ctypes.c_uint], p),
        "vx_destroy": ([p], None),
        "vx_reset": ([p], None),
        "vx_set_clock": ([p, ctypes.c_uint], None),
        "vx_clock": ([p], ctypes.c_uint),
        "vx_sample_rate": ([p], ctypes.c_double),
        "vx_set_mask": ([p, ctypes.c_int], None),
        "vx_mask": ([p], ctypes.c_int),
        "vx_inflection": ([p, ctypes.c_ubyte], None),
        "vx_get_inflection": ([p], ctypes.c_int),
        "vx_set_speed": ([p, ctypes.c_double], None),
        "vx_speed": ([p], ctypes.c_double),
        "vx_phone_samples": ([p, ctypes.c_ubyte], ctypes.c_int),
        "vx_write": ([p, ctypes.c_ubyte], None),
        "vx_ready": ([p], ctypes.c_int),
        "vx_speak": ([p, u8p, ctypes.c_int], ctypes.c_int),
        "vx_pending": ([p], ctypes.c_int),
        "vx_cancel": ([p], None),
        "vx_render": ([p, i16p, ctypes.c_int], ctypes.c_int),
        "ttv_translate": ([ctypes.c_char_p, u8p, ctypes.c_int], ctypes.c_int),
        "ttv_translate_flat": ([ctypes.c_char_p, u8p, ctypes.c_int], ctypes.c_int),
        "ttv_spell": ([ctypes.c_char_p, u8p, ctypes.c_int], ctypes.c_int),
        "vx_phone_name": ([ctypes.c_int], ctypes.c_char_p),
        "vx_phone_by_name": ([ctypes.c_char_p], ctypes.c_int),
    }
    for name, (args, ret) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ret
    return lib


def translate(fn, text):
    buf = (ctypes.c_ubyte * 4096)()
    n = fn(text.encode("utf-8"), buf, 4096)
    return list(buf[:min(n, 4096)])


def render(lib, chip, count):
    buf = (ctypes.c_int16 * count)()
    lib.vx_render(chip, buf, count)
    return bytes(memoryview(buf).cast("B"))


def digest(data):
    return hashlib.sha256(data).hexdigest()[:32]


def capture(path):
    lib = bind(path)
    out = {}

    # -- the tables, independent of any chip ---------------------------------
    out["phone_names"] = [
        (lib.vx_phone_name(i) or b"").decode("ascii") for i in range(64)
    ]
    out["phone_by_name"] = [
        lib.vx_phone_by_name(n.encode("ascii")) for n in out["phone_names"]
    ]

    # -- the front end -------------------------------------------------------
    out["translate"] = {t: translate(lib.ttv_translate, t) for t in CORPUS}
    out["translate_flat"] = {
        t: translate(lib.ttv_translate_flat, t) for t in CORPUS
    }
    out["spell"] = {t: translate(lib.ttv_spell, t) for t in CORPUS[:3]}

    # -- every phone, in isolation, on both masks ----------------------------
    # Each phone gets its own chip so no interpolation state carries over: this
    # measures a phone's own sound, not a transition into it.
    for mask in MASKS:
        rows = []
        for phone in range(64):
            chip = lib.vx_create(mask, 0)
            n = lib.vx_phone_samples(chip, phone)
            lib.vx_write(chip, phone)
            rows.append([n, digest(render(lib, chip, n + 64))])
            lib.vx_destroy(chip)
        out["phones_mask%d" % mask] = rows

    # -- whole utterances through the scheduler ------------------------------
    for mask in MASKS:
        for clock in CLOCKS:
            for speed in SPEEDS:
                chip = lib.vx_create(mask, clock)
                lib.vx_set_speed(chip, speed)
                acc = hashlib.sha256()
                for text in CORPUS:
                    phones = translate(lib.ttv_translate, text)
                    buf = (ctypes.c_ubyte * len(phones))(*phones)
                    lib.vx_speak(chip, buf, len(phones))
                    # Render until the queue drains, plus a tail, so the last
                    # phone's decay is in the hash too.
                    while lib.vx_pending(chip):
                        acc.update(render(lib, chip, 512))
                    acc.update(render(lib, chip, 4096))
                key = "speak_m%d_c%d_s%s" % (mask, clock, speed)
                out[key] = [
                    acc.hexdigest()[:32],
                    round(lib.vx_sample_rate(chip), 9),
                    lib.vx_clock(chip),
                    round(lib.vx_speed(chip), 9),
                ]
                lib.vx_destroy(chip)

    # -- inflection: all four levels -----------------------------------------
    rows = []
    for level in range(4):
        chip = lib.vx_create(0, 0)
        lib.vx_inflection(chip, level)
        phones = translate(lib.ttv_translate, "Hello world.")
        buf = (ctypes.c_ubyte * len(phones))(*phones)
        lib.vx_speak(chip, buf, len(phones))
        acc = hashlib.sha256()
        while lib.vx_pending(chip):
            acc.update(render(lib, chip, 512))
        acc.update(render(lib, chip, 2048))
        rows.append([lib.vx_get_inflection(chip), acc.hexdigest()[:32]])
        lib.vx_destroy(chip)
    out["inflection"] = rows

    # -- cancel and reset must leave no tail of the cancelled utterance ------
    chip = lib.vx_create(0, 0)
    phones = translate(lib.ttv_translate, CORPUS[1])
    buf = (ctypes.c_ubyte * len(phones))(*phones)
    lib.vx_speak(chip, buf, len(phones))
    render(lib, chip, 3000)
    lib.vx_cancel(chip)
    out["after_cancel"] = [lib.vx_pending(chip), digest(render(lib, chip, 2048))]
    lib.vx_reset(chip)
    out["after_reset"] = [lib.vx_get_inflection(chip),
                          digest(render(lib, chip, 2048))]
    lib.vx_destroy(chip)

    # -- switching mask and clock on a live chip -----------------------------
    # 4096 samples, not fewer: a freshly reset chip is silent for its first
    # ~1200 samples while the formant interpolators ramp up from zero, so a
    # short window here would hash silence and pass no matter what changed.
    chip = lib.vx_create(0, 0)
    lib.vx_write(chip, 0x00)
    a = digest(render(lib, chip, 4096))
    lib.vx_set_mask(chip, 1)
    lib.vx_write(chip, 0x00)
    b = digest(render(lib, chip, 4096))
    lib.vx_set_clock(chip, 900000)
    lib.vx_write(chip, 0x00)
    c = digest(render(lib, chip, 4096))
    out["live_switch"] = [a, b, c, lib.vx_mask(chip), lib.vx_clock(chip),
                          round(lib.vx_sample_rate(chip), 9)]
    lib.vx_destroy(chip)

    # -- vx_ready: the hardware handshake, not the scheduler ------------------
    chip = lib.vx_create(0, 0)
    lib.vx_write(chip, 0x0A)
    ready_at = None
    for i in range(20000):
        if lib.vx_ready(chip):
            ready_at = i
            break
        render(lib, chip, 1)
    out["ready_at"] = ready_at
    lib.vx_destroy(chip)

    return out


def _brief(value):
    text = json.dumps(value)
    return text if len(text) <= 160 else text[:157] + "..."


def compare(a_path, b_path):
    with open(a_path, encoding="utf-8") as f:
        a = json.load(f)
    with open(b_path, encoding="utf-8") as f:
        b = json.load(f)

    keys = sorted(set(a) | set(b))
    bad = []
    for key in keys:
        if key not in a:
            bad.append("%s: missing from %s" % (key, a_path))
        elif key not in b:
            bad.append("%s: missing from %s" % (key, b_path))
        elif a[key] != b[key]:
            bad.append("%s: differs\n    %s: %s\n    %s: %s"
                       % (key, a_path, _brief(a[key]), b_path, _brief(b[key])))

    if bad:
        print("%d of %d entries differ:\n" % (len(bad), len(keys)))
        for line in bad:
            print("  " + line)
        return 1
    print("identical: all %d entries match" % len(keys))
    return 0


def main(argv):
    if len(argv) == 4 and argv[1] == "capture":
        with open(argv[3], "w", encoding="utf-8") as f:
            json.dump(capture(argv[2]), f, indent=1, sort_keys=True)
        print("captured %s -> %s" % (argv[2], argv[3]))
        return 0
    if len(argv) == 4 and argv[1] == "compare":
        return compare(argv[2], argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
