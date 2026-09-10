"""The C API's limits and edge cases.

These are the parts the rewrite could not check against the C++ it replaced,
because the C++ had no limits: it grew its buffers and its phone queue until
memory ran out. The C works in fixed storage, so truncation and overflow are
new behaviour, and new behaviour needs its own tests rather than a differential
one.

The other half is defensive: every entry point takes a chip pointer a caller
could pass as NULL, and a screen-reader add-on driving this over ctypes is
exactly the setting where that happens -- a failed vx_create returning NULL
that nobody checked, and then a crash inside NVDA rather than a synth that does
not speak.

Skipped unless the library has been built (nvda-addon/package.py).
"""

import ctypes
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

DLL = ROOT / "nvda-addon" / "addon" / "synthDrivers" / (
    "votraxNative-x64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8
    else "votraxNative-x86.dll"
)

pytestmark = pytest.mark.skipif(
    not DLL.is_file(),
    reason=f"{DLL.name} not built - run nvda-addon/package.py",
)

#: Matching VX_QUEUE_CAPACITY and TTV_TEXT_MAX in the headers.
QUEUE_CAPACITY = 1024
TEXT_MAX = 4096


@pytest.fixture(scope="module")
def lib():
    import goldens
    return goldens.bind(str(DLL))


@pytest.fixture
def chip(lib):
    handle = lib.vx_create(0, 0)
    assert handle, "vx_create returned NULL"
    yield handle
    lib.vx_destroy(handle)


@pytest.fixture
def buf():
    return (ctypes.c_ubyte * 65536)()


class TestNullSafety:
    """Every entry point has to survive a NULL chip. A caller who did not
    check vx_create's result should get silence, not a crash inside NVDA."""

    def test_lifecycle_calls_ignore_null(self, lib):
        lib.vx_destroy(None)
        lib.vx_reset(None)
        lib.vx_cancel(None)
        lib.vx_write(None, 0)
        lib.vx_set_clock(None, 720000)
        lib.vx_set_mask(None, 1)
        lib.vx_inflection(None, 2)
        lib.vx_set_speed(None, 2.0)

    def test_queries_return_defaults_on_null(self, lib):
        assert lib.vx_pending(None) == 0
        assert lib.vx_ready(None) == 0
        assert lib.vx_clock(None) == 0
        assert lib.vx_sample_rate(None) == 0.0
        assert lib.vx_speed(None) == 1.0
        assert lib.vx_phone_samples(None, 0) == 0
        assert lib.vx_mask(None) == 0

    def test_render_into_null_buffer(self, lib, chip):
        assert lib.vx_render(chip, None, 16) == 0
        assert lib.vx_render(None, (ctypes.c_int16 * 16)(), 16) == 0

    def test_translate_null_text(self, lib, buf):
        assert lib.ttv_translate(None, buf, 4096) == 0
        assert lib.ttv_translate_flat(None, buf, 4096) == 0
        assert lib.ttv_spell(None, buf, 4096) == 0

    def test_phone_name_out_of_range(self, lib):
        assert lib.vx_phone_name(-1) is None
        assert lib.vx_phone_name(64) is None
        assert lib.vx_phone_name(0) == b"EH3"
        assert lib.vx_phone_by_name(b"not a phone") == -1


class TestCapacityReporting:
    """A short output buffer must be detectable: the count returned is what the
    text produced, not what fitted."""

    def test_zero_and_negative_capacity_still_count(self, lib, buf):
        full = lib.ttv_translate(b"Hello world.", buf, 4096)
        assert full > 0
        assert lib.ttv_translate(b"Hello world.", buf, 0) == full
        assert lib.ttv_translate(b"Hello world.", buf, -5) == full

    def test_short_capacity_writes_only_what_fits(self, lib, buf):
        full = lib.ttv_translate(b"Hello world.", buf, 4096)
        expected = list(buf[:full])

        short = (ctypes.c_ubyte * 64)()
        for i in range(64):
            short[i] = 0xAA
        assert lib.ttv_translate(b"Hello world.", short, 3) == full
        assert list(short[:3]) == expected[:3]
        assert short[3] == 0xAA, "wrote past the capacity it was given"

    def test_empty_text(self, lib, buf):
        assert lib.ttv_translate(b"", buf, 4096) == 0
        assert lib.ttv_spell(b"", buf, 4096) == 0


class TestInputTruncation:
    """Input past TTV_TEXT_MAX is truncated rather than growing a buffer."""

    @pytest.mark.parametrize("chars", [4000, TEXT_MAX, 10000, 60000])
    def test_oversized_input_is_bounded_and_safe(self, lib, buf, chars):
        got = lib.ttv_translate(("a " * (chars // 2)).encode(), buf, 65536)
        assert got > 0
        assert got <= 4096, "phone output should be bounded too"

    def test_truncation_is_stable(self, lib, buf):
        """Two inputs that both overflow give the same result: the cut is at a
        fixed point in the text, not wherever the buffer happened to be."""
        long_a = ("a " * 20000).encode()
        long_b = ("a " * 40000).encode()
        n_a = lib.ttv_translate(long_a, buf, 65536)
        first = list(buf[:n_a])
        n_b = lib.ttv_translate(long_b, buf, 65536)
        assert n_a == n_b
        assert first == list(buf[:n_b])


class TestQueueOverflow:
    """vx_speak drops what will not fit instead of growing without bound."""

    def test_caps_at_capacity(self, lib, chip):
        many = (ctypes.c_ubyte * 5000)(*([0x24] * 5000))
        assert lib.vx_speak(chip, many, 5000) == QUEUE_CAPACITY
        assert lib.vx_pending(chip) == QUEUE_CAPACITY

    def test_cancel_empties_the_queue(self, lib, chip):
        many = (ctypes.c_ubyte * 2000)(*([0x24] * 2000))
        lib.vx_speak(chip, many, 2000)
        lib.vx_cancel(chip)
        assert lib.vx_pending(chip) == 0

    def test_ring_buffer_wraps(self, lib, chip):
        """Drain and refill past the capacity, so head wraps the array. A ring
        that wrapped wrongly would replay stale phones."""
        block = (ctypes.c_ubyte * 600)(*([0x24] * 600))
        buffer = (ctypes.c_int16 * 4096)()
        for _ in range(4):
            lib.vx_speak(chip, block, 600)
            while lib.vx_pending(chip):
                lib.vx_render(chip, buffer, 4096)
        assert lib.vx_pending(chip) == 0

    def test_empty_speak_reports_pending(self, lib, chip):
        block = (ctypes.c_ubyte * 10)(*([0x24] * 10))
        lib.vx_speak(chip, block, 10)
        assert lib.vx_speak(chip, None, 0) == 10


class TestClamping:
    @pytest.mark.parametrize("requested,expected", [
        (0.0, 0.1), (-3.0, 0.1), (0.05, 0.1),
        (1e9, 10.0), (11.0, 10.0), (2.5, 2.5), (1.0, 1.0),
    ])
    def test_speed_is_clamped(self, lib, chip, requested, expected):
        lib.vx_set_speed(chip, requested)
        assert lib.vx_speed(chip) == pytest.approx(expected)

    @pytest.mark.parametrize("requested", [0, 1, 2, 3, 4, 7, 255])
    def test_inflection_is_masked_to_two_bits(self, lib, chip, requested):
        lib.vx_inflection(chip, requested)
        assert lib.vx_get_inflection(chip) == requested & 3

    @pytest.mark.parametrize("phone", [0, 63, 64, 200, 255])
    def test_phone_codes_are_masked(self, lib, chip, phone):
        """A phone above 63 must wrap into the ROM's 64 rows, not index off
        the end of it."""
        assert lib.vx_phone_samples(chip, phone) == \
            lib.vx_phone_samples(chip, phone & 0x3F)
        lib.vx_write(chip, phone)
        assert lib.vx_render(chip, (ctypes.c_int16 * 256)(), 256) == 256

    def test_zero_clock_is_ignored(self, lib, chip):
        before = lib.vx_clock(chip)
        lib.vx_set_clock(chip, 0)
        assert lib.vx_clock(chip) == before
