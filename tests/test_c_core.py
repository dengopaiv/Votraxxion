"""The C synthesizer must keep producing exactly the samples it produces now.

`tests/data/golden.json` is a fingerprint of the library's entire observable
output -- every phone on both masks, whole utterances at five speeds and three
clocks, the front end, cancel and reset, the ready handshake. It was captured
from the C++ implementation the C one replaced, and the C library reproduced it
entry for entry.

That makes it the one test here that cannot be satisfied by accident. Every
other test in this suite asserts a property somebody thought to write down; this
one asserts that nothing at all changed. A filter coefficient off in the
sixteenth digit is inaudible on one phone and a different voice on a paragraph,
and only a comparison like this one catches it.

So a failure here is not necessarily a bug -- it may be a deliberate change to
the DSP. But it has to be deliberate: re-bless the file with

    python tools/goldens.py capture <the built dll> tests/data/golden.json

and say in the commit message what moved and why.

Skipped unless the library has been built (nvda-addon/package.py).
"""

import ctypes
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "data" / "golden.json"
DLL = ROOT / "nvda-addon" / "addon" / "synthDrivers" / (
    "votraxsc01-x64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8
    else "votraxsc01-x86.dll"
)

pytestmark = pytest.mark.skipif(
    not DLL.is_file(),
    reason=f"{DLL.name} not built - run nvda-addon/package.py",
)


@pytest.fixture(scope="module")
def captured():
    import goldens
    return goldens.capture(str(DLL))


@pytest.fixture(scope="module")
def expected():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def test_golden_file_is_present():
    """Without it the whole module would skip silently and prove nothing."""
    assert GOLDEN.is_file(), f"{GOLDEN} is missing"


def test_every_entry_is_accounted_for(captured, expected):
    """A capture that grew or lost entries means the harness changed, which is
    worth noticing separately from a sample changing."""
    assert sorted(captured) == sorted(expected)


@pytest.mark.parametrize("key", [
    "phone_names", "phone_by_name", "translate", "translate_flat", "spell",
    "phones_mask0", "phones_mask1", "inflection", "after_cancel",
    "after_reset", "live_switch", "ready_at",
])
def test_entry_matches(captured, expected, key):
    assert captured[key] == expected[key], (
        f"{key} differs from the committed fingerprint -- see this file's "
        f"docstring before re-blessing it"
    )


def test_scheduler_entries_match(captured, expected):
    """The speak_* entries: every mask, clock and speed combination."""
    keys = sorted(k for k in expected if k.startswith("speak_"))
    assert keys, "the golden file has no scheduler entries"
    mismatched = [k for k in keys if captured[k] != expected[k]]
    assert not mismatched, f"scheduler output differs for: {mismatched}"
