"""votrax-say, the command-line tool, run as a user would run it.

Skipped unless it has been built:

    cmake -S . -B build/cmake
    cmake --build build/cmake --config Release
"""

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXE_NAME = "votrax-say.exe" if sys.platform == "win32" else "votrax-say"
CANDIDATES = [
    ROOT / "build" / "cmake" / "Release" / EXE_NAME,   # multi-config (MSVC)
    ROOT / "build" / "cmake" / EXE_NAME,               # single-config (make, ninja)
]
EXE = next((p for p in CANDIDATES if p.is_file()), None)

pytestmark = pytest.mark.skipif(
    EXE is None, reason="votrax-say not built - see this module's docstring")


def run(*args, stdin=None):
    return subprocess.run([str(EXE), *args], input=stdin, capture_output=True,
                          text=True, timeout=60)


def wav_info(path):
    with wave.open(str(path)) as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


def test_text_to_wav(tmp_path):
    out = tmp_path / "hello.wav"
    r = run("-o", str(out), "Hello world.")
    assert r.returncode == 0, r.stderr
    channels, width, rate, frames = wav_info(out)
    assert (channels, width, rate) == (1, 2, 40000)
    assert 0.5 * rate < frames < 3 * rate


def test_stdin_is_the_same_as_an_argument(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    assert run("-o", str(a), "From a pipe.").returncode == 0
    assert run("-o", str(b), stdin="From a pipe.").returncode == 0
    assert a.read_bytes() == b.read_bytes()


def test_print_round_trips_through_phones(tmp_path):
    """--print emits the --phones grammar, contour levels included, so the
    audio from the printed phones is the audio from the text."""
    text, phones = tmp_path / "text.wav", tmp_path / "phones.wav"
    printed = run("--print", "Is this a question?")
    assert printed.returncode == 0
    assert ":3" in printed.stdout          # the question's rise is in there
    assert run("-o", str(text), "Is this a question?").returncode == 0
    assert run("-o", str(phones), "--phones", printed.stdout.strip()).returncode == 0
    assert text.read_bytes() == phones.read_bytes()


def test_a_bad_phone_is_named(tmp_path):
    r = run("-o", str(tmp_path / "x.wav"), "--phones", "H EH1 XYZZY")
    assert r.returncode == 2
    assert "phone 3" in r.stderr and "XYZZY" in r.stderr


def test_clock_moves_the_sample_rate(tmp_path):
    out = tmp_path / "fast.wav"
    assert run("-o", str(out), "--clock", "1080000", "test").returncode == 0
    assert wav_info(out)[2] == 60000


def test_speed_shortens_without_changing_rate(tmp_path):
    slow, fast = tmp_path / "slow.wav", tmp_path / "fast.wav"
    text = "testing one two three four five"
    assert run("-o", str(slow), text).returncode == 0
    assert run("-o", str(fast), "--speed", "2", text).returncode == 0
    s, f = wav_info(slow), wav_info(fast)
    assert s[2] == f[2]
    # The quarter-second tail is fixed, so compare what precedes it.
    tail = s[2] // 4
    assert 0.4 < (f[3] - tail) / (s[3] - tail) < 0.65


def test_masks_differ(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    assert run("-o", str(a), "--mask", "sc01a", "father").returncode == 0
    assert run("-o", str(b), "--mask", "sc01", "father").returncode == 0
    assert a.read_bytes() != b.read_bytes()


def test_table_timings_add_up(tmp_path):
    out = tmp_path / "table.wav"
    r = run("--table", "-o", str(out))
    assert r.returncode == 0, r.stderr
    rows = [line.split() for line in r.stdout.splitlines()[1:]]
    assert len(rows) == 64
    assert [row[1] for row in rows[:3]] == ["EH3", "EH2", "EH1"]
    pa1_ms = float(rows[0x3E][3])
    for this, nxt in zip(rows, rows[1:]):
        gap_ms = (float(nxt[2]) - float(this[2])) * 1000
        assert abs(gap_ms - (float(this[3]) + pa1_ms)) < 1.5


def test_names_lists_the_datasheet(tmp_path):
    r = run("--names")
    lines = r.stdout.split("\n")
    assert lines[0].split() == ["00", "EH3"]
    assert lines[0x3F].split() == ["3F", "STOP"]


def test_knob_sets_the_clock_from_the_datasheet_circuit(tmp_path):
    """Figure 8's 6.8 k + 50 k audio taper against 120 pF: position 0.6 is
    within 2% of the datasheet's 720 kHz, and the WAV rate is clock / 18."""
    out = tmp_path / "knob.wav"
    assert run("-o", str(out), "--knob", "0.6", "test").returncode == 0
    assert wav_info(out)[2] == pytest.approx(40000, rel=0.02)
    ends = tmp_path / "ends.wav"
    assert run("-o", str(ends), "--knob", "0", "test").returncode == 0
    assert wav_info(ends)[2] == round(round(1.25 / (6800 * 120e-12)) / 18)


def test_rc_uses_the_datasheet_relation(tmp_path):
    out = tmp_path / "rc.wav"
    assert run("-o", str(out), "--rc", "6500,300e-12", "test").returncode == 0
    assert wav_info(out)[2] == round(round(1.25 / (6500 * 300e-12)) / 18)


def test_figure8_output_stage_changes_the_audio_not_the_length(tmp_path):
    chip, fig8 = tmp_path / "chip.wav", tmp_path / "fig8.wav"
    assert run("-o", str(chip), "hello").returncode == 0
    assert run("-o", str(fig8), "--output-stage", "figure8", "hello").returncode == 0
    assert wav_info(chip) == wav_info(fig8)
    assert chip.read_bytes() != fig8.read_bytes()


@pytest.mark.parametrize("args", [
    ["--clock", "5"], ["--speed", "0"], ["--inflection", "4"],
    ["--mask", "sc02"], ["--bogus"], ["--knob", "1.5"], ["--rc", "6800"],
    ["--rc", "5,5"], ["--output-stage", "radio"],
])
def test_bad_options_fail_cleanly(args, tmp_path):
    r = run("-o", str(tmp_path / "x.wav"), *args, "text")
    assert r.returncode == 2
    assert r.stderr.startswith("votrax-say:") or "usage" in r.stderr
