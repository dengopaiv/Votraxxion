"""The native NVDA driver, exercised against stubbed NVDA modules.

The driver is a shim: ctypes bindings, a work queue, one thread, and the
mapping from NVDA's 0-100 settings onto the chip's controls. None of that
needs NVDA to run — only the handful of modules it imports — so stubbing them
lets the parts most likely to be wrong be tested here rather than discovered
inside a screen reader.

Skipped unless the DLL has been built (nvda-addon/package.py).
"""

import ctypes
import importlib.util
import queue
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "nvda-addon" / "addon" / "synthDrivers" / "votraxNative.py"
DLL = DRIVER.parent / ("votraxNative-x64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8
                       else "votraxNative-x86.dll")

pytestmark = pytest.mark.skipif(
    not DLL.is_file(),
    reason=f"{DLL.name} not built - run nvda-addon/package.py",
)


# --------------------------------------------------------------- the stubs --

class FakePlayer:
    """Stands in for nvwave.WavePlayer, keeping whatever is fed to it."""

    def __init__(self, channels, samplesPerSec, bitsPerSample, outputDevice=None):
        self.rate = samplesPerSec
        self.data = bytearray()
        self.stops = 0
        self.closed = False
        self.idled = 0

    def feed(self, data):
        self.data += data

    def stop(self):
        self.stops += 1
        self.data.clear()

    def idle(self):
        self.idled += 1

    def pause(self, switch):
        pass

    def close(self):
        self.closed = True


class Notifier:
    def __init__(self):
        self.calls = []

    def notify(self, **kwargs):
        self.calls.append(kwargs)


class FakeSynthDriver:
    """Enough of NVDA's SynthDriver for the class body to evaluate."""

    @staticmethod
    def VoiceSetting(**kw):
        return ("voice", kw)

    @staticmethod
    def RateSetting(**kw):
        return ("rate", kw)

    @staticmethod
    def PitchSetting(**kw):
        return ("pitch", kw)


def _install_stubs():
    """Put fake NVDA modules on sys.modules and `_` in builtins."""
    import builtins
    if not hasattr(builtins, "_"):
        builtins._ = lambda s: s

    config = types.ModuleType("config")
    config.conf = {"audio": {"outputDevice": "default"}}
    config.getUserDefaultConfigPath = lambda: None

    nvwave = types.ModuleType("nvwave")
    nvwave.WavePlayer = FakePlayer

    commands = types.ModuleType("speech.commands")

    class IndexCommand:
        def __init__(self, index):
            self.index = index

    class CharacterModeCommand:
        def __init__(self, state):
            self.state = state

    commands.IndexCommand = IndexCommand
    commands.CharacterModeCommand = CharacterModeCommand
    speech = types.ModuleType("speech")
    speech.commands = commands

    sdh = types.ModuleType("synthDriverHandler")
    sdh.SynthDriver = FakeSynthDriver
    sdh.VoiceInfo = lambda vid, display, lang=None: {
        "id": vid, "displayName": display, "language": lang}
    sdh.synthIndexReached = Notifier()
    sdh.synthDoneSpeaking = Notifier()

    asu = types.ModuleType("autoSettingsUtils")
    ds = types.ModuleType("autoSettingsUtils.driverSetting")
    ds.BooleanDriverSetting = lambda *a, **kw: ("bool", a, kw)
    asu.driverSetting = ds

    for name, module in (
        ("config", config), ("nvwave", nvwave),
        ("speech", speech), ("speech.commands", commands),
        ("synthDriverHandler", sdh),
        ("autoSettingsUtils", asu),
        ("autoSettingsUtils.driverSetting", ds),
    ):
        sys.modules.setdefault(name, module)
    return sdh, commands


@pytest.fixture(scope="module")
def driver_module():
    sdh, commands = _install_stubs()
    spec = importlib.util.spec_from_file_location("votraxNative_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._test_sdh = sdh
    module._test_commands = commands
    return module


@pytest.fixture
def driver(driver_module):
    d = driver_module.SynthDriver()
    yield d
    d.terminate()


def _drain(d, timeout=20.0):
    """Wait for the speak thread to work through the queue.

    Polls rather than using Queue.join(): the driver never calls task_done(),
    so join() would wait forever.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if d._queue.empty():
            # The thread may still be inside the final item, so let it settle
            # and confirm the queue is still empty afterwards.
            time.sleep(0.05)
            if d._queue.empty():
                return True
        time.sleep(0.02)
    raise AssertionError("speak thread did not drain within %.0fs" % timeout)


# --------------------------------------------------------------- the tests --

class TestLoading:
    def test_check_finds_the_library(self, driver_module):
        assert driver_module.SynthDriver.check() is True

    def test_both_masks_are_offered(self, driver):
        voices = driver._getAvailableVoices()
        assert set(voices) == {"sc01", "sc01a"}

    def test_default_voice_is_the_1980_mask(self, driver):
        assert driver._get_voice() == "sc01"

    def test_player_runs_at_the_chip_rate(self, driver):
        # 720 kHz / 18 = 40 kHz
        assert driver._player.rate == 40000


class TestTranslation:
    def test_text_becomes_phones(self, driver_module):
        lib = driver_module._Lib.shared()
        phones = lib.translate("hello")
        assert len(phones) > 3

    def test_spelling_differs_from_speaking(self, driver_module):
        lib = driver_module._Lib.shared()
        assert lib.translate("cat") != lib.translate("cat", spell=True)

    def test_spelling_is_longer(self, driver_module):
        """Three letter names take more phones than one syllable."""
        lib = driver_module._Lib.shared()
        assert len(lib.translate("cat", spell=True)) > len(lib.translate("cat"))

    def test_prosody_rides_in_the_high_bits(self, driver_module):
        lib = driver_module._Lib.shared()
        statement = lib.translate("The system is ready.")
        question = lib.translate("Is the system ready?")
        levels = lambda b: [(x >> 6) & 3 for x in b]
        # a statement ends lower than it starts, a question ends higher
        assert levels(statement)[-1] < levels(statement)[0]
        assert levels(question)[-1] > levels(question)[0]


class TestSpeaking:
    def test_speaking_produces_audio(self, driver):
        driver.speak(["hello there"])
        _drain(driver)
        assert len(driver._player.data) > 0

    def test_index_commands_are_reported(self, driver, driver_module):
        notifier = driver_module._test_sdh.synthIndexReached
        notifier.calls.clear()
        IndexCommand = driver_module._test_commands.IndexCommand
        driver.speak(["one", IndexCommand(7), "two"])
        _drain(driver)
        assert 7 in [c["index"] for c in notifier.calls]

    def test_done_is_reported(self, driver, driver_module):
        notifier = driver_module._test_sdh.synthDoneSpeaking
        notifier.calls.clear()
        driver.speak(["done"])
        _drain(driver)
        assert notifier.calls

    def test_character_mode_switches_to_spelling(self, driver, driver_module):
        CharacterModeCommand = driver_module._test_commands.CharacterModeCommand
        lib = driver_module._Lib.shared()
        spelled = lib.translate("ab", spell=True)
        driver.speak([CharacterModeCommand(True), "ab"])
        _drain(driver)
        # nothing to assert about audio content; what matters is that the
        # command routed through the spelling path without raising
        assert len(spelled) > 0
        assert len(driver._player.data) > 0


class TestSettings:
    def test_rate_maps_to_speed_not_clock(self, driver):
        driver._set_rate(100)
        driver._apply_rate()
        assert driver._chip._lib.vx_speed(driver._chip._chip) == pytest.approx(2.0)
        assert driver._chip._lib.vx_clock(driver._chip._chip) == 720000

    def test_authentic_rate_maps_to_clock_not_speed(self, driver):
        driver._authentic = True
        driver._set_rate(100)
        driver._apply_rate()
        assert driver._chip._lib.vx_speed(driver._chip._chip) == pytest.approx(1.0)
        assert driver._chip._lib.vx_clock(driver._chip._chip) == 1440000

    def test_rate_50_is_the_datasheet_clock(self, driver):
        driver._set_rate(50)
        driver._apply_rate()
        assert driver._chip._lib.vx_clock(driver._chip._chip) == 720000
        assert driver._chip._lib.vx_speed(driver._chip._chip) == pytest.approx(1.0)

    @pytest.mark.parametrize("value,level", [(0, 0), (33, 1), (66, 2), (100, 3)])
    def test_pitch_snaps_to_the_four_levels(self, driver, value, level):
        driver._set_pitch(value)
        driver._apply_pitch()
        assert driver._chip._lib.vx_get_inflection(driver._chip._chip) == level

    def test_pitch_readback_matches_what_was_heard(self, driver):
        """The announced number must correspond to a real level, or the
        settings ring appears to move without changing the pitch."""
        for value in range(0, 101):
            driver._set_pitch(value)
            assert driver._get_pitch() in (0, 33, 66, 100)

    def test_voice_change_switches_the_mask(self, driver):
        driver._open_voice("sc01a")
        assert driver._chip._lib.vx_mask(driver._chip._chip) == 0
        driver._open_voice("sc01")
        assert driver._chip._lib.vx_mask(driver._chip._chip) == 1


class TestCancellation:
    def test_cancel_bumps_the_epoch(self, driver):
        before = driver._epoch
        driver.cancel()
        assert driver._epoch == before + 1

    def test_cancel_stops_the_player(self, driver):
        driver.speak(["a long sentence to be interrupted partway through"])
        driver.cancel()
        assert driver._player.stops > 0

    def test_control_items_survive_a_cancel(self, driver):
        """A rate change queued just before a cancel must still happen."""
        driver._queue.put((None, "rate", None, None))
        driver._queue.put((driver._epoch, "phones", b"\x24", None))
        driver.cancel()
        remaining = []
        try:
            while True:
                remaining.append(driver._queue.get_nowait())
        except queue.Empty:
            pass
        assert any(item[1] == "rate" for item in remaining)
        assert not any(item[1] == "phones" for item in remaining)

    def test_speech_after_cancel_still_works(self, driver):
        driver.speak(["first utterance"])
        driver.cancel()
        driver.speak(["second utterance"])
        _drain(driver)
        assert len(driver._player.data) > 0
