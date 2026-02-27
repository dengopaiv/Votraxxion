"""Smoke tests for the wxPython GUI."""

import pytest
import numpy as np

# wxPython may not be installed; skip all tests if missing.
wx = pytest.importorskip("wx")

from pyvotrax.gui import VotraxFrame


@pytest.fixture
def app():
    """Create a wx.App for testing (headless)."""
    application = wx.App(False)
    yield application
    application.Destroy()


class TestVotraxFrame:
    """Smoke tests for VotraxFrame."""

    def test_instantiation(self, app):
        """Frame creates without error."""
        frame = VotraxFrame()
        assert frame is not None
        frame.Destroy()

    def test_speak_empty_text(self, app):
        """Speak callback does not crash on empty text."""
        frame = VotraxFrame()
        frame._text_ctrl.SetValue("")
        frame.on_speak(None)
        frame.Destroy()

    def test_stop_no_playback(self, app):
        """Stop callback does not crash when nothing is playing."""
        frame = VotraxFrame()
        frame.on_stop(None)
        frame.Destroy()


class TestVotraxFrameControls:
    """Tests for new GUI controls and audio processing."""

    def test_sliders_exist(self, app):
        """Rate, pitch, and volume sliders are present."""
        frame = VotraxFrame()
        assert isinstance(frame._rate_slider, wx.Slider)
        assert isinstance(frame._pitch_slider, wx.Slider)
        assert isinstance(frame._volume_slider, wx.Slider)
        frame.Destroy()

    def test_slider_defaults(self, app):
        """Default slider values are rate=50, pitch=50, volume=100."""
        frame = VotraxFrame()
        assert frame._rate_slider.GetValue() == 50
        assert frame._pitch_slider.GetValue() == 50
        assert frame._volume_slider.GetValue() == 100
        frame.Destroy()

    def test_enhanced_checkbox_exists(self, app):
        """Enhanced mode checkbox is present and unchecked by default."""
        frame = VotraxFrame()
        assert isinstance(frame._enhanced_cb, wx.CheckBox)
        assert frame._enhanced_cb.GetValue() is False
        frame.Destroy()

    def test_enhanced_toggle_recreates_tts(self, app):
        """Toggling enhanced mode creates a new TTS instance."""
        frame = VotraxFrame()
        old_tts = frame._tts
        frame._enhanced_cb.SetValue(True)
        frame._recreate_tts(True)
        assert frame._tts is not old_tts
        frame.Destroy()

    def test_phoneme_display_exists(self, app):
        """Phoneme display is a read-only text control."""
        frame = VotraxFrame()
        assert isinstance(frame._phoneme_display, wx.TextCtrl)
        # Check read-only by verifying style
        style = frame._phoneme_display.GetWindowStyleFlag()
        assert style & wx.TE_READONLY
        frame.Destroy()

    def test_progress_bar_exists(self, app):
        """Progress gauge is present."""
        frame = VotraxFrame()
        assert isinstance(frame._progress, wx.Gauge)
        frame.Destroy()

    def test_process_audio_returns_array(self, app):
        """_process_audio returns a valid float32 numpy array."""
        frame = VotraxFrame()
        # Create a simple test signal (1 second of sine at SCLOCK rate)
        from pyvotrax.filters import SCLOCK
        t = np.linspace(0, 0.01, int(SCLOCK * 0.01))
        test_audio = np.sin(2 * np.pi * 440 * t)
        result = frame._process_audio(test_audio)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) > 0
        frame.Destroy()

    def test_process_audio_volume_zero(self, app):
        """Volume=0 produces silence."""
        frame = VotraxFrame()
        frame._volume_slider.SetValue(0)
        from pyvotrax.filters import SCLOCK
        t = np.linspace(0, 0.01, int(SCLOCK * 0.01))
        test_audio = np.sin(2 * np.pi * 440 * t)
        result = frame._process_audio(test_audio)
        assert np.allclose(result, 0.0)
        frame.Destroy()
