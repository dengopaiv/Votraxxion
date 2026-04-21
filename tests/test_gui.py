"""Smoke tests for the wxPython music-production workbench GUI."""

import pytest
import numpy as np

wx = pytest.importorskip("wx")

from pyvotrax.gui import VotraxFrame
from pyvotrax.presets import Preset, factory_presets_dir, load_preset


@pytest.fixture
def app():
    application = wx.App(False)
    yield application
    application.Destroy()


class TestVotraxFrameBasic:
    def test_instantiation(self, app):
        frame = VotraxFrame()
        assert frame is not None
        frame.Destroy()

    def test_stop_no_playback(self, app):
        frame = VotraxFrame()
        frame.on_stop(None)
        frame.Destroy()


class TestParameterWidgets:
    """Every backend knob must have a widget."""

    def test_master_clock_slider_default(self, app):
        frame = VotraxFrame()
        # Default is 720 kHz (integer Hz)
        assert frame._master_clock.get_int() == 720_000
        frame.Destroy()

    def test_fx_fudge_slider_default(self, app):
        frame = VotraxFrame()
        # Default fx_fudge is 150/4000 = 0.0375 → slider stores round(0.0375*100) = 4
        assert frame._fx_fudge.get_int() == 4
        frame.Destroy()

    def test_closure_slider_default(self, app):
        frame = VotraxFrame()
        assert frame._closure.get_int() == 100  # 1.00
        frame.Destroy()

    def test_rd_slider_default(self, app):
        frame = VotraxFrame()
        assert frame._rd.get_int() == 10  # 1.0
        frame.Destroy()

    def test_volume_default(self, app):
        frame = VotraxFrame()
        assert frame._volume_slider.GetValue() == 90
        frame.Destroy()

    def test_boolean_checkboxes_exist_and_default_unchecked(self, app):
        frame = VotraxFrame()
        assert isinstance(frame._dc_block_cb, wx.CheckBox)
        assert isinstance(frame._radiation_filter_cb, wx.CheckBox)
        assert isinstance(frame._enhanced_dsp_cb, wx.CheckBox)
        assert isinstance(frame._enhanced_cb, wx.CheckBox)
        assert frame._dc_block_cb.GetValue() is False
        assert frame._radiation_filter_cb.GetValue() is False
        assert frame._enhanced_dsp_cb.GetValue() is False
        assert frame._enhanced_cb.GetValue() is False
        frame.Destroy()

    def test_mode_choice_defaults_to_text(self, app):
        frame = VotraxFrame()
        assert frame._mode_choice.GetSelection() == 0
        assert frame._current_mode_key() == "text"
        frame.Destroy()

    def test_phoneme_display_is_readonly(self, app):
        frame = VotraxFrame()
        assert isinstance(frame._phoneme_display, wx.TextCtrl)
        assert frame._phoneme_display.GetWindowStyleFlag() & wx.TE_READONLY
        frame.Destroy()

    def test_progress_bar_exists(self, app):
        frame = VotraxFrame()
        assert isinstance(frame._progress, wx.Gauge)
        frame.Destroy()


class TestParameterEnablement:
    def test_rd_disabled_when_enhanced_dsp_off(self, app):
        frame = VotraxFrame()
        frame._enhanced_dsp_cb.SetValue(False)
        frame._sync_rd_enabled()
        assert frame._rd.slider.IsEnabled() is False
        frame.Destroy()

    def test_rd_enabled_when_enhanced_dsp_on(self, app):
        frame = VotraxFrame()
        frame._enhanced_dsp_cb.SetValue(True)
        frame._sync_rd_enabled()
        assert frame._rd.slider.IsEnabled() is True
        frame.Destroy()

    def test_prosody_disabled_in_phoneme_mode(self, app):
        frame = VotraxFrame()
        frame._mode_choice.SetSelection(1)  # phoneme mode
        frame._current_mode = frame._current_mode_key()
        frame._sync_prosody_enabled()
        assert frame._enhanced_cb.IsEnabled() is False
        frame.Destroy()

    def test_prosody_enabled_in_text_mode(self, app):
        frame = VotraxFrame()
        frame._mode_choice.SetSelection(0)
        frame._current_mode = frame._current_mode_key()
        frame._sync_prosody_enabled()
        assert frame._enhanced_cb.IsEnabled() is True
        frame.Destroy()


class TestInputModeBufferring:
    def test_mode_toggle_preserves_other_buffer(self, app):
        frame = VotraxFrame()
        # Text mode is default, edit it
        frame._text_ctrl.SetValue("HELLO")
        # Switch to phoneme mode
        frame._mode_choice.SetSelection(1)
        frame._on_mode_changed(None)
        assert frame._current_mode == "phoneme_string"
        # Phoneme-mode default buffer should now be in the edit box
        assert frame._text_ctrl.GetValue().startswith("I3 M P")
        # Switch back
        frame._mode_choice.SetSelection(0)
        frame._on_mode_changed(None)
        assert frame._text_ctrl.GetValue() == "HELLO"
        frame.Destroy()


class TestPresetRoundtrip:
    def test_current_preset_reflects_widget_state(self, app):
        frame = VotraxFrame()
        frame._master_clock.set_int(360_000)
        frame._fx_fudge.set_int(100)  # 1.00 = as-schematic
        frame._closure.set_int(50)    # 0.50
        frame._dc_block_cb.SetValue(True)
        frame._radiation_filter_cb.SetValue(False)
        frame._enhanced_dsp_cb.SetValue(True)
        frame._rd.set_int(20)         # 2.0
        frame._enhanced_cb.SetValue(True)

        p = frame._current_preset()
        assert p.synth["master_clock"] == 360_000.0
        assert p.synth["fx_fudge"] == pytest.approx(1.0)
        assert p.synth["closure_strength"] == pytest.approx(0.5)
        assert p.synth["dc_block"] is True
        assert p.synth["radiation_filter"] is False
        assert p.synth["enhanced_dsp"] is True
        assert p.synth["rd"] == pytest.approx(2.0)
        assert p.tts["enhanced"] is True
        frame.Destroy()

    def test_apply_preset_populates_widgets(self, app):
        frame = VotraxFrame()
        p = Preset(
            name="X",
            mode="phoneme_string",
            input="AH STOP",
            synth={
                "master_clock": 540_000.0,
                "fx_fudge": 1.0,
                "closure_strength": 0.25,
                "dc_block": True,
                "radiation_filter": True,
                "enhanced_dsp": True,
                "rd": 1.7,
            },
            tts={"enhanced": True},
        )
        frame._apply_preset(p)
        assert frame._master_clock.get_int() == 540_000
        assert frame._fx_fudge.get_int() == 100
        assert frame._closure.get_int() == 25
        assert frame._dc_block_cb.GetValue() is True
        assert frame._radiation_filter_cb.GetValue() is True
        assert frame._enhanced_dsp_cb.GetValue() is True
        assert frame._rd.get_int() == 17
        assert frame._enhanced_cb.GetValue() is True
        assert frame._current_mode == "phoneme_string"
        assert frame._text_ctrl.GetValue() == "AH STOP"
        frame.Destroy()

    def test_factory_presets_load_via_gui(self, app):
        """Every factory preset should apply cleanly through _apply_preset."""
        frame = VotraxFrame()
        for f in sorted(factory_presets_dir().glob("*.json")):
            p = load_preset(f)
            frame._apply_preset(p)  # should not raise
        frame.Destroy()

    def test_preset_picker_populated_with_factory_entries(self, app):
        frame = VotraxFrame()
        labels = [lbl for lbl, _ in frame._preset_entries]
        assert any(lbl.startswith("[F] ") for lbl in labels)
        frame.Destroy()


class TestAudioProcessing:
    def test_process_audio_returns_float32(self, app):
        frame = VotraxFrame()
        audio = np.random.RandomState(0).randn(4000) * 0.1
        out = frame._process_audio(audio, source_sclock=40_000.0)
        assert out.dtype == np.float32
        assert len(out) > 0
        frame.Destroy()

    def test_process_audio_volume_zero(self, app):
        frame = VotraxFrame()
        frame._volume_slider.SetValue(0)
        audio = np.random.RandomState(1).randn(4000) * 0.1
        out = frame._process_audio(audio, source_sclock=40_000.0)
        assert np.allclose(out, 0.0)
        frame.Destroy()

    def test_process_audio_resamples_custom_sclock(self, app):
        """With source_sclock=20kHz, output should be ~2x longer at 44.1kHz."""
        frame = VotraxFrame()
        audio = np.random.RandomState(2).randn(4000) * 0.1
        out = frame._process_audio(audio, source_sclock=20_000.0)
        # 4000 samples @ 20 kHz = 0.2 s → 8820 @ 44.1 kHz (exact rational upsample)
        assert abs(len(out) - 8820) < 50
        frame.Destroy()

    def test_process_audio_empty_input(self, app):
        frame = VotraxFrame()
        out = frame._process_audio(np.array([], dtype=np.float64), source_sclock=40_000.0)
        assert out.dtype == np.float32
        assert len(out) == 0
        frame.Destroy()


class TestPresetsAffectRendering:
    """Regression: the GUI's Load → Speak pipeline must actually honor the
    loaded preset's parameters. A user report had 'none of the presets change
    anything' — this test locks that down."""

    def _render_with(self, frame, preset):
        frame._apply_preset(preset)
        audio, sclock, names, err = frame._render_raw()
        assert err is None, f"Preset {preset.name!r} failed to render: {err}"
        return np.asarray(audio), sclock

    def test_master_clock_changes_output_sclock(self, app):
        frame = VotraxFrame()
        default_p = load_preset(factory_presets_dir() / "default.json")
        slow_p = load_preset(factory_presets_dir() / "slow_robot.json")
        _, default_sclock = self._render_with(frame, default_p)
        _, slow_sclock = self._render_with(frame, slow_p)
        # default is master_clock=720k → sclock 40k; slow_robot is 360k → 20k
        assert default_sclock == pytest.approx(40_000.0, rel=0.01)
        assert slow_sclock == pytest.approx(20_000.0, rel=0.01)
        frame.Destroy()

    def test_as_schematic_muffles_audio_energy(self, app):
        frame = VotraxFrame()
        default_p = load_preset(factory_presets_dir() / "default.json")
        muffled_p = load_preset(factory_presets_dir() / "as_schematic.json")
        d_audio, _ = self._render_with(frame, default_p)
        m_audio, _ = self._render_with(frame, muffled_p)
        # The 150 Hz as-schematic cutoff should remove most voice energy.
        d_rms = float(np.sqrt(np.mean(d_audio ** 2)))
        m_rms = float(np.sqrt(np.mean(m_audio ** 2)))
        assert m_rms < d_rms * 0.5, (
            f"as_schematic RMS {m_rms:.4f} not clearly below default "
            f"{d_rms:.4f} — the preset is not affecting rendering"
        )
        frame.Destroy()

    def test_enhanced_dsp_preset_switches_backend(self, app):
        frame = VotraxFrame()
        natural_p = load_preset(factory_presets_dir() / "natural_lf.json")
        frame._apply_preset(natural_p)
        # _render_raw should succeed; build_tts should route through py_emu
        audio, sclock, names, err = frame._render_raw()
        assert err is None
        assert len(audio) > 0
        assert sclock == pytest.approx(40_000.0)
        frame.Destroy()


class TestPresetAutoLoad:
    def test_choice_selection_triggers_load(self, app):
        """Selecting a preset in the Choice should auto-load it (not require
        a separate Load button click)."""
        frame = VotraxFrame()
        # Find the slow_robot entry in the picker
        target_idx = None
        for i, (label, _) in enumerate(frame._preset_entries):
            if "Slow robot" in label:
                target_idx = i
                break
        assert target_idx is not None, "slow_robot preset not in picker"
        frame._preset_choice.SetSelection(target_idx)
        # Fire the EVT_CHOICE handler directly (SetSelection doesn't auto-fire it)
        frame.on_load_preset(None)
        # After loading slow_robot, master_clock should be 360k
        assert frame._master_clock.get_int() == 360_000
        frame.Destroy()


class TestRenderRaw:
    def test_render_raw_text_mode(self, app):
        frame = VotraxFrame()
        frame._mode_choice.SetSelection(0)
        frame._current_mode = "text"
        frame._text_ctrl.SetValue("hi")
        audio, sclock, names, err = frame._render_raw()
        assert err is None
        assert len(audio) > 0
        assert sclock > 0
        assert names  # phoneme display string
        frame.Destroy()

    def test_render_raw_phoneme_mode(self, app):
        frame = VotraxFrame()
        frame._mode_choice.SetSelection(1)
        frame._current_mode = "phoneme_string"
        frame._text_ctrl.SetValue("AH STOP")
        audio, sclock, names, err = frame._render_raw()
        assert err is None
        assert len(audio) > 0
        assert sclock > 0
        frame.Destroy()

    def test_render_raw_empty_input(self, app):
        frame = VotraxFrame()
        frame._text_ctrl.SetValue("")
        audio, sclock, names, err = frame._render_raw()
        assert err is not None  # surfaced as status message
        assert len(audio) == 0
        frame.Destroy()
