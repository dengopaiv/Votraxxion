"""wxPython GUI for the Votrax SC-01A music-production workbench.

Launch with: python -m pyvotrax   OR   python pyvotrax/gui.py

The window is a compact parameter workbench (not a text editor): a small input
edit box with a Text / Phoneme-string mode toggle, and a dense parameter panel
exposing every knob the backend supports — master clock, FX cutoff fudge,
closure strength, dc_block, radiation_filter, enhanced-DSP path + LF voice
quality (Rd), TTS prosody mode, output gain. Factory and user presets can be
loaded and saved from the preset picker at the top.
"""

import os as _os
import sys as _sys
import threading
from math import gcd
from pathlib import Path

import numpy as np
import wx

if __package__:
    from .tts import VotraxTTS
    from .synth import VotraxSynthesizer
    from .constants import MASTER_CLOCK as _DEFAULT_MASTER_CLOCK
    from .phonemes import code_to_name
    from .presets import (
        Preset,
        load_preset,
        save_preset,
        user_presets_dir,
        factory_presets_dir,
        preset_filename,
    )
else:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pyvotrax.tts import VotraxTTS
    from pyvotrax.synth import VotraxSynthesizer
    from pyvotrax.constants import MASTER_CLOCK as _DEFAULT_MASTER_CLOCK
    from pyvotrax.phonemes import code_to_name
    from pyvotrax.presets import (
        Preset,
        load_preset,
        save_preset,
        user_presets_dir,
        factory_presets_dir,
        preset_filename,
    )

_HAS_SOUNDDEVICE = False
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    sd = None


_PLAYBACK_RATE = 44_100  # device-side rate; any synth.sclock is resampled to this
_DEFAULT_FX_FUDGE = 150.0 / 4000.0
# Trailing silence appended to every playback buffer so the sound card's
# hardware output buffer drains through zeros before PortAudio tears the
# stream down. Without this, Windows WASAPI/DirectSound can produce an
# audible click on stream-close even when the last signal sample is zero.
_PLAYBACK_TRAILING_SILENCE_MS = 150

# Slider scaling: wx.Slider is integer-only. Each scalar param has a (min, max,
# step, decoder) tuple. decoder maps slider int → float backend value.
_MASTER_CLOCK_RANGE = (180_000, 1_440_000)  # 0.25x to 2x of nominal 720k
_FX_FUDGE_STEPS = 200       # 0.00–2.00 in 0.01 increments
_CLOSURE_STEPS = 200        # 0.00–2.00
_RD_MIN, _RD_MAX = 3, 27    # 0.3–2.7 in 0.1 increments


class _TokenEntry:
    """Helper: a labeled slider + float readout + preset-value row."""

    def __init__(self, parent, sizer, label, int_min, int_max, int_default, formatter):
        """
        Args:
            formatter: callable(int_val) -> displayed string (e.g. lambda v: f"{v/100:.2f}")
        """
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.label = wx.StaticText(parent, label=label, size=(110, -1))
        self.slider = wx.Slider(parent, value=int_default, minValue=int_min, maxValue=int_max)
        self.value_txt = wx.StaticText(parent, label=formatter(int_default), size=(70, -1))
        row.Add(self.label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=4)
        row.Add(self.slider, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=4)
        row.Add(self.value_txt, flag=wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(row, flag=wx.EXPAND | wx.ALL, border=2)
        self._formatter = formatter
        self.slider.Bind(wx.EVT_SLIDER, self._on_change)

    def _on_change(self, _evt):
        self.value_txt.SetLabel(self._formatter(self.slider.GetValue()))

    def set_int(self, v):
        self.slider.SetValue(int(v))
        self.value_txt.SetLabel(self._formatter(int(v)))

    def get_int(self):
        return self.slider.GetValue()


class VotraxFrame(wx.Frame):
    """Main application window: Votrax music-production workbench."""

    def __init__(self, parent=None):
        super().__init__(parent, title="Votrax SC-01A Workbench", size=(920, 620))

        self._player_thread = None
        self._stop_flag = threading.Event()

        # Per-mode input buffers so toggling modes doesn't clobber the other.
        self._buffers = {
            "text": "Hello world.",
            "phoneme_string": "I3 M P O1:2 R:1 T AH N T STOP",
        }
        self._current_mode = "text"

        self._build_menu()
        self._build_ui()
        self._refresh_preset_picker()
        self._sync_rd_enabled()
        self._sync_prosody_enabled()

        self.Bind(wx.EVT_CLOSE, self._on_close)

        if not _HAS_SOUNDDEVICE:
            wx.CallAfter(self._warn_no_sounddevice)

    # ------------------------------------------------------------------ UI

    def _warn_no_sounddevice(self):
        wx.MessageBox(
            "sounddevice is not installed.\n"
            "Playback is disabled (WAV export still works). "
            "Install with: pip install sounddevice",
            "Missing Dependency",
            wx.OK | wx.ICON_WARNING,
        )

    def _build_menu(self):
        menubar = wx.MenuBar()

        file_menu = wx.Menu()
        item_load_preset = file_menu.Append(wx.ID_ANY, "&Load Preset...\tCtrl+O")
        item_save_preset = file_menu.Append(wx.ID_ANY, "Save Preset &As...\tCtrl+S")
        file_menu.AppendSeparator()
        item_export_wav = file_menu.Append(wx.ID_ANY, "Export &WAV...\tCtrl+E")
        file_menu.AppendSeparator()
        item_exit = file_menu.Append(wx.ID_EXIT, "E&xit\tAlt+F4")
        menubar.Append(file_menu, "&File")

        edit_menu = wx.Menu()
        edit_menu.Append(wx.ID_CUT, "Cu&t\tCtrl+X")
        edit_menu.Append(wx.ID_COPY, "&Copy\tCtrl+C")
        edit_menu.Append(wx.ID_PASTE, "&Paste\tCtrl+V")
        edit_menu.AppendSeparator()
        item_select_all = edit_menu.Append(wx.ID_SELECTALL, "Select &All\tCtrl+A")
        menubar.Append(edit_menu, "&Edit")

        speech_menu = wx.Menu()
        item_speak = speech_menu.Append(wx.ID_ANY, "&Speak\tF5")
        item_stop = speech_menu.Append(wx.ID_ANY, "S&top\tEscape")
        menubar.Append(speech_menu, "&Speech")

        self.SetMenuBar(menubar)

        self.Bind(wx.EVT_MENU, self.on_load_preset, item_load_preset)
        self.Bind(wx.EVT_MENU, self.on_save_preset_as, item_save_preset)
        self.Bind(wx.EVT_MENU, self.on_export_wav, item_export_wav)
        self.Bind(wx.EVT_MENU, self.on_exit, item_exit)
        self.Bind(wx.EVT_MENU, self.on_select_all, item_select_all)
        self.Bind(wx.EVT_MENU, self.on_speak, item_speak)
        self.Bind(wx.EVT_MENU, self.on_stop, item_stop)

    def _build_ui(self):
        panel = wx.Panel(self)
        root = wx.BoxSizer(wx.VERTICAL)

        # --- Preset bar -------------------------------------------------
        preset_bar = wx.BoxSizer(wx.HORIZONTAL)
        preset_bar.Add(
            wx.StaticText(panel, label="Preset:"),
            flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5,
        )
        self._preset_choice = wx.Choice(panel, choices=[])
        preset_bar.Add(self._preset_choice, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        self._btn_load_preset = wx.Button(panel, label="Load")
        self._btn_save_preset = wx.Button(panel, label="Save As…")
        self._btn_delete_preset = wx.Button(panel, label="Delete")
        preset_bar.Add(self._btn_load_preset, flag=wx.RIGHT, border=3)
        preset_bar.Add(self._btn_save_preset, flag=wx.RIGHT, border=3)
        preset_bar.Add(self._btn_delete_preset)
        root.Add(preset_bar, flag=wx.EXPAND | wx.ALL, border=6)

        self._btn_load_preset.Bind(wx.EVT_BUTTON, self.on_load_preset)
        self._btn_save_preset.Bind(wx.EVT_BUTTON, self.on_save_preset_as)
        self._btn_delete_preset.Bind(wx.EVT_BUTTON, self.on_delete_preset)
        # Auto-load on selection change: the two-step (select + click Load) flow
        # is easy to forget, so selecting a preset immediately applies it.
        self._preset_choice.Bind(wx.EVT_CHOICE, self.on_load_preset)

        # --- Main split: input pane | parameter pane --------------------
        body = wx.BoxSizer(wx.HORIZONTAL)

        # Left pane
        left = wx.BoxSizer(wx.VERTICAL)
        mode_row = wx.BoxSizer(wx.HORIZONTAL)
        mode_row.Add(
            wx.StaticText(panel, label="Input mode:"),
            flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5,
        )
        self._mode_choice = wx.Choice(panel, choices=["Text", "Phoneme string"])
        self._mode_choice.SetSelection(0)
        mode_row.Add(self._mode_choice, flag=wx.ALIGN_CENTER_VERTICAL)
        left.Add(mode_row, flag=wx.EXPAND | wx.BOTTOM, border=4)

        self._text_ctrl = wx.TextCtrl(
            panel,
            value=self._buffers[self._current_mode],
            style=wx.TE_MULTILINE | wx.TE_RICH2 | wx.HSCROLL,
        )
        left.Add(self._text_ctrl, proportion=1, flag=wx.EXPAND)

        phoneme_row = wx.BoxSizer(wx.HORIZONTAL)
        phoneme_row.Add(
            wx.StaticText(panel, label="Phonemes:"),
            flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5,
        )
        self._phoneme_display = wx.TextCtrl(panel, style=wx.TE_READONLY)
        phoneme_row.Add(self._phoneme_display, proportion=1, flag=wx.EXPAND)
        left.Add(phoneme_row, flag=wx.EXPAND | wx.TOP, border=4)

        self._progress = wx.Gauge(panel, range=100)
        left.Add(self._progress, flag=wx.EXPAND | wx.TOP, border=4)

        transport = wx.BoxSizer(wx.HORIZONTAL)
        self._btn_speak = wx.Button(panel, label="Speak (F5)")
        btn_stop = wx.Button(panel, label="Stop (Esc)")
        btn_export = wx.Button(panel, label="Export WAV…")
        transport.Add(self._btn_speak, flag=wx.RIGHT, border=5)
        transport.Add(btn_stop, flag=wx.RIGHT, border=5)
        transport.Add(btn_export)
        left.Add(transport, flag=wx.TOP, border=6)

        if not _HAS_SOUNDDEVICE:
            self._btn_speak.Disable()
        self._btn_speak.Bind(wx.EVT_BUTTON, self.on_speak)
        btn_stop.Bind(wx.EVT_BUTTON, self.on_stop)
        btn_export.Bind(wx.EVT_BUTTON, self.on_export_wav)
        self._mode_choice.Bind(wx.EVT_CHOICE, self._on_mode_changed)

        body.Add(left, proportion=3, flag=wx.EXPAND | wx.RIGHT, border=8)

        # Right pane: parameter panel
        right = wx.BoxSizer(wx.VERTICAL)

        clock_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Clock")
        self._master_clock = _TokenEntry(
            clock_box.GetStaticBox(), clock_box, "Master clock",
            _MASTER_CLOCK_RANGE[0], _MASTER_CLOCK_RANGE[1],
            int(_DEFAULT_MASTER_CLOCK),
            lambda v: f"{v/1000:6.1f} kHz",
        )
        right.Add(clock_box, flag=wx.EXPAND | wx.BOTTOM, border=4)

        filter_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Filters")
        self._fx_fudge = _TokenEntry(
            filter_box.GetStaticBox(), filter_box, "FX fudge",
            1, _FX_FUDGE_STEPS, int(round(_DEFAULT_FX_FUDGE * 100)),
            lambda v: f"{v/100:.3f}",
        )
        self._dc_block_cb = wx.CheckBox(filter_box.GetStaticBox(), label="DC block (20 Hz)")
        filter_box.Add(self._dc_block_cb, flag=wx.LEFT | wx.BOTTOM, border=4)
        self._radiation_filter_cb = wx.CheckBox(filter_box.GetStaticBox(), label="Radiation filter (+6 dB/oct)")
        filter_box.Add(self._radiation_filter_cb, flag=wx.LEFT | wx.BOTTOM, border=4)
        right.Add(filter_box, flag=wx.EXPAND | wx.BOTTOM, border=4)

        closure_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Closure")
        self._closure = _TokenEntry(
            closure_box.GetStaticBox(), closure_box, "Strength",
            0, _CLOSURE_STEPS, 100,  # default 1.00
            lambda v: f"{v/100:.2f}",
        )
        right.Add(closure_box, flag=wx.EXPAND | wx.BOTTOM, border=4)

        dsp_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "DSP path")
        self._enhanced_dsp_cb = wx.CheckBox(dsp_box.GetStaticBox(), label="Enhanced DSP (LF glottal, slower)")
        dsp_box.Add(self._enhanced_dsp_cb, flag=wx.LEFT | wx.BOTTOM, border=4)
        self._rd = _TokenEntry(
            dsp_box.GetStaticBox(), dsp_box, "Rd (voice)",
            _RD_MIN, _RD_MAX, 10,  # default 1.0
            lambda v: f"{v/10:.1f}",
        )
        self._enhanced_dsp_cb.Bind(wx.EVT_CHECKBOX, lambda _e: self._sync_rd_enabled())
        right.Add(dsp_box, flag=wx.EXPAND | wx.BOTTOM, border=4)

        tts_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "TTS prosody")
        self._enhanced_cb = wx.CheckBox(tts_box.GetStaticBox(), label="Enhanced prosody (text mode only)")
        tts_box.Add(self._enhanced_cb, flag=wx.LEFT | wx.BOTTOM, border=4)
        right.Add(tts_box, flag=wx.EXPAND | wx.BOTTOM, border=4)

        out_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Output")
        self._volume_slider = wx.Slider(out_box.GetStaticBox(), value=90, minValue=0, maxValue=100)
        volume_row = wx.BoxSizer(wx.HORIZONTAL)
        volume_row.Add(
            wx.StaticText(out_box.GetStaticBox(), label="Gain", size=(110, -1)),
            flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=4,
        )
        volume_row.Add(self._volume_slider, proportion=1, flag=wx.EXPAND)
        out_box.Add(volume_row, flag=wx.EXPAND | wx.ALL, border=2)
        right.Add(out_box, flag=wx.EXPAND | wx.BOTTOM, border=4)

        body.Add(right, proportion=2, flag=wx.EXPAND)
        root.Add(body, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=6)

        panel.SetSizer(root)

        # Status bar
        self._status = self.CreateStatusBar()
        self._status.SetStatusText("Ready.")

    # ------------------------------------------------------------------ input mode

    def _on_mode_changed(self, _evt):
        # Save the current buffer
        self._buffers[self._current_mode] = self._text_ctrl.GetValue()
        # Swap
        self._current_mode = self._current_mode_key()
        self._text_ctrl.SetValue(self._buffers[self._current_mode])
        self._sync_prosody_enabled()

    def _current_mode_key(self) -> str:
        return "text" if self._mode_choice.GetSelection() == 0 else "phoneme_string"

    def _sync_prosody_enabled(self):
        """Enable Enhanced-prosody checkbox only in text mode (it has no effect in phoneme mode)."""
        self._enhanced_cb.Enable(self._current_mode_key() == "text")

    def _sync_rd_enabled(self):
        """Enable the Rd slider only when Enhanced-DSP is on."""
        self._rd.slider.Enable(self._enhanced_dsp_cb.GetValue())

    # ------------------------------------------------------------------ presets

    def _refresh_preset_picker(self):
        """Repopulate the preset Choice with '[U] name' and '[F] name' entries."""
        self._preset_entries: list[tuple[str, Path]] = []
        user_dir = user_presets_dir()
        factory_dir = factory_presets_dir()
        for f in sorted(user_dir.glob("*.json")):
            try:
                self._preset_entries.append((f"[U] {load_preset(f).name}", f))
            except Exception:
                continue
        if factory_dir.exists():
            for f in sorted(factory_dir.glob("*.json")):
                try:
                    self._preset_entries.append((f"[F] {load_preset(f).name}", f))
                except Exception:
                    continue
        self._preset_choice.Set([label for label, _ in self._preset_entries])
        if self._preset_entries:
            self._preset_choice.SetSelection(0)

    def _current_preset(self) -> Preset:
        """Gather the current UI state into a Preset object."""
        mode_key = self._current_mode_key()
        # Keep the other mode's buffer up to date so Save captures both accurately
        self._buffers[mode_key] = self._text_ctrl.GetValue()

        synth: dict = {
            "master_clock": float(self._master_clock.get_int()),
            "fx_fudge": self._fx_fudge.get_int() / 100.0,
            "closure_strength": self._closure.get_int() / 100.0,
            "dc_block": self._dc_block_cb.GetValue(),
            "radiation_filter": self._radiation_filter_cb.GetValue(),
            "enhanced_dsp": self._enhanced_dsp_cb.GetValue(),
            "rd": self._rd.get_int() / 10.0,
        }
        tts: dict = {"enhanced": self._enhanced_cb.GetValue()}
        return Preset(
            name="Current",
            description="",
            mode=mode_key,
            input=self._text_ctrl.GetValue(),
            synth=synth,
            tts=tts,
        )

    def _apply_preset(self, preset: Preset):
        """Populate all UI widgets from a loaded preset."""
        s = preset.synth
        t = preset.tts
        self._master_clock.set_int(int(s.get("master_clock", _DEFAULT_MASTER_CLOCK)))
        self._fx_fudge.set_int(int(round(s.get("fx_fudge", _DEFAULT_FX_FUDGE) * 100)))
        self._closure.set_int(int(round(s.get("closure_strength", 1.0) * 100)))
        self._dc_block_cb.SetValue(bool(s.get("dc_block", False)))
        self._radiation_filter_cb.SetValue(bool(s.get("radiation_filter", False)))
        self._enhanced_dsp_cb.SetValue(bool(s.get("enhanced_dsp", False)))
        self._rd.set_int(int(round(s.get("rd", 1.0) * 10)))
        self._enhanced_cb.SetValue(bool(t.get("enhanced", False)))

        self._buffers[preset.mode] = preset.input
        self._mode_choice.SetSelection(0 if preset.mode == "text" else 1)
        self._current_mode = preset.mode
        self._text_ctrl.SetValue(preset.input)

        self._sync_rd_enabled()
        self._sync_prosody_enabled()
        self._status.SetStatusText(f"Loaded preset: {preset.name}")

    def on_load_preset(self, _evt):
        idx = self._preset_choice.GetSelection()
        if idx < 0 or idx >= len(self._preset_entries):
            return
        _, path = self._preset_entries[idx]
        try:
            preset = load_preset(path)
        except Exception as e:
            wx.MessageBox(f"Failed to load preset:\n{e}", "Error", wx.OK | wx.ICON_ERROR)
            return
        self._apply_preset(preset)

    def on_save_preset_as(self, _evt):
        preset = self._current_preset()
        with wx.TextEntryDialog(self, "Preset name:", "Save preset as…", "My preset") as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            preset.name = dlg.GetValue().strip() or "Untitled"
        fname = preset_filename(preset.name) + ".json"
        path = user_presets_dir() / fname
        try:
            save_preset(path, preset)
        except Exception as e:
            wx.MessageBox(f"Failed to save preset:\n{e}", "Error", wx.OK | wx.ICON_ERROR)
            return
        self._refresh_preset_picker()
        # Select the newly saved preset
        label = f"[U] {preset.name}"
        for i, (lbl, _) in enumerate(self._preset_entries):
            if lbl == label:
                self._preset_choice.SetSelection(i)
                break
        self._status.SetStatusText(f"Saved preset: {preset.name}")

    def on_delete_preset(self, _evt):
        idx = self._preset_choice.GetSelection()
        if idx < 0 or idx >= len(self._preset_entries):
            return
        label, path = self._preset_entries[idx]
        if not label.startswith("[U] "):
            wx.MessageBox(
                "Factory presets cannot be deleted.",
                "Not allowed",
                wx.OK | wx.ICON_INFORMATION,
            )
            return
        if wx.MessageBox(
            f"Delete preset {label[4:]!r}?",
            "Confirm",
            wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        try:
            _os.remove(path)
        except Exception as e:
            wx.MessageBox(f"Failed to delete:\n{e}", "Error", wx.OK | wx.ICON_ERROR)
            return
        self._refresh_preset_picker()
        self._status.SetStatusText(f"Deleted preset: {label[4:]}")

    # ------------------------------------------------------------------ synthesis

    def _render_raw(self):
        """Render raw audio from the current UI state at synth.sclock.

        Returns: (audio_float64, sclock, phoneme_names_display, error_str_or_None)
        """
        preset = self._current_preset()
        if not preset.input.strip():
            return np.array([], dtype=np.float64), 0.0, "", "Nothing to speak."
        try:
            if preset.mode == "text":
                tts = preset.build_tts()
                phonemes = tts.text_to_phonemes(preset.input)
                names = " ".join(code_to_name(c) for c, _ in phonemes)
                audio = tts.speak(preset.input)
                sclock = tts._synth.sclock
            else:
                synth = preset.build_synthesizer()
                from .phonemes import parse_phoneme_sequence
                tokens = parse_phoneme_sequence(preset.input)
                names = " ".join(code_to_name(c) for c, _ in tokens)
                audio = synth.synthesize_phoneme_string(preset.input)
                sclock = synth.sclock
            return np.asarray(audio, dtype=np.float64), float(sclock), names, None
        except Exception as e:
            return np.array([], dtype=np.float64), 0.0, "", str(e)

    def _process_audio(self, audio: np.ndarray, source_sclock: float = 0.0) -> np.ndarray:
        """Resample raw audio to 44.1 kHz and apply output gain. Returns float32.

        ``source_sclock=0`` is treated as "already at _PLAYBACK_RATE" (no resample).
        """
        if len(audio) == 0:
            return np.array([], dtype=np.float32)
        if source_sclock and int(round(source_sclock)) != _PLAYBACK_RATE:
            from scipy.signal import resample_poly
            src = int(round(source_sclock))
            g = gcd(_PLAYBACK_RATE, src)
            audio = resample_poly(audio, _PLAYBACK_RATE // g, src // g)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak * 0.9
        gain = self._volume_slider.GetValue() / 100.0
        audio = audio * gain
        return audio.astype(np.float32)

    # ------------------------------------------------------------------ playback / export

    def on_speak(self, _evt):
        if not _HAS_SOUNDDEVICE:
            return
        self.on_stop(None)
        if self._player_thread is not None:
            self._player_thread.join(timeout=1)
            self._player_thread = None
        self._stop_flag.clear()
        self._player_thread = threading.Thread(target=self._play_audio, daemon=True)
        self._player_thread.start()

    def _play_audio(self):
        import time as _time
        try:
            audio, sclock, names, err = self._render_raw()
            if err:
                wx.CallAfter(self._status.SetStatusText, err)
                return
            wx.CallAfter(self._phoneme_display.SetValue, names)
            if len(audio) == 0 or self._stop_flag.is_set():
                return
            processed = self._process_audio(audio, sclock)
            if len(processed) == 0 or self._stop_flag.is_set():
                return

            # sd.play is simpler and doesn't require a user-provided callback
            # running in the audio thread. Earlier versions used an OutputStream
            # + callback that invoked wx.CallAfter from the real-time audio
            # thread on every 2048-frame block; that was causing silent audio
            # on longer renders (slow_robot / breathy_lf). Progress polling and
            # cancellation now happen entirely on this worker thread.
            #
            # Append a short silence buffer so the hardware output drains
            # through zeros before the stream closes (end-of-stream click
            # mitigation; see _PLAYBACK_TRAILING_SILENCE_MS comment above).
            pad = int(_PLAYBACK_TRAILING_SILENCE_MS * _PLAYBACK_RATE / 1000)
            padded = np.concatenate([processed, np.zeros(pad, dtype=processed.dtype)])

            total_sec = len(processed) / _PLAYBACK_RATE
            wx.CallAfter(self._status.SetStatusText, f"Playing {total_sec:.2f}s…")
            sd.play(padded, samplerate=_PLAYBACK_RATE, blocking=False)

            start_time = _time.time()
            while True:
                if self._stop_flag.is_set():
                    sd.stop()
                    break
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    break
                elapsed = _time.time() - start_time
                pct = min(100, int(100 * elapsed / max(total_sec, 1e-6)))
                wx.CallAfter(self._progress.SetValue, pct)
                sd.sleep(50)
        except Exception as e:
            wx.CallAfter(
                wx.MessageBox,
                f"Playback error: {e}",
                "Error",
                wx.OK | wx.ICON_ERROR,
            )
        finally:
            wx.CallAfter(self._progress.SetValue, 0)
            wx.CallAfter(self._status.SetStatusText, "Ready.")

    def on_stop(self, _evt):
        self._stop_flag.set()
        if _HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass

    def on_export_wav(self, _evt):
        audio, sclock, names, err = self._render_raw()
        if err:
            wx.MessageBox(err, "Nothing to export", wx.OK | wx.ICON_INFORMATION)
            return
        self._phoneme_display.SetValue(names)

        rates = ["22050", "44100", "48000", "96000"]
        depths = ["16-bit int", "32-bit int", "32-bit float"]
        with _ExportDialog(self, rates, depths) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            rate = int(dlg.rate_choice.GetStringSelection())
            depth = dlg.depth_choice.GetStringSelection()
            path = dlg.path

        try:
            from scipy.io import wavfile
            from scipy.signal import resample_poly

            processed = np.asarray(audio, dtype=np.float64)
            src = int(round(sclock))
            if src != rate:
                g = gcd(rate, src)
                processed = resample_poly(processed, rate // g, src // g)

            peak = float(np.max(np.abs(processed))) if len(processed) else 0.0
            if peak > 0:
                processed = processed / peak * 0.9
            processed = processed * (self._volume_slider.GetValue() / 100.0)

            if depth == "16-bit int":
                data = np.clip(processed * 32767, -32767, 32767).astype(np.int16)
            elif depth == "32-bit int":
                data = np.clip(processed * 2147483647, -2147483647, 2147483647).astype(np.int32)
            else:  # 32-bit float
                data = np.clip(processed, -1.0, 1.0).astype(np.float32)
            wavfile.write(path, rate, data)
        except Exception as e:
            wx.MessageBox(f"Error saving WAV:\n{e}", "Error", wx.OK | wx.ICON_ERROR)
            return
        self._status.SetStatusText(f"Exported {path} ({rate} Hz, {depth})")

    # ------------------------------------------------------------------ misc

    def on_select_all(self, _evt):
        self._text_ctrl.SelectAll()

    def on_exit(self, _evt):
        self.Close()

    def _on_close(self, event):
        self.on_stop(None)
        if self._player_thread is not None:
            self._player_thread.join(timeout=1)
            self._player_thread = None
        event.Skip()


class _ExportDialog(wx.Dialog):
    """Dialog for WAV export options (rate + bit depth)."""

    def __init__(self, parent, rates: list[str], depths: list[str]):
        super().__init__(parent, title="Export WAV", size=(380, 200))
        self.path = ""
        sizer = wx.BoxSizer(wx.VERTICAL)

        rate_row = wx.BoxSizer(wx.HORIZONTAL)
        rate_row.Add(wx.StaticText(self, label="Sample rate:"), flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)
        self.rate_choice = wx.Choice(self, choices=rates)
        self.rate_choice.SetStringSelection("44100")
        rate_row.Add(self.rate_choice, proportion=1, flag=wx.EXPAND)
        sizer.Add(rate_row, flag=wx.EXPAND | wx.ALL, border=8)

        depth_row = wx.BoxSizer(wx.HORIZONTAL)
        depth_row.Add(wx.StaticText(self, label="Bit depth:"), flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)
        self.depth_choice = wx.Choice(self, choices=depths)
        self.depth_choice.SetStringSelection("16-bit int")
        depth_row.Add(self.depth_choice, proportion=1, flag=wx.EXPAND)
        sizer.Add(depth_row, flag=wx.EXPAND | wx.ALL, border=8)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        btn_save = wx.Button(self, wx.ID_OK, "Choose file and save…")
        btn_row.AddStretchSpacer(1)
        btn_row.Add(btn_cancel, flag=wx.RIGHT, border=6)
        btn_row.Add(btn_save)
        sizer.Add(btn_row, flag=wx.EXPAND | wx.ALL, border=8)
        self.SetSizer(sizer)

        btn_save.Bind(wx.EVT_BUTTON, self._on_save)

    def _on_save(self, _evt):
        with wx.FileDialog(
            self, "Save WAV File",
            wildcard="WAV files (*.wav)|*.wav",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            self.path = dlg.GetPath()
        self.EndModal(wx.ID_OK)


def main():
    app = wx.App()
    frame = VotraxFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
