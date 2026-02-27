"""wxPython GUI for Votrax SC-01A text-to-speech.

Launch with: python -m pyvotrax   OR   python pyvotrax/gui.py
"""

import os as _os
import sys as _sys
import threading
from math import gcd

import numpy as np
import wx

if __package__:
    from .tts import VotraxTTS
    from .synth import VotraxSynthesizer
    from .filters import SCLOCK
    from .phonemes import code_to_name
else:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from pyvotrax.tts import VotraxTTS
    from pyvotrax.synth import VotraxSynthesizer
    from pyvotrax.filters import SCLOCK
    from pyvotrax.phonemes import code_to_name

# Sounddevice availability check at module level
_HAS_SOUNDDEVICE = False
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    sd = None


class VotraxFrame(wx.Frame):
    """Main application window."""

    def __init__(self, parent=None):
        super().__init__(parent, title="Votrax SC-01A Text-to-Speech", size=(650, 550))

        self._tts = VotraxTTS()
        self._player_thread = None
        self._stop_flag = threading.Event()
        self._stream = None

        self._build_menu()
        self._build_ui()

        self.Bind(wx.EVT_CLOSE, self._on_close)

        if not _HAS_SOUNDDEVICE:
            wx.CallAfter(self._warn_no_sounddevice)

    def _warn_no_sounddevice(self):
        wx.MessageBox(
            "sounddevice is not installed.\n"
            "Playback is disabled. Install with: pip install sounddevice",
            "Missing Dependency",
            wx.OK | wx.ICON_WARNING,
        )

    def _build_menu(self):
        menubar = wx.MenuBar()

        # File menu
        file_menu = wx.Menu()
        item_open = file_menu.Append(wx.ID_OPEN, "&Open Text\tCtrl+O")
        item_save_text = file_menu.Append(wx.ID_SAVE, "&Save Text\tCtrl+S")
        file_menu.AppendSeparator()
        item_save_wav = file_menu.Append(wx.ID_ANY, "Save &WAV...\tCtrl+Shift+S")
        file_menu.AppendSeparator()
        item_exit = file_menu.Append(wx.ID_EXIT, "E&xit\tAlt+F4")
        menubar.Append(file_menu, "&File")

        # Edit menu
        edit_menu = wx.Menu()
        edit_menu.Append(wx.ID_CUT, "Cu&t\tCtrl+X")
        edit_menu.Append(wx.ID_COPY, "&Copy\tCtrl+C")
        edit_menu.Append(wx.ID_PASTE, "&Paste\tCtrl+V")
        edit_menu.AppendSeparator()
        item_select_all = edit_menu.Append(wx.ID_SELECTALL, "Select &All\tCtrl+A")
        menubar.Append(edit_menu, "&Edit")

        # Speech menu
        speech_menu = wx.Menu()
        item_speak = speech_menu.Append(wx.ID_ANY, "&Speak\tF5")
        item_stop = speech_menu.Append(wx.ID_ANY, "S&top\tEscape")
        menubar.Append(speech_menu, "&Speech")

        # Options menu
        options_menu = wx.Menu()
        self._menu_enhanced = options_menu.AppendCheckItem(
            wx.ID_ANY, "&Enhanced Mode"
        )
        menubar.Append(options_menu, "&Options")

        self.SetMenuBar(menubar)

        # Bind menu events
        self.Bind(wx.EVT_MENU, self.on_open, item_open)
        self.Bind(wx.EVT_MENU, self.on_save_text, item_save_text)
        self.Bind(wx.EVT_MENU, self.on_save_wav, item_save_wav)
        self.Bind(wx.EVT_MENU, self.on_exit, item_exit)
        self.Bind(wx.EVT_MENU, self.on_select_all, item_select_all)
        self.Bind(wx.EVT_MENU, self.on_speak, item_speak)
        self.Bind(wx.EVT_MENU, self.on_stop, item_stop)
        self.Bind(wx.EVT_MENU, self._on_menu_enhanced, self._menu_enhanced)

    def _build_ui(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Text editor
        self._text_ctrl = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_RICH2 | wx.HSCROLL
        )
        sizer.Add(self._text_ctrl, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        # Phoneme display (read-only)
        phoneme_label = wx.StaticText(panel, label="Phonemes:")
        self._phoneme_display = wx.TextCtrl(
            panel, style=wx.TE_READONLY
        )
        phoneme_sizer = wx.BoxSizer(wx.HORIZONTAL)
        phoneme_sizer.Add(phoneme_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        phoneme_sizer.Add(self._phoneme_display, proportion=1, flag=wx.EXPAND)
        sizer.Add(phoneme_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=5)

        # Progress bar
        self._progress = wx.Gauge(panel, range=100)
        sizer.Add(self._progress, flag=wx.EXPAND | wx.ALL, border=5)

        # Sliders row
        slider_sizer = wx.FlexGridSizer(rows=2, cols=4, vgap=4, hgap=10)
        slider_sizer.AddGrowableCol(1)
        slider_sizer.AddGrowableCol(3)

        # Rate slider
        slider_sizer.Add(
            wx.StaticText(panel, label="Rate:"),
            flag=wx.ALIGN_CENTER_VERTICAL,
        )
        self._rate_slider = wx.Slider(panel, value=50, minValue=1, maxValue=100)
        slider_sizer.Add(self._rate_slider, flag=wx.EXPAND)

        # Pitch slider
        slider_sizer.Add(
            wx.StaticText(panel, label="Pitch:"),
            flag=wx.ALIGN_CENTER_VERTICAL,
        )
        self._pitch_slider = wx.Slider(panel, value=50, minValue=1, maxValue=100)
        slider_sizer.Add(self._pitch_slider, flag=wx.EXPAND)

        # Volume slider
        slider_sizer.Add(
            wx.StaticText(panel, label="Volume:"),
            flag=wx.ALIGN_CENTER_VERTICAL,
        )
        self._volume_slider = wx.Slider(panel, value=100, minValue=0, maxValue=100)
        slider_sizer.Add(self._volume_slider, flag=wx.EXPAND)

        # Enhanced checkbox
        slider_sizer.Add((0, 0))  # spacer
        self._enhanced_cb = wx.CheckBox(panel, label="Enhanced mode")
        slider_sizer.Add(self._enhanced_cb, flag=wx.ALIGN_CENTER_VERTICAL)

        sizer.Add(slider_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=5)

        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._btn_speak = wx.Button(panel, label="Speak")
        btn_stop = wx.Button(panel, label="Stop")
        btn_save_wav = wx.Button(panel, label="Save WAV...")

        btn_sizer.Add(self._btn_speak, flag=wx.RIGHT, border=5)
        btn_sizer.Add(btn_stop, flag=wx.RIGHT, border=5)
        btn_sizer.Add(btn_save_wav)

        sizer.Add(btn_sizer, flag=wx.ALL, border=5)

        panel.SetSizer(sizer)

        # Disable Speak if no sounddevice
        if not _HAS_SOUNDDEVICE:
            self._btn_speak.Disable()

        # Bind button events
        self._btn_speak.Bind(wx.EVT_BUTTON, self.on_speak)
        btn_stop.Bind(wx.EVT_BUTTON, self.on_stop)
        btn_save_wav.Bind(wx.EVT_BUTTON, self.on_save_wav)

        # Sync enhanced checkbox with menu
        self._enhanced_cb.Bind(wx.EVT_CHECKBOX, self._on_enhanced_cb)

    # --- Enhanced mode sync ---

    def _on_menu_enhanced(self, event):
        checked = self._menu_enhanced.IsChecked()
        self._enhanced_cb.SetValue(checked)
        self._recreate_tts(checked)

    def _on_enhanced_cb(self, event):
        checked = self._enhanced_cb.GetValue()
        self._menu_enhanced.Check(checked)
        self._recreate_tts(checked)

    def _recreate_tts(self, enhanced):
        self._tts = VotraxTTS(enhanced=enhanced)

    # --- Audio processing ---

    def _process_audio(self, audio):
        """Apply pitch, rate, and volume to raw audio.

        Args:
            audio: Raw audio at SCLOCK sample rate (float64).

        Returns:
            float32 array at 44100 Hz ready for playback.
        """
        from scipy.signal import resample_poly

        if len(audio) == 0:
            return np.array([], dtype=np.float32)

        pitch_val = self._pitch_slider.GetValue()
        rate_val = self._rate_slider.GetValue()
        volume_val = self._volume_slider.GetValue()

        # Pitch: resample from shifted source rate to 44100
        # pitch_factor 0.5 (low) to 1.5 (high)
        pitch_factor = 0.5 + (pitch_val / 100.0)
        source_rate = int(SCLOCK * pitch_factor)
        target_rate = 44100

        g = gcd(target_rate, source_rate)
        up = target_rate // g
        down = source_rate // g
        resampled = resample_poly(audio, up, down)

        # Rate: second resample to change duration
        # speed_factor 0.5 (slow) to 1.5 (fast)
        speed_factor = 0.5 + (rate_val / 100.0)
        speed_down = int(1000 * speed_factor)
        if speed_down != 1000:
            resampled = resample_poly(resampled, 1000, speed_down)

        # Volume: normalize then scale
        peak = np.max(np.abs(resampled))
        if peak > 0:
            resampled = resampled / peak * 0.9

        resampled = resampled * (volume_val / 100.0)

        return resampled.astype(np.float32)

    # --- Event handlers ---

    def on_speak(self, event):
        """Synthesize and play text in a background thread."""
        if not _HAS_SOUNDDEVICE:
            return

        text = self._text_ctrl.GetValue().strip()
        if not text:
            return

        self.on_stop(None)

        # Join old thread before starting new one
        if self._player_thread is not None:
            self._player_thread.join(timeout=1)
            self._player_thread = None

        self._stop_flag.clear()
        self._player_thread = threading.Thread(
            target=self._play_audio, args=(text,), daemon=True
        )
        self._player_thread.start()

    def _play_audio(self, text):
        """Background thread: synthesize and play audio with progress."""
        try:
            # Convert text to phonemes and update display
            phonemes = self._tts.text_to_phonemes(text)
            if not phonemes or self._stop_flag.is_set():
                return

            # Show phoneme names
            names = " ".join(code_to_name(code) for code, _ in phonemes)
            wx.CallAfter(self._phoneme_display.SetValue, names)

            # Synthesize audio
            audio = self._tts._synth.synthesize(phonemes)
            if len(audio) == 0 or self._stop_flag.is_set():
                return

            # Process with pitch/rate/volume
            processed = self._process_audio(audio)
            if len(processed) == 0 or self._stop_flag.is_set():
                return

            # Play via OutputStream with progress callback
            total_frames = len(processed)
            position = [0]

            def callback(outdata, frames, time_info, status):
                start = position[0]
                end = min(start + frames, total_frames)
                n = end - start

                if n <= 0 or self._stop_flag.is_set():
                    outdata[:] = 0
                    raise sd.CallbackStop()

                outdata[:n, 0] = processed[start:end]
                if n < frames:
                    outdata[n:] = 0

                position[0] = end

                # Update progress
                pct = int(end * 100 / total_frames)
                wx.CallAfter(self._progress.SetValue, pct)

                if end >= total_frames:
                    raise sd.CallbackStop()

            self._stream = sd.OutputStream(
                samplerate=44100,
                channels=1,
                dtype="float32",
                callback=callback,
                blocksize=2048,
            )
            self._stream.start()

            # Wait for stream to finish
            while self._stream.active and not self._stop_flag.is_set():
                sd.sleep(50)

            self._stream.stop()
            self._stream.close()
            self._stream = None

        except Exception as e:
            wx.CallAfter(
                wx.MessageBox,
                f"Playback error: {e}",
                "Error",
                wx.OK | wx.ICON_ERROR,
            )
        finally:
            wx.CallAfter(self._progress.SetValue, 0)

    def on_stop(self, event):
        """Stop current playback."""
        self._stop_flag.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def on_save_wav(self, event):
        """Save text as WAV file, respecting rate/pitch/volume settings."""
        text = self._text_ctrl.GetValue().strip()
        if not text:
            wx.MessageBox("No text to save.", "Info", wx.OK | wx.ICON_INFORMATION)
            return

        with wx.FileDialog(
            self,
            "Save WAV File",
            wildcard="WAV files (*.wav)|*.wav",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()

        try:
            from scipy.io import wavfile

            # Synthesize raw audio
            audio = self._tts.speak(text)
            if len(audio) == 0:
                wx.MessageBox("No audio produced.", "Info", wx.OK | wx.ICON_INFORMATION)
                return

            # Process with pitch/rate/volume
            processed = self._process_audio(audio)

            # Convert to 16-bit PCM
            samples_16 = (processed * 32767).astype(np.int16)
            wavfile.write(path, 44100, samples_16)

            wx.MessageBox(f"Saved to {path}", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Error saving: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_open(self, event):
        """Open a text file."""
        with wx.FileDialog(
            self,
            "Open Text File",
            wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()

        try:
            with open(path, "r", encoding="utf-8") as f:
                self._text_ctrl.SetValue(f.read())
        except Exception as e:
            wx.MessageBox(f"Error opening: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_save_text(self, event):
        """Save text to a file."""
        with wx.FileDialog(
            self,
            "Save Text File",
            wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._text_ctrl.GetValue())
        except Exception as e:
            wx.MessageBox(f"Error saving: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_select_all(self, event):
        self._text_ctrl.SelectAll()

    def on_exit(self, event):
        self.Close()

    def _on_close(self, event):
        """Clean up threads before closing."""
        self.on_stop(None)
        if self._player_thread is not None:
            self._player_thread.join(timeout=1)
            self._player_thread = None
        event.Skip()


def main():
    app = wx.App()
    frame = VotraxFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
