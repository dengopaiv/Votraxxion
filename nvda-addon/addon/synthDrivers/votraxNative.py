# license:BSD-3-Clause
# copyright-holders:tgeczy, Päiv Dengo
#
# Derived from the driver in Tamas Geczy's votraxsc01 NVDA add-on
# (https://github.com/tgeczy/votraxsc01-nvda); see NOTICE.md.
"""NVDA synth driver for the Votrax SC-01, over the native synthesizer.

The whole synthesizer — both mask ROMs, the letter-to-sound rules, the phone
mapping, the chip itself — is inside votraxNative.dll.  This file is only the
shim: it turns NVDA's speech sequences into work items on a queue, and a
single thread turns those into audio.

  votraxNative.dll (ctypes)  -- everything: ttv_* text to phones, vx_* the chip
  _Lib                       -- the ctypes bindings, declared once
  SynthDriver                -- NVDA-facing; queues work
  _speak_thread              -- the only thread that touches the chip after
                                startup, so nothing needs locking

There is no numpy, no scipy, no pronunciation dictionary and no ROM file: the
add-on is this file plus one library per architecture — x64 for NVDA 2026,
which is 64-bit only, and x86 for 2025 and earlier.  The previous driver
bundled about 40 MB of Python wheels to do the same job.

Cancellation is an epoch counter.  cancel() bumps it, and work items carrying
an older epoch are dropped when the thread reaches them.  Control items (rate,
pitch, voice) carry no epoch, because a settings change must survive a cancel.

Rate is constant-pitch: the chip runs at its datasheet clock and each phone is
held for a shorter time than its natural length, so tempo moves and pitch does
not.  "Authentic rate" instead varies the master clock, which is what the 1980
hardware's single knob did — faster is also higher.
"""

import ctypes
import os
import queue
import threading

import config
import nvwave
from autoSettingsUtils.driverSetting import BooleanDriverSetting
from speech.commands import CharacterModeCommand, IndexCommand
from synthDriverHandler import SynthDriver as BaseSynthDriver
from synthDriverHandler import VoiceInfo, synthDoneSpeaking, synthIndexReached

_DIR = os.path.dirname(__file__)

#: The chip's datasheet master clock.  Rate 50 maps here.
_BASE_CLOCK = 720000

#: Phone code the chip idles on; also what closes an utterance.
_STOP = 0x3F

#: Mask revisions, matching VX_MASK_* in votrax_capi.h.
_MASK_SC01A = 0
_MASK_SC01 = 1

#: voice id -> (display name, mask constant)
_VOICES = {
	"sc01": (_("SC-01 (1980 mask)"), _MASK_SC01),
	"sc01a": (_("SC-01-A (later mask)"), _MASK_SC01A),
}


def _dll_name():
	# Chosen from the bitness of the process we are actually running in, not
	# from the NVDA version: NVDA 2026 is 64-bit only, 2025 and earlier were
	# 32-bit, and the add-on ships both libraries so one build serves both.
	return "votraxNative-x64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8 \
		else "votraxNative-x86.dll"


def _output_device():
	# The output-device setting moved between config sections across the NVDA
	# versions this add-on spans.
	try:
		return config.conf["audio"]["outputDevice"]
	except KeyError:
		return config.conf["speech"]["outputDevice"]


class _Lib:
	"""The DLL, with every argtype declared up front.

	Declaring argtypes is not optional detail: without them ctypes guesses,
	and a guessed 32-bit handle in a 64-bit process is a crash that only
	happens once the heap wanders past 4 GB.
	"""

	_shared = None

	def __init__(self):
		self.lib = lib = ctypes.cdll.LoadLibrary(os.path.join(_DIR, _dll_name()))
		p = ctypes.c_void_p
		lib.vx_create.restype = p
		lib.vx_create.argtypes = [ctypes.c_int, ctypes.c_uint]
		for name, res, args in (
			("vx_destroy", None, [p]),
			("vx_reset", None, [p]),
			("vx_set_clock", None, [p, ctypes.c_uint]),
			("vx_clock", ctypes.c_uint, [p]),
			("vx_sample_rate", ctypes.c_double, [p]),
			("vx_set_mask", None, [p, ctypes.c_int]),
			("vx_mask", ctypes.c_int, [p]),
			("vx_inflection", None, [p, ctypes.c_ubyte]),
			("vx_get_inflection", ctypes.c_int, [p]),
			("vx_set_speed", None, [p, ctypes.c_double]),
			("vx_speed", ctypes.c_double, [p]),
			("vx_phone_samples", ctypes.c_int, [p, ctypes.c_ubyte]),
			("vx_write", None, [p, ctypes.c_ubyte]),
			("vx_ready", ctypes.c_int, [p]),
			("vx_speak", ctypes.c_int, [p, ctypes.c_char_p, ctypes.c_int]),
			("vx_pending", ctypes.c_int, [p]),
			("vx_cancel", None, [p]),
			("vx_render", ctypes.c_int, [p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]),
			("ttv_translate", ctypes.c_int, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
			("ttv_spell", ctypes.c_int, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
			("vx_phone_name", ctypes.c_char_p, [ctypes.c_int]),
		):
			fn = getattr(lib, name)
			fn.restype, fn.argtypes = res, args

	@classmethod
	def shared(cls):
		if cls._shared is None:
			cls._shared = cls()
		return cls._shared

	def translate(self, text, spell=False):
		"""Text to packed phone bytes.  Touches no chip state, so it is safe
		to call from NVDA's main thread while the speak thread renders."""
		out = ctypes.create_string_buffer(8192)
		fn = self.lib.ttv_spell if spell else self.lib.ttv_translate
		n = fn(text.encode("utf-8", "replace"), out, 8192)
		return bytes(out.raw[:min(n, 8192)])


class _Chip:
	"""One vx_chip."""

	def __init__(self, lib, mask):
		self._lib = lib.lib
		self._chip = self._lib.vx_create(mask, _BASE_CLOCK)
		if not self._chip:
			raise RuntimeError("vx_create failed")

	def close(self):
		if self._chip:
			self._lib.vx_destroy(self._chip)
			self._chip = None

	@property
	def sample_rate(self):
		return int(self._lib.vx_sample_rate(self._chip))

	def set_mask(self, mask):
		self._lib.vx_set_mask(self._chip, mask)

	def set_clock(self, hz):
		self._lib.vx_set_clock(self._chip, int(hz))

	def set_speed(self, factor):
		self._lib.vx_set_speed(self._chip, float(factor))

	def inflection(self, level):
		self._lib.vx_inflection(self._chip, level)

	def speak(self, phones):
		self._lib.vx_speak(self._chip, phones, len(phones))

	def pending(self):
		return self._lib.vx_pending(self._chip)

	def cancel(self):
		self._lib.vx_cancel(self._chip)

	def render(self, count):
		buf = (ctypes.c_int16 * count)()
		n = self._lib.vx_render(self._chip, buf, count)
		return ctypes.string_at(buf, n * 2)


class SynthDriver(BaseSynthDriver):
	name = "votraxNative"
	# Translators: description of the Votrax speech synthesizer.
	description = _("Votrax Native (SC-01)")

	supportedSettings = (
		BaseSynthDriver.VoiceSetting(),
		BaseSynthDriver.RateSetting(),
		# The SC-01's pitch input is two bits — four levels, nothing between.
		# minStep=33 makes the settings ring step one level per press, and
		# _set_pitch snaps the stored value onto {0, 33, 66, 100} so the
		# announced number always matches the level you hear.
		BaseSynthDriver.PitchSetting(minStep=33),
		BooleanDriverSetting(
			"authenticRate",
			# Translators: a Votrax driver setting: rate varies the chip clock,
			# changing pitch with speed, as the real hardware did.
			_("&Authentic rate (vary the chip clock; pitch rises with speed)"),
			defaultVal=False,
		),
	)
	supportedCommands = {IndexCommand, CharacterModeCommand}
	supportedNotifications = {synthIndexReached, synthDoneSpeaking}

	@classmethod
	def check(cls):
		try:
			return os.path.isfile(os.path.join(_DIR, _dll_name()))
		except Exception:
			return False

	def __init__(self):
		# Driver.__init__ is what registers the config save action;
		# without it every setting on this driver is forgotten at the
		# next restart.
		super().__init__()
		self._lib = _Lib.shared()
		self._chip = None
		self._player = None
		self._player_rate = 0
		self._rate = 50
		self._pitch = 33     # canonical level 1, the contour's neutral
		self._authentic = False
		self._voice = "sc01"
		self._epoch = 0
		self._queue = queue.Queue()
		# First open happens here on the main thread, before the speak thread
		# exists; afterwards the chip belongs to that thread alone.
		self._open_voice(self._voice)
		self._thread = threading.Thread(
			target=self._speak_thread, name="votraxNative", daemon=True)
		self._thread.start()

	def terminate(self):
		self.cancel()
		self._queue.put(None)
		self._thread.join(timeout=2)
		if self._player:
			self._player.close()
			self._player = None
		if self._chip:
			self._chip.close()
			self._chip = None

	# ---- chip lifecycle ----------------------------------------------

	def _open_voice(self, voice_id):
		_display, mask = _VOICES[voice_id]
		if self._chip is None:
			self._chip = _Chip(self._lib, mask)
		else:
			self._chip.set_mask(mask)
		self._voice = voice_id
		self._apply_rate()
		self._apply_pitch()
		self._ensure_player()

	def _ensure_player(self):
		rate = self._chip.sample_rate
		if self._player is None or self._player_rate != rate:
			old = self._player
			self._player = nvwave.WavePlayer(
				channels=1, samplesPerSec=rate, bitsPerSample=16,
				outputDevice=_output_device())
			self._player_rate = rate
			if old:
				old.close()

	def _apply_rate(self):
		# 0..100 maps to half..double speed, exponentially, so equal slider
		# steps sound like equal speed ratios.  Two regimes:
		#   default   -- constant pitch: the clock stays at the datasheet
		#                value and the scheduler truncates phones;
		#   authentic -- rate IS the master clock, pitch and all, as the 1980
		#                hardware's one knob really behaved.
		factor = 2.0 ** ((self._rate - 50) / 50.0)
		if self._authentic:
			self._chip.set_clock(_BASE_CLOCK * factor)
			self._chip.set_speed(1.0)
		else:
			self._chip.set_clock(_BASE_CLOCK)
			self._chip.set_speed(factor)

	def _apply_pitch(self):
		# Quantize NVDA's 0..100 onto the chip's four levels.  This is the base
		# the sentence contour shifts around, so leaving it at level 1 gives
		# the contour exactly as computed.
		self._chip.inflection(min(3, self._pitch * 4 // 101))

	# ---- NVDA settings -----------------------------------------------
	# Setters only record the value and queue a control item; the speak thread
	# applies it.  Control items carry epoch None on purpose: a settings change
	# must survive a cancel.

	def _get_voice(self):
		return self._voice

	def _set_voice(self, value):
		if value in _VOICES and value != self._voice:
			self._queue.put((None, "voice", value, None))

	def _getAvailableVoices(self):
		return {vid: VoiceInfo(vid, display, "en")
			for vid, (display, _mask) in _VOICES.items()}

	def _get_rate(self):
		return self._rate

	def _set_rate(self, value):
		self._rate = max(0, min(100, value))
		self._queue.put((None, "rate", None, None))

	def _get_pitch(self):
		return self._pitch

	def _set_pitch(self, value):
		# Snap onto the four real levels' canonical values so the number NVDA
		# announces matches the pitch produced, and each ring step lands
		# cleanly on the next level.  v*4//101 maps 0->0, 33->1, 66->2,
		# 100->3; level*100//3 is its inverse.
		level = min(3, max(0, value) * 4 // 101)
		self._pitch = level * 100 // 3
		self._queue.put((None, "pitch", None, None))

	def _get_authenticRate(self):
		return self._authentic

	def _set_authenticRate(self, value):
		self._authentic = bool(value)
		self._queue.put((None, "rate", None, None))

	# ---- speaking ----------------------------------------------------

	def speak(self, speechSequence):
		epoch = self._epoch
		spell = False
		for item in speechSequence:
			if isinstance(item, str):
				phones = self._lib.translate(item, spell=spell)
				if phones:
					self._queue.put((epoch, "phones", phones, None))
			elif isinstance(item, IndexCommand):
				self._queue.put((epoch, "index", None, item.index))
			elif isinstance(item, CharacterModeCommand):
				spell = item.state
		self._queue.put((epoch, "done", None, None))

	def cancel(self):
		self._epoch += 1
		# Drain speech items but put surviving control items back: a voice or
		# rate change queued just before a cancel must still happen.
		keep = []
		try:
			while True:
				item = self._queue.get_nowait()
				if item and item[0] is None:
					keep.append(item)
		except queue.Empty:
			pass
		for item in keep:
			self._queue.put(item)
		if self._player:
			self._player.stop()

	def pause(self, switch):
		if self._player:
			self._player.pause(switch)

	# ---- the chip thread ---------------------------------------------

	def _speak_thread(self):
		while True:
			item = self._queue.get()
			if item is None:
				return
			epoch, kind, payload, index = item
			if epoch is not None and epoch != self._epoch:
				continue
			if kind == "rate":
				self._apply_rate()
				self._ensure_player()
			elif kind == "pitch":
				self._apply_pitch()
			elif kind == "voice":
				self._open_voice(payload)
			elif kind == "index":
				synthIndexReached.notify(synth=self, index=index)
			elif kind == "phones":
				self._feed(payload, epoch)
			elif kind == "done":
				# Close the utterance the way the hardware would: STOP, then a
				# short tail so the last phone rings out.
				self._feed(bytes([_STOP]), epoch)
				self._render_tail(epoch, seconds=0.25)
				if epoch == self._epoch:
					self._player.idle()
					synthDoneSpeaking.notify(synth=self)

	def _block(self):
		return max(1, int(self._chip.sample_rate * 0.012))

	def _feed_player(self, data, epoch):
		"""Chip audio straight to the player, re-checking the epoch at the
		moment of feeding.

		A cancel can bump the epoch between the caller's check and here, and
		audio fed to a just-stopped player starts sounding before any later
		stop() can discard it — heard as the previous utterance's tail at the
		head of the next.  So drop the block if the epoch has moved."""
		if epoch == self._epoch and data:
			self._player.feed(data)

	def _feed(self, phones, epoch):
		"""Hand a run of phones to the scheduler and stream out the audio.

		Blocks of ~12 ms keep cancel latency low; the epoch check between
		blocks is what makes a cancel take effect mid-word.  vx_cancel is what
		silences a chip still voicing a cancelled phone — dropping the queue
		alone would leave that phone's remainder to come out at the head of
		the next utterance."""
		self._chip.speak(phones)
		block = self._block()
		while self._chip.pending() and epoch == self._epoch:
			self._feed_player(self._chip.render(block), epoch)
		if epoch != self._epoch:
			self._chip.cancel()
			self._apply_pitch()      # cancel resets the base level
			if self._player:
				self._player.stop()

	def _render_tail(self, epoch, seconds):
		# Ring out the last phone after STOP so the utterance's true ending is
		# heard.
		block = self._block()
		for _ in range(int(seconds / 0.012) + 1):
			if epoch != self._epoch:
				self._chip.cancel()
				self._apply_pitch()
				if self._player:
					self._player.stop()
				return
			self._feed_player(self._chip.render(block), epoch)
