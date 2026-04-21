"""Tests for the JSON preset system."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyvotrax.presets import (
    Preset,
    PRESET_SCHEMA_VERSION,
    load_preset,
    save_preset,
    preset_filename,
    factory_presets_dir,
    list_presets,
)


class TestPresetDataclass:
    def test_defaults(self):
        p = Preset()
        assert p.schema_version == PRESET_SCHEMA_VERSION
        assert p.mode == "phoneme_string"
        assert p.synth == {}
        assert p.tts == {}

    def test_to_dict_roundtrip(self):
        p = Preset(
            name="Robot",
            description="slow",
            mode="phoneme_string",
            input="AH STOP",
            synth={"master_clock": 360_000.0, "dc_block": True},
            tts={"enhanced": False},
        )
        d = p.to_dict()
        p2 = Preset.from_dict(d)
        assert p2 == p

    def test_from_dict_rejects_bad_version(self):
        with pytest.raises(ValueError):
            Preset.from_dict({"schema_version": 9999})

    def test_from_dict_rejects_bad_mode(self):
        with pytest.raises(ValueError):
            Preset.from_dict({"schema_version": PRESET_SCHEMA_VERSION, "mode": "bogus"})

    def test_synth_coercion(self):
        p = Preset.from_dict({
            "schema_version": PRESET_SCHEMA_VERSION,
            "mode": "text",
            "synth": {"master_clock": 360000, "dc_block": 1},
        })
        assert isinstance(p.synth["master_clock"], float)
        assert isinstance(p.synth["dc_block"], bool)
        assert p.synth["dc_block"] is True


class TestBuildFromPreset:
    def test_build_synthesizer(self):
        p = Preset(synth={"master_clock": 360_000.0, "dc_block": True})
        synth = p.build_synthesizer()
        assert synth.master_clock == 360_000.0

    def test_build_tts_and_synthesize(self):
        p = Preset(
            mode="phoneme_string",
            input="AH STOP",
            synth={"master_clock": 720_000.0},
        )
        synth = p.build_synthesizer()
        audio = synth.synthesize_phoneme_string(p.input)
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))


class TestLoadSave:
    def test_roundtrip(self, tmp_path):
        p = Preset(
            name="Test preset",
            description="hello",
            mode="text",
            input="Hello.",
            synth={"master_clock": 540_000.0, "fx_fudge": 1.0},
            tts={"enhanced": True},
        )
        path = tmp_path / "test.json"
        save_preset(path, p)
        loaded = load_preset(path)
        assert loaded == p

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "preset.json"
        save_preset(path, Preset(name="Nested"))
        assert path.exists()

    def test_save_is_pretty_printed(self, tmp_path):
        path = tmp_path / "preset.json"
        save_preset(path, Preset(name="Pretty"))
        raw = path.read_text(encoding="utf-8")
        assert "\n  " in raw


class TestFilenameHelpers:
    @pytest.mark.parametrize("name,expected", [
        ("Default", "Default"),
        ("Slow robot", "Slow_robot"),
        ("As-schematic (muffled)", "As-schematic_muffled"),
        ("  trailing  ", "trailing"),
        ("", "untitled"),
        ("....", "untitled"),
    ])
    def test_preset_filename(self, name, expected):
        assert preset_filename(name) == expected


class TestFactoryPresets:
    def test_factory_dir_exists(self):
        d = factory_presets_dir()
        assert d.exists(), f"Factory presets directory missing: {d}"

    def test_all_factory_presets_load(self):
        """Every factory preset JSON must parse cleanly into a Preset."""
        files = sorted(factory_presets_dir().glob("*.json"))
        assert len(files) >= 1, "No factory presets found"
        for f in files:
            loaded = load_preset(f)
            assert loaded.name  # non-empty
            assert loaded.schema_version == PRESET_SCHEMA_VERSION

    def test_factory_presets_are_buildable(self):
        """Every factory preset should build a working VotraxSynthesizer."""
        for f in sorted(factory_presets_dir().glob("*.json")):
            preset = load_preset(f)
            synth = preset.build_synthesizer()
            assert synth.master_clock > 0

    def test_list_presets_filters_directory(self):
        files = list_presets(factory_presets_dir())
        assert all(p.suffix == ".json" for p in files)
        assert len(files) >= 1
