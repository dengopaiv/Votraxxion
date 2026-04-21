# PyInstaller spec for the Votrax SC-01A Workbench (Windows standalone .exe).
#
# Build from the repo root with:
#     python -m PyInstaller packaging/votrax-gui.spec
#
# Produces dist/VotraxWorkbench/ (one-dir layout). The .exe is at
# dist/VotraxWorkbench/VotraxWorkbench.exe and depends on sibling _internal/.
#
# One-dir was chosen over one-file: wxPython + scipy + numpy is large enough
# that one-file's temp-extract path has flaky startup and slow cold launches.
# One-dir also plays nicely with Inno Setup, which bundles the whole folder.

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO_ROOT = Path(SPECPATH).parent.resolve()

# cmudict ships its pronunciation dictionary as package data; pull everything
# (data + binaries + hidden imports) so the lookup path works in the frozen app.
cmudict_datas, cmudict_binaries, cmudict_hiddenimports = collect_all("cmudict")

# Factory presets: shipped under presets/factory/ relative to the bundle root
# so pyvotrax.presets.factory_presets_dir() (which uses __file__.parent.parent)
# resolves correctly once the package lives under _internal/pyvotrax/.
factory_presets = [
    (str(p), "presets/factory")
    for p in (REPO_ROOT / "presets" / "factory").glob("*.json")
]

datas = cmudict_datas + factory_presets
binaries = list(cmudict_binaries)

# Hidden imports that PyInstaller's static analysis might miss.
# collect_submodules("pyvotrax") guarantees every submodule lands in the bundle
# even if the import graph from the launcher misses something (e.g. py_emu is
# imported lazily when enhanced_dsp=True).
hiddenimports = list(cmudict_hiddenimports) + [
    "scipy.signal",
    "scipy.signal.windows",
    "scipy.io.wavfile",
    "wx.lib.agw",
] + collect_submodules("pyvotrax") + collect_submodules("py_emu")

# The compiled C++ extension is auto-collected via the pyvotrax.chip import
# graph, so we do NOT list _votrax_core*.pyd here.

a = Analysis(
    [str(REPO_ROOT / "packaging" / "votrax_workbench_launcher.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Dead-weight things pulled in by numpy/scipy we don't use.
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
        "pybind11",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VotraxWorkbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # windowed Win32 app; no console popup
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Icon can be added later by passing icon="packaging/votrax.ico".
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VotraxWorkbench",
)
