#!/usr/bin/env python3
"""Build the native Votrax SC-01 NVDA add-on.

The add-on is one Python file and one DLL per architecture — no bundled
wheels, no dictionary, no ROM files. Everything the synthesizer needs is
compiled into the library.

    python package.py

produces votraxsc01-<version>.nvda-addon, on the order of 300 KB.

NVDA 2026 is 64-bit only, so x64 is the build that matters and the one this
script insists on. The x86 build ships alongside it for NVDA 2025 and earlier,
which ran as 32-bit processes; the driver picks between them at load time from
the bitness of the interpreter it finds itself in, so one add-on serves both.

Building needs MSVC — run from a Developer Command Prompt, or let the script
find vcvars itself.
"""

import configparser
import os
import shutil
import subprocess
import sys
import zipfile

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ADDON_DIR)
CSRC = os.path.join(ROOT_DIR, "csrc")
SOURCE = os.path.join(CSRC, "votrax_capi.cpp")
DRIVER_DIR = os.path.join(ADDON_DIR, "addon", "synthDrivers")
BUILD_DIR = os.path.join(ADDON_DIR, "build")

#: (vcvars batch file, output DLL name, required). x64 is what NVDA 2026 runs;
#: x86 serves NVDA 2025 and earlier and is nice to have, not essential.
TARGETS = (
    ("vcvars64.bat", "votraxsc01-x64.dll", True),
    ("vcvars32.bat", "votraxsc01-x86.dll", False),
)

_VS_ROOTS = (
    r"C:\Program Files\Microsoft Visual Studio",
    r"C:\Program Files (x86)\Microsoft Visual Studio",
)


def find_vcvars(name):
    """Locate a vcvars batch file, newest Visual Studio first."""
    candidates = []
    for root in _VS_ROOTS:
        if not os.path.isdir(root):
            continue
        for version in sorted(os.listdir(root), reverse=True):
            for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
                path = os.path.join(root, version, edition, "VC", "Auxiliary",
                                    "Build", name)
                if os.path.isfile(path):
                    candidates.append(path)
    return candidates[0] if candidates else None


def build_dll(vcvars_name, output_name, required):
    """Compile the C API into a DLL for one architecture."""
    vcvars = find_vcvars(vcvars_name)
    if not vcvars:
        level = "ERROR" if required else "SKIP"
        print(f"  {level} {output_name}: {vcvars_name} not found")
        return False

    os.makedirs(BUILD_DIR, exist_ok=True)
    out = os.path.join(DRIVER_DIR, output_name)
    obj_prefix = os.path.join(BUILD_DIR, output_name.replace(".dll", "_"))

    # Run it from a batch file rather than `cmd /c "..."`: the nested quoting
    # a vcvars path with spaces needs does not survive being handed to cmd as
    # a single argument.
    script = os.path.join(BUILD_DIR, output_name.replace(".dll", ".bat"))
    lines = [
        "@echo off",
        f'call "{vcvars}" >nul',
        (f'cl /nologo /std:c++17 /EHsc /W4 /O2 /LD '
         f'/I"{CSRC}" /Fe:"{out}" /Fo:"{obj_prefix}" "{SOURCE}"'),
    ]
    with open(script, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    result = subprocess.run([script], capture_output=True, text=True, shell=True)
    if result.returncode != 0 or not os.path.isfile(out):
        print(f"  FAILED {output_name}")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return False

    # cl leaves an import library and export table next to the DLL; the add-on
    # only needs the DLL itself.
    for stray in (".lib", ".exp"):
        path = out.replace(".dll", stray)
        if os.path.isfile(path):
            os.remove(path)

    print(f"  built {output_name} ({os.path.getsize(out) / 1024:.0f} KB)")
    return True


def addon_version():
    parser = configparser.ConfigParser()
    parser.read(os.path.join(ADDON_DIR, "manifest.ini"), encoding="utf-8")
    return parser["addon"]["version"].strip()


def build_addon(version):
    output = os.path.join(ADDON_DIR, f"votraxsc01-{version}.nvda-addon")
    if os.path.exists(output):
        os.remove(output)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(ADDON_DIR, "manifest.ini"), "manifest.ini")
        for name in sorted(os.listdir(DRIVER_DIR)):
            if name.endswith(".py") or name.endswith(".dll"):
                zf.write(os.path.join(DRIVER_DIR, name), f"synthDrivers/{name}")

    print(f"\nbuilt {os.path.basename(output)} "
          f"({os.path.getsize(output) / 1024:.0f} KB)")
    return output


def verify(output):
    """An add-on missing the library for the host's architecture is useless,
    and the failure would only show up as NVDA silently not listing the
    synth — so check here rather than letting a user find out."""
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    required = ["manifest.ini", "synthDrivers/votraxsc01.py"]
    missing = [n for n in required if n not in names]
    dlls = [n for n in names if n.endswith(".dll")]
    if missing:
        print(f"ERROR: missing from the add-on: {missing}")
        sys.exit(1)
    if not any("x64" in n for n in dlls):
        print("ERROR: no x64 library - NVDA 2026 is 64-bit only and would not "
              "load this.")
        sys.exit(1)
    if not any("x86" in n for n in dlls):
        print("WARNING: no x86 library - this add-on will not work on NVDA "
              "2025 or earlier, which ran 32-bit.")
    print("contents:")
    for name in sorted(names):
        print(f"  {name}")


def main():
    if not os.path.isfile(SOURCE):
        print(f"ERROR: {SOURCE} not found")
        sys.exit(1)

    print("Building native libraries...")
    for vcvars, name, required in TARGETS:
        if not build_dll(vcvars, name, required) and required:
            print("ERROR: the x64 library is required - NVDA 2026 is 64-bit "
                  "only. Is MSVC installed?")
            sys.exit(1)

    output = build_addon(addon_version())
    verify(output)


if __name__ == "__main__":
    main()
