#!/usr/bin/env python3
"""Build the Votrax Native NVDA add-on -- the SC-01, in C.

The add-on is one Python file and one DLL per architecture — no bundled
wheels, no dictionary, no ROM files. Everything the synthesizer needs is
compiled into the library.

    python package.py

produces votraxNative-<version>.nvda-addon, on the order of 300 KB.

NVDA 2026 is 64-bit only, so x64 is the build that matters and the one this
script insists on. The x86 build ships alongside it for NVDA 2025 and earlier,
which ran as 32-bit processes; the driver picks between them at load time from
the bitness of the interpreter it finds itself in, so one add-on serves both.

Building needs a C compiler. This script drives MSVC, and finds vcvars
itself if you are not already in a Developer Command Prompt; the sources are
plain C11 and build under MinGW or clang just as well.
"""

import os
import shutil
import subprocess
import sys
import zipfile

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ADDON_DIR)
SRC = os.path.join(ROOT_DIR, "src")

#: The whole synthesizer.  C11, no libc beyond string.h and math.h, and no
#: dependency on the C++ runtime -- which is the point: an add-on that needs a
#: redistributable a user may not have fails by NVDA silently not listing the
#: synth, with no error anyone can act on.
SOURCES = ("votrax.c", "votrax_core.c", "votrax_filters.c", "votrax_rom.c",
           "ttv.c", "ttv_tables.c")
DRIVER_DIR = os.path.join(ADDON_DIR, "addon", "synthDrivers")
BUILD_DIR = os.path.join(ADDON_DIR, "build")

#: (vcvars batch file, output DLL name, required). x64 is what NVDA 2026 runs;
#: x86 serves NVDA 2025 and earlier and is nice to have, not essential.
TARGETS = (
    ("vcvars64.bat", "votraxNative-x64.dll", True),
    ("vcvars32.bat", "votraxNative-x86.dll", False),
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
    obj_dir = os.path.join(BUILD_DIR, output_name.replace(".dll", ""))
    os.makedirs(obj_dir, exist_ok=True)

    # Run it from a batch file rather than `cmd /c "..."`: the nested quoting
    # a vcvars path with spaces needs does not survive being handed to cmd as
    # a single argument.
    script = os.path.join(BUILD_DIR, output_name.replace(".dll", ".bat"))

    # Everything below is shaped around two MSVC command-line quirks that only
    # bite when a path contains a space, which this repository's does.
    #
    # /Fo must name a directory when there are several source files, and a
    # directory name ends in a backslash -- but a trailing backslash inside
    # quotes escapes the closing quote and swallows the rest of the line. So
    # /Fo cannot be quoted, and therefore cannot contain a space. Compiling
    # from inside the source directory and naming the object directory
    # relatively keeps it space-free whatever the repository is called.
    obj_rel = os.path.relpath(obj_dir, SRC)
    names = " ".join(SOURCES)
    lines = [
        "@echo off",
        f'call "{vcvars}" >nul',
        f'cd /d "{SRC}"',
        (f'cl /nologo /std:c11 /W4 /O2 /LD '
         f'/I. /Fe:"{out}" /Fo:{obj_rel}\\ {names}'),
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


#: Keys NVDA's manifest specification requires; everything else has a default.
MANIFEST_REQUIRED = ("name", "summary", "author", "version")


def read_manifest():
    """Parse manifest.ini the way NVDA does, and refuse a file it would reject.

    NVDA reads the manifest with ConfigObj against a specification whose keys
    sit at the top level: there is no [addon] section. A section header buries
    every key below where the specification looks, so the whole manifest
    validates as missing and the add-on fails to install with nothing but
    "failed" to go on. An unquoted value containing a comma fails the same way
    for a different reason -- ConfigObj reads it as a list where the
    specification wants a string. Neither mistake is visible by reading the
    file, so check for both here rather than at the far end of a download.
    """
    values, errors = {}, []
    with open(os.path.join(ADDON_DIR, "manifest.ini"), encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                errors.append(f"line {number}: {line} -- NVDA's manifest has no "
                              "sections; every key sits at the top level")
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if "," in value and value[:1] not in ('"', "'"):
                errors.append(f"line {number}: {key} contains a comma and is not "
                              "quoted -- ConfigObj would read it as a list")
            values[key] = value.strip('"').strip("'")
    missing = [k for k in MANIFEST_REQUIRED if not values.get(k)]
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    if errors:
        print("ERROR: manifest.ini is not one NVDA would accept:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    return values


def addon_version():
    return read_manifest()["version"]


def build_addon(version):
    output = os.path.join(ADDON_DIR, f"votraxNative-{version}.nvda-addon")
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
    required = ["manifest.ini", "synthDrivers/votraxNative.py"]
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
    missing = [n for n in SOURCES if not os.path.isfile(os.path.join(SRC, n))]
    if missing:
        print(f"ERROR: not found in {SRC}: {', '.join(missing)}")
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
