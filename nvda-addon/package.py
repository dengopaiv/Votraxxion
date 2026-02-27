#!/usr/bin/env python3
"""Build script for the Votrax NVDA addon.

Creates a .nvda-addon file (ZIP archive) containing:
  - manifest.ini
  - synthDrivers/votrax.py
  - lib/  (bundled Python dependencies: pyvotrax, numpy, scipy, pronouncing)

Usage:
    python package.py

Output:
    votrax-0.2.0.nvda-addon
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ADDON_DIR)
LIB_DIR = os.path.join(ADDON_DIR, "lib")
OUTPUT_NAME = "votrax-0.2.0.nvda-addon"

# Files and patterns to exclude from the addon ZIP
_EXCLUDE_FILES = {"gui.py", "__main__.py"}
_EXCLUDE_DIRS = {"__pycache__"}
_EXCLUDE_EXTS = {".pyc"}
_EXCLUDE_PREFIXES = {"test_"}


def _should_exclude(filepath):
    """Check if a file should be excluded from the addon."""
    basename = os.path.basename(filepath)
    dirname = os.path.basename(os.path.dirname(filepath))

    if basename in _EXCLUDE_FILES:
        return True
    if dirname in _EXCLUDE_DIRS:
        return True
    if any(basename.endswith(ext) for ext in _EXCLUDE_EXTS):
        return True
    if any(basename.startswith(prefix) for prefix in _EXCLUDE_PREFIXES):
        return True
    return False


def build_extension():
    """Build the C++ extension in the source tree."""
    print("Building C++ extension...")
    subprocess.check_call([
        sys.executable, "setup.py", "build_ext", "--inplace",
    ], cwd=ROOT_DIR)
    print("C++ extension built.")


def install_deps():
    """Install pyvotrax and dependencies into lib/."""
    if os.path.isdir(LIB_DIR):
        shutil.rmtree(LIB_DIR)
    os.makedirs(LIB_DIR)

    print(f"Installing pyvotrax and dependencies into {LIB_DIR}...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--target", LIB_DIR,
        "--no-user",
        ROOT_DIR,
    ])
    print("Dependencies installed.")


def copy_native_extension():
    """Copy the compiled .pyd into the bundled lib. Fails if missing."""
    pyd_pattern = os.path.join(ROOT_DIR, "pyvotrax", "_votrax_core*.pyd")
    matches = glob.glob(pyd_pattern)
    if not matches:
        print("ERROR: No native extension (.pyd) found in pyvotrax/.")
        print("Build the extension first: python setup.py build_ext --inplace")
        sys.exit(1)

    dest_dir = os.path.join(LIB_DIR, "pyvotrax")
    os.makedirs(dest_dir, exist_ok=True)

    for pyd_path in matches:
        dest = os.path.join(dest_dir, os.path.basename(pyd_path))
        shutil.copy2(pyd_path, dest)
        print(f"Copied native extension: {os.path.basename(pyd_path)}")


def build_addon():
    """Create the .nvda-addon ZIP file."""
    output_path = os.path.join(ADDON_DIR, OUTPUT_NAME)

    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Building {OUTPUT_NAME}...")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add manifest
        zf.write(
            os.path.join(ADDON_DIR, "manifest.ini"),
            "manifest.ini",
        )

        # Add synth driver
        driver_path = os.path.join(ADDON_DIR, "addon", "synthDrivers", "votrax.py")
        zf.write(driver_path, "synthDrivers/votrax.py")

        # Add bundled libraries (with filtering)
        for dirpath, dirnames, filenames in os.walk(LIB_DIR):
            # Prune excluded directories
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if _should_exclude(filepath):
                    continue
                arcname = "lib/" + os.path.relpath(filepath, LIB_DIR).replace("\\", "/")
                zf.write(filepath, arcname)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Built {output_path} ({size_mb:.1f} MB)")


def verify_addon():
    """Verify the addon ZIP contains the native extension."""
    output_path = os.path.join(ADDON_DIR, OUTPUT_NAME)
    with zipfile.ZipFile(output_path, "r") as zf:
        pyd_entries = [n for n in zf.namelist() if "_votrax_core" in n and n.endswith(".pyd")]
        if not pyd_entries:
            print("ERROR: Addon ZIP does not contain the native extension!")
            sys.exit(1)
        print("Addon verification passed:")
        for entry in pyd_entries:
            print(f"  Verified: {entry}")


def main():
    build_extension()
    install_deps()
    copy_native_extension()
    build_addon()
    verify_addon()


if __name__ == "__main__":
    main()
