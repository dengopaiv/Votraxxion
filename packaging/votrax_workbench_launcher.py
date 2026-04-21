"""Entry point for the frozen Windows build of the Votrax Workbench.

Kept separate from ``pyvotrax/__main__.py`` because PyInstaller treats its
entry-point script as top-level and strips the ``__package__``, which breaks
relative imports inside that file. This launcher uses absolute imports and
does nothing else, so the whole ``pyvotrax`` package is pulled in via the
``pyvotrax.gui`` import.
"""

import multiprocessing

from pyvotrax.gui import main


if __name__ == "__main__":
    # Needed when a frozen wx app ever spawns a subprocess on Windows.
    multiprocessing.freeze_support()
    main()
