#!/usr/bin/env python3
"""Check that the Votrax GUI can be driven from the keyboard alone.

    python tools/verify_gui_keyboard.py [x64|x86]

verify_gui.py proves the executable renders the right audio. It says
nothing about whether anyone can reach the button that renders it.

This launches the real GUI, puts focus where the program puts it, and
walks the tab order by posting actual VK_TAB messages into its queue --
the same messages a keypress delivers, processed by the same
IsDialogMessage call in the same loop. After each one it reads back where
the focus actually went. Nothing is simulated except the finger.

What it demands:

  * Tab moves focus off every control, the multiline text box included.
    A control that swallows Tab is a keyboard trap: someone navigating
    without a mouse cannot get past it, and for a screen reader voice
    that is not a cosmetic problem.
  * Tab reaches every control that declares WS_TABSTOP.
  * The order cycles rather than dead-ending, forwards and backwards.
  * No two controls claim the same Alt accelerator. Windows does not
    complain about a duplicate -- the key cycles between the claimants
    instead of activating either -- so a clash is invisible until someone
    navigating by keyboard cannot reach a button.

Exit status is 0 when the whole ring is reachable, 1 otherwise. Skips
with 0 if the executable has not been built.
"""
import ctypes
import ctypes.wintypes as w
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'gui-native', 'build')
EXE = 'votrax_native-{arch}.exe'
WINDOW_CLASS = 'VotraxNativeMainWindow'

WM_KEYDOWN, WM_KEYUP, WM_CLOSE, WM_GETDLGCODE = 0x0100, 0x0101, 0x0010, 0x0087
VK_TAB, VK_SHIFT = 0x09, 0x10
GWL_STYLE, GWL_ID = -16, -12
WS_TABSTOP, WS_DISABLED = 0x00010000, 0x08000000

# WM_GETDLGCODE bits, for explaining a failure rather than just reporting
# it. DLGC_WANTALLKEYS and DLGC_WANTMESSAGE are the same bit (0x0004) in
# winuser.h; the name is just which way you read it.
DLGC = [
    (0x0001, 'WANTARROWS'), (0x0002, 'WANTTAB'),
    (0x0004, 'WANTALLKEYS/WANTMESSAGE'), (0x0008, 'HASSETSEL'),
    (0x0010, 'DEFPUSHBUTTON'), (0x0020, 'UNDEFPUSHBUTTON'),
    (0x0040, 'RADIOBUTTON'), (0x0080, 'WANTCHARS'),
    (0x0100, 'STATIC'), (0x2000, 'BUTTON'),
]

# gui-native/resource.h, so a failure names a control instead of a handle.
NAMES = {
    1000: 'text label', 1001: 'text box', 1002: 'phoneme mode',
    1003: 'mask label', 1004: 'mask', 1005: 'preset label',
    1006: 'voice preset', 1007: 'clock label', 1008: 'clock',
    1009: 'clock spin', 1010: 'speed label', 1011: 'speed',
    1012: 'speed spin', 1013: 'inflection label', 1014: 'inflection',
    1015: 'inflection spin', 1016: 'clock help', 1017: 'phoneme help',
    1018: 'Preview', 1019: 'Convert', 1020: 'Render',
}

u32 = ctypes.WinDLL('user32', use_last_error=True)


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [('cbSize', w.DWORD), ('flags', w.DWORD),
                ('hwndActive', w.HWND), ('hwndFocus', w.HWND),
                ('hwndCapture', w.HWND), ('hwndMenuOwner', w.HWND),
                ('hwndMoveSize', w.HWND), ('hwndCaret', w.HWND),
                ('rcCaret', w.RECT)]


ENUMPROC = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)


def find_window(cls, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = u32.FindWindowW(cls, None)
        if hwnd:
            return hwnd
        time.sleep(0.05)
    return None


def children(parent):
    """Direct children in z-order, which is what tab order follows."""
    out = []

    def cb(hwnd, _):
        out.append(hwnd)
        return True

    u32.EnumChildWindows(parent, ENUMPROC(cb), 0)
    return out


def describe(hwnd):
    if not hwnd:
        return 'nothing'
    cid = u32.GetWindowLongW(hwnd, GWL_ID)
    return NAMES.get(cid, 'control %d' % cid)


def tabstops(parent):
    """The controls that claim to be reachable by Tab."""
    out = []
    for hwnd in children(parent):
        style = u32.GetWindowLongW(hwnd, GWL_STYLE)
        if (style & WS_TABSTOP) and not (style & WS_DISABLED) \
                and u32.IsWindowVisible(hwnd):
            out.append(hwnd)
    return out


def dlgcode(hwnd):
    code = u32.SendMessageW(hwnd, WM_GETDLGCODE, 0, 0)
    return code, [n for bit, n in DLGC if code & bit]


def focus_of(thread_id):
    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(info)
    if not u32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
        return None
    return info.hwndFocus


def settle(hwnd, thread_id, timeout=10.0):
    """Wait until the window has finished building itself.

    FindWindow succeeds while WM_CREATE is still running, so enumerating
    children straight away can catch half a window. Wait for the child
    count to stop changing, then for focus to reach a control -- the
    program sets that last.
    """
    deadline = time.time() + timeout

    count = -1
    while time.time() < deadline:
        now = len(children(hwnd))
        if now == count and now > 0:
            break
        count = now
        time.sleep(0.1)

    while time.time() < deadline:
        focus = focus_of(thread_id)
        if focus and focus != hwnd and u32.IsChild(hwnd, focus):
            return focus
        time.sleep(0.05)
    return None


def press_tab(hwnd, thread_id, shift=False, timeout=2.0):
    """Post a real Tab keypress and wait for the focus to actually move.

    Returns where the focus ended up, which is the same window it started
    on if nothing moved -- that is the keyboard-trap case, and it costs
    the full timeout to establish.

    Waiting for the change rather than pausing a fixed time is what makes
    a pass mean the tab order is right, rather than that the machine
    happened to be idle when the key was posted.
    """
    before = focus_of(thread_id)

    if shift:
        u32.PostMessageW(hwnd, WM_KEYDOWN, VK_SHIFT, 1)
    u32.PostMessageW(hwnd, WM_KEYDOWN, VK_TAB, 1)
    u32.PostMessageW(hwnd, WM_KEYUP, VK_TAB, 0xC0000001)
    if shift:
        u32.PostMessageW(hwnd, WM_KEYUP, VK_SHIFT, 0xC0000001)

    deadline = time.time() + timeout
    while time.time() < deadline:
        now = focus_of(thread_id)
        if now and now != before:
            return now
        time.sleep(0.02)
    return focus_of(thread_id)


def walk(thread_id, start, steps, shift=False):
    """Press Tab `steps` times, returning where focus went each time."""
    seen = []
    cur = start
    for _ in range(steps):
        nxt = press_tab(cur, thread_id, shift)
        seen.append(nxt)
        if not nxt or nxt == cur:
            break
        cur = nxt
    return seen


def check_accelerators(hwnd):
    """No two controls may claim the same Alt key."""
    seen, clashes = {}, []

    for h in children(hwnd):
        buf = ctypes.create_unicode_buffer(256)
        u32.GetWindowTextW(h, buf, 256)
        text = buf.value
        i = text.find('&')
        # "&&" is a literal ampersand, not an accelerator.
        while i >= 0 and i + 1 < len(text) and text[i + 1] == '&':
            i = text.find('&', i + 2)
        if i < 0 or i + 1 >= len(text):
            continue
        key = text[i + 1].upper()
        if key in seen:
            clashes.append((key, seen[key], text))
        else:
            seen[key] = text

    for key, first, second in clashes:
        print('  FAIL Alt+%s is claimed by both "%s" and "%s"'
              % (key, first, second))
    if not clashes:
        print('  ok   %d accelerators, all distinct: %s'
              % (len(seen), ' '.join(sorted(seen))))
    return len(clashes)


def check(arch):
    exe = os.path.join(BUILD, EXE.format(arch=arch))
    if not os.path.isfile(exe):
        print('  %s: not built, skipping' % os.path.basename(exe))
        return None

    proc = subprocess.Popen([exe])
    failures = 0
    try:
        hwnd = find_window(WINDOW_CLASS)
        if not hwnd:
            print('  FAIL the window never appeared')
            return 1

        thread_id = u32.GetWindowThreadProcessId(hwnd, None)
        start = settle(hwnd, thread_id)
        if not start:
            print('  FAIL focus never reached a control')
            return 1

        stops = tabstops(hwnd)
        print('  start: focus on the %s; %d tab stops'
              % (describe(start), len(stops)))

        # One extra step so the ring is seen to close rather than end.
        seen = walk(thread_id, start, len(stops) + 1)

        # A control that Tab does not leave is the trap this exists for.
        cur = start
        for hit in seen:
            if hit == cur:
                code, bits = dlgcode(cur)
                print('  FAIL Tab does not leave the %s '
                      '(WM_GETDLGCODE = 0x%04x: %s)'
                      % (describe(cur), code, ', '.join(bits) or 'none'))
                failures += 1
                break
            cur = hit

        reached = set(seen) | {start}
        missed = [h for h in stops if h not in reached]
        if missed:
            for h in missed:
                print('  FAIL Tab never reaches the %s' % describe(h))
            failures += 1
        else:
            print('  ok   forwards: every tab stop reached -- %s'
                  % ' -> '.join(describe(h) for h in [start] + seen[:len(stops)]))

        if seen and seen[len(stops) - 1:] and seen[len(stops) - 1] != start:
            print('  FAIL the ring does not close: %d tabs from the %s '
                  'landed on the %s'
                  % (len(stops), describe(start),
                     describe(seen[len(stops) - 1])))
            failures += 1

        back = walk(thread_id, focus_of(thread_id), len(stops), shift=True)
        reached_back = set(back)
        missed_back = [h for h in stops if h not in reached_back]
        if missed_back:
            for h in missed_back:
                print('  FAIL Shift+Tab never reaches the %s' % describe(h))
            failures += 1
        else:
            print('  ok   backwards: Shift+Tab walks the same ring')

        failures += check_accelerators(hwnd)

        u32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return failures


def main(argv):
    arches = [argv[0]] if argv else ['x64', 'x86']
    total, ran = 0, 0

    for arch in arches:
        print('%s:' % arch)
        result = check(arch)
        if result is None:
            continue
        ran += 1
        total += result

    if ran == 0:
        print('\nNothing to check. Build with gui-native\\build.cmd.')
        return 0
    if total:
        print('\n%d keyboard problem(s).' % total)
        return 1
    print('\nOK: the whole window is reachable from the keyboard.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
