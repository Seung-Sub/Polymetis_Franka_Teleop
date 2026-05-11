#!/usr/bin/env python3
"""Standalone OpenCV viewer subprocess.

Why a subprocess: cv2.imshow + Qt5 inside the demo's main process deadlocks
under multi-subprocess load (ZED grab, polymetis client, recorders all
contending for the X mutex via Qt5). A fresh Python interpreter with no
other multiprocessing has none of that contention and cv2.imshow renders
reliably.

Inputs:
  - JPEG file path (polled every ~30 ms)
  - --signal-pid PID  (optional): when user presses 'q' inside the cv2
    window, send SIGINT to that PID so the demo shuts down cleanly.

Run via:
  python bin/cv2_viewer.py /tmp/teleop_vis.jpg --signal-pid <DEMO_PID>
"""
import argparse
import os
import signal
import sys
import time

import cv2
import numpy as np


def _read_jpeg_with_retry(path, last_mtime):
    """Read the JPEG only when its mtime advances. Returns (img, mtime) or
    (None, last_mtime) if the file is still the same / unreadable."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None, last_mtime
    if st.st_mtime_ns == last_mtime:
        return None, last_mtime
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        return None, last_mtime
    return img, st.st_mtime_ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path', help='JPEG file the demo writes')
    p.add_argument('--signal-pid', type=int, default=0,
                   help='Send SIGINT to this PID when user presses q in the window')
    p.add_argument('--poll-ms', type=int, default=30,
                   help='File poll interval (ms). Lower = more responsive, more CPU')
    p.add_argument('--win-name', default='Franka Vive Demo')
    args = p.parse_args()

    cv2.setNumThreads(1)

    last_mtime = 0
    img = None
    # Wait up to 10 s for the demo to write the first frame
    t0 = time.monotonic()
    while img is None and (time.monotonic() - t0) < 10.0:
        img, last_mtime = _read_jpeg_with_retry(args.path, last_mtime)
        if img is None:
            time.sleep(0.1)
    if img is None:
        # Bail out cleanly — demo never wrote an image.
        print(f'[cv2_viewer] no image at {args.path} after 10 s, exiting',
              file=sys.stderr, flush=True)
        return 1

    cv2.imshow(args.win_name, img)
    cv2.waitKey(1)

    poll_s = max(0.005, args.poll_ms / 1000.0)
    # Double-press 'q' guard — accidental q presses (e.g. user typing in
    # the wrong window, focus stolen by another app, fingertip slip while
    # reaching for the controller) used to silently kill ongoing recording
    # sessions. Now we require two q presses within 1.5 s.
    last_q_t = 0.0
    # Drop confirmation state — Backspace arms a 5 s window during which a
    # single 'y' confirms the drop (relayed to the demo via SIGRTMIN); 'n'
    # cancels. Bundling the 2-step confirmation here means only one signal
    # (SIGRTMIN) crosses the process boundary instead of needing bespoke
    # signals for Backspace / y / n individually.
    drop_armed_t = 0.0
    while True:
        new_img, new_mtime = _read_jpeg_with_retry(args.path, last_mtime)
        if new_img is not None:
            img = new_img
            last_mtime = new_mtime
        cv2.imshow(args.win_name, img)
        # waitKey returns the keycode (-1 if none). 1 ms event-pump is enough
        # to keep the window alive.
        key = cv2.waitKey(1) & 0xFF
        # Auto-expire the drop confirmation window after 5 s of no y/n.
        if drop_armed_t > 0.0 and (time.monotonic() - drop_armed_t) > 5.0:
            print('[cv2_viewer] drop confirmation timed out (no y within 5 s)',
                  flush=True)
            drop_armed_t = 0.0
        if args.signal_pid > 0 and key != 0xFF:  # 0xFF (=255) means no key
            try:
                if key == ord('q'):
                    now = time.monotonic()
                    if now - last_q_t < 1.5:
                        print('[cv2_viewer] q pressed twice -> SIGINT (clean shutdown)',
                              flush=True)
                        os.kill(args.signal_pid, signal.SIGINT)
                        break
                    else:
                        print('[cv2_viewer] q pressed (1/2) -- press q again within '
                              '1.5 s to confirm shutdown', flush=True)
                        last_q_t = now
                elif key == ord('c'):
                    print('[cv2_viewer] c pressed -> SIGUSR1 (record start)', flush=True)
                    os.kill(args.signal_pid, signal.SIGUSR1)
                elif key == ord('s'):
                    print('[cv2_viewer] s pressed -> SIGUSR2 (record stop)', flush=True)
                    os.kill(args.signal_pid, signal.SIGUSR2)
                elif key == ord('h'):
                    print('[cv2_viewer] h pressed -> SIGHUP (home)', flush=True)
                    os.kill(args.signal_pid, signal.SIGHUP)
                elif key == ord('d') or key in (8, 127):
                    # 'd' is the primary drop-arm key (mnemonic, always
                    # delivered by cv2.waitKey under all X / Wayland / Qt /
                    # GTK builds). Backspace (8) / Delete (127) kept as
                    # fallbacks for keyboards/WMs that don't intercept them
                    # at the system level (some GTK setups grab Backspace
                    # as a navigation shortcut and the keypress never
                    # reaches cv2.waitKey, which is the failure mode
                    # observed at KIST 2026-05-11).
                    drop_armed_t = time.monotonic()
                    print('[cv2_viewer] d/Backspace -- press y within 5 s to '
                          'confirm drop, n to cancel', flush=True)
                elif key == ord('y'):
                    if drop_armed_t > 0.0:
                        print('[cv2_viewer] y pressed -> SIGRTMIN (drop episode)',
                              flush=True)
                        os.kill(args.signal_pid, signal.SIGRTMIN)
                        drop_armed_t = 0.0
                elif key == ord('n'):
                    if drop_armed_t > 0.0:
                        print('[cv2_viewer] n pressed -- drop cancelled', flush=True)
                        drop_armed_t = 0.0
                else:
                    # Unrecognised key — print the keycode (after the
                    # & 0xFF mask) so future "key X doesn't work"
                    # debugging has data instead of silence. ASCII chr if
                    # printable so the operator immediately sees which
                    # key fired.
                    ch = chr(key) if 32 <= key < 127 else '?'
                    print(f'[cv2_viewer] unrecognised keycode {key} '
                          f'({ch!r}); no signal sent', flush=True)
            except ProcessLookupError:
                # demo is gone — exit viewer too
                break
        time.sleep(poll_s)

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
