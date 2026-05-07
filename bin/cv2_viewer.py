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
    while True:
        new_img, new_mtime = _read_jpeg_with_retry(args.path, last_mtime)
        if new_img is not None:
            img = new_img
            last_mtime = new_mtime
        cv2.imshow(args.win_name, img)
        # waitKey returns the keycode (-1 if none). 1 ms event-pump is enough
        # to keep the window alive.
        key = cv2.waitKey(1) & 0xFF
        if args.signal_pid > 0 and key != 0xFF:  # 0xFF (=255) means no key
            try:
                if key == ord('q'):
                    print('[cv2_viewer] q pressed -> SIGINT', flush=True)
                    os.kill(args.signal_pid, signal.SIGINT)
                    break
                elif key == ord('c'):
                    print('[cv2_viewer] c pressed -> SIGUSR1 (record start)', flush=True)
                    os.kill(args.signal_pid, signal.SIGUSR1)
                elif key == ord('s'):
                    print('[cv2_viewer] s pressed -> SIGUSR2 (record stop)', flush=True)
                    os.kill(args.signal_pid, signal.SIGUSR2)
                elif key == ord('h'):
                    print('[cv2_viewer] h pressed -> SIGHUP (home)', flush=True)
                    os.kill(args.signal_pid, signal.SIGHUP)
            except ProcessLookupError:
                # demo is gone — exit viewer too
                break
        time.sleep(poll_s)

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
