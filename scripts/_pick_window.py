#!/usr/bin/env python3
"""Print the start time (s) of the most active LEN-second window of a video (largest summed frame-to-frame change).
Usage: _pick_window.py VIDEO LEN   -> prints a float; prints 0 if opencv is missing or the video is not longer than LEN."""
import sys
try:
    import cv2
    import numpy as np
except ImportError:
    print(0); sys.exit(0)
path, length = sys.argv[1], float(sys.argv[2])
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS) or 10
prev, diffs = None, []
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(f, (80, 60)), cv2.COLOR_BGR2GRAY).astype("float32")
    diffs.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
    prev = g
n = len(diffs); w = int(length * fps)
if n <= w:
    print(0); sys.exit(0)
c = np.concatenate([[0], np.cumsum(diffs)])
best = max(range(0, n - w + 1), key=lambda i: c[i + w] - c[i])
print(f"{best / fps:.2f}")
