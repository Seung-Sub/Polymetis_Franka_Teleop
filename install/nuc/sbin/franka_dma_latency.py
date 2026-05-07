#!/usr/bin/env python3
# franka_dma_latency.py -- keep /dev/cpu_dma_latency at 0us forever.
#
# Holds the file open with PM_QOS request 0us, which forbids the CPU from
# entering any wake-up-latency-inducing C-state. Required so the Franka
# 1 kHz inner loop never misses a deadline because a core was asleep.
#
# Run as a long-lived systemd Service (Type=simple, Restart=always).
import struct, time, signal
fd = open("/dev/cpu_dma_latency", "wb")
fd.write(struct.pack("i", 0)); fd.flush()
signal.signal(signal.SIGTERM, lambda *_: (fd.close(), exit(0)))
while True: time.sleep(3600)
