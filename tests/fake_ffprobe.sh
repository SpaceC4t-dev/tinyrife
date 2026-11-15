#!/bin/sh
# Stand-in for ffprobe in tests: reports the size and frame rate given by FAKE_W, FAKE_H, FAKE_RATE (default 64x48 at 30/1),
# or fails when FAKE_PROBE_FAIL is set.
[ -n "$FAKE_PROBE_FAIL" ] && { echo "fake ffprobe: failing as asked" >&2; exit 1; }
echo "width=${FAKE_W:-64}"
echo "height=${FAKE_H:-48}"
echo "r_frame_rate=${FAKE_RATE:-30/1}"
