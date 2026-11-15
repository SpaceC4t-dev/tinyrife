#!/bin/sh
# Stand-in for ffmpeg in tests (no codecs): as the decoder ("-i FILE ... -") it streams FILE, which is raw RGB24 frames;
# as the encoder ("-i -") it copies stdin to the output file, the last argument. FAKE_LOG gets every command line,
# FAKE_ENC_FAIL makes the encoder exit at once with status 3, FAKE_ENC_FAIL_LATE only after it has read everything (a failure
# when finalizing the file), FAKE_DEC_EXIT sets the decoder's exit status.
[ -n "$FAKE_LOG" ] && echo "$*" >> "$FAKE_LOG"
case " $* " in
*" -i - "*)
  [ -n "$FAKE_ENC_FAIL" ] && { echo "fake ffmpeg: encoder failing as asked" >&2; exit 3; }
  for last; do :; done
  cat > "$last"
  if [ -n "$FAKE_ENC_FAIL_LATE" ]; then echo "fake ffmpeg: encoder failing at the end as asked" >&2; exit 3; fi
  ;;
*)
  while [ $# -gt 0 ]; do
    [ "$1" = "-i" ] && { cat "$2"; exit "${FAKE_DEC_EXIT:-0}"; }
    shift
  done
  echo "fake ffmpeg: no -i" >&2; exit 2
  ;;
esac
