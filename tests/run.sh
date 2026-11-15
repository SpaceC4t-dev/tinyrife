#!/bin/sh
# End-to-end checks of the C programs: golden outputs, CPU-vs-GL agreement, bad-model rejection.
# Run from anywhere after `make` (make test does both). Needs the model from ./fetch_model.sh.
# BUILD=dir runs against binaries in another directory (make sanitize uses build/san).
#
# Golden images come from the verified implementations (CPU: the original pre-refactor binary; GL: matches the PyTorch
# reference, see python/test_gl.py). Tolerance is 1 uint8 level, the project's pass criterion: the CPU Winograd tile
# choice depends on cache sizes and GL drivers differ in the last bit.
cd "$(dirname "$0")/.." || exit 2
B=${BUILD:-build}
MODEL=${MODEL:-ext/rife-ncnn-vulkan/models/rife-v4.6}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
pass=0; fail=0; skip=0
ok()  { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1${2:+ -- $2}"; fail=$((fail + 1)); }
skp() { echo "SKIP  $1 ($2)"; skip=$((skip + 1)); }

for f in $B/rife_cpu $B/rife_gl $B/pngdiff; do
  [ -x "$f" ] || { echo "missing $f: run make first (or make sanitize)"; exit 2; }
done
[ -f "$MODEL/flownet.bin" ] || { echo "SKIP  all end-to-end tests (no model at $MODEL: run ./fetch_model.sh)"; exit 0; }

# cpu NAME PADARGS TIMESTEP A B
cpu() {
  if $B/rife_cpu $2 "$MODEL/flownet.bin" "$4" "$5" "$3" "$T/cpu_$1.png" >"$T/log" 2>&1; then
    out=$($B/pngdiff "$T/cpu_$1.png" "tests/golden/cpu_$1.png" 1) && ok "cpu $1" || bad "cpu $1" "$out"
  else bad "cpu $1" "$(tail -1 "$T/log")"; fi
}

cpu frames    ""         0.5 data/frame0.png data/frame1.png
cpu odd       ""         0.3 data/odd0.png   data/odd1.png
cpu odd_pad64 "--pad 64" 0.7 data/odd0.png   data/odd1.png
cpu tiny      ""         0.5 data/tiny0.png  data/tiny1.png

# GL needs a GL 4.3 context; skip (not fail) on machines without one
gl() {
  if $B/rife_gl -m "$MODEL" -t "$2" "$3" "$4" "$T/gl_$1.png" >"$T/log" 2>&1; then
    out=$($B/pngdiff "$T/gl_$1.png" "tests/golden/gl_$1.png" 1) && ok "gl  $1" || bad "gl  $1" "$out"
  elif grep -q "libEGL\|eglInitialize\|EGL config\|OpenGL 4.3\|eglBindAPI\|eglMakeCurrent" "$T/log"; then
    skp "gl  $1" "no OpenGL 4.3 context"; gl_unavailable=1
  else bad "gl  $1" "$(tail -1 "$T/log")"; fi
}
gl_unavailable=
gl frames 0.5 data/frame0.png data/frame1.png
if [ -z "$gl_unavailable" ]; then
  gl odd  0.3 data/odd0.png  data/odd1.png
  gl tiny 0.5 data/tiny0.png data/tiny1.png
  # the two backends implement the same network
  out=$($B/pngdiff "$T/cpu_frames.png" "$T/gl_frames.png" 1) && ok "cpu vs gl agree (frames)" || bad "cpu vs gl (frames)" "$out"

  # the GL host parses the generic ncnn files: damaged ones must give a clear error (regression: these used to read
  # past the buffer / divide by zero). Only meaningful with a GL context, or it would die earlier for another reason.
  mkdir "$T/m_trunc" "$T/m_param"
  cp "$MODEL/flownet.param" "$T/m_trunc/"; head -c 5000000 "$MODEL/flownet.bin" > "$T/m_trunc/flownet.bin"
  cp "$MODEL/flownet.bin" "$T/m_param/"; sed 's/0=192 1=3 4=1 5=1 6=331776/0=0 1=3 4=1 5=1 6=331776/' "$MODEL/flownet.param" > "$T/m_param/flownet.param"
  for c in "m_trunc:truncated weights" "m_param:bad convolution parameters"; do
    dir=${c%%:*}; msg=${c#*:}
    if $B/rife_gl -m "$T/$dir" data/tiny0.png data/tiny1.png "$T/rej.png" >"$T/log" 2>&1; then bad "gl reject $dir" "exited 0"
    elif grep -q "$msg" "$T/log"; then ok "gl reject $dir"; else bad "gl reject $dir" "$(tail -1 "$T/log")"; fi
  done
fi

# malformed models must be rejected, not misread
head -c 1000 "$MODEL/flownet.bin" > "$T/trunc.bin"
{ cat "$MODEL/flownet.bin"; echo x; } > "$T/trail.bin"
head -c 4096 /dev/zero > "$T/zeros.bin"
for m in trunc trail zeros; do
  if $B/rife_cpu "$T/$m.bin" data/tiny0.png data/tiny1.png 0.5 "$T/rej.png" >/dev/null 2>&1; then
    bad "reject $m model" "exited 0"; else ok "reject $m model"; fi
done
if $B/rife_cpu "$T/nonexistent.bin" data/tiny0.png data/tiny1.png 0.5 "$T/rej.png" >/dev/null 2>&1; then
  bad "reject missing model" "exited 0"; else ok "reject missing model"; fi
if $B/rife_cpu --pad 33 "$MODEL/flownet.bin" data/tiny0.png data/tiny1.png 0.5 "$T/rej.png" >/dev/null 2>&1; then
  bad "reject --pad 33" "exited 0"; else ok "reject --pad 33"; fi
if $B/rife_cpu "$MODEL/flownet.bin" data/tiny0.png data/frame1.png 0.5 "$T/rej.png" >/dev/null 2>&1; then
  bad "reject mismatched frame sizes" "exited 0"; else ok "reject mismatched frame sizes"; fi

echo; echo "end-to-end: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
