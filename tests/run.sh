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
TO=; command -v timeout >/dev/null 2>&1 && TO="timeout 10"   # a hung pipe must fail a test, not stall the suite
MODEL=${MODEL:-ext/rife-ncnn-vulkan/models/rife-v4.6}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
pass=0; fail=0; skip=0
ok()  { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1${2:+ -- $2}"; fail=$((fail + 1)); }
skp() { echo "SKIP  $1 ($2)"; skip=$((skip + 1)); }

for f in $B/cpu $B/gl $B/gl_dir $B/gl_dir_fake $B/check_dir $B/gl_video $B/gl_video_fake $B/check_video $B/pngdiff; do
  [ -x "$f" ] || { echo "missing $f: run make first (or make sanitize)"; exit 2; }
done
# ---- gl_dir with the fake GL host (average of the two frames): structure, copies, threading, errors; no GPU, no model ----
S1="$T/seq"; mkdir "$S1"
i=0; for f in frame0 frame1 frame0 frame1 frame0; do cp "data/$f.png" "$S1/f$i.png"; i=$((i + 1)); done
for j in 1 3 16; do
  rm -rf "$T/o"
  if $B/gl_dir_fake -j $j "$S1" "$T/o" >"$T/log" 2>&1; then
    out=$($B/check_dir "$S1" "$T/o" avg) && ok "dir (fake host) -j $j" || bad "dir (fake host) -j $j" "$out"
  else bad "dir (fake host) -j $j" "$(tail -1 "$T/log")"; fi
done
S2="$T/seq_odd"; mkdir "$S2"; cp data/odd0.png "$S2/a.png"; cp data/odd1.png "$S2/b.png"; cp data/odd0.png "$S2/c.png"
$B/gl_dir_fake "$S2" "$T/deep/nested/out/" >"$T/log" 2>&1 && out=$($B/check_dir "$S2" "$T/deep/nested/out" avg) && ok "dir: odd size, nested output dir, trailing slash" || bad "dir: odd size / nested output" "$out $(tail -1 "$T/log")"
$B/gl_dir_fake "$S2" "$T/deep/nested/out" >"$T/log" 2>&1 && out=$($B/check_dir "$S2" "$T/deep/nested/out" avg) && ok "dir: re-run over an existing output dir" || bad "dir: re-run" "$out"
S3="$T/seq_noise"; mkdir "$S3"; cp data/tiny0.png "$S3/a.png"; cp data/tiny1.png "$S3/b.png"
cp data/tiny0.png "$S3/.hidden.png"; cp data/tiny0.png "$S3/UPPER.PNG"; echo notes > "$S3/readme.txt"
$B/gl_dir_fake "$S3" "$T/o3" >"$T/log" 2>&1 && out=$($B/check_dir "$S3" "$T/o3" avg) && ok "dir: hidden, .txt and .PNG files are ignored" || bad "dir: noise files" "$out"

rejects() { # description, expected message, command...
  d=$1; want=$2; shift 2
  if $TO "$@" >"$T/log" 2>&1; then bad "$d" "exited 0"; elif grep -q "$want" "$T/log"; then ok "$d"; else bad "$d" "$(tail -1 "$T/log")"; fi
}
mkdir "$T/one"; cp data/tiny0.png "$T/one/a.png"
mkdir "$T/mismatch"; cp data/tiny0.png "$T/mismatch/a.png"; cp data/frame0.png "$T/mismatch/b.png"; cp data/tiny1.png "$T/mismatch/c.png"
mkdir "$T/corrupt"; cp data/tiny0.png "$T/corrupt/a.png"; head -c 100 data/tiny1.png > "$T/corrupt/b.png"; cp data/tiny0.png "$T/corrupt/c.png"
: > "$T/afile"
rejects "dir rejects a single frame"        "need at least 2"       $B/gl_dir_fake "$T/one" "$T/oe1"
rejects "dir rejects a size mismatch"       "size mismatch"         $B/gl_dir_fake "$T/mismatch" "$T/oe2"
rejects "dir rejects a corrupt PNG"         "cannot read"           $B/gl_dir_fake "$T/corrupt" "$T/oe3"
rejects "dir rejects a missing input dir"   "cannot open directory" $B/gl_dir_fake "$T/nonexistent" "$T/oe4"
rejects "dir rejects output path = a file"  "not a directory"       $B/gl_dir_fake "$S1" "$T/afile"
rejects "dir rejects -j 0"                  "usage"                 $B/gl_dir_fake -j 0 "$S1" "$T/oe5"

# the reader/queues/writer pool under ThreadSanitizer (only built by make sanitize)
# (many tiny frames: a race between the reader and the consumer only shows up if they get many chances to collide)
if [ -x "$B/gl_dir_fake_tsan" ]; then
  S5="$T/seq_many"; mkdir "$S5"; i=0
  while [ $i -lt 60 ]; do cp "data/tiny$((i % 2)).png" "$(printf "$S5/m%03d.png" $i)"; i=$((i + 1)); done
  racy=0
  for j in 2 5; do
    rm -rf "$T/ts"
    TSAN_OPTIONS=halt_on_error=0 $B/gl_dir_fake_tsan -j $j "$S5" "$T/ts" >"$T/log" 2>&1 || racy=1
    grep -q "ThreadSanitizer" "$T/log" && racy=1
    [ $racy -eq 1 ] && break
  done
  if [ $racy -eq 0 ]; then ok "dir threading under ThreadSanitizer"; else bad "dir threading under ThreadSanitizer" "$(grep -m1 -A3 ThreadSanitizer "$T/log" | tr '\n' ' ')"; fi
fi

# ---- gl_video with fake ffmpeg/ffprobe scripts and the fake GL host: exact output, the ffmpeg command lines, error paths.
# No ffmpeg, GPU, codec or model needed (raw RGB24 files stand in for videos; see tests/fake_ffmpeg.sh). ----
FK="env FFMPEG=tests/fake_ffmpeg.sh FFPROBE=tests/fake_ffprobe.sh FAKE_W=64 FAKE_H=48 FAKE_RATE=30000/1001 FAKE_LOG=$T/ff.log"
$B/check_video gen "$T/v.raw" 64 48 6 1
$B/check_video gen "$T/v1.raw" 64 48 1 2
{ cat "$T/v.raw"; head -c 1000 /dev/zero; } > "$T/v_partial.raw"     # a cut-off last frame must be dropped
vtest() { # name, multiplier, input, extra args...
  name=$1; mult=$2; vin=$3; shift 3
  rm -f "$T/ff.log" "$T/vo.raw"
  if $TO $FK $B/gl_video_fake -x "$mult" "$@" "$vin" "$T/vo.raw" >"$T/log" 2>&1; then
    out=$($B/check_video verify "$vin" "$T/vo.raw" 64 48 "$mult") && ok "video: $name" || bad "video: $name" "$out"
  else bad "video: $name" "$(tail -1 "$T/log")"; fi
}
logs_have() { # description, patterns that must all appear in the ffmpeg command lines
  d=$1; shift
  for pat in "$@"; do grep -q -- "$pat" "$T/ff.log" || { bad "$d" "not in the ffmpeg command lines: $pat"; return; }; done
  ok "$d"
}
vtest "x1 copies every frame"                 1 "$T/v.raw"
vtest "x3 blends at 1/3 and 2/3"              3 "$T/v.raw"
vtest "x5"                                    5 "$T/v.raw"
vtest "a single frame is repeated x times"    3 "$T/v1.raw"
vtest "a cut-off last frame is dropped"       2 "$T/v_partial.raw"
vtest "x2, default codec, no trim"            2 "$T/v.raw"
logs_have "video: encoder gets the doubled rate, size, default codec, audio copy" \
  "-r 60000/1001 -i -" "-s 64x48 " "-map 1:a? -c:a copy -c:v libsvtav1 -preset 8 -crf 28 -pix_fmt yuv420p"
logs_have "video: decoder asks for raw rgb24 on stdout" "-f rawvideo -pix_fmt rgb24 -$"
grep -q -- " -t " "$T/ff.log" && bad "video: no -t unless asked" "-t was passed" || ok "video: no -t unless asked"
vtest "custom codec and -t reach both ffmpegs" 2 "$T/v.raw" --vcodec "libx264 -crf 20 -preset fast" -t 2.5
logs_have "video: --vcodec options are passed through" "-c:v libx264 -crf 20 -preset fast -pix_fmt yuv420p"
[ "$(grep -c -- '-t 2.5' "$T/ff.log")" = 2 ] && ok "video: -t goes to the decoder and the encoder" || bad "video: -t goes to the decoder and the encoder" "$(grep -c -- '-t 2.5' "$T/ff.log") of 2"

: > "$T/empty.raw"
rejects "video rejects no arguments"          "usage"                       $B/gl_video_fake
rejects "video rejects -x 0"                  "usage"                       $B/gl_video_fake -x 0 "$T/v.raw" "$T/vo.raw"
rejects "video rejects -x 17"                 "usage"                       $B/gl_video_fake -x 17 "$T/v.raw" "$T/vo.raw"
rejects "video rejects a negative -t"         "usage"                       $B/gl_video_fake -t -1 "$T/v.raw" "$T/vo.raw"
rejects "video: ffprobe fails"                "cannot read the video stream" $FK FAKE_PROBE_FAIL=1 $B/gl_video_fake "$T/v.raw" "$T/vo.raw"
rejects "video: ffprobe missing"              "cannot read the video stream" $FK FFPROBE=/no/such/ffprobe $B/gl_video_fake "$T/v.raw" "$T/vo.raw"
rejects "video: no frames decoded"            "no frames could be decoded"  $FK $B/gl_video_fake "$T/empty.raw" "$T/vo.raw"
rejects "video: ffmpeg missing"               "no frames could be decoded"  $FK FFMPEG=/no/such/ffmpeg $B/gl_video_fake "$T/v.raw" "$T/vo.raw"
rejects "video: the encoder fails at once"    "the encoder failed"          $FK FAKE_ENC_FAIL=1 $B/gl_video_fake "$T/v.raw" "$T/vo.raw"
rejects "video: the encoder fails at the end" "the encoder failed"          $FK FAKE_ENC_FAIL_LATE=1 $B/gl_video_fake "$T/v.raw" "$T/vo.raw"
# a decoder that dies after producing frames: keep what was decoded, but say so
if $TO $FK FAKE_DEC_EXIT=3 $B/gl_video_fake -x 2 "$T/v.raw" "$T/vo.raw" >"$T/log" 2>&1 && grep -q "decoder exited with status 3" "$T/log" \
   && $B/check_video verify "$T/v.raw" "$T/vo.raw" 64 48 2 >/dev/null; then ok "video: a failing decoder gives a warning and the frames so far"
else bad "video: a failing decoder gives a warning and the frames so far" "$(tail -1 "$T/log")"; fi

if [ -x "$B/gl_video_fake_tsan" ]; then
  $B/check_video gen "$T/v60.raw" 64 48 60 3
  racy=0
  for m in 2 3; do
    TSAN_OPTIONS=halt_on_error=0 $TO $FK $B/gl_video_fake_tsan -x $m "$T/v60.raw" "$T/vo.raw" >"$T/log" 2>&1 || racy=1
    grep -q "ThreadSanitizer" "$T/log" && racy=1
  done
  if [ $racy -eq 0 ]; then ok "video threading under ThreadSanitizer"; else bad "video threading under ThreadSanitizer" "$(grep -m1 -A3 ThreadSanitizer "$T/log" | tr '\n' ' ')"; fi
fi

# with a real ffmpeg on PATH: a real video through real encoding, audio copied bit for bit
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  ffmpeg -v error -y -f lavfi -i "testsrc2=size=64x48:rate=10:duration=1" -f lavfi -i "sine=frequency=440:duration=1" \
    -c:v ffv1 -pix_fmt yuv420p -c:a flac -shortest "$T/real_in.mkv" >/dev/null 2>&1
  real_ffmpeg=1
  if $B/gl_video_fake -x 2 --vcodec ffv1 "$T/real_in.mkv" "$T/real_out.mkv" >"$T/log" 2>&1; then
    info=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=r_frame_rate,nb_read_frames -of csv=p=0 "$T/real_out.mkv")
    a_in=$(ffmpeg -v error -i "$T/real_in.mkv" -map 0:a -f md5 - 2>&1); a_out=$(ffmpeg -v error -i "$T/real_out.mkv" -map 0:a -f md5 - 2>&1)
    [ "$info" = "20/1,20" ] && [ -n "$a_in" ] && [ "$a_in" = "$a_out" ] && ok "video (real ffmpeg): 10 -> 20 frames at 20 fps, audio copied bit for bit" \
      || bad "video (real ffmpeg)" "video: $info, audio in/out: $a_in / $a_out"
  else bad "video (real ffmpeg)" "$(tail -1 "$T/log")"; fi
else skp "video (real ffmpeg)" "no ffmpeg/ffprobe on PATH"; fi

[ -f "$MODEL/flownet.bin" ] || { echo "SKIP  all end-to-end tests (no model at $MODEL: run ./fetch_model.sh)"; exit 0; }

# cpu NAME PADARGS TIMESTEP A B
cpu() {
  if $B/cpu $2 "$MODEL/flownet.bin" "$4" "$5" "$3" "$T/cpu_$1.png" >"$T/log" 2>&1; then
    out=$($B/pngdiff "$T/cpu_$1.png" "tests/golden/cpu_$1.png" 1) && ok "cpu $1" || bad "cpu $1" "$out"
  else bad "cpu $1" "$(tail -1 "$T/log")"; fi
}

cpu frames    ""         0.5 data/frame0.png data/frame1.png
cpu odd       ""         0.3 data/odd0.png   data/odd1.png
cpu odd_pad64 "--pad 64" 0.7 data/odd0.png   data/odd1.png
cpu tiny      ""         0.5 data/tiny0.png  data/tiny1.png

# GL needs a GL 4.3 context; skip (not fail) on machines without one
gl() {
  if $B/gl -m "$MODEL" -t "$2" "$3" "$4" "$T/gl_$1.png" >"$T/log" 2>&1; then
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

  # gl_dir on the GPU: same frames as gl produces for each pair (frames a, b, a -> pairs (a,b) and (b,a))
  S4="$T/seq_gl"; mkdir "$S4"; cp data/frame0.png "$S4/a.png"; cp data/frame1.png "$S4/b.png"; cp data/frame0.png "$S4/c.png"
  if $B/gl_dir -m "$MODEL" "$S4" "$T/o4" >"$T/log" 2>&1; then
    out=$($B/check_dir "$S4" "$T/o4") || bad "dir (gl): structure and copies" "$out"
    $B/gl -m "$MODEL" -t 0.5 "$S4/a.png" "$S4/b.png" "$T/p1.png" >/dev/null 2>&1
    $B/gl -m "$MODEL" -t 0.5 "$S4/b.png" "$S4/c.png" "$T/p2.png" >/dev/null 2>&1
    out1=$($B/pngdiff "$T/p1.png" "$T/o4/00000002.png" 0) && out2=$($B/pngdiff "$T/p2.png" "$T/o4/00000004.png" 0) \
      && ok "dir (gl): in-between frames identical to gl's" || bad "dir (gl): in-between frames" "$out1 | $out2"
  else bad "dir (gl)" "$(tail -1 "$T/log")"; fi

  # gl_video on the GPU with a real ffmpeg: 10 frames -> 20 at double the rate, audio copied
  if [ -n "$real_ffmpeg" ]; then
    if $TO $B/gl_video -m "$MODEL" -x 2 --vcodec ffv1 "$T/real_in.mkv" "$T/gl_out.mkv" >"$T/log" 2>&1; then
      info=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=r_frame_rate,nb_read_frames -of csv=p=0 "$T/gl_out.mkv")
      a_in=$(ffmpeg -v error -i "$T/real_in.mkv" -map 0:a -f md5 - 2>&1); a_out=$(ffmpeg -v error -i "$T/gl_out.mkv" -map 0:a -f md5 - 2>&1)
      [ "$info" = "20/1,20" ] && [ -n "$a_in" ] && [ "$a_in" = "$a_out" ] && ok "video (gl + real ffmpeg): 10 -> 20 frames, audio copied" \
        || bad "video (gl + real ffmpeg)" "video: $info, audio in/out: $a_in / $a_out"
    else bad "video (gl + real ffmpeg)" "$(tail -1 "$T/log")"; fi
  fi

  # the GL host parses the generic ncnn files: damaged ones must give a clear error (regression: these used to read
  # past the buffer / divide by zero). Only meaningful with a GL context, or it would die earlier for another reason.
  mkdir "$T/m_trunc" "$T/m_param"
  cp "$MODEL/flownet.param" "$T/m_trunc/"; head -c 5000000 "$MODEL/flownet.bin" > "$T/m_trunc/flownet.bin"
  cp "$MODEL/flownet.bin" "$T/m_param/"; sed 's/0=192 1=3 4=1 5=1 6=331776/0=0 1=3 4=1 5=1 6=331776/' "$MODEL/flownet.param" > "$T/m_param/flownet.param"
  for c in "m_trunc:truncated weights" "m_param:bad convolution parameters"; do
    dir=${c%%:*}; msg=${c#*:}
    if $B/gl -m "$T/$dir" data/tiny0.png data/tiny1.png "$T/rej.png" >"$T/log" 2>&1; then bad "gl reject $dir" "exited 0"
    elif grep -q "$msg" "$T/log"; then ok "gl reject $dir"; else bad "gl reject $dir" "$(tail -1 "$T/log")"; fi
  done
fi

# malformed models must be rejected, not misread
head -c 1000 "$MODEL/flownet.bin" > "$T/trunc.bin"
{ cat "$MODEL/flownet.bin"; echo x; } > "$T/trail.bin"
head -c 4096 /dev/zero > "$T/zeros.bin"
for m in trunc trail zeros; do
  if $B/cpu "$T/$m.bin" data/tiny0.png data/tiny1.png 0.5 "$T/rej.png" >/dev/null 2>&1; then
    bad "reject $m model" "exited 0"; else ok "reject $m model"; fi
done
if $B/cpu "$T/nonexistent.bin" data/tiny0.png data/tiny1.png 0.5 "$T/rej.png" >/dev/null 2>&1; then
  bad "reject missing model" "exited 0"; else ok "reject missing model"; fi
if $B/cpu --pad 33 "$MODEL/flownet.bin" data/tiny0.png data/tiny1.png 0.5 "$T/rej.png" >/dev/null 2>&1; then
  bad "reject --pad 33" "exited 0"; else ok "reject --pad 33"; fi
if $B/cpu "$MODEL/flownet.bin" data/tiny0.png data/frame1.png 0.5 "$T/rej.png" >/dev/null 2>&1; then
  bad "reject mismatched frame sizes" "exited 0"; else ok "reject mismatched frame sizes"; fi

echo; echo "end-to-end: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
