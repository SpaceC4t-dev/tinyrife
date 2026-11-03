"""Video -> video frame interpolation with the OpenGL RIFE 4.6, piping raw RGB24 through ffmpeg (no intermediate files).
Usage: python run_gl_video.py in.webm out.mkv [-m 2] [-t SECONDS] [--vcodec ...]
Output has m x the frame rate; each source frame is followed by m-1 interpolated frames (last frame repeated m times).
Audio is copied when present."""
import argparse, json, queue, subprocess, threading, time
import gl_rife

ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("output")
ap.add_argument("-m", "--multi", type=int, default=2, help="frame-rate multiplier")
ap.add_argument("-t", "--duration", type=float, help="only process the first N seconds")
ap.add_argument("--vcodec", default="libsvtav1 -preset 8 -crf 28", help="ffmpeg video encoder + options")
args = ap.parse_args()

info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                           "stream=width,height,r_frame_rate", "-of", "json", args.input]))["streams"][0]
w, h = info["width"], info["height"]
num, den = map(int, info["r_frame_rate"].split("/"))
nbytes = w * h * 3
trim = ["-t", str(args.duration)] if args.duration else []

dec = subprocess.Popen(["ffmpeg", "-v", "error", *trim, "-i", args.input, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, bufsize=nbytes * 4)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{num * args.multi}/{den}", "-i", "-",
                        *trim, "-i", args.input, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
                        "-c:v", *args.vcodec.split(), "-pix_fmt", "yuv420p", args.output],
                       stdin=subprocess.PIPE, bufsize=nbytes * 4)

# decode and encode run on threads (pipe I/O releases the GIL); all GL calls stay on the main thread
q_in, q_out = queue.Queue(8), queue.Queue(8)
def reader():
  while (buf := dec.stdout.read(nbytes)) and len(buf) == nbytes: q_in.put(buf)
  q_in.put(None)
def writer():
  while (buf := q_out.get()) is not None: enc.stdin.write(buf)
  enc.stdin.close()
threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=writer, daemon=True)]
for t in threads: t.start()

g = gl_rife.GLRife(h, w)
t0, n_in, n_interp, gpu_s = time.perf_counter(), 0, 0, 0.0
prev = None
while (frame := q_in.get()) is not None:
  g.push(frame); n_in += 1
  if prev is not None:
    q_out.put(prev)
    for k in range(1, args.multi):
      tg = time.perf_counter(); q_out.put(g.interpolate_bytes(k / args.multi)); gpu_s += time.perf_counter() - tg
      n_interp += 1
    if n_in % 50 == 0:
      el = time.perf_counter() - t0
      print(f"  {n_in} frames in, {n_interp} interpolated, {n_interp / el:.2f} interp/s", flush=True)
  prev = frame
if prev is not None:
  for _ in range(args.multi): q_out.put(prev)
q_out.put(None)
for t in threads: t.join()
enc.wait(); dec.wait()
el = time.perf_counter() - t0
print(f"done: {n_in} -> {n_in * args.multi} frames ({w}x{h}, {num/den:g} -> {num*args.multi/den:g} fps) in {el:.1f}s | "
      f"{n_interp / el:.2f} interp/s overall, {gpu_s / max(1, n_interp) * 1e3:.0f} ms per interpolation")
if enc.returncode: raise SystemExit(f"encoder failed ({enc.returncode})")
