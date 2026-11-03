"""Benchmark the OpenGL RIFE. Usage: python bench_gl.py WxH [-n N] [--px 2 --ot 4 --lx 16 --ly 4] [--breakdown]"""
import argparse, time, numpy as np
import gl_rife

ap = argparse.ArgumentParser()
ap.add_argument("res", nargs="?", default="1280x720")
ap.add_argument("-n", type=int, default=10)
ap.add_argument("--px", type=int, default=1); ap.add_argument("--ot", type=int, default=4)
ap.add_argument("--lx", type=int, default=16); ap.add_argument("--ly", type=int, default=4)
ap.add_argument("--breakdown", action="store_true")
args = ap.parse_args()
w, h = map(int, args.res.split("x"))
rng = np.random.default_rng(0)
a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8); b = np.roll(a, 7, axis=1)
g = gl_rife.GLRife(h, w, px=args.px, ot=args.ot, lx=args.lx, ly=args.ly)
ctx = g.ctx
for _ in range(2): g(a, b)

q = ctx.query(time=True)
gpu = []
for _ in range(args.n):
  with q: g.run()
  gpu.append(q.elapsed / 1e6)
ctx.finish(); t0 = time.perf_counter()
for _ in range(args.n): g.run()
ctx.finish(); wall = (time.perf_counter() - t0) / args.n * 1e3
t0 = time.perf_counter()
for _ in range(args.n): g(a, b)
e2e = (time.perf_counter() - t0) / args.n * 1e3
gm = float(np.median(gpu))
print(f"GL {w}x{h} (padded {g.W}x{g.H}) px={args.px} ot={args.ot} lx={args.lx} ly={args.ly}: "
      f"gpu {gm:.1f} ms ({1e3/gm:.2f} fps) | wall {wall:.1f} ms | end-to-end {e2e:.1f} ms ({1e3/e2e:.2f} fps)")

if args.breakdown:
  names = []
  for bi, s in enumerate(gl_rife.SCALES):
    names += [f"b{bi} prep", f"b{bi} conv0a", f"b{bi} conv0b"] + [f"b{bi} res{k}" for k in range(8)] + [f"b{bi} deconv", f"b{bi} update"]
  names.append("final")
  times = np.zeros(len(g.steps))
  for _ in range(3):
    for i, (prog, binds, groups, has_t) in enumerate(g.steps):
      for k, bb in g.shared.items(): bb.bind_to_storage_buffer(k)
      for k, bb in binds.items(): bb.bind_to_storage_buffer(k)
      if has_t: prog["timestep"] = 0.5
      with q: prog.run(*groups); ctx.memory_barrier()
      times[i] += q.elapsed / 1e6 / 3
  agg = {}
  for n, t in zip(names, times):
    k = n if " " not in n else n.split()[0] + " " + ("res" if "res" in n else n.split()[1])
    agg[k] = agg.get(k, 0) + t
  for k, t in agg.items(): print(f"  {k:12s} {t:7.2f} ms")
  print(f"  total        {times.sum():7.2f} ms")
