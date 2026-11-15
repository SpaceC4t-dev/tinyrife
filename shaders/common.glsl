
layout(std430, binding = 4) readonly buffer Img0 { uint img0[]; };
layout(std430, binding = 5) readonly buffer Img1 { uint img1[]; };
layout(std430, binding = 6) buffer Flow { vec4 flow[]; };   // full res: (f0x, f0y, f1x, f1y)
layout(std430, binding = 7) buffer Mask { float mask[]; };  // full res
vec4 px0(int x, int y) { return unpackUnorm4x8(img0[y * W + x]); }
vec4 px1(int x, int y) { return unpackUnorm4x8(img1[y * W + x]); }
// backward warp == grid_sample(bilinear, border, align_corners=True), flow in pixels
#define WARP(PX, fx, fy) { \
  float gx = clamp(float(x) + fx, 0.0, float(W - 1)), gy = clamp(float(y) + fy, 0.0, float(H - 1)); \
  int x0 = int(floor(gx)), y0 = int(floor(gy)); int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1); \
  float ax = gx - float(x0), ay = gy - float(y0); \
  res = mix(mix(PX(x0, y0), PX(x1, y0), ax), mix(PX(x0, y1), PX(x1, y1), ax), ay); }
vec3 warp0(int x, int y, vec2 f) { vec4 res; WARP(px0, f.x, f.y) return res.rgb; }
vec3 warp1(int x, int y, vec2 f) { vec4 res; WARP(px1, f.x, f.y) return res.rgb; }
