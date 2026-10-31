// Bevy's sun shadow cascades (`calculate_cascade_bounds`, `build_directional_light_cascades`,
// `calculate_cascade`), ported from render-engine's `SunShadows` and `Cascade::fit`. Done in f32
// (Math.fround), like glam, because the fit rounds the diameter up to whole metres and snaps the
// centre to texels: a float64 result could land one texel away from Bevy's.

const f = Math.fround;

/** Bevy `CascadeShadowConfigBuilder` defaults. */
export const DEFAULT_SUN_SHADOWS = Object.freeze({
  numCascades: 4,
  minDistance: 0.1,
  firstCascadeFarBound: 10.0,
  maxDistance: 150.0,
  overlapProportion: 0.2,
  depthBias: 0.02,
  normalBias: 1.8,
});

/** One cascade covering `minDistance..maxDistance`. */
export function singleCascade(maxDistance) {
  return { ...DEFAULT_SUN_SHADOWS, numCascades: 1, maxDistance };
}

/** Far bound of each cascade (Bevy `calculate_cascade_bounds`). */
export function farBounds(s) {
  if (s.numCascades < 1 || s.numCascades > 4) throw new Error(`numCascades must be 1..=4, was ${s.numCascades}`);
  if (s.numCascades === 1) return [f(s.maxDistance)];
  const base = f(Math.pow(f(s.maxDistance / s.firstCascadeFarBound), f(1 / (s.numCascades - 1))));
  return Array.from({ length: s.numCascades }, (_, i) => f(f(s.firstCascadeFarBound) * f(Math.pow(base, i))));
}

/** `[near, far]` view distances of each cascade. */
export function cascadeRanges(s) {
  const far = farBounds(s);
  return far.map((fb, i) => [i === 0 ? f(s.minDistance) : f(f(1 - s.overlapProportion) * far[i - 1]), fb]);
}

const v3 = (x, y, z) => [f(x), f(y), f(z)];
const sub = (a, b) => v3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const dot = (a, b) => f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
const cross = (a, b) => v3(
  f(a[1] * b[2]) - f(a[2] * b[1]),
  f(a[2] * b[0]) - f(a[0] * b[2]),
  f(a[0] * b[1]) - f(a[1] * b[0]),
);
const normalize = (a) => {
  const inv = f(1 / f(Math.sqrt(dot(a, a))));
  return v3(a[0] * inv, a[1] * inv, a[2] * inv);
};

// Column-major 4×4 (glam/three layout): m[c * 4 + r].
function mul(a, b) {
  const out = new Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      // glam: a.x_axis * b.x + a.y_axis * b.y + a.z_axis * b.z + a.w_axis * b.w
      let s = f(a[r] * b[c * 4]);
      s = f(s + f(a[4 + r] * b[c * 4 + 1]));
      s = f(s + f(a[8 + r] * b[c * 4 + 2]));
      s = f(s + f(a[12 + r] * b[c * 4 + 3]));
      out[c * 4 + r] = s;
    }
  }
  return out;
}

function transformPoint(m, p) {
  const out = [0, 1, 2].map((r) => {
    let s = f(m[r] * p[0]);
    s = f(s + f(m[4 + r] * p[1]));
    s = f(s + f(m[8 + r] * p[2]));
    return f(s + m[12 + r]);
  });
  return out;
}

/**
 * One cascade fitted to the camera frustum slice `near..far` (Bevy `calculate_cascade`).
 * @param {number[]} worldFromView column-major camera world matrix
 * @returns {{ clipFromWorld: number[], texelSize: number, cascadeFromWorld: number[], diameter: number, depthRange: number }}
 */
export function fitCascade(near, far, mapSize, sunDirection, fovY, aspect, worldFromView) {
  const back = normalize(v3(-sunDirection[0], -sunDirection[1], -sunDirection[2]));
  const right = normalize(cross([0, 1, 0], back));
  const up = cross(back, right);
  // transpose(world_from_light): rows are right, up, back.
  const lightFromWorld = [
    right[0], up[0], back[0], 0,
    right[1], up[1], back[1], 0,
    right[2], up[2], back[2], 0,
    0, 0, 0, 1,
  ];
  const lightFromView = mul(lightFromWorld, worldFromView.map(f));

  const t = f(Math.tan(f(fovY * 0.5)));
  const corners = [];
  for (const z of [-near, -far]) {
    const a = f(f(Math.abs(z)) * t);
    const ax = f(a * aspect);
    corners.push(v3(ax, -a, z), v3(ax, a, z), v3(-ax, a, z), v3(-ax, -a, z));
  }
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const c of corners) {
    const p = transformPoint(lightFromView, c);
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  const body = dot(sub(corners[0], corners[6]), sub(corners[0], corners[6]));
  const farDiag = dot(sub(corners[4], corners[6]), sub(corners[4], corners[6]));
  const diameter = Math.ceil(f(Math.sqrt(Math.max(body, farDiag))));
  const texelSize = f(diameter / mapSize);
  const centre = [
    f(Math.floor(f(f(f(0.5 * f(min[0] + max[0])) / texelSize))) * texelSize),
    f(Math.floor(f(f(f(0.5 * f(min[1] + max[1])) / texelSize))) * texelSize),
    max[2],
  ];
  const cascadeFromWorld = [
    lightFromWorld[0], lightFromWorld[1], lightFromWorld[2], 0,
    lightFromWorld[4], lightFromWorld[5], lightFromWorld[6], 0,
    lightFromWorld[8], lightFromWorld[9], lightFromWorld[10], 0,
    -centre[0], -centre[1], -centre[2], 1,
  ];
  const depthRange = f(max[2] - min[2]);
  const r = f(1 / depthRange);
  const clipFromCascade = [
    f(2 / diameter), 0, 0, 0,
    0, f(2 / diameter), 0, 0,
    0, 0, r, 0,
    0, 0, 1, 1,
  ];
  return {
    clipFromWorld: mul(clipFromCascade, cascadeFromWorld),
    cascadeFromWorld,
    texelSize,
    diameter,
    depthRange,
  };
}
