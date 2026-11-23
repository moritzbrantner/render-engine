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

// The eight view-space corners of the frustum slice `near..far`, near ones first.
function sliceCorners(near, far, fovY, aspect) {
  const t = f(Math.tan(f(fovY * 0.5)));
  const corners = [];
  for (const z of [-near, -far]) {
    const a = f(f(Math.abs(z)) * t);
    const ax = f(a * aspect);
    corners.push(v3(ax, -a, z), v3(ax, a, z), v3(-ax, a, z), v3(-ax, -a, z));
  }
  return corners;
}

/**
 * One cascade fitted to the camera frustum slice `near..far` (Bevy `calculate_cascade`).
 * `marginProportion` (default 0, Bevy's fit) widens it on every side by that share of its
 * diameter, rounded up to whole metres, so a camera that moves on stays inside it for a while
 * (`StaggeredCascades`); whole metres keep the texel size and its snapping stable.
 * @param {number[]} worldFromView column-major camera world matrix
 * @returns {{ clipFromWorld: number[], texelSize: number, cascadeFromWorld: number[], diameter: number, depthRange: number }}
 */
export function fitCascade(near, far, mapSize, sunDirection, fovY, aspect, worldFromView, marginProportion = 0) {
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

  const corners = sliceCorners(near, far, fovY, aspect);
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
  const fitted = Math.ceil(f(Math.sqrt(Math.max(body, farDiag))));
  const margin = marginProportion > 0 ? Math.ceil(fitted * marginProportion) : 0;
  const diameter = fitted + 2 * margin;
  const texelSize = f(diameter / mapSize);
  const centre = [
    f(Math.floor(f(f(f(0.5 * f(min[0] + max[0])) / texelSize))) * texelSize),
    f(Math.floor(f(f(f(0.5 * f(min[1] + max[1])) / texelSize))) * texelSize),
    f(max[2] + margin),
  ];
  min[2] = f(min[2] - margin);
  const cascadeFromWorld = [
    lightFromWorld[0], lightFromWorld[1], lightFromWorld[2], 0,
    lightFromWorld[4], lightFromWorld[5], lightFromWorld[6], 0,
    lightFromWorld[8], lightFromWorld[9], lightFromWorld[10], 0,
    -centre[0], -centre[1], -centre[2], 1,
  ];
  const depthRange = f(centre[2] - min[2]);
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

/**
 * Whether the camera frustum slice `near..far` lies inside `cascade` (a `fitCascade` result) with
 * `marginTexels` of its texels to spare on every side: then every receiver in the slice, moved by
 * the normal and depth biases and sampled by the filter, still finds its texels in that map.
 */
export function sliceInside(cascade, near, far, fovY, aspect, worldFromView, mapSize, marginTexels = 8) {
  const xy = 1 - (2 * marginTexels) / mapSize;
  const z = (marginTexels * cascade.texelSize) / cascade.depthRange;
  const m = cascade.clipFromWorld;
  for (const c of sliceCorners(near, far, fovY, aspect)) {
    const w = transformPoint(worldFromView, c);
    const x = m[0] * w[0] + m[4] * w[1] + m[8] * w[2] + m[12];
    const y = m[1] * w[0] + m[5] * w[1] + m[9] * w[2] + m[13];
    const d = m[2] * w[0] + m[6] * w[1] + m[10] * w[2] + m[14];
    if (Math.abs(x) > xy || Math.abs(y) > xy || d < z || d > 1 - z) return false;
  }
  return true;
}

/** Share of a staggered cascade's diameter added on every side (`fitCascade`'s margin). */
export const STAGGER_MARGIN = 0.05;

/** A kept cascade follows a sun that turned less than this since it was drawn (cos 0.5°). */
const SUN_TURN_COS = Math.cos((0.5 * Math.PI) / 180);

const unit = (d) => {
  const l = Math.hypot(d[0], d[1], d[2]);
  return [d[0] / l, d[1] / l, d[2] / l];
};

/**
 * Which sun cascades a frame redraws. Cascade `i` with an update interval `n` above 1 is redrawn
 * on the frames where `(frame + i) % n === 0`, so cascades of one interval take turns; in between
 * it keeps its map and the matrices it was drawn with, so its texels stay where they were and
 * nothing swims. A kept cascade is redrawn early when the current view slice no longer fits inside
 * it (`sliceInside`) or the sun turned more than half a degree since; every cascade is when the
 * cascade settings, the intervals, the map size or the scene change. Staggered cascades are fitted
 * with a margin (`STAGGER_MARGIN`); the others are Bevy's fit, drawn every frame.
 */
export class StaggeredCascades {
  #key = null;
  #scene = null;
  #kept = [];
  #frame = 0;

  /** Forgets every kept cascade: the next frame redraws all. */
  reset() {
    this.#key = null;
    this.#kept = [];
  }

  /**
   * This frame's cascades, each `{ far, redraw, ...fitCascade }`; `redraw` false keeps the map and
   * the matrices of an earlier frame. `intervals[i]` (default 1) is cascade `i`'s interval;
   * `scene` is any identity of what the maps hold. With `peek`, fresh fits that change nothing
   * kept and advance no frame (`Lighting.prepare`).
   */
  update({ shadows, mapSize, sunDirection, fovY, aspect, worldFromView, intervals = [], scene = null, peek = false }) {
    const ranges = cascadeRanges(shadows);
    if (peek) {
      return ranges.map(([near, far]) => ({ far, redraw: true, ...fitCascade(near, far, mapSize, sunDirection, fovY, aspect, worldFromView) }));
    }
    const key = JSON.stringify([shadows, mapSize, intervals]);
    if (key !== this.#key || scene !== this.#scene) {
      this.reset();
      this.#key = key;
      this.#scene = scene;
    }
    const sun = unit(sunDirection);
    const frame = this.#frame++;
    return ranges.map(([near, far], i) => {
      const n = Math.max(1, Math.floor(intervals[i] ?? 1));
      const kept = this.#kept[i];
      if (
        n > 1 &&
        kept &&
        (frame + i) % n !== 0 &&
        kept.sun[0] * sun[0] + kept.sun[1] * sun[1] + kept.sun[2] * sun[2] >= SUN_TURN_COS &&
        sliceInside(kept, near, far, fovY, aspect, worldFromView, mapSize)
      ) {
        return { ...kept, redraw: false };
      }
      const margin = n > 1 ? STAGGER_MARGIN : 0;
      const cascade = { far, redraw: true, sun, ...fitCascade(near, far, mapSize, sunDirection, fovY, aspect, worldFromView, margin) };
      this.#kept[i] = cascade;
      return cascade;
    });
  }
}
