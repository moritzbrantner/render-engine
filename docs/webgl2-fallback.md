# WebGL 2 fallback in the browser

What a page does when browser WebGPU is unavailable: it falls back to the three.js port on
WebGL 2 (`packages/three-lighting`, D8), and shows an error naming both reasons when neither
starts. render-engine's own wgpu renderer is **not** run on wgpu's WebGL 2 backend; the probe
below shows why. Native rendering is unchanged, and no engine code changed for this.

## Result

- **wgpu 30's WebGL 2 backend cannot run the existing renderer** without a second binding model:
  storage buffers in the vertex and fragment stages, a cube-array texture, 11 compute pipelines,
  indirect buffers and a sampled multisampled depth texture, the last of which aborts the wasm
  module. Working around them is a renderer rewrite, so it was not attempted.
- **The fallback is `three-lighting`**, which already is the WebGL 2 port of `lit.wgsl`,
  `tonemap.wgsl` and the sky passes, gated against the same Bevy scenarios on CI (D8,
  `three_side::GAPS` is empty). No second renderer was written.
- **Backend selection** is `selectBackend` in `packages/three-lighting/backend.js`
  (`@moritzbrantner/three-lighting/backend`): WebGPU first, then WebGL 2, then
  `NoGraphicsBackendError`. A backend whose initialiser throws falls through to the next one.
- In headless Chromium, the WebGL 2 frames of `sun_shadows` and `textured_props` (textured glTF
  props, sun shadows, a blended sphere) are within D3's gates against Bevy (FLIP p99 ≤ 0.0006), and
  within 0.0005 of native render-engine.

## Feasibility: render-engine on wgpu's WebGL 2 backend

`oracle/webgpu-smoke` has a `probe` mode: wgpu with only `Backends::GL` (the `webgl` feature, in
the smoke's manifest only), an adapter from a canvas, a device with the adapter's own limits
(`Limits::default()` is refused: WebGPU's compute and storage limits are above WebGL 2's zero),
then `Renderer::new` and one `sun_shadows` frame with every validation error collected:

```sh
node oracle/webgpu-smoke/run.mjs target/oracle probe sun_shadows
```

Adapter: `Gl`, ANGLE on llvmpipe (OpenGL ES 3.2), Chromium 159 headless shell. Its limits:
0 storage buffers and storage textures per stage, 0 compute workgroups, 16 sampled textures per
stage, 64 KiB uniform bindings. Downlevel flags it lacks include `COMPUTE_SHADERS`,
`VERTEX_STORAGE`, `FRAGMENT_STORAGE`, `INDIRECT_EXECUTION`, `CUBE_ARRAY_TEXTURES`,
`VIEW_FORMATS`, `SURFACE_VIEW_FORMATS` and `DEPTH_BIAS_CLAMP`.

| Engine use | Error on WebGL 2 | What running it there would take |
| --- | --- | --- |
| Point-light shadows: `texture_depth_cube_array` in `lit.wgsl` and the shaded particles | `CUBE_ARRAY_TEXTURES` required | One cube per light, or an atlas: new bindings and shadow sampling |
| Instances, visible indices and joint matrices in storage buffers (group 0, vertex stage) | `VERTEX_STORAGE` required | Instance data as vertex attributes or uniform arrays, joints in a texture: every lit and shadow pipeline and `prepare` |
| The light list in a fragment storage buffer (group 1) | 0 fragment storage buffers | A fixed-size uniform array |
| Sky LUTs (4), environment filter (4), particle simulation (3) | `COMPUTE_SHADERS` required, at `Renderer::new` | Fragment-pass versions of the sky, a CPU particle simulation |
| Particle counters and indirect args | `INDIRECT_EXECUTION` required | Direct draws with CPU counts |
| The 4× MSAA depth, sampled by the sky pass | **Aborts the module** in glow (`texStorage2DMultisample` is ES 3.1) | A resolved depth copy |
| The sRGB view of an `rgba8unorm` canvas | `VIEW_FORMATS` absent (not reached) | A different presentation path |

Every row is a renderer-wide change: together, a second backend of the renderer. Each of them
is something `three-lighting` already does in GLSL ES 3.0 (point lights in their own cube
targets, PCSS through an R32F depth copy, the sky's LUTs in fragment passes; D8).

## Backend selection

```js
import { selectBackend, NoGraphicsBackendError } from '@moritzbrantner/three-lighting/backend';

try {
  const { backend, value, attempts } = await selectBackend({
    webgpu: async () => startEngine(canvas), // render-engine compiled to wasm
    webgl2: async () => startThreeLighting(canvas), // WebGLRenderer + Lighting
  });
  // attempts: [{ backend, reason }] for what failed before `backend`
} catch (e) {
  if (e instanceof NoGraphicsBackendError) showError(e.message); // names both reasons
  else throw e;
}
```

- **Probe, then initialise.** WebGPU needs `navigator.gpu` and an adapter (it is not kept: wgpu
  requests its own). WebGL 2 needs a context with `EXT_color_buffer_float` (the port's half-float
  HDR targets); the probe context is released. A sky also needs `OES_texture_float_linear`.
- **Initialisation failures fall through**: an initialiser that throws (device creation, pipeline
  validation, a scene the build cannot load) is recorded and the next backend is tried.
- **`backends`** restricts and orders the candidates, e.g. from a page's `?backend=webgl2`.
- The module imports nothing, so a page decides before loading the wasm or three.js. It does not
  handle device or context loss after start.

`oracle/webgpu-smoke`'s `select` mode is the page that uses it: WebGPU renders the scenario with
the wasm build, WebGL 2 with the oracle's own three.js scene module (`oracle/three/scene.js`, from
the scenario's export), and with neither the page shows the error.

## What the fallback supports

Everything the three.js port supports ([consumer guide](consumer-integration.md#web-client-threejs)):
every lighting scenario of the oracle passes against Bevy, including textured and skinned glTF,
sun cascades, point and spot shadows with PCSS, IBL, fog, the sky and day cycle, and tone mapping.
On WebGL 2 it uses the paths that exist without compute or storage buffers:

| Engine path | On the fallback |
| --- | --- |
| Light assignment | Every point and spot light for every fragment, the engine's reference path (`LightAssignment::PerFragment`). No clustering. |
| Culling | CPU: three.js frustum culling, plus `culling.js` per shadow view for instanced meshes. No `Culling::GpuIndirect`. |
| Particles | CPU-simulated: `ParticleEmitter` replays the engine's emitters on the CPU and draws instanced quads (#126); every bevy_hanabi scenario passes against Bevy, and so do lit, fogged and soft particles (#130). |
| HUD overlay | Not drawn: the web client draws its HUD in CSS (D8). |

Limitations: it is a second implementation of the shading, so a shading change in the engine is
ported by hand (the D8 rule), and its CPU cost is three.js's, not measured here.

## Comparison

Headless Chromium 159 and Firefox 159 on this machine (no hardware GPU: llvmpipe/lavapipe, WSL2),
2026-11-20.
Bevy, native render-engine and the oracle's three.js frame come from the oracle
(`--only sun_shadows --only textured_props`, lavapipe). FLIP mean / p99, D3's gates 0.005 / 0.05.

| Scenario | WebGL 2 fallback vs Bevy | vs native engine | vs browser WebGPU (engine wasm) | vs oracle three.js (differing px) | Native engine vs Bevy |
| --- | --- | --- | --- | --- | --- |
| `sun_shadows` | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0243 / 0.0493 | 0.0000 / 0.0000 (14) | 0.0000 / 0.0000 |
| `textured_props` | 0.0000 / 0.0006 | 0.0000 / 0.0005 | not run (glTF from disk; the page fetches it since #129) | 0.0000 / 0.0002 (96) | 0.0000 / 0.0005 |

Playwright's Firefox 159 gives the same WebGL 2 numbers (its WebGL 2 is also on llvmpipe), and its
WebGPU (behind `dom.webgpu.enabled`; wgpu on the same llvmpipe Vulkan driver) renders
`sun_shadows` byte-identically to native render-engine: 0 differing pixels, FLIP 0.0000 / 0.0000
against Bevy.

The WebGL 2 frames ran through ANGLE on llvmpipe (OpenGL ES); the oracle's three.js side uses
ANGLE on Vulkan, hence the few differing pixels. The browser's WebGPU column is the software
adapter Chromium exposes here (wgpu reports no name): 99.9 % of its pixels are within one 8-bit
step of native and about 700 differ on shadow edges. That is the adapter, not the engine: the
same WebGL 2 page rendered through ANGLE on SwiftShader is identical to that WebGPU frame (FLIP
0.0000) and shows the same offset against Bevy, as #75 found for SwiftShader. On the RTX 3060 Ti,
#75 measured WebGPU against Bevy at 0.0000 / 0.0000 on `sun_shadows`.

### Particles

The oracle's three.js side runs every particle scenario (bevy_hanabi and `StandardMaterial` quad
references) with `ParticleEmitter`, driven as it drives render-engine, under D9's gates
(lavapipe, headless Chromium through ANGLE on Vulkan; FLIP mean / p99, and coverage / spatial /
colour distances for statistical captures):

| Scenario | Capture | three.js vs Bevy | render-engine vs Bevy |
| --- | --- | --- | --- |
| `particles_fountain` | 0.5 s exact, 2.0 s statistical | 0.0000 / 0.0000; 0.000 / 0.000 / 0.000 | the same |
| `particles_100k` | 0.75 s exact, 3.0 s statistical | 0.0001 / 0.0016; 0.000 / 0.000 / 0.000 | 0.0000 / 0.0011; 0.000 / 0.000 / 0.000 |
| `particles_alpha` | 1.0 s, 3.0 s statistical | 0.000 / 0.009 / 0.030–0.051 (gate 0.065 / 0.07 / 0.095) | 0.000 / 0.004 / 0.034 |
| `particles_textured` | 0.625 s exact, 2.5 s statistical | 0.0001 / 0.0017; 0.000 / 0.000 / 0.001 | 0.0000 / 0.0011; 0.000 / 0.000 / 0.000 |
| `smoke_and_sparks` | 1.0 s, 3.0 s statistical | 0.000 / 0.002–0.005 / 0.001 | 0.000 / 0.001–0.002 / 0.000 |
| `particles_snapshot` | 0.5 s, 1.0 s exact | 0.0000 / 0.0000 | the same |
| `particles_cone` | 0.5 s exact, 2.5 s statistical | 0.0000 / 0.0000; 0.000 / 0.000 / 0.000 | the same |
| `particles_lit_fog` | 0.5 s, 2.5 s exact | 0.0000 / 0.0000 | the same |
| `particles_soft` | 0.5 s, 2.5 s exact | 0.0000 / 0.0001 | 0.0000 / 0.0000 |

The spawn and alive counts match Bevy's on every frame of every scenario. `particles_alpha`'s
colour distance varies between runs because Bevy's GPU decides the alpha-blended draw order
within an emitter; the three.js side's is deterministic.

## Tests

- `packages/three-lighting/backend.test.js` (`bun test packages`, on CI): WebGPU preferred;
  fallback without `navigator.gpu`, without an adapter and when `requestAdapter` rejects; a
  WebGPU initialiser that throws; the error naming both backends; forced backends; the WebGL 2
  probe's extension check and context release.
- `scripts/browser-backends.sh` (local, like the WebGPU smoke; not on CI): `check-backends.mjs`
  runs the `select` page in headless Chromium with WebGPU available (the engine renders
  `sun_shadows`; `textured_props` with its glTF files withheld, `assets=0`, fails to initialise in
  wasm and falls back), with WebGPU off
  (WebGL 2 renders both) and with WebGPU and WebGL off (the error on the page), then
  `compare select` fails if a WebGL 2 frame is outside D3's gates against Bevy. `SKIP_WEBGPU=1`
  skips the WebGPU cases, `SMOKE_BROWSER=firefox` runs Firefox (the switches become prefs).
- `scripts/browser-hardware.sh` (local, for a machine with a hardware GPU): the oracle on the GPU,
  the WebGPU smoke against native and Bevy in Chromium and Firefox, the backend selection and the
  fallback's frames in both, and browser GPU and CPU time against native, into
  `target/browser-hardware/report.md`. A failed step is recorded and the run goes on.
- The fallback's rendering of every lighting scenario is already gated on CI by the oracle's
  three.js side.

```sh
cargo run --release --locked --manifest-path oracle/Cargo.toml -- --only sun_shadows --only textured_props
bash scripts/browser-backends.sh
```

## Not verified

- Safari, Chrome or Firefox on a real window, mobile browsers and any hardware GPU: everything here
  ran headless in Playwright's Chromium and Firefox on software adapters
  (`scripts/browser-hardware.sh` is the run that covers the hardware GPU).
- A real consumer: the mmorpg web client stays on three.js (mmorpg#80) and has no WebGPU build
  to fall back from; there is no consumer-facing wasm build (D10, #75's prototype scope).
- WebGL context loss and WebGPU device loss after start, and WebGL 1.
- The probe stops at the multisampled depth texture, so later failures on the GL backend (draws,
  presentation) were not reached.
