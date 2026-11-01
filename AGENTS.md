# Agent guidance

## Authority

The current direction is in [technical decisions](docs/technical-decisions.md): Bevy is the
visual oracle and performance baseline (`bevy_pbr` for lighting, `bevy_hanabi` for particles),
render-engine provides the lighting and particle capabilities the owner's games use through its
own wgpu API, and lighting comes before particles. First consumer: the mmorpg native client
(wgpu 30, `three-d-*` crates from 3d-lab), starting with mmorpg#529.

Keep ECS, asset loading, game loops, gameplay and the day clock in the consumers. The engine
never depends on Bevy outside `oracle/`.

## Rendering rules

- Scene values use Bevy's physical units: metres, lux, cd/m², EV100, linear RGB.
- Depth is reverse Z everywhere (clear 0, `GreaterEqual`), like Bevy.
- Shading terms follow Bevy's `bevy_pbr` WESL sources for the pinned Bevy version; when a term
  differs on purpose, say why in the scenario PR.
- Do not tune a scenario until the FLIP numbers pass; fix the renderer. Thresholds may only be
  loosened in the PR that adds the scenario, with measured numbers and the reason.

## Layout and commands

| Path | Role |
| --- | --- |
| `src/` | The `render-engine` crate: `world` (consumer meshes/textures/materials and particle emitters behind handles), `mesh`, `material`, `scene` (camera, lights and sky per frame), `renderer` (wgpu passes, `Viewport`), `sky` (Bevy's atmosphere LUTs, sky pass and runtime environment filter), `particles` (emitters, their spawners and seed chains, replaying bevy_hanabi), `particle_sim` (the GPU simulation: slot pool and compute passes), `particle_pass` (particle quads in the main pass), `headless` (offscreen + GPU timing) |
| `src/shaders/` | WGSL shaders |
| `oracle/` | Bevy oracle (own manifest/lockfile): each scenario on Bevy and render-engine, FLIP + GPU-time gates; particle scenarios on Bevy + `bevy_hanabi` and render-engine (`particles.rs`, decision D9) |
| `assets/ibl/` | CC0 HDRI and its prefiltered environment cube maps (`oracle/src/bin/bake_ibl.rs`; provenance in its README) |
| `assets/props/` | CC0 glTF props for `textured_props` (provenance in its README) |
| `assets/figure/` | CC0 rigged character for `skinned_figure` (provenance in its README) |
| `examples/mesh_upload.rs` | Microbenchmark: per-frame pose uploads (`set_mesh`, `set_mesh_positions_normals`, `set_skin`) |
| `docs/consumer-integration.md` | How a consumer (the mmorpg native client) drives the engine |
| `site/` | Report page over the last oracle run (`scripts/build-pages.sh` → `pages-dist/`) |
| `packages/three-lighting/` | The web client's lighting on three.js (`@moritzbrantner/three-lighting`, root `package.json`): GLSL ports of `lit.wgsl`, `tonemap.wgsl` and the sky passes (`sky.js`, `sky-shaders.js`) and Bevy's cascade fitting (D8). A shading change in `lit.wgsl`, `tonemap.wgsl` or `src/shaders/sky/` makes the same change here |
| `oracle/three/` | The oracle's three.js side: `render.mjs` renders exported scenarios in headless Chromium (Playwright, ANGLE on Vulkan); `oracle/src/three_side.rs` exports them and lists known gaps with their issues |

```sh
bun install                                           # three.js side: three + Playwright (needs node); --no-three skips it
cargo fmt --all -- --check && (cd oracle && cargo fmt -- --check)
cargo clippy --all-targets --locked -- -D warnings
(cd oracle && cargo clippy --release --all-targets --locked -- -D warnings)
cargo test --locked                                   # includes a headless render (needs any wgpu adapter)
(cd oracle && cargo test --release --locked)
cargo run --release --locked --manifest-path oracle/Cargo.toml                    # GPU: image + GPU-time gates
cargo run --release --locked --manifest-path oracle/Cargo.toml -- --only sun_shadows --no-timing   # debugging; --list
WGPU_FORCE_FALLBACK_ADAPTER=1 cargo run --release --locked --manifest-path oracle/Cargo.toml       # as CI: lavapipe
cargo run --release --locked --manifest-path oracle/Cargo.toml -- --calibrate   # particle gates: Bevy vs itself (D9)
bash scripts/build-pages.sh
```

The oracle writes `target/oracle/<scenario>.{bevy,engine,flip}.png`, the three.js side's
`<scenario>.three.png` and `<scenario>.three.flip.png`, and `report.json`. Look at the FLIP heat
map before changing shading code. A lighting scenario counts as done on the web only when its
three.js side passes; one that does not yet is a known gap (`three_side::GAPS`) or not ported,
each with a follow-up issue.

CI (`.github/workflows/validate.yml`) runs all of this on plain Ubuntu with lavapipe
(`WGPU_FORCE_FALLBACK_ADAPTER=1`) in two parallel jobs: `tests` (engine fmt/clippy/test, never
compiles Bevy) and `oracle` (oracle clippy/test and the FLIP gate). Markdown-only PRs skip both;
`main` requires the `merge-ready` check. The FLIP gate is enforced on CI. The GPU-time gate is a
**local gate**: run the oracle on the GPU for any PR that changes passes or shaders and paste its
table into the PR.

## Execution scope

- **One task = one branch = one PR**, targeting `main`, closing its issue (`Closes #n`). A task is one eligible issue. Deliver its
  scope, its oracle scenario and the docs it changes on that branch.
- **Bevy oracle.** Every parity PR adds its scenario to `oracle/src/scenarios.rs` (particles:
  `oracle/src/particles.rs`, with thresholds from `--calibrate`) and its Bevy translation to
  `oracle/src/bevy_side.rs`. Do not add other ratchets, evidence packaging or
  attestation.
- **Experimental API.** Pick the cleanest API for the consumers; no deprecation windows, shims or
  back-compat layers.
- **Free assets only.** CC0 HDRIs/textures (Poly Haven, ambientCG); never paid sources.
- **Validate in tiers.** Focused tests while iterating; CI is the full gate. A red check blocks
  merge; fix it.
- **Short PR descriptions.** What changed, API changes, the oracle table, anything not verified.
