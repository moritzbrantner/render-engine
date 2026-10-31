# Props for `textured_props`

CC0 1.0 models by [Quaternius](https://quaternius.com), from the free Standard tiers:

| File | Pack |
| --- | --- |
| `Anvil.gltf`, `Cauldron.gltf`, `FarmCrate_Empty.gltf` (+ `.bin`), `textures/T_Trim_Metal_*`, `textures/T_Trim_Furniture_*` | Fantasy Props MegaKit (Standard) |
| `Bush_Common.gltf` (+ `.bin`), `textures/Leaves_TwistedTree_C.png` | Stylized Nature MegaKit (Standard) |

Changes from the originals: the textures are downscaled from 2048² (Leaves: 1024²) to 512² with
Lanczos filtering, and the image URIs point at `textures/`. Geometry, materials and samplers are
unchanged. The oracle builds the mip chains when it loads the files.
