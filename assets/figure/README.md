# Figure for `skinned_figure`

`Superhero_Male.gltf` (+ `.bin`, `textures/`) is the male body from the free Standard tier of
[Quaternius](https://quaternius.com)' Universal Base Characters, CC0 1.0: one skinned primitive
(7281 vertices) on a 65-joint humanoid rig.

Changes from `Superhero_Male_FullBody.gltf`: the eyebrow and eye meshes, their materials and
images are dropped (their nodes stay, empty, so the joint indices are unchanged); the body keeps
only `POSITION`, `NORMAL`, `TEXCOORD_0`, `JOINTS_0` and `WEIGHTS_0`; the three body textures
are downscaled from 2048² to 512² with Lanczos filtering and the image URIs point at
`textures/`. The `.bin` is the original buffer.
