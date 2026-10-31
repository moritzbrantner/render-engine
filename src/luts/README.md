# Tone-mapping LUTs

Copied unchanged from Bevy 0.20.1 (`bevy_core_pipeline/src/tonemapping/luts/`), so the oracle
compares the same operators, not re-derived curves. `info.txt` is Bevy's provenance note.

| File | Operator | Format | Origin and licence |
| --- | --- | --- | --- |
| `tony_mc_mapface.ktx2` | Tony McMapface (Bevy's default) | 48³ `E5B9G9R9_UFLOAT`, zstd | [h3r2tic/tony-mc-mapface](https://github.com/h3r2tic/tony-mc-mapface) by Tomasz Stachowiak, MIT or Apache-2.0 ([`LICENSE-MIT.tony-mc-mapface`](LICENSE-MIT.tony-mc-mapface)) |
| `AgX-default_contrast.ktx2` | AgX | 32³ `R16G16B16A16_SFLOAT`, zstd | [MrLixm/AgXc](https://github.com/MrLixm/AgXc) (AgX by Troy Sobotka); the upstream repository carries no licence file, so this follows Bevy's redistribution |

The KTX2 conversions and the shader code ported in `src/shaders/tonemap.wgsl` are Bevy's, MIT or
Apache-2.0: [`LICENSE-MIT.bevy`](LICENSE-MIT.bevy), [`LICENSE-APACHE.bevy`](LICENSE-APACHE.bevy).
