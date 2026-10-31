# Environment maps

| File | What |
| --- | --- |
| `tiergarten_1k.hdr` | "Tiergarten" by Greg Zaal, [Poly Haven](https://polyhaven.com/a/tiergarten), 1k equirectangular Radiance HDR, downloaded 2026-10-31 (md5 `d58a4fc43686a8925b9c676b1166bcab`). Licence: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). |
| `tiergarten_specular.ktx2` | GGX-prefiltered radiance, 256² faces, 9 mips (mip `m` = perceptual roughness `m / 8`), `E5B9G9R9_UFLOAT_PACK32`, zstd. Derived from the HDR above (CC0). |
| `tiergarten_diffuse.ktx2` | Lambertian irradiance / π, 32² faces, `E5B9G9R9_UFLOAT_PACK32`, zstd. Derived from the HDR above (CC0). |

Rebuild the cube maps (needs KTX-Software ≥ 4.3 `ktx` on `PATH`):

```sh
cargo run --release --locked --manifest-path oracle/Cargo.toml --bin bake_ibl -- assets/ibl/tiergarten_1k.hdr assets/ibl/tiergarten
```

The layout matches what Bevy's `EnvironmentMapLight` documents for the glTF IBL Sampler, so the
maps also load in Bevy directly.
