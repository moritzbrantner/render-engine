// GLSL ES 3.0 ports of render-engine's `src/shaders/lit.wgsl` and `src/shaders/tonemap.wgsl`
// (themselves Bevy 0.20 `bevy_pbr` / `bevy_core_pipeline`, MIT or Apache-2.0). Term for term the
// same maths, so the three.js side passes the same FLIP gate against Bevy.
//
// Conventions: reverse-Z shadow maps (clear 0, GreaterEqual) with depth in [0, 1] like wgpu;
// cube maps are looked up with z negated (Bevy's left-handed cube maps); everything is linear
// exposed HDR until the tone-mapping pass.

export const MAX_CASCADES = 4;

export const litVertex = /* glsl */ `
out vec3 vWorld;
out vec3 vNormal;

// Bevy inverse_transpose_3x3m.
mat3 inverseTranspose3x3(mat3 m) {
    vec3 x = cross(m[1], m[2]);
    vec3 y = cross(m[2], m[0]);
    vec3 z = cross(m[0], m[1]);
    float det = dot(m[2], z);
    return mat3(x / det, y / det, z / det);
}

void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    vec3 n = normal;
    if (any(notEqual(n, vec3(0.0)))) {
        n = normalize(inverseTranspose3x3(mat3(modelMatrix)) * n);
    }
    vNormal = n;
    gl_Position = projectionMatrix * viewMatrix * world;
}
`;

export const litFragment = /* glsl */ `
#define PI 3.141592653589793
#define MAX_CASCADES ${MAX_CASCADES}

in vec3 vWorld;
in vec3 vNormal;
layout(location = 0) out highp vec4 fragColor;

// Material (Bevy StandardMaterial, untextured subset).
uniform vec4 baseColor;
uniform vec3 emissive;
uniform float metallic;
uniform float perceptualRoughness;
uniform float reflectance;

// Frame (shared by every LitMaterial, owned by Lighting).
uniform float exposure;
uniform float hasSun;
uniform vec3 directionToLight;
uniform vec3 sunColor;
uniform vec3 ambientColor;
uniform float sunShadows;
uniform float gaussianFilter;
// x: depth bias (m), y: normal bias (texels, ×√2), z: overlap proportion, w: cascade count.
uniform vec4 shadowParams;
uniform mat4 cascadeClipFromWorld[MAX_CASCADES];
// x: far bound (m), y: texel size (m).
uniform vec4 cascadeParams[MAX_CASCADES];
uniform float shadowMapSize;
uniform sampler2DShadow shadowMap0;
uniform sampler2DShadow shadowMap1;
uniform sampler2DShadow shadowMap2;
uniform sampler2DShadow shadowMap3;
// x: intensity (cd/m²), y: largest specular mip level, z: 1 when there is an environment map.
uniform vec3 environment;
uniform samplerCube diffuseEnvironmentMap;
uniform samplerCube specularEnvironmentMap;
// Bevy DistanceFog: colour, sun inscattering colour (a = strength), be.xyz + directional-light
// exponent, bi.xyz + mode (0 off, 1 linear, 2 exponential, 3 exponential squared, 4 atmospheric).
uniform vec4 fogColor;
uniform vec4 fogDirectionalLightColor;
uniform vec4 fogBe;
uniform vec4 fogBi;

float perceptualRoughnessToRoughness(float p) {
    float c = clamp(p, 0.089, 1.0);
    return c * c;
}

float D_GGX(float roughness, float NdotH) {
    float oneMinus = 1.0 - NdotH * NdotH;
    float a = NdotH * roughness;
    float k = roughness / (oneMinus + a * a);
    return k * k * (1.0 / PI);
}

float V_SmithGGXCorrelated(float roughness, float NdotV, float NdotL) {
    float a2 = roughness * roughness;
    float lambdaV = NdotL * sqrt((NdotV - a2 * NdotV) * NdotV + a2);
    float lambdaL = NdotV * sqrt((NdotL - a2 * NdotL) * NdotL + a2);
    return 0.5 / (lambdaV + lambdaL);
}

float F_Schlick(float f0, float f90, float VdotH) {
    return f0 + (f90 - f0) * pow(1.0 - VdotH, 5.0);
}

vec3 F_Schlick_vec(vec3 f0, float f90, float VdotH) {
    return f0 + (f90 - f0) * pow(1.0 - VdotH, 5.0);
}

vec3 fresnelTerm(vec3 f0, float LdotH) {
    float f90 = clamp(dot(f0, vec3(50.0 * 0.33)), 0.0, 1.0);
    return F_Schlick_vec(f0, f90, LdotH);
}

// Polynomial approximation of the split-sum DFG terms (Karis, mobile).
vec2 F_AB(float pr, float NdotV) {
    vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = pr * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return max(vec2(-1.04, 1.04) * a004 + r.zw, vec2(0.00005));
}

vec3 envBrdfApprox(vec3 f0, vec2 fab) {
    return f0 * fab.x + fab.y;
}

float sampleHardware(vec2 uv, float depth, int layer) {
    vec3 p = vec3(uv, depth);
    if (layer == 0) return texture(shadowMap0, p);
    if (layer == 1) return texture(shadowMap1, p);
    if (layer == 2) return texture(shadowMap2, p);
    return texture(shadowMap3, p);
}

// Bevy sample_shadow_map_castano_thirteen: a 13-tap Gaussian from 9 hardware PCF taps.
float sampleGaussian(vec2 lightLocal, float depth, int layer) {
    vec2 size = vec2(shadowMapSize);
    vec2 inv = 1.0 / size;
    vec2 uv = lightLocal * size;
    vec2 baseUv = floor(uv + 0.5);
    float s = uv.x + 0.5 - baseUv.x;
    float t = uv.y + 0.5 - baseUv.y;
    baseUv = (baseUv - 0.5) * inv;

    vec3 uw = vec3(4.0 - 3.0 * s, 7.0, 1.0 + 3.0 * s);
    vec3 u = vec3((3.0 - 2.0 * s) / uw.x - 2.0, (3.0 + s) / uw.y, s / uw.z + 2.0);
    vec3 vw = vec3(4.0 - 3.0 * t, 7.0, 1.0 + 3.0 * t);
    vec3 v = vec3((3.0 - 2.0 * t) / vw.x - 2.0, (3.0 + t) / vw.y, t / vw.z + 2.0);

    float sum = 0.0;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            sum += uw[i] * vw[j] * sampleHardware(baseUv + vec2(u[i], v[j]) * inv, depth, layer);
        }
    }
    return sum * (1.0 / 144.0);
}

// Bevy sample_directional_cascade. GL textures put row 0 at the bottom, so v is not flipped.
float sampleCascade(int cascade, vec3 world, vec3 normal) {
    float texelSize = cascadeParams[cascade].y;
    vec3 offset = world + shadowParams.y * texelSize * normal + shadowParams.x * directionToLight;
    vec4 clip = cascadeClipFromWorld[cascade] * vec4(offset, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    vec3 ndc = clip.xyz / clip.w;
    if (any(lessThan(ndc.xy, vec2(-1.0))) || ndc.z < 0.0 || any(greaterThan(ndc, vec3(1.0)))) {
        return 1.0;
    }
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (gaussianFilter > 0.5) {
        return sampleGaussian(uv, ndc.z, cascade);
    }
    return sampleHardware(uv, ndc.z, cascade);
}

// Bevy fetch_directional_shadow: pick the cascade by view depth, blend into the next one across
// the overlap.
float sunShadow(vec3 world, vec3 normal) {
    float depth = -(viewMatrix * vec4(world, 1.0)).z;
    int count = int(shadowParams.w + 0.5);
    int cascade = count;
    for (int i = 0; i < MAX_CASCADES; i++) {
        if (i < count && depth < cascadeParams[i].x) {
            cascade = i;
            break;
        }
    }
    if (cascade >= count) {
        return 1.0;
    }
    float shadow = sampleCascade(cascade, world, normal);
    int next = cascade + 1;
    if (next < count) {
        float thisFar = cascadeParams[cascade].x;
        float nextNear = (1.0 - shadowParams.z) * thisFar;
        if (depth >= nextNear) {
            float nextShadow = sampleCascade(next, world, normal);
            shadow = mix(shadow, nextShadow, (depth - nextNear) / (thisFar - nextNear));
        }
    }
    return shadow;
}

// Bevy radiance_sample_direction ("Moving Frostbite to PBR 3.0", listing 22).
vec3 radianceSampleDirection(vec3 N, vec3 R, float roughness) {
    float smoothness = clamp(1.0 - roughness, 0.0, 1.0);
    float lerpFactor = smoothness * (sqrt(smoothness) + roughness);
    return mix(N, R, lerpFactor);
}

// Bevy compute_multiscatter (Fdez-Agüera 2019): FssEss, FmsEms, Edss.
void computeMultiscatter(vec3 F0, vec2 fab, float Ems, float so, out vec3 FssEss, out vec3 FmsEms, out vec3 Edss) {
    FssEss = (F0 * fab.x + fab.y) * so;
    vec3 Favg = F0 + (1.0 - F0) / 21.0;
    FmsEms = FssEss * Favg / (1.0 - Ems * Favg) * Ems;
    Edss = 1.0 - (FssEss + FmsEms);
}

// Bevy environment_map_light for a view environment map (no light probes, no rotation).
void environmentMapLight(vec3 N, vec3 V, float pr, float roughness, vec3 diffuseColor, float metal,
    vec3 F0Dielectric, vec3 F0Metallic, vec2 fab, out vec3 diffuse, out vec3 specular) {
    float intensity = environment.x;
    vec3 R = reflect(-V, N);
    float radianceLevel = pr * environment.y;
    vec3 irradianceDir = N;
    irradianceDir.z = -irradianceDir.z;
    vec3 irradiance = textureLod(diffuseEnvironmentMap, irradianceDir, 0.0).rgb * intensity;
    vec3 radianceDir = radianceSampleDirection(N, R, roughness);
    radianceDir.z = -radianceDir.z;
    vec3 radiance = textureLod(specularEnvironmentMap, radianceDir, radianceLevel).rgb * intensity;
    if (all(equal(irradiance, vec3(0.0))) && all(equal(radiance, vec3(0.0)))) {
        diffuse = vec3(0.0);
        specular = vec3(0.0);
        return;
    }
    vec3 F0Surface = mix(F0Dielectric, F0Metallic, metal);
    float so = clamp(dot(F0Surface, vec3(50.0 * 0.33)), 0.0, 1.0);
    float Ems = 1.0 - (fab.x + fab.y);
    vec3 dFss, dFms, dEdss, mFss, mFms, mEdss;
    computeMultiscatter(F0Dielectric, fab, Ems, so, dFss, dFms, dEdss);
    computeMultiscatter(F0Metallic, fab, Ems, so, mFss, mFms, mEdss);
    vec3 FssEss = mix(dFss, mFss, metal);
    vec3 FmsEms = mix(dFms, mFms, metal);
    vec3 kD = diffuseColor * dEdss;
    diffuse = (FmsEms + kD) * irradiance;
    specular = FssEss * radiance;
}

// Bevy directional_light (without the shadow): Burley diffuse and GGX specular with the
// multiscatter compensation.
vec3 directionalLight(vec3 L, vec3 lightColor, vec3 N, vec3 V, float NdotV, float roughness,
    vec3 diffuseColor, vec3 F0, vec2 fab) {
    vec3 H = normalize(L + V);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotH = clamp(dot(N, H), 0.0, 1.0);
    float LdotH = clamp(dot(L, H), 0.0, 1.0);

    float f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    float fd = F_Schlick(1.0, f90, NdotL) * F_Schlick(1.0, f90, NdotV) * (1.0 / PI);
    vec3 diffuse = diffuseColor * fd;

    float D = D_GGX(roughness, NdotH);
    float Vis = V_SmithGGXCorrelated(roughness, NdotV, NdotL);
    vec3 F = fresnelTerm(F0, LdotH);
    vec3 spec = (D * Vis) * F;
    spec *= 1.0 + F0 * (1.0 / (fab.x + fab.y) - 1.0);

    return (diffuse + spec) * NdotL * lightColor;
}

vec3 litRadiance(vec3 worldNormal, vec3 N) {
    vec3 base = baseColor.rgb;
    float roughness = perceptualRoughnessToRoughness(perceptualRoughness);
    vec3 V = normalize(cameraPosition - vWorld);
    float NdotV = max(dot(N, V), 0.0001);

    vec3 diffuseColor = base * (1.0 - metallic);
    vec3 f0Dielectric = vec3(0.16 * reflectance * reflectance);
    vec3 F0 = mix(f0Dielectric, base, metallic);
    vec2 fab = F_AB(perceptualRoughness, NdotV);

    vec3 direct = vec3(0.0);
    if (hasSun > 0.5) {
        float shadow = 1.0;
        if (sunShadows > 0.5) {
            shadow = sunShadow(vWorld, worldNormal);
        }
        direct += directionalLight(directionToLight, sunColor, N, V, NdotV, roughness, diffuseColor, F0, fab) * shadow;
    }

    vec3 diffuseAmbient = envBrdfApprox(diffuseColor, F_AB(1.0, NdotV));
    vec3 specularAmbient = envBrdfApprox(F0, fab);
    float specularOcclusion = clamp(dot(F0, vec3(50.0 * 0.33)), 0.0, 1.0);
    vec3 indirect = vec3(0.0);
    if (environment.z > 0.5) {
        vec3 envDiffuse, envSpecular;
        environmentMapLight(N, V, perceptualRoughness, roughness, diffuseColor, metallic, f0Dielectric, base, fab, envDiffuse, envSpecular);
        indirect += envDiffuse;
        indirect += envSpecular;
    }
    indirect += (diffuseAmbient + specularAmbient * specularOcclusion) * ambientColor;

    // Emissive light is scaled by alpha and exposed like the rest (Bevy, emissive.a = 1).
    return exposure * (direct + indirect) + emissive * baseColor.a * exposure;
}

vec4 scatteringAdjustedFogColor(vec3 scattering) {
    vec4 glow = fogDirectionalLightColor;
    if (glow.a > 0.0) {
        return vec4(fogColor.rgb + scattering * glow.rgb * glow.a, fogColor.a);
    }
    return fogColor;
}

// Bevy apply_fog (one directional light) and the falloffs of fog.wesl, on the exposed colour.
vec3 applyFog(vec3 inputColor, vec3 world) {
    vec3 viewToWorld = world - cameraPosition;
    float distance = length(viewToWorld);

    vec3 scattering = vec3(0.0);
    if (fogDirectionalLightColor.a > 0.0 && hasSun > 0.5) {
        vec3 contribution = pow(max(dot(viewToWorld / distance, directionToLight), 0.0), fogBe.w) * sunColor * exposure;
        float shadow = 1.0;
        if (sunShadows > 0.5) {
            // Bevy approximates the surface normal by the direction to the camera here.
            shadow = sunShadow(world, normalize(-viewToWorld));
        }
        scattering += contribution * shadow;
    }

    vec4 color = scatteringAdjustedFogColor(scattering);
    int mode = int(fogBi.w + 0.5);
    if (mode == 1) {
        float start = fogBe.x;
        float end = fogBe.y;
        color.a *= 1.0 - clamp((end - distance) / (end - start), 0.0, 1.0);
        return mix(inputColor, color.rgb, color.a);
    } else if (mode == 2) {
        color.a *= 1.0 - 1.0 / exp(distance * fogBe.x);
        return mix(inputColor, color.rgb, color.a);
    } else if (mode == 3) {
        float dd = distance * fogBe.x;
        color.a *= 1.0 - 1.0 / exp(dd * dd);
        return mix(inputColor, color.rgb, color.a);
    }
    vec3 extinctionFactor = 1.0 - 1.0 / exp(distance * fogBe.xyz);
    vec3 inscatteringFactor = 1.0 - 1.0 / exp(distance * fogBi.xyz);
    return inputColor * (1.0 - extinctionFactor * color.a) + color.rgb * inscatteringFactor * color.a;
}

void main() {
    vec3 worldNormal = vNormal;
    vec3 N = normalize(worldNormal);
    vec3 color = litRadiance(worldNormal, N);
    if (fogBi.w > 0.5) {
        color = applyFog(color, vWorld);
    }
    fragColor = vec4(color, 1.0);
}
`;

// Shadow pass: reverse Z with depth in [0, 1] like wgpu, and Bevy's shadow pancaking (casters
// between the sun and the cascade's near plane clamp onto it). With EXT_clip_control and
// EXT_depth_clamp (nativeDepth) this is Bevy's pipeline: zero-to-one clip space and unclipped,
// rasterizer-clamped depth. Without them, GL clip z = 2·z − w and the clamp happens per fragment.
export const shadowVertex = /* glsl */ `
uniform mat4 cascadeClip;
out float vDepth;
void main() {
    vec4 clip = cascadeClip * (modelMatrix * vec4(position, 1.0));
    vDepth = clip.z;
#ifdef NATIVE_DEPTH
    gl_Position = clip;
#else
    clip.z = min(clip.z, 1.0);
    gl_Position = vec4(clip.xy, 2.0 * clip.z - clip.w, clip.w);
#endif
}
`;

export const shadowFragment = /* glsl */ `
in float vDepth;
layout(location = 0) out highp vec4 fragColor;
void main() {
#ifndef NATIVE_DEPTH
    gl_FragDepth = min(vDepth, 1.0);
#endif
    fragColor = vec4(0.0);
}
`;

export const fullscreenVertex = /* glsl */ `
void main() {
    gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// Bevy tonemapping.wesl operators. ACES fitted is three.js' own ACESFilmicToneMapping (the same
// Stephen Hill fit), with toneMappingExposure = 0.6 to cancel its built-in 1 / 0.6.
export const tonemapFragment = /* glsl */ `
#include <tonemapping_pars_fragment>
uniform sampler2D hdrTexture;
uniform highp sampler3D lutTexture;
// 0: ACES fitted, 1: Tony McMapface, 2: AgX.
uniform int method;
layout(location = 0) out highp vec4 fragColor;

vec3 sampleLut(vec3 p) {
    return textureLod(lutTexture, p, 0.0).rgb;
}

float bevyLuminance(vec3 v) {
    return dot(v, vec3(0.2126, 0.7152, 0.0722));
}

const float TONY_MC_MAPFACE_LUT_DIMS = 48.0;

vec3 tonyMcMapface(vec3 stimulus) {
    vec3 uv = (stimulus / (stimulus + 1.0)) * ((TONY_MC_MAPFACE_LUT_DIMS - 1.0) / TONY_MC_MAPFACE_LUT_DIMS)
        + 0.5 / TONY_MC_MAPFACE_LUT_DIMS;
    return sampleLut(clamp(uv, 0.0, 1.0));
}

vec3 openDomainToNormalizedLog2(vec3 color, float minimumEv, float maximumEv) {
    float inMidgray = 0.18;
    vec3 c = max(vec3(0.0), color);
    c = mix(c, 0.00001525878 + c, vec3(lessThan(c, vec3(0.00003051757))));
    c = clamp(log2(c / inMidgray), vec3(minimumEv), vec3(maximumEv));
    return (c - minimumEv) / (maximumEv - minimumEv);
}

vec3 agxLog(vec3 image) {
    vec3 c = max(vec3(0.0), image);
    float r = dot(c, vec3(0.84247906, 0.0784336, 0.07922375));
    float g = dot(c, vec3(0.04232824, 0.87846864, 0.07916613));
    float b = dot(c, vec3(0.04237565, 0.0784336, 0.87914297));
    c = openDomainToNormalizedLog2(vec3(r, g, b), -10.0, 6.5);
    return clamp(c, vec3(0.0), vec3(1.0));
}

vec3 agx(vec3 color) {
    float blockSize = 32.0;
    return sampleLut(agxLog(color) * ((blockSize - 1.0) / blockSize) + 0.5 / blockSize);
}

void main() {
    vec4 hdr = texelFetch(hdrTexture, ivec2(gl_FragCoord.xy), 0);
    vec3 color = max(hdr.rgb, vec3(0.0));
    vec3 outColor;
    if (method == 0) {
        outColor = ACESFilmicToneMapping(color);
    } else if (method == 2) {
        outColor = agx(color);
    } else {
        outColor = tonyMcMapface(color);
    }
    float luma = bevyLuminance(outColor);
    outColor = mix(vec3(luma), outColor, vec3(1.0));
    fragColor = vec4(outColor, hdr.a);
}
`;

// Copies the tone-mapped Rgba16Float frame into the 8-bit output with the sRGB transfer function
// (Bevy's blit into an sRGB target).
export const outputFragment = /* glsl */ `
uniform sampler2D postTexture;
// 0 for an sRGB target, which encodes on store.
uniform float encodeSrgb;
layout(location = 0) out highp vec4 fragColor;
vec3 srgbEncode(vec3 c) {
    return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, 12.92 * c, vec3(lessThanEqual(c, vec3(0.0031308))));
}
void main() {
    vec4 c = texelFetch(postTexture, ivec2(gl_FragCoord.xy), 0);
    vec3 rgb = clamp(c.rgb, 0.0, 1.0);
    fragColor = vec4(encodeSrgb > 0.5 ? srgbEncode(rgb) : rgb, 1.0);
}
`;
