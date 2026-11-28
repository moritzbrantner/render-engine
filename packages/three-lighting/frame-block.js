// The frame's light, shadow, fog and sky values as one uniform block (#181).
//
// three uploads every plain uniform of a ShaderMaterial each time a draw switches to it, so as
// plain uniforms these values, the same for every material of a frame, were set again at each of
// the main pass's material switches. In a uniform block (three's `UniformsGroup`) they are
// written once a frame and bound per program. Samplers cannot be block members; they stay plain
// uniforms (`Lighting.uniforms`).
//
// The block is one Float32Array laid out here by std140's rules, which three uploads whole; the
// GLSL declaration is `LitFrame` in shaders.js (`litShading`). Keep the two in step: the offsets
// below are in floats, each `vec3` shares its 16 bytes with the `float` after it, and the local
// lights' array comes last, so a program compiled for fewer lights reads a prefix of the buffer.

import { Uniform, UniformsGroup } from 'three';
import { MAX_CASCADES } from './shaders.js';

const CASCADE_CLIP_FROM_WORLD = 0;
const WORLD_TO_ATMOSPHERE = CASCADE_CLIP_FROM_WORLD + 16 * MAX_CASCADES;
const CASCADE_PARAMS = WORLD_TO_ATMOSPHERE + 16;
const SHADOW_PARAMS = CASCADE_PARAMS + 4 * MAX_CASCADES;
const FOG_COLOR = SHADOW_PARAMS + 4;
const FOG_DIRECTIONAL_LIGHT_COLOR = FOG_COLOR + 4;
const FOG_BE = FOG_DIRECTIONAL_LIGHT_COLOR + 4;
const FOG_BI = FOG_BE + 4;
const MOON_DIRECTION_TO_LIGHT = FOG_BI + 4;
const ATMOSPHERE = MOON_DIRECTION_TO_LIGHT + 4;
const DIRECTION_TO_LIGHT = ATMOSPHERE + 4; // + exposure
const SUN_COLOR = DIRECTION_TO_LIGHT + 4; // + hasSun
const AMBIENT_COLOR = SUN_COLOR + 4; // + sunShadows
const ENVIRONMENT = AMBIENT_COLOR + 4; // + gaussianFilter
const MOON_COLOR = ENVIRONMENT + 4; // + shadowMapSize
const DISKS = MOON_COLOR + 4; // a vec2, padded to 16 bytes by the array after it
/** Where the local lights' vec4s start, in floats. */
export const LOCAL_LIGHTS = DISKS + 4;
/** The block's name in GLSL. */
export const FRAME_BLOCK = 'LitFrame';

// Room for the local lights grows by this many floats (16 lights of five vec4s).
const LIGHT_ROOM = 16 * 5 * 4;

/** The block's size in bytes in a program compiled for `lights` floats of local lights. */
export function frameBlockBytes(lights) {
  return (LOCAL_LIGHTS + lights) * 4;
}

export class FrameBlock {
  /** The group a material lists in `uniformsGroups`; a new one when the lights outgrow it. */
  group = null;
  /** The block's floats. */
  data = new Float32Array(0);

  #room(lights) {
    if (this.group && this.data.length >= LOCAL_LIGHTS + lights) return;
    this.group?.dispose();
    this.data = new Float32Array(LOCAL_LIGHTS + Math.ceil(lights / LIGHT_ROOM) * LIGHT_ROOM);
    this.group = new UniformsGroup();
    this.group.setName(FRAME_BLOCK);
    this.group.add(new Uniform(this.data));
  }

  /**
   * Writes the frame's values (`Lighting`'s, by uniform name) and the local lights' flat vec4s.
   */
  write(v, lights) {
    this.#room(lights.length);
    const f = this.data;
    f.set(v.cascadeClipFromWorld.value, CASCADE_CLIP_FROM_WORLD);
    v.worldToAtmosphere.value.toArray(f, WORLD_TO_ATMOSPHERE);
    f.set(v.cascadeParams.value, CASCADE_PARAMS);
    v.shadowParams.value.toArray(f, SHADOW_PARAMS);
    v.fogColor.value.toArray(f, FOG_COLOR);
    v.fogDirectionalLightColor.value.toArray(f, FOG_DIRECTIONAL_LIGHT_COLOR);
    v.fogBe.value.toArray(f, FOG_BE);
    v.fogBi.value.toArray(f, FOG_BI);
    v.moonDirectionToLight.value.toArray(f, MOON_DIRECTION_TO_LIGHT);
    v.atmosphere.value.toArray(f, ATMOSPHERE);
    v.directionToLight.value.toArray(f, DIRECTION_TO_LIGHT);
    f[DIRECTION_TO_LIGHT + 3] = v.exposure.value;
    v.sunColor.value.toArray(f, SUN_COLOR);
    f[SUN_COLOR + 3] = v.hasSun.value;
    v.ambientColor.value.toArray(f, AMBIENT_COLOR);
    f[AMBIENT_COLOR + 3] = v.sunShadows.value;
    v.environment.value.toArray(f, ENVIRONMENT);
    f[ENVIRONMENT + 3] = v.gaussianFilter.value;
    v.moonColor.value.toArray(f, MOON_COLOR);
    f[MOON_COLOR + 3] = v.shadowMapSize.value;
    v.disks.value.toArray(f, DISKS);
    f.set(lights, LOCAL_LIGHTS);
  }

  dispose() {
    this.group?.dispose();
    this.group = null;
  }
}
