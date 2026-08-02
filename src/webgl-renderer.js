/**
 * Draws the field as one instanced capsule per particle, accumulating into an
 * offscreen target so the motion trail survives between frames.
 */

/** x, y of the streak tail and head, then hue, lightness, fade and glow. */
export const INSTANCE_FLOATS = 8

/** How far the halo reaches past the particle edge, in radii. */
const GLOW_REACH = 3.5
/** Peak halo brightness, at full speed. Trails accumulate the halo across
 *  frames, so this stays well under one. */
const GLOW_STRENGTH = 0.15

const PARTICLE_VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 segment;
layout(location = 2) in vec4 shade;

uniform vec2 viewport;
uniform float radius;
uniform float feather;
uniform float glow;
uniform float haloPass;

out vec2 pixel;
flat out vec2 tail;
flat out vec2 head;
flat out vec3 color;
flat out float glowScale;

/** Saturation is always full, so only hue and lightness vary. */
vec3 hueToRgb(float hue, float lightness) {
  float chroma = 1.0 - abs(2.0 * lightness - 1.0);
  float sector = fract(hue) * 6.0;
  float second = chroma * (1.0 - abs(mod(sector, 2.0) - 1.0));
  vec3 wheel;

  if (sector < 1.0) {
    wheel = vec3(chroma, second, 0.0);
  } else if (sector < 2.0) {
    wheel = vec3(second, chroma, 0.0);
  } else if (sector < 3.0) {
    wheel = vec3(0.0, chroma, second);
  } else if (sector < 4.0) {
    wheel = vec3(0.0, second, chroma);
  } else if (sector < 5.0) {
    wheel = vec3(second, 0.0, chroma);
  } else {
    wheel = vec3(chroma, 0.0, second);
  }

  return wheel + (lightness - chroma * 0.5);
}

void main() {
  tail = segment.xy;
  head = segment.zw;
  color = hueToRgb(shade.x, shade.y) * shade.z;
  glowScale = shade.w;

  vec2 axis = head - tail;
  float span = length(axis);
  vec2 forward = span > 0.0001 ? axis / span : vec2(1.0, 0.0);
  vec2 side = vec2(-forward.y, forward.x);
  // Padded past the radius, or the quad clips off its own soft edge. Only the
  // halo pass needs the wider footprint.
  float extent = radius + feather + glow * haloPass;

  pixel =
    tail +
    forward * (corner.x * (span + 2.0 * extent) - extent) +
    side * (corner.y * extent);

  gl_Position = vec4(
    pixel.x / viewport.x * 2.0 - 1.0,
    1.0 - pixel.y / viewport.y * 2.0,
    0.0,
    1.0
  );
}`

const PARTICLE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 pixel;
flat in vec2 tail;
flat in vec2 head;
flat in vec3 color;
flat in float glowScale;

uniform float radius;
uniform float feather;
uniform float glow;
uniform float glowStrength;
uniform float haloPass;

out vec4 fragment;

void main() {
  vec2 toPixel = pixel - tail;
  vec2 axis = head - tail;
  float along = clamp(dot(toPixel, axis) / max(dot(axis, axis), 1e-6), 0.0, 1.0);
  float distance = length(toPixel - axis * along);
  // Ramp centred on the edge, so coverage matches a rasterised stroke instead
  // of eroding the particle inward.
  float core = 1.0 - smoothstep(radius - feather, radius + feather, distance);

  if (haloPass < 0.5) {
    if (core <= 0.0) {
      discard;
    }

    fragment = vec4(color * core, core);
    return;
  }

  // Squared for a softer knee, and compactly supported so the quad edge never
  // shows as a seam.
  float falloff = 1.0 - smoothstep(radius, radius + glow, distance);
  float halo = falloff * falloff * glowStrength * glowScale;

  if (halo <= 0.0) {
    discard;
  }

  // Zero coverage, so this only ever adds to the cores and halos already laid
  // down. Overlapping glow in a cluster piles up instead of being replaced.
  fragment = vec4(color * halo, 0.0);
}`

const VEIL_VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec2 corner;

void main() {
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}`

const VEIL_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform float fade;

out vec4 fragment;

void main() {
  fragment = vec4(0.0, 0.0, 0.0, fade);
}`

const COPY_VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec2 corner;

out vec2 uv;

void main() {
  uv = corner;
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}`

const COPY_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 uv;

uniform sampler2D trail;

out vec4 fragment;

/** Emitters accumulate past full brightness, so the top end is rolled off
 *  instead of clipped. Linear below the knee keeps a lone particle's colour. */
vec3 toneMap(vec3 light) {
  const float knee = 0.8;
  vec3 over = max(light - knee, vec3(0.0));

  return min(light, vec3(knee)) + (1.0 - knee) * (1.0 - exp(-over / (1.0 - knee)));
}

void main() {
  fragment = vec4(toneMap(texture(trail, uv).rgb), 1.0);
}`

function compile(gl, type, source) {
  const shader = gl.createShader(type)

  gl.shaderSource(shader, source)
  gl.compileShader(shader)

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`Shader failed to compile: ${gl.getShaderInfoLog(shader)}`)
  }

  return shader
}

function link(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram()

  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertexSource))
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource))
  gl.linkProgram(program)

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program failed to link: ${gl.getProgramInfoLog(program)}`)
  }

  return program
}

export function createRenderer(canvas, radius) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: true,
    powerPreference: 'high-performance',
  })

  if (!gl) {
    throw new Error('WebGL2 is required for the particle field.')
  }

  const particleProgram = link(
    gl,
    PARTICLE_VERTEX_SHADER,
    PARTICLE_FRAGMENT_SHADER,
  )
  const veilProgram = link(gl, VEIL_VERTEX_SHADER, VEIL_FRAGMENT_SHADER)
  const copyProgram = link(gl, COPY_VERTEX_SHADER, COPY_FRAGMENT_SHADER)
  // Multiplicative decay on an 8 bit target rounds back up and leaves a
  // permanent smear, so the trail accumulates at float precision. Blending
  // only ever reads source alpha, so the packed no-alpha format is enough and
  // halves the trail's memory. It is no faster: the fullscreen passes are fill
  // bound rather than bandwidth bound.
  const trail = gl.getExtension('EXT_color_buffer_float')
    ? {
        internalFormat: gl.R11F_G11F_B10F,
        format: gl.RGB,
        type: gl.HALF_FLOAT,
      }
    : { internalFormat: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE }
  const particleUniforms = {
    viewport: gl.getUniformLocation(particleProgram, 'viewport'),
    radius: gl.getUniformLocation(particleProgram, 'radius'),
    feather: gl.getUniformLocation(particleProgram, 'feather'),
    glow: gl.getUniformLocation(particleProgram, 'glow'),
    glowStrength: gl.getUniformLocation(particleProgram, 'glowStrength'),
    haloPass: gl.getUniformLocation(particleProgram, 'haloPass'),
  }
  const veilUniforms = { fade: gl.getUniformLocation(veilProgram, 'fade') }

  const capsuleQuad = gl.createBuffer()

  gl.bindBuffer(gl.ARRAY_BUFFER, capsuleQuad)
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]),
    gl.STATIC_DRAW,
  )

  const veilQuad = gl.createBuffer()

  gl.bindBuffer(gl.ARRAY_BUFFER, veilQuad)
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]),
    gl.STATIC_DRAW,
  )

  const instanceBuffer = gl.createBuffer()
  const particleArray = gl.createVertexArray()

  gl.bindVertexArray(particleArray)
  gl.bindBuffer(gl.ARRAY_BUFFER, capsuleQuad)
  gl.enableVertexAttribArray(0)
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer)

  const stride = INSTANCE_FLOATS * 4

  gl.enableVertexAttribArray(1)
  gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 0)
  gl.vertexAttribDivisor(1, 1)
  gl.enableVertexAttribArray(2)
  gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 16)
  gl.vertexAttribDivisor(2, 1)

  const veilArray = gl.createVertexArray()

  gl.bindVertexArray(veilArray)
  gl.bindBuffer(gl.ARRAY_BUFFER, veilQuad)
  gl.enableVertexAttribArray(0)
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
  gl.bindVertexArray(null)

  const trailTarget = gl.createFramebuffer()

  let trailTexture = null
  let instanceCapacity = 0

  gl.disable(gl.DEPTH_TEST)
  gl.enable(gl.BLEND)
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)

  return {
    /** Trail history cannot survive a resize, so the target restarts black. */
    resize(width, height, pixelRatio) {
      if (trailTexture) {
        gl.deleteTexture(trailTexture)
      }

      trailTexture = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, trailTexture)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        trail.internalFormat,
        canvas.width,
        canvas.height,
        0,
        trail.format,
        trail.type,
        null,
      )
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      gl.bindFramebuffer(gl.FRAMEBUFFER, trailTarget)
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        trailTexture,
        0,
      )
      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)

      gl.useProgram(particleProgram)
      gl.uniform2f(particleUniforms.viewport, width, height)
      gl.uniform1f(particleUniforms.radius, radius)
      gl.uniform1f(particleUniforms.glow, radius * GLOW_REACH)
      gl.uniform1f(particleUniforms.glowStrength, GLOW_STRENGTH)
      // Half a device pixel either side of the edge, in the CSS pixels the
      // shader measures distance in.
      gl.uniform1f(particleUniforms.feather, 0.5 / pixelRatio)
    },

    draw(instances, instanceCount, veilAlpha) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, trailTarget)
      gl.viewport(0, 0, canvas.width, canvas.height)

      gl.useProgram(veilProgram)
      gl.uniform1f(veilUniforms.fade, veilAlpha)
      gl.bindVertexArray(veilArray)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

      if (instanceCount > 0) {
        gl.bindVertexArray(particleArray)
        gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer)

        if (instanceCapacity < instanceCount) {
          instanceCapacity = instanceCount * 2
          gl.bufferData(
            gl.ARRAY_BUFFER,
            instanceCapacity * stride,
            gl.DYNAMIC_DRAW,
          )
        }

        gl.bufferSubData(
          gl.ARRAY_BUFFER,
          0,
          instances,
          0,
          instanceCount * INSTANCE_FLOATS,
        )
        gl.useProgram(particleProgram)
        // Cores first, so the halos that follow add on top of neighbours
        // instead of being covered by them.
        gl.uniform1f(particleUniforms.haloPass, 0)
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, instanceCount)
        gl.uniform1f(particleUniforms.haloPass, 1)
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, instanceCount)
      }

      gl.bindVertexArray(null)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.disable(gl.BLEND)
      gl.useProgram(copyProgram)
      gl.bindTexture(gl.TEXTURE_2D, trailTexture)
      gl.bindVertexArray(veilArray)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      gl.bindVertexArray(null)
      gl.enable(gl.BLEND)
    },
  }
}
