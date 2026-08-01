/**
 * Draws the field as one instanced capsule per particle, accumulating into an
 * offscreen target so the motion trail survives between frames.
 */

/** x, y of the streak tail and head, then the colour. */
export const INSTANCE_FLOATS = 7

const PARTICLE_VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 segment;
layout(location = 2) in vec3 tint;

uniform vec2 viewport;
uniform float radius;
uniform float feather;

out vec2 pixel;
flat out vec2 tail;
flat out vec2 head;
flat out vec3 color;

void main() {
  tail = segment.xy;
  head = segment.zw;
  color = tint;

  vec2 axis = head - tail;
  float span = length(axis);
  vec2 forward = span > 0.0001 ? axis / span : vec2(1.0, 0.0);
  vec2 side = vec2(-forward.y, forward.x);
  // Padded past the radius, or the quad clips off its own soft edge.
  float extent = radius + feather;

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

uniform float radius;
uniform float feather;

out vec4 fragment;

void main() {
  vec2 toPixel = pixel - tail;
  vec2 axis = head - tail;
  float along = clamp(dot(toPixel, axis) / max(dot(axis, axis), 1e-6), 0.0, 1.0);
  float distance = length(toPixel - axis * along);
  // Ramp centred on the edge, so coverage matches a rasterised stroke instead
  // of eroding the particle inward.
  float alpha = 1.0 - smoothstep(radius - feather, radius + feather, distance);

  if (alpha <= 0.0) {
    discard;
  }

  fragment = vec4(color * alpha, alpha);
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

void main() {
  fragment = vec4(texture(trail, uv).rgb, 1.0);
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
  // permanent smear, so the trail accumulates at half float precision.
  const trailFormat = gl.getExtension('EXT_color_buffer_float')
    ? gl.RGBA16F
    : gl.RGBA8
  const particleUniforms = {
    viewport: gl.getUniformLocation(particleProgram, 'viewport'),
    radius: gl.getUniformLocation(particleProgram, 'radius'),
    feather: gl.getUniformLocation(particleProgram, 'feather'),
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
  gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 16)
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
        trailFormat,
        canvas.width,
        canvas.height,
        0,
        gl.RGBA,
        trailFormat === gl.RGBA16F ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE,
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
