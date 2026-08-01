import './style.css'

import {
  BLACK_HOLE_POLARITY,
  DEFAULT_PARTICLE_LIMIT,
  FRAME_DURATION,
  FULL_BRIGHTNESS_SPEED,
  HUE_COUNT,
  HUE_STEP,
  MAX_CONTROL_PERCENT,
  MAX_RETIRING_PARTICLES,
  MESSAGE_AIR_RESISTANCE,
  MESSAGE_EMISSION,
  MESSAGE_GRAVITY,
  MESSAGE_INIT,
  MESSAGE_PARTICLE_LIMIT,
  MESSAGE_POINTER,
  MESSAGE_POINTER_RESET,
  MESSAGE_READY,
  MESSAGE_RECYCLE_BUFFER,
  MESSAGE_RESIZE,
  MESSAGE_SNAPSHOT,
  MESSAGE_VISIBILITY,
  PARTICLE_FADE_LEVELS,
  PARTICLE_FADE_SHIFT,
  PARTICLE_FLAG_RESPAWNED,
  PARTICLE_LIMIT_STEP,
  PARTICLE_RADIUS,
  SNAPSHOT_FLAG_DISCONTINUOUS,
  WHITE_HOLE_POLARITY,
  clamp,
  createSnapshotViews,
  getSnapshotByteLength,
} from './simulation-shared.js'
import { INSTANCE_FLOATS, createRenderer } from './webgl-renderer.js'

const CONTROL_STEP = 10
const BRIGHTNESS_LEVELS = 8
const MAX_PIXEL_RATIO = 2
/** Trail decay constant; keeps the veil independent of frame rate. */
const TRAIL_TAU = 28
/** Beyond this a streak is a teleport, not motion. */
const MAX_STREAK_LENGTH = 160
const SNAPSHOT_POOL_SIZE = 4
const SNAPSHOT_INTERVAL_SMOOTHING = 0.15
const MAX_INTERPOLATION_SPAN = 250

/** Fully saturated hue to linear-ish rgb, matching the old hsl() palette. */
function hueToChannels(hue, lightness) {
  const chroma = (1 - Math.abs(2 * lightness - 1)) * 1
  const sector = hue / 60
  const second = chroma * (1 - Math.abs((sector % 2) - 1))
  const base = lightness - chroma / 2
  const wheel = [
    [chroma, second, 0],
    [second, chroma, 0],
    [0, chroma, second],
    [0, second, chroma],
    [second, 0, chroma],
    [chroma, 0, second],
  ][Math.floor(sector) % 6]

  return wheel.map((channel) => channel + base)
}

const PARTICLE_COLORS = new Float32Array(HUE_COUNT * BRIGHTNESS_LEVELS * 3)

for (let bucket = 0; bucket < HUE_COUNT * BRIGHTNESS_LEVELS; bucket += 1) {
  const colorIndex = Math.floor(bucket / BRIGHTNESS_LEVELS)
  const brightnessIndex = bucket % BRIGHTNESS_LEVELS
  const lightness = (25 + (37 * brightnessIndex) / (BRIGHTNESS_LEVELS - 1)) / 100

  PARTICLE_COLORS.set(
    hueToChannels(colorIndex * HUE_STEP, lightness),
    bucket * 3,
  )
}

const getBrightnessIndex = (speed) =>
  Math.round(
    Math.sqrt(clamp(speed / FULL_BRIGHTNESS_SPEED, 0, 1)) *
      (BRIGHTNESS_LEVELS - 1),
  )

document.querySelector('#app').innerHTML = `
  <canvas
    id="particle-canvas"
    aria-label="Interactive rainbow particle simulation. Hold the left mouse button for a black hole or the right mouse button for a white hole."
  >
    Move your pointer across the screen to emit rainbow particles.
  </canvas>
  <div id="pointer-hole" aria-hidden="true"></div>
  <div
    id="control-status"
    role="status"
    aria-live="polite"
    aria-atomic="true"
  ></div>
  <dialog id="controls-dialog" aria-labelledby="controls-title">
    <h1 id="controls-title">Controls</h1>
    <dl>
      <dt>Move</dt><dd>Emit particles</dd>
      <dt>Left click</dt><dd>Black hole</dd>
      <dt>Right click</dt><dd>White hole</dd>
      <dt>Space</dt><dd>Toggle emission</dd>
      <dt>-<span>/</span>=</dt><dd>Gravity</dd>
      <dt>[<span>/</span>]</dt><dd>Air resistance</dd>
      <dt>;<span>/</span>'</dt><dd>Particle limit</dd>
    </dl>
    <button id="controls-dismiss" type="button" autofocus>OK</button>
  </dialog>
`

const canvas = document.querySelector('#particle-canvas')
const pointerHole = document.querySelector('#pointer-hole')
const controlStatus = document.querySelector('#control-status')
const controlsDialog = document.querySelector('#controls-dialog')
const context = createRenderer(canvas, PARTICLE_RADIUS)
const physicsWorker = new Worker(
  new URL('./physics.worker.js', import.meta.url),
  { type: 'module' },
)

let instanceData = new Float32Array(0)
let previousSnapshot = null
let latestSnapshot = null
let renderX = new Float32Array(0)
let renderY = new Float32Array(0)
let snapshotInterval = FRAME_DURATION
let previousFrameTime = performance.now()
let viewportWidth = window.innerWidth
let viewportHeight = window.innerHeight
let pixelRatio = 1
let gravityPercent = 0
let airResistancePercent = 0
let isParticleEmissionEnabled = true
let particleLimit = DEFAULT_PARTICLE_LIMIT

const pointer = {
  x: null,
  y: null,
  holePolarity: 0,
}

let isWorkerReady = false
const queuedWorkerMessages = []

/**
 * The worker's module init is asynchronous, and a dedicated worker's port is
 * live before its `onmessage` is assigned, so anything sent early is dropped.
 */
function sendToWorker(message, transfer) {
  if (isWorkerReady) {
    physicsWorker.postMessage(message, transfer)
    return
  }

  queuedWorkerMessages.push({ message, transfer })
}

function configureCanvas() {
  pixelRatio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO)
  canvas.width = Math.round(viewportWidth * pixelRatio)
  canvas.height = Math.round(viewportHeight * pixelRatio)
  canvas.style.width = `${viewportWidth}px`
  canvas.style.height = `${viewportHeight}px`

  context.resize(viewportWidth, viewportHeight, pixelRatio)
}

function showControlStatus(message) {
  controlStatus.textContent = message
  controlStatus.classList.remove('is-visible')
  void controlStatus.offsetWidth
  controlStatus.classList.add('is-visible')
}

function toggleParticleEmission() {
  isParticleEmissionEnabled = !isParticleEmissionEnabled
  canvas.classList.toggle('is-emission-disabled', !isParticleEmissionEnabled)
  sendToWorker({
    type: MESSAGE_EMISSION,
    enabled: isParticleEmissionEnabled,
  })
}

function adjustGravity(direction) {
  gravityPercent = clamp(
    gravityPercent + direction * CONTROL_STEP,
    0,
    MAX_CONTROL_PERCENT,
  )
  sendToWorker({ type: MESSAGE_GRAVITY, percent: gravityPercent })
  showControlStatus(`Gravity: ${gravityPercent}%`)
}

function adjustAirResistance(direction) {
  airResistancePercent = clamp(
    airResistancePercent + direction * CONTROL_STEP,
    0,
    MAX_CONTROL_PERCENT,
  )
  sendToWorker({
    type: MESSAGE_AIR_RESISTANCE,
    percent: airResistancePercent,
  })
  showControlStatus(`Air Resistance: ${airResistancePercent}%`)
}

function adjustParticleLimit(direction) {
  particleLimit = Math.max(particleLimit + direction * PARTICLE_LIMIT_STEP, 0)
  sendToWorker({
    type: MESSAGE_PARTICLE_LIMIT,
    limit: particleLimit,
  })
  showControlStatus(`Particle Limit: ${particleLimit}`)
}

function handleKeyDown(event) {
  // The dialog owns the keyboard while it is open.
  if (controlsDialog.open) {
    if (event.key === 'Escape') {
      controlsDialog.close()
    }

    return
  }

  if (event.ctrlKey || event.altKey || event.metaKey) {
    return
  }

  switch (event.key) {
    case '-':
      adjustGravity(-1)
      break
    case '=':
      adjustGravity(1)
      break
    case '[':
      adjustAirResistance(-1)
      break
    case ']':
      adjustAirResistance(1)
      break
    case ';':
      adjustParticleLimit(-1)
      break
    case "'":
      adjustParticleLimit(1)
      break
    case ' ':
      if (!event.repeat) {
        toggleParticleEmission()
      }
      break
    default:
      return
  }

  event.preventDefault()
}

function resizeCanvas() {
  viewportWidth = window.innerWidth
  viewportHeight = window.innerHeight
  configureCanvas()
  sendToWorker({
    type: MESSAGE_RESIZE,
    width: viewportWidth,
    height: viewportHeight,
  })
}

function updatePointerHoleIndicator(x = pointer.x, y = pointer.y) {
  const isActive = pointer.holePolarity !== 0 && x !== null && y !== null

  pointerHole.classList.toggle('is-active', isActive)
  pointerHole.classList.toggle(
    'is-black-hole',
    pointer.holePolarity === BLACK_HOLE_POLARITY,
  )
  pointerHole.classList.toggle(
    'is-white-hole',
    pointer.holePolarity === WHITE_HOLE_POLARITY,
  )
  canvas.classList.toggle('has-pointer-hole', isActive)

  if (isActive) {
    pointerHole.style.left = `${x}px`
    pointerHole.style.top = `${y}px`
  }
}

function updatePointerHole(event) {
  if (event.pointerType !== 'mouse') {
    return false
  }

  let nextPolarity = 0

  if (event.buttons & 1) {
    nextPolarity = BLACK_HOLE_POLARITY
  } else if (event.buttons & 2) {
    nextPolarity = WHITE_HOLE_POLARITY
  }

  const hasPolarityChanged = pointer.holePolarity !== nextPolarity

  pointer.holePolarity = nextPolarity
  updatePointerHoleIndicator(event.clientX, event.clientY)

  return hasPolarityChanged
}

function sendPointerState(x, y, timeStamp) {
  pointer.x = x
  pointer.y = y

  sendToWorker({
    type: MESSAGE_POINTER,
    x,
    y,
    timeStamp,
    polarity: pointer.holePolarity,
  })
}

function handlePointerMove(event) {
  updatePointerHole(event)

  const samples = event.getCoalescedEvents?.() ?? [event]
  const sample = samples[samples.length - 1] ?? event

  sendPointerState(sample.clientX, sample.clientY, sample.timeStamp)
}

function handlePointerButtons(event) {
  if (updatePointerHole(event)) {
    sendPointerState(event.clientX, event.clientY, event.timeStamp)
  }
}

function resetPointer(event) {
  pointer.holePolarity = 0
  pointer.x = event?.clientX ?? null
  pointer.y = event?.clientY ?? null
  updatePointerHoleIndicator()
  sendToWorker({ type: MESSAGE_POINTER_RESET })
}

function ensureRenderCapacity(count) {
  if (renderX.length >= count) {
    return
  }

  const grownX = new Float32Array(count)
  const grownY = new Float32Array(count)

  grownX.set(renderX)
  grownY.set(renderY)
  renderX = grownX
  renderY = grownY
}

function handleSnapshot(data) {
  const arrivedAt = performance.now()
  const views = createSnapshotViews(data.buffer, data.count)

  if (previousSnapshot) {
    sendToWorker(
      { type: MESSAGE_RECYCLE_BUFFER, buffer: previousSnapshot.buffer },
      [previousSnapshot.buffer],
    )
  }

  if (latestSnapshot) {
    snapshotInterval +=
      (arrivedAt - latestSnapshot.arrivedAt - snapshotInterval) *
      SNAPSHOT_INTERVAL_SMOOTHING
  }

  previousSnapshot = latestSnapshot
  latestSnapshot = { buffer: data.buffer, count: data.count, views, arrivedAt }

  ensureRenderCapacity(data.count)
}

function renderParticles(time, veilAlpha) {
  const { count, views } = latestSnapshot
  const previousViews = previousSnapshot?.views
  const previousCount = previousSnapshot?.count ?? 0
  const canInterpolate =
    Boolean(previousViews) &&
    (views.header[0] & SNAPSHOT_FLAG_DISCONTINUOUS) === 0
  const span = clamp(snapshotInterval, FRAME_DURATION, MAX_INTERPOLATION_SPAN)
  const alpha = clamp((time - latestSnapshot.arrivedAt) / span, 0, 1)

  if (instanceData.length < count * INSTANCE_FLOATS) {
    instanceData = new Float32Array(count * INSTANCE_FLOATS)
  }

  for (let index = 0; index < count; index += 1) {
    const currentX = views.positions[index * 2]
    const currentY = views.positions[index * 2 + 1]
    const isRespawned = (views.flags[index] & PARTICLE_FLAG_RESPAWNED) !== 0
    let x = currentX
    let y = currentY

    // A recycled particle's slot still holds the old particle's position in
    // the previous snapshot, so interpolating would drag it across the screen.
    if (canInterpolate && index < previousCount && !isRespawned) {
      const startX = previousViews.positions[index * 2]
      const startY = previousViews.positions[index * 2 + 1]

      x = startX + (currentX - startX) * alpha
      y = startY + (currentY - startY) * alpha
    }

    if (
      isRespawned ||
      Math.abs(x - renderX[index]) + Math.abs(y - renderY[index]) >
        MAX_STREAK_LENGTH
    ) {
      renderX[index] = x
      renderY[index] = y
    }

    // Speed picks the palette entry, then the fade scales it continuously to
    // black. Bucket 0 is still 25% lightness, so quantising the fade too would
    // make a retiring particle vanish from a visible dot.
    const fade =
      (views.flags[index] >> PARTICLE_FADE_SHIFT) & PARTICLE_FADE_LEVELS
    const fadeScale = fade / PARTICLE_FADE_LEVELS
    const palette =
      (views.colors[index] * BRIGHTNESS_LEVELS +
        getBrightnessIndex(views.speeds[index])) *
      3
    const offset = index * INSTANCE_FLOATS

    // The streak spans motion since the last drawn frame, so it stays the
    // right length at any display rate.
    instanceData[offset] = renderX[index]
    instanceData[offset + 1] = renderY[index]
    instanceData[offset + 2] = x
    instanceData[offset + 3] = y
    instanceData[offset + 4] = PARTICLE_COLORS[palette] * fadeScale
    instanceData[offset + 5] = PARTICLE_COLORS[palette + 1] * fadeScale
    instanceData[offset + 6] = PARTICLE_COLORS[palette + 2] * fadeScale
    renderX[index] = x
    renderY[index] = y
  }

  context.draw(instanceData, count, veilAlpha)
}

function drawFrame(time) {
  const delta = Math.min(time - previousFrameTime, FRAME_DURATION * 2)
  const veilAlpha = 1 - Math.exp(-delta / TRAIL_TAU)

  previousFrameTime = time

  if (latestSnapshot) {
    renderParticles(time, veilAlpha)
  } else {
    context.draw(instanceData, 0, veilAlpha)
  }

  requestAnimationFrame(drawFrame)
}

canvas.addEventListener('pointerenter', (event) => {
  resetPointer(event)
  handlePointerButtons(event)
})
canvas.addEventListener('pointermove', handlePointerMove)
canvas.addEventListener('pointerleave', () => resetPointer())
canvas.addEventListener('pointerdown', (event) => {
  resetPointer(event)
  handlePointerButtons(event)
  canvas.setPointerCapture(event.pointerId)
})
canvas.addEventListener('pointerup', (event) => {
  handlePointerButtons(event)

  if (event.pointerType !== 'mouse') {
    resetPointer()
  }
})
canvas.addEventListener('pointercancel', () => resetPointer())
canvas.addEventListener('contextmenu', (event) => event.preventDefault())
window.addEventListener('resize', resizeCanvas)
window.addEventListener('keydown', handleKeyDown)
window.addEventListener('blur', () => resetPointer())
document.addEventListener('visibilitychange', () => {
  sendToWorker({
    type: MESSAGE_VISIBILITY,
    hidden: document.hidden,
  })
  resetPointer()
})

physicsWorker.onerror = (event) => {
  // A module worker that fails during init dies silently otherwise.
  console.error('Physics worker failed:', event.message)
}

physicsWorker.onmessage = ({ data }) => {
  if (data.type === MESSAGE_READY) {
    isWorkerReady = true

    for (const { message, transfer } of queuedWorkerMessages) {
      physicsWorker.postMessage(message, transfer)
    }

    queuedWorkerMessages.length = 0
    return
  }

  if (data.type === MESSAGE_SNAPSHOT) {
    handleSnapshot(data)
  }
}

const initialSnapshotBuffers = Array.from(
  { length: SNAPSHOT_POOL_SIZE },
  () =>
    new ArrayBuffer(
      getSnapshotByteLength(DEFAULT_PARTICLE_LIMIT + MAX_RETIRING_PARTICLES),
    ),
)

configureCanvas()
document
  .querySelector('#controls-dismiss')
  .addEventListener('click', () => controlsDialog.close())
controlsDialog.showModal()
sendToWorker(
  {
    type: MESSAGE_INIT,
    width: viewportWidth,
    height: viewportHeight,
    gravityPercent,
    airResistancePercent,
    particleLimit,
    emissionEnabled: isParticleEmissionEnabled,
    hidden: document.hidden,
    buffers: initialSnapshotBuffers,
  },
  initialSnapshotBuffers,
)
requestAnimationFrame(drawFrame)
