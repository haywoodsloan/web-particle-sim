import './style.css'

import {
  BLACK_HOLE_POLARITY,
  DEFAULT_PARTICLE_LIMIT,
  FRAME_DURATION,
  FULL_BRIGHTNESS_SPEED,
  HUE_COUNT,
  MAX_CONTROL_PERCENT,
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
  PARTICLE_FLAG_RESPAWNED,
  PARTICLE_LIMIT_STEP,
  SNAPSHOT_FLAG_DISCONTINUOUS,
  WHITE_HOLE_POLARITY,
  clamp,
  createSnapshotViews,
  getSnapshotByteLength,
} from './simulation-shared.js'

const CONTROL_STEP = 10
const PARTICLE_DIAMETER = 1.8
const BRIGHTNESS_LEVELS = 4
const MAX_PIXEL_RATIO = 2
const SNAPSHOT_POOL_SIZE = 4
const SNAPSHOT_INTERVAL_SMOOTHING = 0.15
const MAX_INTERPOLATION_SPAN = 250

const PARTICLE_COLORS = Array.from(
  { length: HUE_COUNT * BRIGHTNESS_LEVELS },
  (_, bucketIndex) => {
    const colorIndex = Math.floor(bucketIndex / BRIGHTNESS_LEVELS)
    const brightnessIndex = bucketIndex % BRIGHTNESS_LEVELS
    const lightness = 25 + (37 * brightnessIndex) / (BRIGHTNESS_LEVELS - 1)

    return `hsl(${colorIndex * 10}, 100%, ${lightness}%)`
  },
)

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
`

const canvas = document.querySelector('#particle-canvas')
const pointerHole = document.querySelector('#pointer-hole')
const controlStatus = document.querySelector('#control-status')
const context = canvas.getContext('2d', { alpha: false })
const physicsWorker = new Worker(
  new URL('./physics.worker.js', import.meta.url),
  { type: 'module' },
)

const colorBuckets = Array.from(
  { length: HUE_COUNT * BRIGHTNESS_LEVELS },
  () => [],
)
let previousSnapshot = null
let latestSnapshot = null
let snapshotInterval = FRAME_DURATION
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

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
  context.globalCompositeOperation = 'source-over'
  context.globalAlpha = 1
  context.lineCap = 'round'
  context.lineWidth = PARTICLE_DIAMETER
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
  sendToWorker({
    type: MESSAGE_POINTER_RESET,
    x: pointer.x,
    y: pointer.y,
    timeStamp: event?.timeStamp ?? null,
  })
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
}

function renderParticles(time) {
  const { count, views } = latestSnapshot

  if (count === 0) {
    return
  }

  const previousViews = previousSnapshot?.views
  const previousCount = previousSnapshot?.count ?? 0
  const canInterpolate =
    Boolean(previousViews) &&
    (views.header[0] & SNAPSHOT_FLAG_DISCONTINUOUS) === 0
  const span = clamp(snapshotInterval, FRAME_DURATION, MAX_INTERPOLATION_SPAN)
  const alpha = clamp((time - latestSnapshot.arrivedAt) / span, 0, 1)

  for (let index = 0; index < count; index += 1) {
    const brightnessIndex = getBrightnessIndex(views.speeds[index])

    colorBuckets[
      views.colors[index] * BRIGHTNESS_LEVELS + brightnessIndex
    ].push(index)
  }

  for (
    let bucketIndex = 0;
    bucketIndex < colorBuckets.length;
    bucketIndex += 1
  ) {
    const bucket = colorBuckets[bucketIndex]

    if (bucket.length === 0) {
      continue
    }

    context.beginPath()

    for (const index of bucket) {
      const currentX = views.positions[index * 2]
      const currentY = views.positions[index * 2 + 1]
      let x = currentX
      let y = currentY

      // A recycled particle's slot still holds the old particle's position in
      // the previous snapshot, so interpolating would drag it across the screen.
      if (
        canInterpolate &&
        index < previousCount &&
        (views.flags[index] & PARTICLE_FLAG_RESPAWNED) === 0
      ) {
        const startX = previousViews.positions[index * 2]
        const startY = previousViews.positions[index * 2 + 1]

        x = startX + (currentX - startX) * alpha
        y = startY + (currentY - startY) * alpha
      }

      context.moveTo(x, y)
      context.lineTo(x + 0.01, y)
    }

    context.strokeStyle = PARTICLE_COLORS[bucketIndex]
    context.stroke()
    bucket.length = 0
  }
}

function drawFrame(time) {
  context.fillStyle = '#000000'
  context.fillRect(0, 0, viewportWidth, viewportHeight)

  if (latestSnapshot) {
    renderParticles(time)
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
  () => new ArrayBuffer(getSnapshotByteLength(DEFAULT_PARTICLE_LIMIT)),
)

canvas.dataset.renderQuality = 'baseline'
configureCanvas()
sendToWorker(
  {
    type: MESSAGE_INIT,
    width: viewportWidth,
    height: viewportHeight,
    gravityPercent,
    airResistancePercent,
    particleLimit,
    emissionEnabled: isParticleEmissionEnabled,
    buffers: initialSnapshotBuffers,
  },
  initialSnapshotBuffers,
)
requestAnimationFrame(drawFrame)
