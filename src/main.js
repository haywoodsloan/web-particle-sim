import './style.css'

import {
  BLACK_HOLE_POLARITY,
  DEFAULT_PARTICLE_LIMIT,
  FRAME_DURATION,
  FULL_BRIGHTNESS_SPEED,
  HUE_COUNT,
  MAX_AIR_RESISTANCE,
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

const GLOW_LEVELS = 8
const RETIRE_FADE_DURATION = 500
const MAX_RETIRING_PARTICLES = 180
const SIMPLIFY_FRAME_DURATION = 1000 / 45
const RESTORE_FRAME_DURATION = 1000 / 55
const SIMPLIFY_AFTER_DURATION = 600
const RESTORE_AFTER_DURATION = 2000
/** Minimum time at a quality level, so the ladder cannot oscillate. */
const QUALITY_DWELL_DURATION = 3000
const FRAME_DURATION_SMOOTHING = 0.08
const CONTROL_STEP = 10
const PARTICLE_CORE_DIAMETER = 1.8
/** Bright inner glow, drawn over the halo. */
const PARTICLE_GLOW_WIDTH = 1.6
/** Wide, dim pass that gives the glow its reach. */
const PARTICLE_GLOW_HALO_WIDTH = 4.2
/** Trail decay constant; keeps the veil independent of frame rate. */
const TRAIL_TAU = 28
const MAX_STREAK_LENGTH = 160
const SNAPSHOT_POOL_SIZE = 4
const SNAPSHOT_INTERVAL_SMOOTHING = 0.15
const MAX_INTERPOLATION_SPAN = 250

const RAINBOW_COLORS = Array.from(
  { length: HUE_COUNT },
  (_, index) => `hsl(${index * 10}, 100%, 62%)`,
)
const PARTICLE_CORE_COLORS = Array.from(
  { length: HUE_COUNT * GLOW_LEVELS },
  (_, bucketIndex) => {
    const colorIndex = Math.floor(bucketIndex / GLOW_LEVELS)
    const brightnessIndex = bucketIndex % GLOW_LEVELS
    const lightness = 25 + (37 * brightnessIndex) / 7

    return `hsl(${colorIndex * 10}, 100%, ${lightness}%)`
  },
)
// Geometric so every step is the same ratio. A linear ramp banded badly at the
// dim end, where 0.15 -> 0.32 more than doubles the brightness.
const buildGlowIntensities = (peak) =>
  Array.from(
    { length: GLOW_LEVELS },
    (_, index) => 0.15 * (peak / 0.15) ** (index / (GLOW_LEVELS - 1)),
  )
const FULL_GLOW_INTENSITIES = buildGlowIntensities(1.35)
// Rescaled rather than clipped, so brightness stays monotonic once the second
// additive pass is dropped.
const SINGLE_PASS_GLOW_INTENSITIES = buildGlowIntensities(1)
const MERGED_BRIGHTNESS_INDICES = [0, 0, 2, 2, 4, 4, 7, 7]
// The glow is fill-rate bound, so it renders to a half-resolution layer and is
// upscaled. The bilinear upscale also softens it, which suits a glow.
const QUALITY_LEVELS = [
  {
    name: 'full',
    brightnessMap: null,
    intensities: FULL_GLOW_INTENSITIES,
    maxPixelRatio: 2,
    glowScale: 0.5,
    haloIntensity: 0.3,
  },
  {
    name: 'reduced',
    brightnessMap: null,
    intensities: FULL_GLOW_INTENSITIES,
    maxPixelRatio: 1,
    glowScale: 0.5,
    haloIntensity: 0.3,
  },
  {
    name: 'minimal',
    brightnessMap: MERGED_BRIGHTNESS_INDICES,
    intensities: SINGLE_PASS_GLOW_INTENSITIES,
    maxPixelRatio: 1,
    glowScale: 0.35,
    haloIntensity: 0,
  },
]

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
// Detached, so it never takes part in layout.
const glowCanvas = document.createElement('canvas')
const glowContext = glowCanvas.getContext('2d', { alpha: false })
const physicsWorker = new Worker(
  new URL('./physics.worker.js', import.meta.url),
  { type: 'module' },
)

const colorBuckets = Array.from({ length: HUE_COUNT * GLOW_LEVELS }, () => [])
const retiringParticles = Array.from(
  { length: MAX_RETIRING_PARTICLES },
  () => ({ active: false }),
)
let retiringCursor = 0
let renderX = new Float32Array(0)
let renderY = new Float32Array(0)
let previousSnapshot = null
let latestSnapshot = null
let snapshotInterval = FRAME_DURATION
let viewportWidth = window.innerWidth
let viewportHeight = window.innerHeight
let pixelRatio = 1
let previousFrameTime = performance.now()
let smoothedFrameDuration = FRAME_DURATION
let simplifyPressureDuration = 0
let restorePressureDuration = 0
let qualityDwellRemaining = 0
let qualityLevel = 0
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

const getBrightnessIndex = (speed) =>
  Math.round(
    Math.sqrt(clamp(speed / FULL_BRIGHTNESS_SPEED, 0, 1)) * (GLOW_LEVELS - 1),
  )

const getAirResistance = () =>
  MAX_AIR_RESISTANCE * (airResistancePercent / MAX_CONTROL_PERCENT)

function strokeGlow(intensity) {
  const { haloIntensity } = QUALITY_LEVELS[qualityLevel]

  if (haloIntensity > 0) {
    glowContext.lineWidth = PARTICLE_GLOW_HALO_WIDTH
    glowContext.globalAlpha = Math.min(intensity * haloIntensity, 1)
    glowContext.stroke()
  }

  glowContext.lineWidth = PARTICLE_GLOW_WIDTH
  glowContext.globalAlpha = Math.min(intensity, 1)
  glowContext.stroke()

  if (intensity > 1) {
    glowContext.globalAlpha = intensity - 1
    glowContext.stroke()
  }
}

const getTargetPixelRatio = () =>
  Math.min(
    window.devicePixelRatio || 1,
    QUALITY_LEVELS[qualityLevel].maxPixelRatio,
  )

function configureCanvas() {
  pixelRatio = getTargetPixelRatio()
  canvas.width = Math.round(viewportWidth * pixelRatio)
  canvas.height = Math.round(viewportHeight * pixelRatio)
  canvas.style.width = `${viewportWidth}px`
  canvas.style.height = `${viewportHeight}px`

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
  context.globalCompositeOperation = 'source-over'
  context.globalAlpha = 1
  context.fillStyle = '#000000'
  context.fillRect(0, 0, viewportWidth, viewportHeight)
  context.lineCap = 'round'

  const glowRatio = pixelRatio * QUALITY_LEVELS[qualityLevel].glowScale

  glowCanvas.width = Math.max(1, Math.round(viewportWidth * glowRatio))
  glowCanvas.height = Math.max(1, Math.round(viewportHeight * glowRatio))

  glowContext.setTransform(glowRatio, 0, 0, glowRatio, 0, 0)
  glowContext.globalCompositeOperation = 'source-over'
  glowContext.globalAlpha = 1
  glowContext.fillStyle = '#000000'
  glowContext.fillRect(0, 0, viewportWidth, viewportHeight)
  glowContext.lineCap = 'round'
}

function setQualityLevel(level) {
  const nextLevel = clamp(level, 0, QUALITY_LEVELS.length - 1)

  if (nextLevel === qualityLevel) {
    return
  }

  qualityLevel = nextLevel
  canvas.dataset.renderQuality = QUALITY_LEVELS[qualityLevel].name
  qualityDwellRemaining = QUALITY_DWELL_DURATION
  smoothedFrameDuration = FRAME_DURATION
  simplifyPressureDuration = 0
  restorePressureDuration = 0

  // Every level changes either the pixel ratio or the glow scale.
  configureCanvas()
}

function updateAdaptiveRendering(frameDuration) {
  if (document.hidden || frameDuration <= 0) {
    return
  }

  const boundedFrameDuration = Math.min(frameDuration, 100)

  smoothedFrameDuration +=
    (boundedFrameDuration - smoothedFrameDuration) * FRAME_DURATION_SMOOTHING

  if (qualityDwellRemaining > 0) {
    qualityDwellRemaining -= boundedFrameDuration
    return
  }

  if (smoothedFrameDuration > SIMPLIFY_FRAME_DURATION) {
    simplifyPressureDuration += boundedFrameDuration
    restorePressureDuration = 0
  } else if (smoothedFrameDuration < RESTORE_FRAME_DURATION) {
    restorePressureDuration += boundedFrameDuration
    simplifyPressureDuration = 0
  } else {
    simplifyPressureDuration = 0
    restorePressureDuration = 0
  }

  if (simplifyPressureDuration >= SIMPLIFY_AFTER_DURATION) {
    setQualityLevel(qualityLevel + 1)
  } else if (restorePressureDuration >= RESTORE_AFTER_DURATION) {
    setQualityLevel(qualityLevel - 1)
  }
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

function retireParticle(retirement, startedAt) {
  const retiringParticle = retiringParticles[retiringCursor]
  const { brightnessMap } = QUALITY_LEVELS[qualityLevel]
  let brightnessIndex = getBrightnessIndex(retirement.speed)

  if (brightnessMap) {
    brightnessIndex = brightnessMap[brightnessIndex]
  }

  retiringCursor = (retiringCursor + 1) % MAX_RETIRING_PARTICLES
  retiringParticle.active = true
  retiringParticle.x = retirement.x
  retiringParticle.y = retirement.y
  retiringParticle.velocityX = retirement.velocityX
  retiringParticle.velocityY = retirement.velocityY
  retiringParticle.colorIndex = retirement.colorIndex
  retiringParticle.brightnessIndex = brightnessIndex
  retiringParticle.startedAt = startedAt
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

  // Worker timestamps use a different time origin, so fades start on receipt.
  for (const retirement of data.retired) {
    retireParticle(retirement, arrivedAt)
  }
}

function renderParticleGlow(time) {
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
  const { brightnessMap, intensities } = QUALITY_LEVELS[qualityLevel]

  for (let index = 0; index < count; index += 1) {
    let brightnessIndex = getBrightnessIndex(views.speeds[index])

    if (brightnessMap) {
      brightnessIndex = brightnessMap[brightnessIndex]
    }

    colorBuckets[views.colors[index] * GLOW_LEVELS + brightnessIndex].push(index)
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

    glowContext.beginPath()

    for (const index of bucket) {
      const currentX = views.positions[index * 2]
      const currentY = views.positions[index * 2 + 1]
      let x = currentX
      let y = currentY

      if (canInterpolate && index < previousCount) {
        const startX = previousViews.positions[index * 2]
        const startY = previousViews.positions[index * 2 + 1]

        x = startX + (currentX - startX) * alpha
        y = startY + (currentY - startY) * alpha
      }

      if (
        (views.flags[index] & PARTICLE_FLAG_RESPAWNED) !== 0 ||
        Math.abs(x - renderX[index]) + Math.abs(y - renderY[index]) >
          MAX_STREAK_LENGTH
      ) {
        renderX[index] = x
        renderY[index] = y
      }

      // The streak spans motion since the last drawn frame, so it stays the
      // right length at any display rate.
      glowContext.moveTo(renderX[index], renderY[index])
      glowContext.lineTo(x, y)
      renderX[index] = x
      renderY[index] = y
    }

    glowContext.strokeStyle =
      RAINBOW_COLORS[Math.floor(bucketIndex / GLOW_LEVELS)]
    strokeGlow(intensities[bucketIndex % GLOW_LEVELS])
  }
}

/** Runs after the glow layer is upscaled, so cores stay sharp. */
function renderParticleCores() {
  context.globalAlpha = 1
  context.globalCompositeOperation = 'source-over'
  context.lineWidth = PARTICLE_CORE_DIAMETER

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
      context.moveTo(renderX[index], renderY[index])
      context.lineTo(renderX[index] + 0.01, renderY[index])
    }

    context.strokeStyle = PARTICLE_CORE_COLORS[bucketIndex]
    context.stroke()
    bucket.length = 0
  }
}

function renderRetiringParticles(time, frameScale) {
  const retirementDrag = (1 - getAirResistance()) ** frameScale
  const { intensities } = QUALITY_LEVELS[qualityLevel]

  for (const particle of retiringParticles) {
    if (!particle.active) {
      continue
    }

    const fadeProgress = clamp(
      (time - particle.startedAt) / RETIRE_FADE_DURATION,
      0,
      1,
    )

    if (fadeProgress >= 1) {
      particle.active = false
      continue
    }

    const previousX = particle.x
    const previousY = particle.y

    particle.x += particle.velocityX * frameScale
    particle.y += particle.velocityY * frameScale
    particle.velocityX *= retirementDrag
    particle.velocityY *= retirementDrag

    glowContext.beginPath()
    glowContext.moveTo(previousX, previousY)
    glowContext.lineTo(particle.x, particle.y)
    glowContext.strokeStyle = RAINBOW_COLORS[particle.colorIndex]
    strokeGlow(intensities[particle.brightnessIndex] * (1 - fadeProgress) ** 2)
  }
}

function drawFrame(time) {
  const rawFrameDuration = time - previousFrameTime

  previousFrameTime = time
  updateAdaptiveRendering(rawFrameDuration)

  const delta = Math.min(rawFrameDuration, FRAME_DURATION * 2)

  // The glow layer keeps the trail, so only it gets the fading veil.
  glowContext.globalCompositeOperation = 'source-over'
  glowContext.globalAlpha = 1
  glowContext.fillStyle = `rgba(0, 0, 0, ${1 - Math.exp(-delta / TRAIL_TAU)})`
  glowContext.fillRect(0, 0, viewportWidth, viewportHeight)
  glowContext.globalCompositeOperation = 'lighter'

  renderRetiringParticles(time, delta / FRAME_DURATION)

  if (latestSnapshot) {
    renderParticleGlow(time)
  }

  context.globalCompositeOperation = 'source-over'
  context.globalAlpha = 1
  context.drawImage(glowCanvas, 0, 0, viewportWidth, viewportHeight)

  if (latestSnapshot) {
    renderParticleCores()
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
  previousFrameTime = performance.now()
  smoothedFrameDuration = FRAME_DURATION
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

canvas.dataset.renderQuality = QUALITY_LEVELS[qualityLevel].name
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
