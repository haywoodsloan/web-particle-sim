import * as RAPIER from '@dimforge/rapier2d'

import {
  EMISSION_VELOCITY_SCALE,
  FRAME_DURATION,
  GRAVITY_SCALE,
  HUE_COUNT,
  MAX_AIR_RESISTANCE,
  MAX_CONTROL_PERCENT,
  MAX_PARTICLES_PER_FRAME,
  MAX_WORLD_GRAVITY,
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
  PARTICLE_GRAVITY,
  PARTICLE_GRAVITY_SOFTENING,
  PARTICLE_RADIUS,
  SNAPSHOT_FLAG_DISCONTINUOUS,
  clamp,
  createSnapshotViews,
  getPointerHoleAccelerationScale,
  getSnapshotByteLength,
} from './simulation-shared.js'

const STEPS_PER_SECOND = 1000 / FRAME_DURATION
/**
 * The tuning constants are in px/ms^2, which is what the previous integrator
 * used. Rapier works in px/s^2, hence the 1000^2.
 */
const ACCELERATION_TO_SECONDS = 1e6
const PARTICLE_DENSITY = 0.0007
const BOUNDARY_THICKNESS = 120
/**
 * Softer than Rapier's default 30. Measured over 24k steps, 30 compounds the
 * kinetic energy to 1.098 while 10 holds it inside 1.02 indefinitely.
 */
const CONTACT_NATURAL_FREQUENCY = 10
const SOLVER_ITERATIONS = 4
/** Rapier scales allowed penetration by this; particles are only ~1 px. */
const LENGTH_UNIT = 1
const SNAPSHOT_POOL_SIZE = 4

const world = new RAPIER.World({ x: 0, y: 0 })

world.integrationParameters.dt = FRAME_DURATION / 1000
world.integrationParameters.numSolverIterations = SOLVER_ITERATIONS
world.integrationParameters.lengthUnit = LENGTH_UNIT
world.integrationParameters.contact_natural_frequency = CONTACT_NATURAL_FREQUENCY

const bodies = []
const boundaries = []
const pendingRetirements = []
const snapshotBufferPool = []

let capacity = 0
let recycleCursor = 0
let viewportWidth = 0
let viewportHeight = 0
let particleLimit = 0
let gravityPercent = 0
let airResistancePercent = 0
let isEmissionEnabled = true
let isRunning = false
let isHidden = false
let stepTimerId = null
let nextStepTime = 0
let stepIndex = 0
let hasDiscontinuity = true

let px = new Float32Array(0)
let py = new Float32Array(0)
let previousPx = new Float32Array(0)
let previousPy = new Float32Array(0)
let mass = new Float32Array(0)
let colorIndex = new Uint8Array(0)
let respawned = new Uint8Array(0)

const pointer = {
  x: null,
  y: null,
  time: null,
  hue: Math.random() * 360,
  spawnBudget: 0,
  holePolarity: 0,
}

const randomBetween = (minimum, maximum) =>
  minimum + Math.random() * (maximum - minimum)

const getAirResistance = () =>
  MAX_AIR_RESISTANCE * (airResistancePercent / MAX_CONTROL_PERCENT)

/** Rapier damps as v /= (1 + dt * damping); solve for the old per-step factor. */
const getLinearDamping = () => {
  const retained = 1 - getAirResistance()

  return retained <= 0 ? 0 : (1 / retained - 1) * STEPS_PER_SECOND
}

function ensureCapacity(required) {
  if (capacity >= required) {
    return
  }

  const grown = Math.max(required, 64)
  const copy = (source) => {
    const next = new source.constructor(grown)

    next.set(source)

    return next
  }

  px = copy(px)
  py = copy(py)
  previousPx = copy(previousPx)
  previousPy = copy(previousPy)
  mass = copy(mass)
  colorIndex = copy(colorIndex)
  respawned = copy(respawned)
  capacity = grown
}

function rebuildBoundaries() {
  for (const boundary of boundaries) {
    world.removeRigidBody(boundary)
  }

  boundaries.length = 0

  const half = BOUNDARY_THICKNESS / 2
  const walls = [
    [viewportWidth / 2, -half, viewportWidth / 2 + BOUNDARY_THICKNESS, half],
    [
      viewportWidth / 2,
      viewportHeight + half,
      viewportWidth / 2 + BOUNDARY_THICKNESS,
      half,
    ],
    [-half, viewportHeight / 2, half, viewportHeight / 2 + BOUNDARY_THICKNESS],
    [
      viewportWidth + half,
      viewportHeight / 2,
      half,
      viewportHeight / 2 + BOUNDARY_THICKNESS,
    ],
  ]

  for (const [x, y, halfWidth, halfHeight] of walls) {
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(x, y),
    )

    world.createCollider(
      RAPIER.ColliderDesc.cuboid(halfWidth, halfHeight)
        .setRestitution(1)
        .setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Max)
        .setFriction(0)
        .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Min),
      body,
    )
    boundaries.push(body)
  }
}

function recordRetirement(index) {
  pendingRetirements.push({
    x: px[index],
    y: py[index],
    velocityX: px[index] - previousPx[index],
    velocityY: py[index] - previousPy[index],
    colorIndex: colorIndex[index],
    speed: Math.hypot(px[index] - previousPx[index], py[index] - previousPy[index]),
  })
}

function applyParticleLimit(limit) {
  particleLimit = Math.max(limit, 0)

  while (bodies.length > particleLimit) {
    const index = bodies.length - 1

    recordRetirement(index)
    world.removeRigidBody(bodies[index])
    bodies.pop()
  }

  ensureCapacity(particleLimit)
  recycleCursor = particleLimit === 0 ? 0 : recycleCursor % particleLimit
}

function spawnParticle(x, y, heading, pointerSpeed, hue) {
  if (particleLimit === 0) {
    return
  }

  const spread = randomBetween(-0.34, 0.34)
  const speed =
    (pointerSpeed + randomBetween(2.2, 4.8)) *
    randomBetween(0.72, 1.12) *
    EMISSION_VELOCITY_SCALE *
    STEPS_PER_SECOND
  const positionX = clamp(
    x + randomBetween(-2.5, 2.5),
    PARTICLE_RADIUS,
    Math.max(PARTICLE_RADIUS, viewportWidth - PARTICLE_RADIUS),
  )
  const positionY = clamp(
    y + randomBetween(-2.5, 2.5),
    PARTICLE_RADIUS,
    Math.max(PARTICLE_RADIUS, viewportHeight - PARTICLE_RADIUS),
  )
  const velocity = {
    x: Math.cos(heading + spread) * speed,
    y: Math.sin(heading + spread) * speed,
  }
  let index

  if (bodies.length < particleLimit) {
    const visualRadius = randomBetween(0.35, 0.95)

    ensureCapacity(bodies.length + 1)
    index = bodies.length

    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(positionX, positionY)
        .setLinvel(velocity.x, velocity.y)
        .lockRotations()
        .setLinearDamping(getLinearDamping())
        .setCanSleep(false),
    )

    world.createCollider(
      RAPIER.ColliderDesc.ball(PARTICLE_RADIUS)
        .setMass(PARTICLE_DENSITY * Math.PI * visualRadius ** 2)
        .setRestitution(1)
        .setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Max)
        .setFriction(0)
        .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Min),
      body,
    )
    bodies.push(body)
    mass[index] = body.mass()
  } else {
    index = recycleCursor
    recycleCursor = (recycleCursor + 1) % particleLimit
    recordRetirement(index)

    const body = bodies[index]

    body.setTranslation({ x: positionX, y: positionY }, true)
    body.setLinvel(velocity, true)
  }

  px[index] = positionX
  py[index] = positionY
  // A recycled particle must not streak in from wherever it used to be.
  previousPx[index] = positionX
  previousPy[index] = positionY
  colorIndex[index] = Math.floor(hue / 10) % HUE_COUNT
  respawned[index] = 1
}

function emitFromPointer(x, y, timeStamp) {
  if (pointer.x === null || pointer.y === null || pointer.time === null) {
    pointer.x = x
    pointer.y = y
    pointer.time = timeStamp
    return
  }

  const deltaX = x - pointer.x
  const deltaY = y - pointer.y
  const distance = Math.hypot(deltaX, deltaY)

  if (distance < 0.8) {
    return
  }

  const elapsed = clamp(timeStamp - pointer.time, 4, 40)
  const heading = Math.atan2(deltaY, deltaX)
  const pointerSpeedPerMillisecond = distance / elapsed
  const pointerSpeed = clamp(pointerSpeedPerMillisecond * FRAME_DURATION, 1.4, 8)
  const spawnSpeedRatio = clamp(
    (pointerSpeedPerMillisecond * 1000 - 30) / 1000,
    0,
    1,
  )
  const spawnRate = 300 * spawnSpeedRatio ** 1.35

  pointer.spawnBudget = Math.min(
    pointer.spawnBudget + spawnRate * (elapsed / 1000),
    MAX_PARTICLES_PER_FRAME,
  )

  const particleCount = Math.floor(pointer.spawnBudget)

  pointer.spawnBudget -= particleCount

  for (let index = 0; index < particleCount; index += 1) {
    const progress = (index + 1) / particleCount

    spawnParticle(
      pointer.x + deltaX * progress,
      pointer.y + deltaY * progress,
      heading,
      pointerSpeed,
      (pointer.hue + index * 17) % 360,
    )
  }

  pointer.x = x
  pointer.y = y
  pointer.time = timeStamp
  pointer.hue = (pointer.hue + distance * 1.8 + particleCount * 8) % 360
}

const appliedImpulse = { x: 0, y: 0 }

function applyFieldImpulses() {
  const count = bodies.length
  const hasHole =
    pointer.holePolarity !== 0 && pointer.x !== null && pointer.y !== null
  let totalMass = 0
  let weightedX = 0
  let weightedY = 0

  for (let index = 0; index < count; index += 1) {
    totalMass += mass[index]
    weightedX += px[index] * mass[index]
    weightedY += py[index] * mass[index]
  }

  for (let index = 0; index < count; index += 1) {
    const bodyMass = mass[index]
    // A monopole must not attract itself.
    const otherMass = totalMass - bodyMass
    let accelerationX = 0
    let accelerationY = 0

    if (otherMass > 0) {
      const offsetX = (weightedX - px[index] * bodyMass) / otherMass - px[index]
      const offsetY = (weightedY - py[index] * bodyMass) / otherMass - py[index]
      const softened =
        offsetX ** 2 + offsetY ** 2 + PARTICLE_GRAVITY_SOFTENING ** 2
      const scale =
        (PARTICLE_GRAVITY * otherMass * ACCELERATION_TO_SECONDS) /
        (softened * Math.sqrt(softened))

      accelerationX += offsetX * scale
      accelerationY += offsetY * scale
    }

    if (hasHole) {
      const deltaX = pointer.x - px[index]
      const deltaY = pointer.y - py[index]
      const scale =
        getPointerHoleAccelerationScale(deltaX, deltaY, pointer.holePolarity) *
        ACCELERATION_TO_SECONDS

      accelerationX += deltaX * scale
      accelerationY += deltaY * scale
    }

    if (accelerationX === 0 && accelerationY === 0) {
      continue
    }

    // An impulse is spent on contact, so it needs no per-step reset the way a
    // persistent force would.
    appliedImpulse.x = (accelerationX * bodyMass) / STEPS_PER_SECOND
    appliedImpulse.y = (accelerationY * bodyMass) / STEPS_PER_SECOND
    bodies[index].applyImpulse(appliedImpulse, false)
  }
}

function readBackState() {
  for (let index = 0; index < bodies.length; index += 1) {
    const translation = bodies[index].translation()

    previousPx[index] = px[index]
    previousPy[index] = py[index]
    px[index] = translation.x
    py[index] = translation.y
  }
}

function acquireSnapshotBuffer(byteLength) {
  for (let index = 0; index < snapshotBufferPool.length; index += 1) {
    if (snapshotBufferPool[index].byteLength >= byteLength) {
      return snapshotBufferPool.splice(index, 1)[0]
    }
  }

  // Pooled buffers are all too small; replace one rather than growing the pool.
  if (snapshotBufferPool.length > 0) {
    snapshotBufferPool.pop()
    return new ArrayBuffer(byteLength)
  }

  return null
}

function sendSnapshot() {
  const count = bodies.length
  const buffer = acquireSnapshotBuffer(getSnapshotByteLength(count))

  // No free buffer means the main thread is behind. Keep simulating and skip
  // this snapshot so the message queue can never grow unbounded.
  if (!buffer) {
    return
  }

  const views = createSnapshotViews(buffer, count)

  views.header[0] = hasDiscontinuity ? SNAPSHOT_FLAG_DISCONTINUOUS : 0
  views.header[1] = count
  views.header[2] = stepIndex

  for (let index = 0; index < count; index += 1) {
    views.positions[index * 2] = px[index]
    views.positions[index * 2 + 1] = py[index]
    // Displacement over one step IS the velocity in px/step.
    views.speeds[index] = Math.hypot(
      px[index] - previousPx[index],
      py[index] - previousPy[index],
    )
    views.colors[index] = colorIndex[index]
    views.flags[index] = respawned[index] ? PARTICLE_FLAG_RESPAWNED : 0
    respawned[index] = 0
  }

  hasDiscontinuity = false

  const retired = pendingRetirements.splice(0, pendingRetirements.length)

  self.postMessage({ type: MESSAGE_SNAPSHOT, buffer, count, retired }, [buffer])
}

function step() {
  if (bodies.length > 0) {
    applyFieldImpulses()
  }

  world.step()
  readBackState()
  stepIndex += 1
  sendSnapshot()
}

function scheduleNextStep() {
  if (!isRunning) {
    return
  }

  stepTimerId = setTimeout(runStep, Math.max(0, nextStepTime - performance.now()))
}

function runStep() {
  step()
  nextStepTime += FRAME_DURATION

  const now = performance.now()

  if (now - nextStepTime > 100) {
    nextStepTime = now
  }

  scheduleNextStep()
}

function startLoop() {
  if (isRunning || isHidden) {
    return
  }

  isRunning = true
  nextStepTime = performance.now()
  scheduleNextStep()
}

function stopLoop() {
  isRunning = false

  if (stepTimerId !== null) {
    clearTimeout(stepTimerId)
    stepTimerId = null
  }
}

function resetPointerEmission() {
  pointer.spawnBudget = 0
  pointer.time = null
}

self.onmessage = ({ data }) => {
  switch (data.type) {
    case MESSAGE_INIT:
      viewportWidth = data.width
      viewportHeight = data.height
      gravityPercent = data.gravityPercent
      airResistancePercent = data.airResistancePercent
      isEmissionEnabled = data.emissionEnabled
      world.gravity = {
        x: 0,
        y:
          MAX_WORLD_GRAVITY *
          (gravityPercent / MAX_CONTROL_PERCENT) *
          GRAVITY_SCALE *
          ACCELERATION_TO_SECONDS,
      }
      applyParticleLimit(data.particleLimit)
      snapshotBufferPool.push(...data.buffers)
      rebuildBoundaries()
      startLoop()
      break

    case MESSAGE_RESIZE:
      viewportWidth = data.width
      viewportHeight = data.height
      rebuildBoundaries()
      break

    case MESSAGE_POINTER:
      pointer.holePolarity = data.polarity

      if (isEmissionEnabled && data.polarity === 0) {
        emitFromPointer(data.x, data.y, data.timeStamp)
      } else {
        pointer.x = data.x
        pointer.y = data.y
        pointer.time = data.timeStamp
        pointer.spawnBudget = 0
      }
      break

    case MESSAGE_POINTER_RESET:
      pointer.x = data.x
      pointer.y = data.y
      pointer.time = data.timeStamp
      pointer.holePolarity = 0
      resetPointerEmission()
      break

    case MESSAGE_EMISSION:
      isEmissionEnabled = data.enabled
      resetPointerEmission()
      break

    case MESSAGE_GRAVITY:
      gravityPercent = data.percent
      world.gravity = {
        x: 0,
        y:
          MAX_WORLD_GRAVITY *
          (gravityPercent / MAX_CONTROL_PERCENT) *
          GRAVITY_SCALE *
          ACCELERATION_TO_SECONDS,
      }
      break

    case MESSAGE_AIR_RESISTANCE: {
      airResistancePercent = data.percent

      const damping = getLinearDamping()

      for (const body of bodies) {
        body.setLinearDamping(damping)
      }
      break
    }

    case MESSAGE_PARTICLE_LIMIT:
      applyParticleLimit(data.limit)
      break

    case MESSAGE_VISIBILITY:
      isHidden = data.hidden

      if (isHidden) {
        stopLoop()
        resetPointerEmission()
      } else {
        hasDiscontinuity = true
        startLoop()
      }
      break

    case MESSAGE_RECYCLE_BUFFER:
      if (snapshotBufferPool.length < SNAPSHOT_POOL_SIZE) {
        snapshotBufferPool.push(data.buffer)
      }
      break

    default:
      break
  }
}

// vite-plugin-wasm makes this module top-level await, so the port is already
// live before `onmessage` exists. Anything sent earlier would be dropped.
self.postMessage({ type: MESSAGE_READY })
