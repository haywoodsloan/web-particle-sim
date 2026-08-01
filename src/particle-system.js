import RAPIER from '@dimforge/rapier2d-compat'

export const FRAME_DURATION = 1000 / 60
export const MAX_CONTROL_PERCENT = 100
export const MAX_WORLD_GRAVITY = 1
export const MAX_AIR_RESISTANCE = 0.03
export const GRAVITY_SCALE = 0.001
export const DEFAULT_PARTICLE_LIMIT = 500
export const PARTICLE_LIMIT_STEP = 250
export const MAX_PARTICLES_PER_FRAME = 10
export const EMISSION_VELOCITY_SCALE = 0.4
export const FULL_BRIGHTNESS_SPEED = 6
export const PARTICLE_GRAVITY = 0.15
export const PARTICLE_GRAVITY_SOFTENING = 24
export const POINTER_HOLE_RADIUS = 375
export const POINTER_HOLE_MASS = 288
export const POINTER_HOLE_SOFTENING = 90
export const BLACK_HOLE_POLARITY = 1
export const WHITE_HOLE_POLARITY = -1
export const HUE_COUNT = 36
export const PARTICLE_RADIUS = 1.05

const STEPS_PER_SECOND = 1000 / FRAME_DURATION
/**
 * The tuning constants above are in px/ms^2, which is what the previous
 * integrator used. Rapier works in px/s^2, hence the 1000^2.
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

export const clamp = (value, minimum, maximum) =>
  Math.min(Math.max(value, minimum), maximum)

const POINTER_HOLE_RIM_SCALE =
  POINTER_HOLE_MASS /
  (POINTER_HOLE_RADIUS ** 2 + POINTER_HOLE_SOFTENING ** 2) ** 1.5

/** Plummer-softened point mass; callers multiply by the offset vector. */
export function getPointerHoleAccelerationScale(deltaX, deltaY, polarity) {
  const distanceSquared = deltaX ** 2 + deltaY ** 2

  if (distanceSquared >= POINTER_HOLE_RADIUS ** 2 || distanceSquared < 0.0001) {
    return 0
  }

  const softened = distanceSquared + POINTER_HOLE_SOFTENING ** 2
  // Shifted-force truncation keeps the field continuous at the rim.
  const scale =
    POINTER_HOLE_MASS / (softened * Math.sqrt(softened)) - POINTER_HOLE_RIM_SCALE

  return scale <= 0 ? 0 : scale * polarity
}

const randomBetween = (minimum, maximum) =>
  minimum + Math.random() * (maximum - minimum)

await RAPIER.init()

const world = new RAPIER.World({ x: 0, y: 0 })

world.integrationParameters.dt = FRAME_DURATION / 1000
world.integrationParameters.numSolverIterations = SOLVER_ITERATIONS
world.integrationParameters.lengthUnit = LENGTH_UNIT
world.integrationParameters.contact_natural_frequency = CONTACT_NATURAL_FREQUENCY

const bodies = []
const boundaries = []

let capacity = 0
let recycleCursor = 0
let viewportWidth = 0
let viewportHeight = 0
let particleLimit = 0
let gravityPercent = 0
let airResistancePercent = 0
let isEmissionEnabled = true

let px = new Float32Array(0)
let py = new Float32Array(0)
let previousPx = new Float32Array(0)
let previousPy = new Float32Array(0)
let vx = new Float32Array(0)
let vy = new Float32Array(0)
let mass = new Float32Array(0)
let colorIndex = new Uint8Array(0)
let respawned = new Uint8Array(0)

/** Particles recycled this step, drained by the renderer to draw fade-outs. */
export const retirements = []

const pointer = {
  x: null,
  y: null,
  time: null,
  hue: Math.random() * 360,
  spawnBudget: 0,
  holePolarity: 0,
}

export const getCount = () => bodies.length
export const getPositionX = () => px
export const getPositionY = () => py
export const getPreviousPositionX = () => previousPx
export const getPreviousPositionY = () => previousPy
export const getVelocityX = () => vx
export const getVelocityY = () => vy
export const getColorIndex = () => colorIndex
export const getRespawned = () => respawned

export const getAirResistance = () =>
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
  vx = copy(vx)
  vy = copy(vy)
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

export function resize(width, height) {
  viewportWidth = width
  viewportHeight = height
  rebuildBoundaries()
}

function recordRetirement(index) {
  retirements.push({
    x: px[index],
    y: py[index],
    velocityX: vx[index],
    velocityY: vy[index],
    colorIndex: colorIndex[index],
    speed: Math.hypot(vx[index], vy[index]),
  })
}

export function setParticleLimit(limit) {
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

export function setGravityPercent(percent) {
  gravityPercent = percent
  world.gravity = {
    x: 0,
    y:
      MAX_WORLD_GRAVITY *
      (gravityPercent / MAX_CONTROL_PERCENT) *
      GRAVITY_SCALE *
      ACCELERATION_TO_SECONDS,
  }
}

export function setAirResistancePercent(percent) {
  airResistancePercent = percent

  const damping = getLinearDamping()

  for (const body of bodies) {
    body.setLinearDamping(damping)
  }
}

export function setEmissionEnabled(enabled) {
  isEmissionEnabled = enabled
  pointer.spawnBudget = 0
  pointer.time = null
}

export function setPointerHolePolarity(polarity) {
  pointer.holePolarity = polarity
}

export function resetPointer(x, y, timeStamp) {
  pointer.x = x
  pointer.y = y
  pointer.time = timeStamp
  pointer.holePolarity = 0
  pointer.spawnBudget = 0
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
    body.resetForces(true)
  }

  px[index] = positionX
  py[index] = positionY
  // A recycled particle must not streak in from wherever it used to be.
  previousPx[index] = positionX
  previousPy[index] = positionY
  vx[index] = velocity.x / STEPS_PER_SECOND
  vy[index] = velocity.y / STEPS_PER_SECOND
  colorIndex[index] = Math.floor(hue / 10) % HUE_COUNT
  respawned[index] = 1
}

export function emit(x, y, timeStamp) {
  if (!isEmissionEnabled || pointer.holePolarity !== 0) {
    pointer.x = x
    pointer.y = y
    pointer.time = timeStamp
    pointer.spawnBudget = 0
    return
  }

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

function applyFieldForces() {
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

    px[index] = translation.x
    py[index] = translation.y
    // Displacement over one step IS the velocity in px/step, so reading linvel
    // back across the WASM boundary would be redundant.
    vx[index] = px[index] - previousPx[index]
    vy[index] = py[index] - previousPy[index]
  }
}

export function step() {
  const count = bodies.length

  if (count === 0) {
    return
  }

  previousPx.set(px.subarray(0, count))
  previousPy.set(py.subarray(0, count))
  applyFieldForces()
  world.step()
  readBackState()
}
