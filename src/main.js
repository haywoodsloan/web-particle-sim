import Matter from 'matter-js'
import './style.css'

const { Bodies, Body, Composite, Engine, Events } = Matter

const PARTICLE_CATEGORY = 0x0001
const BOUNDARY_CATEGORY = 0x0002
const DEFAULT_PARTICLE_LIMIT = 500
const PARTICLE_LIMIT_STEP = 250
const MAX_PARTICLES_PER_FRAME = 10
const RETIRE_FADE_DURATION = 500
const MAX_RETIRING_PARTICLES = 180
const FRAME_DURATION = 1000 / 60
const CONTROL_STEP = 10
const MAX_CONTROL_PERCENT = 100
const PARTICLE_GRAVITY = 4e-8
const MAX_WORLD_GRAVITY = 1
const MAX_AIR_RESISTANCE = 0.03
const EMISSION_VELOCITY_SCALE = 0.72
const POINTER_HOLE_RADIUS = 500
const POINTER_HOLE_STRENGTH = 0.02
const BLACK_HOLE_POLARITY = 1
const WHITE_HOLE_POLARITY = -1
const RAINBOW_COLORS = Array.from(
  { length: 36 },
  (_, index) => `hsl(${index * 10}, 100%, 62%)`,
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
const engine = Engine.create()

engine.gravity.x = 0
engine.gravity.y = 0
engine.gravity.scale = 0.001
engine.positionIterations = 8
engine.velocityIterations = 2
engine.constraintIterations = 1

const particles = []
const liveColorBuckets = Array.from(
  { length: RAINBOW_COLORS.length },
  () => [],
)
const retiringParticles = Array.from(
  { length: MAX_RETIRING_PARTICLES },
  () => ({ active: false }),
)
const wallInwardNormals = new Map()
const pendingWallImpacts = new Map()
let boundaries = []
let recycleCursor = 0
let retiringCursor = 0
let pendingPointerSample = null
let viewportWidth = window.innerWidth
let viewportHeight = window.innerHeight
let pixelRatio = 1
let previousFrameTime = performance.now()
let physicsAccumulator = 0
let gravityPercent = 0
let airResistancePercent = 0
let isParticleEmissionEnabled = true
let particleLimit = DEFAULT_PARTICLE_LIMIT

const pointer = {
  x: null,
  y: null,
  time: null,
  hue: Math.random() * 360,
  spawnBudget: 0,
  holePolarity: 0,
}

const clamp = (value, minimum, maximum) =>
  Math.min(Math.max(value, minimum), maximum)

const randomBetween = (minimum, maximum) =>
  minimum + Math.random() * (maximum - minimum)

const getAirResistance = () =>
  MAX_AIR_RESISTANCE * (airResistancePercent / MAX_CONTROL_PERCENT)

function showControlStatus(message) {
  controlStatus.textContent = message
  controlStatus.classList.remove('is-visible')
  void controlStatus.offsetWidth
  controlStatus.classList.add('is-visible')
}

function toggleParticleEmission() {
  isParticleEmissionEnabled = !isParticleEmissionEnabled
  pendingPointerSample = null
  pointer.spawnBudget = 0
  canvas.classList.toggle(
    'is-emission-disabled',
    !isParticleEmissionEnabled,
  )
}

function adjustGravity(direction) {
  gravityPercent = clamp(
    gravityPercent + direction * CONTROL_STEP,
    0,
    MAX_CONTROL_PERCENT,
  )
  engine.gravity.y =
    MAX_WORLD_GRAVITY * (gravityPercent / MAX_CONTROL_PERCENT)
  showControlStatus(`Gravity: ${gravityPercent}%`)
}

function adjustAirResistance(direction) {
  airResistancePercent = clamp(
    airResistancePercent + direction * CONTROL_STEP,
    0,
    MAX_CONTROL_PERCENT,
  )

  const frictionAir = getAirResistance()

  for (const particle of particles) {
    particle.body.frictionAir = frictionAir
  }

  showControlStatus(`Air Resistance: ${airResistancePercent}%`)
}

function adjustParticleLimit(direction) {
  particleLimit = Math.max(
    particleLimit + direction * PARTICLE_LIMIT_STEP,
    0,
  )

  const startedAt = performance.now()

  while (particles.length > particleLimit) {
    const particle = particles.pop()

    retireParticle(particle, startedAt)
    Composite.remove(engine.world, particle.body)
  }

  recycleCursor = particleLimit === 0 ? 0 : recycleCursor % particleLimit
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

function rebuildBoundaries() {
  for (const boundary of boundaries) {
    Composite.remove(engine.world, boundary)
  }

  wallInwardNormals.clear()

  const thickness = 120
  const boundaryOptions = {
    isStatic: true,
    restitution: 1,
    friction: 0,
    frictionStatic: 0,
    collisionFilter: {
      category: BOUNDARY_CATEGORY,
      mask: PARTICLE_CATEGORY,
    },
  }

  boundaries = [
    Bodies.rectangle(
      viewportWidth / 2,
      -thickness / 2,
      viewportWidth + thickness * 2,
      thickness,
      boundaryOptions,
    ),
    Bodies.rectangle(
      viewportWidth / 2,
      viewportHeight + thickness / 2,
      viewportWidth + thickness * 2,
      thickness,
      boundaryOptions,
    ),
    Bodies.rectangle(
      -thickness / 2,
      viewportHeight / 2,
      thickness,
      viewportHeight + thickness * 2,
      boundaryOptions,
    ),
    Bodies.rectangle(
      viewportWidth + thickness / 2,
      viewportHeight / 2,
      thickness,
      viewportHeight + thickness * 2,
      boundaryOptions,
    ),
  ]

  const inwardNormals = [
    { x: 0, y: 1 },
    { x: 0, y: -1 },
    { x: 1, y: 0 },
    { x: -1, y: 0 },
  ]

  // Matter resets material properties while converting a body to static.
  for (let index = 0; index < boundaries.length; index += 1) {
    const boundary = boundaries[index]

    boundary.restitution = 1
    boundary.friction = 0
    boundary.frictionStatic = 0
    wallInwardNormals.set(boundary, inwardNormals[index])
  }

  Composite.add(engine.world, boundaries)
}

function getWallImpact(pair) {
  const { parentA, parentB } = pair.collision
  const normalA = wallInwardNormals.get(parentA)

  if (normalA) {
    return { particleBody: parentB, inwardNormal: normalA }
  }

  const normalB = wallInwardNormals.get(parentB)

  if (normalB) {
    return { particleBody: parentA, inwardNormal: normalB }
  }

  return null
}

function captureWallImpacts(event) {
  for (const pair of event.pairs) {
    const wallImpact = getWallImpact(pair)

    if (!wallImpact) {
      continue
    }

    const { particleBody, inwardNormal } = wallImpact
    let pendingImpact = pendingWallImpacts.get(particleBody)

    if (!pendingImpact) {
      pendingImpact = {
        incomingVelocity: Body.getVelocity(particleBody),
        inwardNormals: [],
      }
      pendingWallImpacts.set(particleBody, pendingImpact)
    }

    pendingImpact.inwardNormals.push(inwardNormal)
  }
}

function preserveElasticWallImpacts() {
  for (const [particleBody, pendingImpact] of pendingWallImpacts) {
    const { incomingVelocity, inwardNormals } = pendingImpact
    const correctedVelocity = Body.getVelocity(particleBody)
    let requiresCorrection = false

    for (const inwardNormal of inwardNormals) {
      const incomingNormalSpeed =
        incomingVelocity.x * inwardNormal.x +
        incomingVelocity.y * inwardNormal.y

      if (incomingNormalSpeed >= 0) {
        continue
      }

      const resolvedNormalSpeed =
        correctedVelocity.x * inwardNormal.x +
        correctedVelocity.y * inwardNormal.y
      const targetNormalSpeed = -incomingNormalSpeed
      const normalSpeedCorrection = targetNormalSpeed - resolvedNormalSpeed

      correctedVelocity.x += normalSpeedCorrection * inwardNormal.x
      correctedVelocity.y += normalSpeedCorrection * inwardNormal.y
      requiresCorrection = true
    }

    if (requiresCorrection) {
      Body.setVelocity(particleBody, correctedVelocity)
    }
  }

  pendingWallImpacts.clear()
}

function resizeCanvas() {
  viewportWidth = window.innerWidth
  viewportHeight = window.innerHeight
  pixelRatio = Math.min(window.devicePixelRatio || 1, 2)

  canvas.width = Math.round(viewportWidth * pixelRatio)
  canvas.height = Math.round(viewportHeight * pixelRatio)
  canvas.style.width = `${viewportWidth}px`
  canvas.style.height = `${viewportHeight}px`

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
  context.fillStyle = '#000000'
  context.fillRect(0, 0, viewportWidth, viewportHeight)
  rebuildBoundaries()
}

function retireParticle(particle, startedAt) {
  const { body } = particle
  const retiringParticle = retiringParticles[retiringCursor]

  retiringCursor = (retiringCursor + 1) % MAX_RETIRING_PARTICLES
  retiringParticle.active = true
  retiringParticle.x = body.position.x
  retiringParticle.y = body.position.y
  retiringParticle.velocityX = body.velocity.x
  retiringParticle.velocityY = body.velocity.y
  retiringParticle.colorIndex = particle.colorIndex
  retiringParticle.radius = particle.radius
  retiringParticle.startedAt = startedAt
}

function createParticle(x, y, heading, pointerSpeed, hue, emittedAt) {
  if (particleLimit === 0) {
    return
  }

  const radius = randomBetween(0.35, 0.95)
  const spread = randomBetween(-0.34, 0.34)
  const speed =
    (pointerSpeed + randomBetween(2.2, 4.8)) *
    randomBetween(0.72, 1.12) *
    EMISSION_VELOCITY_SCALE
  const position = {
    x: x + randomBetween(-2.5, 2.5),
    y: y + randomBetween(-2.5, 2.5),
  }
  let particle

  if (particles.length < particleLimit) {
    const body = Bodies.circle(position.x, position.y, radius, {
      density: 0.0007,
      inertia: Infinity,
      friction: 0,
      frictionAir: getAirResistance(),
      frictionStatic: 0,
      restitution: 1,
      slop: 0.01,
      collisionFilter: {
        category: PARTICLE_CATEGORY,
        mask: PARTICLE_CATEGORY | BOUNDARY_CATEGORY,
      },
    })

    particle = { body }
    particles.push(particle)
    Composite.add(engine.world, body)
  } else {
    particle = particles[recycleCursor]
    recycleCursor = (recycleCursor + 1) % particleLimit
    retireParticle(particle, emittedAt)
    Body.setPosition(particle.body, position)
    particle.body.force.x = 0
    particle.body.force.y = 0
    particle.body.torque = 0
  }

  Body.setVelocity(particle.body, {
    x: Math.cos(heading + spread) * speed,
    y: Math.sin(heading + spread) * speed,
  })

  particle.colorIndex = Math.floor(hue / 10) % RAINBOW_COLORS.length
  particle.radius = radius
  particle.previousX = position.x
  particle.previousY = position.y
  particle.renderX = position.x
  particle.renderY = position.y
}

function applyMutualParticleGravity() {
  if (particles.length < 2) {
    return
  }

  let totalMass = 0
  let weightedX = 0
  let weightedY = 0

  for (const particle of particles) {
    const { body } = particle
    totalMass += body.mass
    weightedX += body.position.x * body.mass
    weightedY += body.position.y * body.mass
  }

  const centerX = weightedX / totalMass
  const centerY = weightedY / totalMass

  for (const particle of particles) {
    const { body } = particle
    const forceScale = PARTICLE_GRAVITY * body.mass * totalMass

    body.force.x += (centerX - body.position.x) * forceScale
    body.force.y += (centerY - body.position.y) * forceScale
  }
}

function getPointerHoleAccelerationScale(deltaX, deltaY) {
  const distanceSquared = deltaX ** 2 + deltaY ** 2
  const radiusSquared = POINTER_HOLE_RADIUS ** 2

  if (distanceSquared >= radiusSquared || distanceSquared < 0.0001) {
    return 0
  }

  const distance = Math.sqrt(distanceSquared)
  const proximity = 1 - distance / POINTER_HOLE_RADIUS

  return (
    (POINTER_HOLE_STRENGTH * proximity ** 2 * pointer.holePolarity) /
    distance
  )
}

function applyPointerHoleForce() {
  if (
    pointer.holePolarity === 0 ||
    pointer.x === null ||
    pointer.y === null
  ) {
    return
  }

  for (const particle of particles) {
    const { body } = particle
    const deltaX = pointer.x - body.position.x
    const deltaY = pointer.y - body.position.y
    const accelerationScale = getPointerHoleAccelerationScale(deltaX, deltaY)

    body.force.x += deltaX * body.mass * accelerationScale
    body.force.y += deltaY * body.mass * accelerationScale
  }
}

function emitFromPointer(sample, emittedAt) {
  if (pointer.x === null || pointer.y === null || pointer.time === null) {
    pointer.x = sample.clientX
    pointer.y = sample.clientY
    pointer.time = sample.timeStamp
    return
  }

  const deltaX = sample.clientX - pointer.x
  const deltaY = sample.clientY - pointer.y
  const distance = Math.hypot(deltaX, deltaY)

  if (distance < 0.8) {
    return
  }

  const elapsed = clamp(sample.timeStamp - pointer.time, 4, 40)
  const heading = Math.atan2(deltaY, deltaX)
  const pointerSpeedPerMillisecond = distance / elapsed
  const pointerSpeed = clamp(
    pointerSpeedPerMillisecond * FRAME_DURATION,
    1.4,
    17,
  )
  const pointerSpeedPerSecond = pointerSpeedPerMillisecond * 1000
  const spawnSpeedRatio = clamp((pointerSpeedPerSecond - 30) / 1000, 0, 1)
  const spawnRate = 300 * spawnSpeedRatio ** 1.35

  pointer.spawnBudget = Math.min(
    pointer.spawnBudget + spawnRate * (elapsed / 1000),
    MAX_PARTICLES_PER_FRAME,
  )

  const particleCount = Math.floor(pointer.spawnBudget)
  pointer.spawnBudget -= particleCount

  for (let index = 0; index < particleCount; index += 1) {
    const progress = (index + 1) / particleCount
    const x = pointer.x + deltaX * progress
    const y = pointer.y + deltaY * progress
    const hue = (pointer.hue + index * 17) % 360

    createParticle(x, y, heading, pointerSpeed, hue, emittedAt)
  }

  pointer.x = sample.clientX
  pointer.y = sample.clientY
  pointer.time = sample.timeStamp
  pointer.hue = (pointer.hue + distance * 1.8 + particleCount * 8) % 360
}

function handlePointerMove(event) {
  updatePointerHole(event)

  const samples = event.getCoalescedEvents?.() ?? [event]
  const sample = samples[samples.length - 1] ?? event

  if (!isParticleEmissionEnabled || pointer.holePolarity !== 0) {
    pendingPointerSample = null
    pointer.x = sample.clientX
    pointer.y = sample.clientY
    pointer.time = sample.timeStamp
    pointer.spawnBudget = 0
    return
  }

  pendingPointerSample = {
    clientX: sample.clientX,
    clientY: sample.clientY,
    timeStamp: sample.timeStamp,
  }
}

function updatePointerHole(event) {
  if (event.pointerType !== 'mouse') {
    return
  }

  if (event.buttons & 1) {
    pointer.holePolarity = BLACK_HOLE_POLARITY
  } else if (event.buttons & 2) {
    pointer.holePolarity = WHITE_HOLE_POLARITY
  } else {
    pointer.holePolarity = 0
  }

  updatePointerHoleIndicator(event.clientX, event.clientY)
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

function resetPointer(event) {
  pendingPointerSample = null
  pointer.x = event?.clientX ?? null
  pointer.y = event?.clientY ?? null
  pointer.time = event?.timeStamp ?? null
  pointer.spawnBudget = 0
  pointer.holePolarity = 0
  updatePointerHoleIndicator()
}

function drawFrame(time) {
  if (pendingPointerSample) {
    emitFromPointer(pendingPointerSample, time)
    pendingPointerSample = null
  }

  const delta = Math.min(time - previousFrameTime, FRAME_DURATION)
  previousFrameTime = time
  physicsAccumulator += delta

  if (physicsAccumulator + 0.1 >= FRAME_DURATION) {
    for (const particle of particles) {
      particle.previousX = particle.body.position.x
      particle.previousY = particle.body.position.y
    }

    applyMutualParticleGravity()
    applyPointerHoleForce()
    Engine.update(engine, FRAME_DURATION)
    physicsAccumulator = Math.max(0, physicsAccumulator - FRAME_DURATION)
  }

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
  context.globalCompositeOperation = 'source-over'
  context.globalAlpha = 1
  context.fillStyle = 'rgba(0, 0, 0, 0.24)'
  context.fillRect(0, 0, viewportWidth, viewportHeight)
  context.globalCompositeOperation = 'lighter'
  context.lineCap = 'round'
  const frameScale = delta / FRAME_DURATION
  const retirementDrag = (1 - getAirResistance()) ** frameScale
  const retirementGravity =
    engine.gravity.y * engine.gravity.scale * FRAME_DURATION ** 2
  const pointerHoleVelocityScale = FRAME_DURATION ** 2 * frameScale

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

    if (
      pointer.holePolarity !== 0 &&
      pointer.x !== null &&
      pointer.y !== null
    ) {
      const deltaX = pointer.x - particle.x
      const deltaY = pointer.y - particle.y
      const accelerationScale = getPointerHoleAccelerationScale(deltaX, deltaY)

      particle.velocityX +=
        deltaX * accelerationScale * pointerHoleVelocityScale
      particle.velocityY +=
        deltaY * accelerationScale * pointerHoleVelocityScale
    }

    particle.velocityY += retirementGravity * frameScale
    particle.x += particle.velocityX * frameScale
    particle.y += particle.velocityY * frameScale
    particle.velocityX *= retirementDrag
    particle.velocityY *= retirementDrag

    context.beginPath()
    context.moveTo(previousX, previousY)
    context.lineTo(particle.x, particle.y)
    context.globalAlpha = (1 - fadeProgress) ** 2
    context.lineWidth = 1
    context.strokeStyle = RAINBOW_COLORS[particle.colorIndex]
    context.stroke()
  }

  context.globalAlpha = 1
  context.lineWidth = 1
  const interpolation = physicsAccumulator / FRAME_DURATION

  for (const particle of particles) {
    liveColorBuckets[particle.colorIndex].push(particle)
  }

  for (let colorIndex = 0; colorIndex < liveColorBuckets.length; colorIndex += 1) {
    const bucket = liveColorBuckets[colorIndex]

    if (bucket.length === 0) {
      continue
    }

    context.beginPath()

    for (const particle of bucket) {
      const x =
        particle.previousX +
        (particle.body.position.x - particle.previousX) * interpolation
      const y =
        particle.previousY +
        (particle.body.position.y - particle.previousY) * interpolation

      context.moveTo(particle.renderX, particle.renderY)
      context.lineTo(x, y)
      particle.renderX = x
      particle.renderY = y
    }

    context.strokeStyle = RAINBOW_COLORS[colorIndex]
    context.stroke()
    bucket.length = 0
  }

  context.globalAlpha = 1
  context.globalCompositeOperation = 'source-over'
  requestAnimationFrame(drawFrame)
}

canvas.addEventListener('pointerenter', (event) => {
  resetPointer(event)
  updatePointerHole(event)
})
canvas.addEventListener('pointermove', handlePointerMove)
canvas.addEventListener('pointerleave', () => resetPointer())
canvas.addEventListener('pointerdown', (event) => {
  resetPointer(event)
  updatePointerHole(event)
  canvas.setPointerCapture(event.pointerId)
})
canvas.addEventListener('pointerup', (event) => {
  updatePointerHole(event)

  if (event.pointerType !== 'mouse') {
    resetPointer()
  }
})
canvas.addEventListener('pointercancel', () => resetPointer())
canvas.addEventListener('contextmenu', (event) => event.preventDefault())
Events.on(engine, 'collisionStart', captureWallImpacts)
Events.on(engine, 'afterUpdate', preserveElasticWallImpacts)
window.addEventListener('resize', resizeCanvas)
window.addEventListener('keydown', handleKeyDown)
window.addEventListener('blur', () => resetPointer())
document.addEventListener('visibilitychange', () => {
  previousFrameTime = performance.now()
  physicsAccumulator = 0

  for (const particle of particles) {
    particle.previousX = particle.body.position.x
    particle.previousY = particle.body.position.y
  }

  resetPointer()
})

resizeCanvas()
requestAnimationFrame(drawFrame)
