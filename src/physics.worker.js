import * as RAPIER from '@dimforge/rapier2d'

import {
  EMISSION_VELOCITY_SCALE,
  FRAME_DURATION,
  GRAVITY_SCALE,
  HUE_COUNT,
  HUE_STEP,
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
/**
 * Chosen so a uniform 1.8 px particle weighs what the old varying-radius
 * particles averaged, which keeps the existing gravity tuning valid.
 */
const PARTICLE_DENSITY = 9.8e-5
const PARTICLE_MASS = PARTICLE_DENSITY * Math.PI * PARTICLE_RADIUS ** 2
/**
 * Particles cover several times their own diameter in one 60 Hz step, so the
 * solver runs finer than the snapshot rate or it skips straight past contacts.
 */
const PHYSICS_SUBSTEPS = 8
const BOUNDARY_THICKNESS = 120
/**
 * Contact stiffness has to scale with the solve rate. Measured over 9000
 * frames: freq 5/20/40 at 1/4/8 substeps are each the local optimum, and
 * 8 substeps with freq 40 holds kinetic energy at 0.997.
 */
const CONTACT_NATURAL_FREQUENCY = 5 * PHYSICS_SUBSTEPS
const SOLVER_ITERATIONS = 4
/** Rapier scales allowed penetration by this; particles are only ~1 px. */
const LENGTH_UNIT = 1
/** Barnes-Hut opening angle. 0.5 is the usual accuracy/speed compromise. */
const BARNES_HUT_THETA_SQUARED = 0.5 ** 2
const MAX_TREE_DEPTH = 20
const SNAPSHOT_POOL_SIZE = 4
const SPAWN_JITTER = 2.5
/**
 * Spawning inside another particle makes the solver fling both apart, which
 * invents energy, so positions are rejection sampled over a widening area.
 */
const SPAWN_PLACEMENT_ATTEMPTS = 8

const world = new RAPIER.World({ x: 0, y: 0 })

world.integrationParameters.dt =
  FRAME_DURATION / 1000 / PHYSICS_SUBSTEPS
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
let vx = new Float32Array(0)
let vy = new Float32Array(0)
let mass = new Float32Array(0)
let colorIndex = new Uint8Array(0)
let respawned = new Uint8Array(0)
/** Chains bodies that share a quadtree leaf. */
let bodyNext = new Int32Array(0)

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

/** Rapier damps as v /= (1 + dt * damping) each substep; solve for one frame. */
const getLinearDamping = () => {
  const retained = 1 - getAirResistance()

  return retained <= 0
    ? 0
    : (retained ** (-1 / PHYSICS_SUBSTEPS) - 1) *
        STEPS_PER_SECOND *
        PHYSICS_SUBSTEPS
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
  vx = copy(vx)
  vy = copy(vy)
  mass = copy(mass)
  colorIndex = copy(colorIndex)
  respawned = copy(respawned)
  bodyNext = copy(bodyNext)
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
    velocityX: vx[index],
    velocityY: vy[index],
    colorIndex: colorIndex[index],
    speed: Math.hypot(vx[index], vy[index]),
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

const spawnQueryShape = new RAPIER.Ball(PARTICLE_RADIUS)
const spawnQueryPoint = { x: 0, y: 0 }
/** Bodies created this frame are absent from the broad phase until it steps. */
const batchSpawnX = []
const batchSpawnY = []

function isSpawnPositionFree(positionX, positionY, excludedBody) {
  const contactSquared = (PARTICLE_RADIUS * 2) ** 2

  for (let index = 0; index < batchSpawnX.length; index += 1) {
    const offsetX = batchSpawnX[index] - positionX
    const offsetY = batchSpawnY[index] - positionY

    if (offsetX ** 2 + offsetY ** 2 < contactSquared) {
      return false
    }
  }

  spawnQueryPoint.x = positionX
  spawnQueryPoint.y = positionY

  return (
    world.intersectionWithShape(
      spawnQueryPoint,
      0,
      spawnQueryShape,
      RAPIER.QueryFilterFlags.EXCLUDE_FIXED,
      undefined,
      undefined,
      excludedBody,
    ) === null
  )
}

function spawnParticle(x, y, heading, pointerSpeed, hue) {
  if (particleLimit === 0) {
    return
  }

  const isRecycling = bodies.length >= particleLimit
  // Resolved before the search so it can ignore the body about to move away.
  const recycledBody = isRecycling ? bodies[recycleCursor] : undefined
  let positionX = 0
  let positionY = 0
  let hasPlacement = false

  for (
    let attempt = 0;
    attempt < SPAWN_PLACEMENT_ATTEMPTS && !hasPlacement;
    attempt += 1
  ) {
    const jitter = SPAWN_JITTER + attempt * PARTICLE_RADIUS

    positionX = clamp(
      x + randomBetween(-jitter, jitter),
      PARTICLE_RADIUS,
      Math.max(PARTICLE_RADIUS, viewportWidth - PARTICLE_RADIUS),
    )
    positionY = clamp(
      y + randomBetween(-jitter, jitter),
      PARTICLE_RADIUS,
      Math.max(PARTICLE_RADIUS, viewportHeight - PARTICLE_RADIUS),
    )
    hasPlacement = isSpawnPositionFree(positionX, positionY, recycledBody)
  }

  // Nowhere to put it without an overlap, so the field is simply full here.
  if (!hasPlacement) {
    return
  }

  const spread = randomBetween(-0.34, 0.34)
  const speed =
    (pointerSpeed + randomBetween(2.2, 4.8)) *
    randomBetween(0.72, 1.12) *
    EMISSION_VELOCITY_SCALE *
    STEPS_PER_SECOND
  const velocity = {
    x: Math.cos(heading + spread) * speed,
    y: Math.sin(heading + spread) * speed,
  }
  let index

  if (!isRecycling) {
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
        .setMass(PARTICLE_MASS)
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
  vx[index] = velocity.x / STEPS_PER_SECOND
  vy[index] = velocity.y / STEPS_PER_SECOND
  colorIndex[index] = Math.floor(hue / HUE_STEP) % HUE_COUNT
  respawned[index] = 1
  batchSpawnX.push(positionX)
  batchSpawnY.push(positionY)
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

/**
 * Barnes-Hut quadtree. A centre-of-mass monopole cannot produce local
 * attraction at all, so clusters could never form and existing ones were torn
 * apart. This approximates true pairwise gravity in O(n log n).
 */
let treeCapacity = 0
let treeNodeCount = 0
let treeChildren = new Int32Array(0)
let treeParent = new Int32Array(0)
let treeBody = new Int32Array(0)
let treeMass = new Float64Array(0)
let treeWeightedX = new Float64Array(0)
let treeWeightedY = new Float64Array(0)
let treeCenterX = new Float64Array(0)
let treeCenterY = new Float64Array(0)
let treeMinX = new Float64Array(0)
let treeMinY = new Float64Array(0)
let treeSize = new Float64Array(0)
/** Depth-first over a 4-ary tree never needs more than 3 * depth + 1 slots. */
const treeStack = new Int32Array(512)

let gravityAccelerationX = 0
let gravityAccelerationY = 0

function ensureTreeCapacity(required) {
  if (treeCapacity >= required) {
    return
  }

  const grown = Math.max(required, treeCapacity * 2, 256)

  const copyInt = (source, stride) => {
    const next = new Int32Array(grown * stride)

    next.set(source)

    return next
  }
  const copyFloat = (source) => {
    const next = new Float64Array(grown)

    next.set(source)

    return next
  }

  treeChildren = copyInt(treeChildren, 4)
  treeParent = copyInt(treeParent, 1)
  treeBody = copyInt(treeBody, 1)
  treeMass = copyFloat(treeMass)
  treeWeightedX = copyFloat(treeWeightedX)
  treeWeightedY = copyFloat(treeWeightedY)
  treeCenterX = copyFloat(treeCenterX)
  treeCenterY = copyFloat(treeCenterY)
  treeMinX = copyFloat(treeMinX)
  treeMinY = copyFloat(treeMinY)
  treeSize = copyFloat(treeSize)
  treeCapacity = grown
}

function createTreeNode(minX, minY, size, parent) {
  const node = treeNodeCount

  treeNodeCount += 1
  treeChildren[node * 4] = -1
  treeChildren[node * 4 + 1] = -1
  treeChildren[node * 4 + 2] = -1
  treeChildren[node * 4 + 3] = -1
  treeParent[node] = parent
  treeBody[node] = -1
  treeMass[node] = 0
  treeWeightedX[node] = 0
  treeWeightedY[node] = 0
  treeMinX[node] = minX
  treeMinY[node] = minY
  treeSize[node] = size

  return node
}

function addBodyToNode(node, index) {
  treeMass[node] += mass[index]
  treeWeightedX[node] += px[index] * mass[index]
  treeWeightedY[node] += py[index] * mass[index]
}

function getChildForBody(node, index) {
  const half = treeSize[node] / 2
  const quadrant =
    (px[index] >= treeMinX[node] + half ? 1 : 0) +
    (py[index] >= treeMinY[node] + half ? 2 : 0)

  return treeChildren[node * 4 + quadrant]
}

function insertBody(node, index, depth) {
  if (treeChildren[node * 4] === -1) {
    if (treeBody[node] === -1) {
      treeBody[node] = index
      bodyNext[index] = -1
      addBodyToNode(node, index)
      return
    }

    // Coincident or near-coincident bodies would subdivide forever, so past
    // the depth cap a leaf just holds a chain of them.
    if (depth >= MAX_TREE_DEPTH) {
      bodyNext[index] = treeBody[node]
      treeBody[node] = index
      addBodyToNode(node, index)
      return
    }

    const existing = treeBody[node]
    const half = treeSize[node] / 2
    const minX = treeMinX[node]
    const minY = treeMinY[node]

    ensureTreeCapacity(treeNodeCount + 4)
    treeBody[node] = -1
    treeMass[node] = 0
    treeWeightedX[node] = 0
    treeWeightedY[node] = 0
    treeChildren[node * 4] = createTreeNode(minX, minY, half, node)
    treeChildren[node * 4 + 1] = createTreeNode(minX + half, minY, half, node)
    treeChildren[node * 4 + 2] = createTreeNode(minX, minY + half, half, node)
    treeChildren[node * 4 + 3] = createTreeNode(
      minX + half,
      minY + half,
      half,
      node,
    )

    let moved = existing

    while (moved !== -1) {
      const next = bodyNext[moved]

      insertBody(getChildForBody(node, moved), moved, depth + 1)
      moved = next
    }
  }

  insertBody(getChildForBody(node, index), index, depth + 1)
}

function buildGravityTree(count) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (let index = 0; index < count; index += 1) {
    if (px[index] < minX) minX = px[index]
    if (py[index] < minY) minY = py[index]
    if (px[index] > maxX) maxX = px[index]
    if (py[index] > maxY) maxY = py[index]
  }

  // A hair of slack keeps the far edge inside the root cell.
  const size = Math.max(maxX - minX, maxY - minY, 1) * 1.0001

  ensureTreeCapacity(count * 4 + 16)
  treeNodeCount = 0
  createTreeNode(minX, minY, size, -1)

  for (let index = 0; index < count; index += 1) {
    insertBody(0, index, 0)
  }

  // Children always have a higher index than their parent, so one reverse pass
  // both finishes each node and folds it into its parent.
  for (let node = treeNodeCount - 1; node >= 0; node -= 1) {
    const nodeMass = treeMass[node]

    if (nodeMass > 0) {
      treeCenterX[node] = treeWeightedX[node] / nodeMass
      treeCenterY[node] = treeWeightedY[node] / nodeMass
    }

    const parent = treeParent[node]

    if (parent >= 0) {
      treeMass[parent] += nodeMass
      treeWeightedX[parent] += treeWeightedX[node]
      treeWeightedY[parent] += treeWeightedY[node]
    }
  }
}

function accumulateTreeGravity(index) {
  const particleX = px[index]
  const particleY = py[index]
  let accelerationX = 0
  let accelerationY = 0
  let top = 0

  treeStack[top] = 0
  top += 1

  while (top > 0) {
    top -= 1

    const node = treeStack[top]
    const nodeMass = treeMass[node]

    if (nodeMass <= 0) {
      continue
    }

    if (treeChildren[node * 4] === -1) {
      for (let other = treeBody[node]; other !== -1; other = bodyNext[other]) {
        if (other === index) {
          continue
        }

        const offsetX = px[other] - particleX
        const offsetY = py[other] - particleY
        const softened =
          offsetX ** 2 + offsetY ** 2 + PARTICLE_GRAVITY_SOFTENING ** 2
        const scale =
          (PARTICLE_GRAVITY * mass[other] * ACCELERATION_TO_SECONDS) /
          (softened * Math.sqrt(softened))

        accelerationX += offsetX * scale
        accelerationY += offsetY * scale
      }

      continue
    }

    const offsetX = treeCenterX[node] - particleX
    const offsetY = treeCenterY[node] - particleY
    const distanceSquared = offsetX ** 2 + offsetY ** 2

    if (
      treeSize[node] ** 2 <
      BARNES_HUT_THETA_SQUARED * distanceSquared
    ) {
      const softened = distanceSquared + PARTICLE_GRAVITY_SOFTENING ** 2
      const scale =
        (PARTICLE_GRAVITY * nodeMass * ACCELERATION_TO_SECONDS) /
        (softened * Math.sqrt(softened))

      accelerationX += offsetX * scale
      accelerationY += offsetY * scale
      continue
    }

    for (let quadrant = 0; quadrant < 4; quadrant += 1) {
      treeStack[top] = treeChildren[node * 4 + quadrant]
      top += 1
    }
  }

  gravityAccelerationX = accelerationX
  gravityAccelerationY = accelerationY
}

function applyFieldImpulses() {
  const count = bodies.length
  const hasHole =
    pointer.holePolarity !== 0 && pointer.x !== null && pointer.y !== null
  const hasGravity = count > 1

  if (hasGravity) {
    buildGravityTree(count)
  }

  for (let index = 0; index < count; index += 1) {
    const bodyMass = mass[index]
    let accelerationX = 0
    let accelerationY = 0

    if (hasGravity) {
      accumulateTreeGravity(index)
      accelerationX += gravityAccelerationX
      accelerationY += gravityAccelerationY
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
    const body = bodies[index]
    const translation = body.translation()
    // Displacement matches this for ~99% of particles, but soft-contact
    // position correction throws it off by up to 48x for bodies in contact.
    const velocity = body.linvel()

    px[index] = translation.x
    py[index] = translation.y
    vx[index] = velocity.x / STEPS_PER_SECOND
    vy[index] = velocity.y / STEPS_PER_SECOND
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
    views.speeds[index] = Math.hypot(vx[index], vy[index])
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

  // The impulse above already covers a whole frame, so it is applied once and
  // only the contact solve is subdivided.
  for (let substep = 0; substep < PHYSICS_SUBSTEPS; substep += 1) {
    world.step()
  }

  batchSpawnX.length = 0
  batchSpawnY.length = 0
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
