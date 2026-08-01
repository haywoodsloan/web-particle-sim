import {
  FRAME_DURATION,
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
} from './simulation-shared.js'
import initialisePhysics from './physics.wasm?init'

const SNAPSHOT_POOL_SIZE = 4

const snapshotBufferPool = []
const instance = await initialisePhysics()
const physics = instance.exports

/** Views detach whenever the module grows its memory, so they are revalidated. */
let memoryBuffer = null
let memoryBytes = null
let isEmissionEnabled = true
let isRunning = false
let isHidden = false
let stepTimerId = null
let nextStepTime = 0

function getMemoryBytes() {
  if (memoryBuffer !== physics.memory.buffer) {
    memoryBuffer = physics.memory.buffer
    memoryBytes = new Uint8Array(memoryBuffer)
  }

  return memoryBytes
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
  const byteLength = physics.snapshot_len()
  const buffer = acquireSnapshotBuffer(byteLength)

  // No free buffer means the main thread is behind. Keep simulating and skip
  // this snapshot so the message queue can never grow unbounded.
  if (!buffer) {
    return
  }

  const source = physics.snapshot_ptr()

  physics.write_snapshot()
  new Uint8Array(buffer, 0, byteLength).set(
    getMemoryBytes().subarray(source, source + byteLength),
  )
  self.postMessage(
    { type: MESSAGE_SNAPSHOT, buffer, count: physics.particle_count() },
    [buffer],
  )
}

function scheduleNextStep() {
  if (!isRunning) {
    return
  }

  stepTimerId = setTimeout(
    runStep,
    Math.max(0, nextStepTime - performance.now()),
  )
}

function runStep() {
  physics.step()
  sendSnapshot()
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

self.onmessage = ({ data }) => {
  switch (data.type) {
    case MESSAGE_INIT:
      isEmissionEnabled = data.emissionEnabled
      // A page that loads hidden never fires visibilitychange.
      isHidden = data.hidden
      physics.init(
        data.width,
        data.height,
        (Math.random() * 0xffffffff) >>> 0,
        data.gravityPercent,
        data.airResistancePercent,
        data.particleLimit,
      )
      snapshotBufferPool.push(...data.buffers)
      startLoop()
      break

    case MESSAGE_RESIZE:
      physics.resize(data.width, data.height)
      break

    case MESSAGE_POINTER:
      physics.pointer_move(
        data.x,
        data.y,
        data.timeStamp,
        data.polarity,
        isEmissionEnabled ? 1 : 0,
      )
      break

    case MESSAGE_POINTER_RESET:
      physics.pointer_reset()
      break

    case MESSAGE_EMISSION:
      isEmissionEnabled = data.enabled
      physics.reset_emission()
      break

    case MESSAGE_GRAVITY:
      physics.set_gravity(data.percent)
      break

    case MESSAGE_AIR_RESISTANCE:
      physics.set_air_resistance(data.percent)
      break

    case MESSAGE_PARTICLE_LIMIT:
      physics.set_particle_limit(data.limit)
      break

    case MESSAGE_VISIBILITY:
      isHidden = data.hidden

      if (isHidden) {
        stopLoop()
        physics.reset_emission()
      } else {
        physics.mark_discontinuous()
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

// Instantiating the module makes this a top-level-await module, so the port is
// already live before `onmessage` exists. Anything sent earlier would be lost.
self.postMessage({ type: MESSAGE_READY })
