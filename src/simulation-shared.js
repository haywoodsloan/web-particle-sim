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
/**
 * Roughly the contact diameter. Larger values flatten gravity to a harmonic
 * well at close range, which prevents orbits and bound pairs from forming.
 */
export const PARTICLE_GRAVITY_SOFTENING = 4
export const POINTER_HOLE_RADIUS = 375
export const POINTER_HOLE_MASS = 288
export const POINTER_HOLE_SOFTENING = 90
export const BLACK_HOLE_POLARITY = 1
export const WHITE_HOLE_POLARITY = -1
export const HUE_COUNT = 18
/** Degrees per colour index, so the palette spans the wheel exactly once. */
export const HUE_STEP = 360 / HUE_COUNT
/**
 * Contact diameter must exceed the distance a particle covers in one step, or
 * discrete detection skips straight past the encounter.
 */
export const PARTICLE_RADIUS = 1.8

export const MESSAGE_INIT = 'init'
export const MESSAGE_RESIZE = 'resize'
export const MESSAGE_POINTER = 'pointer'
export const MESSAGE_POINTER_RESET = 'pointerReset'
export const MESSAGE_EMISSION = 'emission'
export const MESSAGE_GRAVITY = 'gravity'
export const MESSAGE_AIR_RESISTANCE = 'airResistance'
export const MESSAGE_PARTICLE_LIMIT = 'particleLimit'
export const MESSAGE_VISIBILITY = 'visibility'
export const MESSAGE_RECYCLE_BUFFER = 'recycleBuffer'
export const MESSAGE_READY = 'ready'
export const MESSAGE_SNAPSHOT = 'snapshot'

export const SNAPSHOT_HEADER_BYTES = 16
export const SNAPSHOT_BYTES_PER_PARTICLE = 14
/** Set when the main thread must not interpolate from its previous snapshot. */
export const SNAPSHOT_FLAG_DISCONTINUOUS = 1
export const PARTICLE_FLAG_RESPAWNED = 1

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

export const getSnapshotByteLength = (capacity) =>
  SNAPSHOT_HEADER_BYTES + capacity * SNAPSHOT_BYTES_PER_PARTICLE

/**
 * Positions, speeds, colours and flags share one transferable buffer so a
 * snapshot can never be torn across messages.
 */
export function createSnapshotViews(buffer, count) {
  return {
    header: new Uint32Array(buffer, 0, 4),
    positions: new Float32Array(buffer, SNAPSHOT_HEADER_BYTES, count * 2),
    speeds: new Float32Array(buffer, SNAPSHOT_HEADER_BYTES + count * 8, count),
    colors: new Uint8Array(buffer, SNAPSHOT_HEADER_BYTES + count * 12, count),
    flags: new Uint8Array(buffer, SNAPSHOT_HEADER_BYTES + count * 13, count),
  }
}
