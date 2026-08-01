// Values the renderer shares with the engine. Their counterparts live in
// physics/src/lib.rs and the two sets must be changed together.
export const FRAME_DURATION = 1000 / 60
export const PARTICLE_RADIUS = 1.8
export const FULL_BRIGHTNESS_SPEED = 6
export const HUE_COUNT = 18
/** Degrees per colour index, so the palette spans the wheel exactly once. */
export const HUE_STEP = 360 / HUE_COUNT

// Control surface, owned entirely by the main thread.
export const MAX_CONTROL_PERCENT = 100
export const DEFAULT_PARTICLE_LIMIT = 500
export const PARTICLE_LIMIT_STEP = 250
export const BLACK_HOLE_POLARITY = 1
export const WHITE_HOLE_POLARITY = -1

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
const SNAPSHOT_BYTES_PER_PARTICLE = 14
/** Set when the main thread must not interpolate from its previous snapshot. */
export const SNAPSHOT_FLAG_DISCONTINUOUS = 1
export const PARTICLE_FLAG_RESPAWNED = 1
/** Fade level occupies the bits above the respawn flag. */
export const PARTICLE_FADE_SHIFT = 1
export const PARTICLE_FADE_LEVELS = 7
/** Retiring particles keep simulating, so the field can exceed its limit. */
export const MAX_RETIRING_PARTICLES = 250

export const clamp = (value, minimum, maximum) =>
  Math.min(Math.max(value, minimum), maximum)

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
