//! Particle solver for the rainbow field.
//!
//! Equal-mass frictionless discs only ever swap their normal velocity on
//! contact, so collisions conserve momentum and kinetic energy exactly rather
//! than approximately. Everything is stored as flat arrays inside the module's
//! linear memory and the host reads a packed snapshot straight out of it.

use core::arch::wasm32::*;
use core::cell::UnsafeCell;

// Values mirrored from simulation-shared.js.
const FRAME_DURATION: f32 = 1000.0 / 60.0;
const MAX_CONTROL_PERCENT: f32 = 100.0;
const MAX_WORLD_GRAVITY: f32 = 1.0;
const MAX_AIR_RESISTANCE: f32 = 0.03;
const GRAVITY_SCALE: f32 = 0.001;
const MAX_PARTICLES_PER_FRAME: f32 = 10.0;
const EMISSION_VELOCITY_SCALE: f32 = 0.4;
const PARTICLE_GRAVITY: f32 = 0.15;
const PARTICLE_GRAVITY_SOFTENING: f32 = 4.0;
const POINTER_HOLE_RADIUS: f32 = 375.0;
const POINTER_HOLE_MASS: f32 = 288.0;
const POINTER_HOLE_SOFTENING: f32 = 90.0;
const PARTICLE_RADIUS: f32 = 1.8;
const SNAPSHOT_HEADER_BYTES: usize = 16;
const SNAPSHOT_BYTES_PER_PARTICLE: usize = 14;
const SNAPSHOT_FLAG_DISCONTINUOUS: u32 = 1;
const PARTICLE_FLAG_RESPAWNED: u8 = 1;
/// Fade level occupies the bits above the respawn flag.
const PARTICLE_FADE_SHIFT: u8 = 1;

/// Tuning is expressed in px/ms^2 but the solver integrates in frames.
const ACCELERATION_TO_FRAMES: f32 = FRAME_DURATION * FRAME_DURATION;
const MIN_PHYSICS_SUBSTEPS: usize = 8;
const MAX_PHYSICS_SUBSTEPS: usize = 32;
/// Headroom for the speed a particle gains during the frame it is measured on.
const SUBSTEP_SPEED_MARGIN: f32 = 1.25;
/// Passes over the contact set per substep. Beyond the first the velocity swap
/// has already separated every pair, so the rest only relax residual overlap.
const CONTACT_ITERATIONS: usize = 2;
const CONTACT_DIAMETER: f32 = PARTICLE_RADIUS * 2.0;
const CONTACT_DIAMETER_SQUARED: f32 = CONTACT_DIAMETER * CONTACT_DIAMETER;
const INVERSE_CELL_SIZE: f32 = 1.0 / CONTACT_DIAMETER;
/// Overlap is removed positionally so that separating never adds energy.
const POSITION_CORRECTION: f32 = 0.8;
const PARTICLE_DENSITY: f32 = 9.8e-5;
const PARTICLE_MASS: f32 =
    PARTICLE_DENSITY * core::f32::consts::PI * PARTICLE_RADIUS * PARTICLE_RADIUS;
const BARNES_HUT_THETA_SQUARED: f32 = 0.5 * 0.5;
const MAX_TREE_DEPTH: u32 = 20;
const SPAWN_JITTER: f32 = 2.5;
const SPAWN_PLACEMENT_ATTEMPTS: usize = 8;
/// A replaced particle keeps colliding while it dims out over these frames.
/// The step rate is fixed, so this is 225 ms on any display.
const RETIRE_FADE_FRAMES: f32 = 13.5;
const RETIRE_FADE_LEVELS: f32 = 7.0;
/// Past this backlog the longest-running fades are cut short.
const MAX_RETIRING: usize = 250;

/// Constant-folded; `sqrt` cannot run in a `const` initialiser.
#[inline]
fn pointer_hole_rim_scale() -> f32 {
    let edge =
        POINTER_HOLE_RADIUS * POINTER_HOLE_RADIUS + POINTER_HOLE_SOFTENING * POINTER_HOLE_SOFTENING;

    POINTER_HOLE_MASS / (edge * edge.sqrt())
}

#[inline]
fn clamp(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value < minimum {
        minimum
    } else if value > maximum {
        maximum
    } else {
        value
    }
}

#[inline]
fn hash_cell(cell_x: i32, cell_y: i32) -> usize {
    let x = (cell_x as u32).wrapping_mul(0x9E37_79B1);
    let y = (cell_y as u32).wrapping_mul(0x85EB_CA77);
    (x ^ y) as usize
}

#[derive(Default)]
struct World {
    x: Vec<f32>,
    y: Vec<f32>,
    velocity_x: Vec<f32>,
    velocity_y: Vec<f32>,
    acceleration_x: Vec<f32>,
    acceleration_y: Vec<f32>,
    color: Vec<u8>,
    respawned: Vec<u8>,
    /// Frames of fade left. Infinite for a particle that is still alive.
    fade: Vec<f32>,
    birth: Vec<u32>,
    count: usize,
    alive_count: usize,
    next_birth: u32,
    limit: usize,

    width: f32,
    height: f32,
    gravity_percent: f32,
    air_percent: f32,

    pointer_x: f32,
    pointer_y: f32,
    pointer_time: f64,
    has_pointer: bool,
    hole_polarity: f32,
    spawn_budget: f32,
    hue: f32,

    step_index: u32,
    discontinuous: bool,
    rng: u32,

    grid_mask: usize,
    grid_start: Vec<u32>,
    grid_cursor: Vec<u32>,
    grid_items: Vec<u32>,

    body_next: Vec<i32>,
    tree_children: Vec<i32>,
    /// Stride 4: centre x, centre y, mass, size. Interleaved because the
    /// traversal touches all four together for every node it visits.
    tree_packed: Vec<f32>,
    tree_body: Vec<i32>,
    tree_mass: Vec<f64>,
    tree_weighted_x: Vec<f64>,
    tree_weighted_y: Vec<f64>,
    tree_min_x: Vec<f64>,
    tree_min_y: Vec<f64>,
    tree_size: Vec<f64>,
    tree_parent: Vec<i32>,
    tree_node_count: usize,
    tree_stack: Vec<i32>,

    /// Which nodes and bodies each particle pulls on, recorded on the first
    /// field evaluation of a frame and replayed by the rest of them. Node
    /// sources are stored inline as parallel arrays, padded to four lanes, so
    /// the replay is a flat SIMD stream instead of an unsupported gather.
    node_source_x: Vec<f32>,
    node_source_y: Vec<f32>,
    node_source_mass: Vec<f32>,
    node_start: Vec<u32>,
    body_items: Vec<u32>,
    body_start: Vec<u32>,
    has_interactions: bool,

    batch_x: Vec<f32>,
    batch_y: Vec<f32>,
    snapshot: Vec<u8>,
}

impl World {
    fn new() -> Self {
        Self {
            discontinuous: true,
            rng: 0x2545_F491,
            // Depth-first over a 4-ary tree never needs more than 3 * depth + 1 slots.
            tree_stack: vec![0; 512],
            snapshot: vec![0; SNAPSHOT_HEADER_BYTES],
            ..Self::default()
        }
    }

    fn next_random(&mut self) -> f32 {
        let mut value = self.rng;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        self.rng = value;
        (value >> 8) as f32 / 16_777_216.0
    }

    fn random_between(&mut self, minimum: f32, maximum: f32) -> f32 {
        minimum + self.next_random() * (maximum - minimum)
    }

    fn ensure_capacity(&mut self, required: usize) {
        if self.x.len() >= required {
            return;
        }

        self.x.resize(required, 0.0);
        self.y.resize(required, 0.0);
        self.velocity_x.resize(required, 0.0);
        self.velocity_y.resize(required, 0.0);
        self.acceleration_x.resize(required, 0.0);
        self.acceleration_y.resize(required, 0.0);
        self.color.resize(required, 0);
        self.respawned.resize(required, 0);
        self.fade.resize(required, f32::INFINITY);
        self.birth.resize(required, 0);
        self.body_next.resize(required, -1);
        self.grid_items.resize(required, 0);
        self.snapshot.resize(
            SNAPSHOT_HEADER_BYTES + required * SNAPSHOT_BYTES_PER_PARTICLE,
            0,
        );

        let mut table = 64;
        while table < required * 2 {
            table *= 2;
        }

        self.grid_mask = table - 1;
        self.grid_start.resize(table + 1, 0);
        self.grid_cursor.resize(table, 0);
    }

    /// Retiring particles stay in the arrays, so a slot can outlive its own
    /// removal; the swapped-in occupant is flagged so the host stops
    /// interpolating it from the previous snapshot.
    fn remove_particle(&mut self, index: usize) {
        if self.fade[index].is_infinite() {
            self.alive_count -= 1;
        }

        self.count -= 1;

        let last = self.count;

        if index == last {
            return;
        }

        self.x[index] = self.x[last];
        self.y[index] = self.y[last];
        self.velocity_x[index] = self.velocity_x[last];
        self.velocity_y[index] = self.velocity_y[last];
        self.acceleration_x[index] = self.acceleration_x[last];
        self.acceleration_y[index] = self.acceleration_y[last];
        self.color[index] = self.color[last];
        self.fade[index] = self.fade[last];
        self.birth[index] = self.birth[last];
        self.respawned[index] = 1;
    }

    fn retire_oldest_alive(&mut self) {
        let mut oldest = usize::MAX;
        let mut oldest_birth = u32::MAX;

        for index in 0..self.count {
            if self.fade[index].is_finite() || self.birth[index] >= oldest_birth {
                continue;
            }

            oldest_birth = self.birth[index];
            oldest = index;
        }

        if oldest != usize::MAX {
            self.fade[oldest] = RETIRE_FADE_FRAMES;
            self.alive_count -= 1;
        }
    }

    /// All fades start equal, so the least remaining is the longest running.
    fn drop_longest_fade(&mut self) {
        let mut oldest = usize::MAX;
        let mut least = f32::INFINITY;

        for index in 0..self.count {
            if self.fade[index] < least {
                least = self.fade[index];
                oldest = index;
            }
        }

        if oldest != usize::MAX {
            self.remove_particle(oldest);
        }
    }

    fn set_limit(&mut self, limit: usize) {
        self.limit = limit;
        self.ensure_capacity(limit + MAX_RETIRING);

        while self.alive_count > limit {
            self.retire_oldest_alive();
        }

        while self.count - self.alive_count > MAX_RETIRING {
            self.drop_longest_fade();
        }
    }

    // ---- broad phase -------------------------------------------------------

    fn build_grid(&mut self) {
        let table = self.grid_mask + 1;

        for slot in self.grid_start[..=table].iter_mut() {
            *slot = 0;
        }

        for index in 0..self.count {
            let bucket = self.bucket_of(self.x[index], self.y[index]);
            self.grid_start[bucket + 1] += 1;
        }

        for bucket in 0..table {
            self.grid_start[bucket + 1] += self.grid_start[bucket];
        }

        self.grid_cursor[..table].copy_from_slice(&self.grid_start[..table]);

        for index in 0..self.count {
            let bucket = self.bucket_of(self.x[index], self.y[index]);
            let slot = self.grid_cursor[bucket] as usize;

            self.grid_items[slot] = index as u32;
            self.grid_cursor[bucket] += 1;
        }
    }

    #[inline]
    fn bucket_of(&self, x: f32, y: f32) -> usize {
        let cell_x = (x * INVERSE_CELL_SIZE).floor() as i32;
        let cell_y = (y * INVERSE_CELL_SIZE).floor() as i32;

        hash_cell(cell_x, cell_y) & self.grid_mask
    }

    // ---- contacts ----------------------------------------------------------

    #[inline]
    fn resolve_pair(&mut self, first: usize, second: usize) {
        let offset_x = self.x[second] - self.x[first];
        let offset_y = self.y[second] - self.y[first];
        let distance_squared = offset_x * offset_x + offset_y * offset_y;

        if distance_squared >= CONTACT_DIAMETER_SQUARED || distance_squared <= 1e-12 {
            return;
        }

        let distance = distance_squared.sqrt();
        let normal_x = offset_x / distance;
        let normal_y = offset_y / distance;
        let approach = (self.velocity_x[second] - self.velocity_x[first]) * normal_x
            + (self.velocity_y[second] - self.velocity_y[first]) * normal_y;

        // Equal masses trade their normal components, which is exactly elastic.
        if approach < 0.0 {
            self.velocity_x[first] += approach * normal_x;
            self.velocity_y[first] += approach * normal_y;
            self.velocity_x[second] -= approach * normal_x;
            self.velocity_y[second] -= approach * normal_y;
        }

        let push = (CONTACT_DIAMETER - distance) * 0.5 * POSITION_CORRECTION;

        self.x[first] -= normal_x * push;
        self.y[first] -= normal_y * push;
        self.x[second] += normal_x * push;
        self.y[second] += normal_y * push;
    }

    fn solve_contacts(&mut self) {
        // One grid serves every pass: a correction moves a particle far less
        // than the cell size, so its neighbourhood cannot change underneath it.
        self.build_grid();

        for _ in 0..CONTACT_ITERATIONS {
            self.relax_contacts();
        }
    }

    fn relax_contacts(&mut self) {
        for index in 0..self.count {
            let cell_x = (self.x[index] * INVERSE_CELL_SIZE).floor() as i32;
            let cell_y = (self.y[index] * INVERSE_CELL_SIZE).floor() as i32;

            for offset_y in -1..=1 {
                for offset_x in -1..=1 {
                    let bucket = hash_cell(cell_x + offset_x, cell_y + offset_y) & self.grid_mask;
                    let start = self.grid_start[bucket] as usize;
                    let end = self.grid_start[bucket + 1] as usize;

                    for slot in start..end {
                        let other = self.grid_items[slot] as usize;

                        // Hashed cells can repeat, so the ordering also dedupes.
                        if other <= index {
                            continue;
                        }

                        self.resolve_pair(index, other);
                    }
                }
            }
        }
    }

    // ---- gravity -----------------------------------------------------------

    fn ensure_tree_capacity(&mut self, required: usize) {
        if self.tree_parent.len() >= required {
            return;
        }

        let grown = required.max(self.tree_parent.len() * 2).max(256);

        self.tree_children.resize(grown * 4, -1);
        self.tree_packed.resize(grown * 4, 0.0);
        self.tree_parent.resize(grown, -1);
        self.tree_body.resize(grown, -1);
        self.tree_mass.resize(grown, 0.0);
        self.tree_weighted_x.resize(grown, 0.0);
        self.tree_weighted_y.resize(grown, 0.0);
        self.tree_min_x.resize(grown, 0.0);
        self.tree_min_y.resize(grown, 0.0);
        self.tree_size.resize(grown, 0.0);
    }

    fn create_tree_node(&mut self, min_x: f64, min_y: f64, size: f64, parent: i32) -> usize {
        let node = self.tree_node_count;

        self.tree_node_count += 1;
        self.tree_children[node * 4] = -1;
        self.tree_children[node * 4 + 1] = -1;
        self.tree_children[node * 4 + 2] = -1;
        self.tree_children[node * 4 + 3] = -1;
        self.tree_parent[node] = parent;
        self.tree_body[node] = -1;
        self.tree_mass[node] = 0.0;
        self.tree_weighted_x[node] = 0.0;
        self.tree_weighted_y[node] = 0.0;
        self.tree_min_x[node] = min_x;
        self.tree_min_y[node] = min_y;
        self.tree_size[node] = size;

        node
    }

    #[inline]
    fn add_body_to_node(&mut self, node: usize, index: usize) {
        self.tree_mass[node] += PARTICLE_MASS as f64;
        self.tree_weighted_x[node] += self.x[index] as f64 * PARTICLE_MASS as f64;
        self.tree_weighted_y[node] += self.y[index] as f64 * PARTICLE_MASS as f64;
    }

    #[inline]
    fn child_for_body(&self, node: usize, index: usize) -> usize {
        let half = self.tree_size[node] / 2.0;
        let quadrant = usize::from(self.x[index] as f64 >= self.tree_min_x[node] + half)
            + usize::from(self.y[index] as f64 >= self.tree_min_y[node] + half) * 2;

        self.tree_children[node * 4 + quadrant] as usize
    }

    fn insert_body(&mut self, node: usize, index: usize, depth: u32) {
        if self.tree_children[node * 4] == -1 {
            if self.tree_body[node] == -1 {
                self.tree_body[node] = index as i32;
                self.body_next[index] = -1;
                self.add_body_to_node(node, index);
                return;
            }

            // Coincident bodies would subdivide forever, so past the depth cap
            // a leaf just holds a chain of them.
            if depth >= MAX_TREE_DEPTH {
                self.body_next[index] = self.tree_body[node];
                self.tree_body[node] = index as i32;
                self.add_body_to_node(node, index);
                return;
            }

            let existing = self.tree_body[node];
            let half = self.tree_size[node] / 2.0;
            let min_x = self.tree_min_x[node];
            let min_y = self.tree_min_y[node];

            self.ensure_tree_capacity(self.tree_node_count + 4);
            self.tree_body[node] = -1;
            self.tree_mass[node] = 0.0;
            self.tree_weighted_x[node] = 0.0;
            self.tree_weighted_y[node] = 0.0;

            let quadrants = [
                self.create_tree_node(min_x, min_y, half, node as i32),
                self.create_tree_node(min_x + half, min_y, half, node as i32),
                self.create_tree_node(min_x, min_y + half, half, node as i32),
                self.create_tree_node(min_x + half, min_y + half, half, node as i32),
            ];

            for (quadrant, child) in quadrants.iter().enumerate() {
                self.tree_children[node * 4 + quadrant] = *child as i32;
            }

            let mut moved = existing;

            while moved != -1 {
                let next = self.body_next[moved as usize];
                let child = self.child_for_body(node, moved as usize);

                self.insert_body(child, moved as usize, depth + 1);
                moved = next;
            }
        }

        let child = self.child_for_body(node, index);

        self.insert_body(child, index, depth + 1);
    }

    fn build_gravity_tree(&mut self) {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for index in 0..self.count {
            min_x = min_x.min(self.x[index]);
            min_y = min_y.min(self.y[index]);
            max_x = max_x.max(self.x[index]);
            max_y = max_y.max(self.y[index]);
        }

        // A hair of slack keeps the far edge inside the root cell.
        let size = (max_x - min_x).max(max_y - min_y).max(1.0) as f64 * 1.0001;

        self.ensure_tree_capacity(self.count * 4 + 16);
        self.tree_node_count = 0;
        self.has_interactions = false;
        self.create_tree_node(min_x as f64, min_y as f64, size, -1);

        for index in 0..self.count {
            self.insert_body(0, index, 0);
        }

        // Children always outrank their parent, so one reverse pass finishes
        // each node, publishes it for the traversal, and folds it into the
        // parent.
        for node in (0..self.tree_node_count).rev() {
            let mass = self.tree_mass[node];
            let base = node * 4;

            if mass > 0.0 {
                self.tree_packed[base] = (self.tree_weighted_x[node] / mass) as f32;
                self.tree_packed[base + 1] = (self.tree_weighted_y[node] / mass) as f32;
            }

            self.tree_packed[base + 2] = mass as f32;
            self.tree_packed[base + 3] = self.tree_size[node] as f32;

            let parent = self.tree_parent[node];

            if parent >= 0 {
                let parent = parent as usize;

                self.tree_mass[parent] += mass;
                self.tree_weighted_x[parent] += self.tree_weighted_x[node];
                self.tree_weighted_y[parent] += self.tree_weighted_y[node];
            }
        }
    }

    fn accumulate_tree_gravity(&mut self, index: usize) -> (f32, f32) {
        let particle_x = self.x[index];
        let particle_y = self.y[index];
        let mut acceleration_x = 0.0f32;
        let mut acceleration_y = 0.0f32;
        let mut top = 0usize;

        self.node_start.push(self.node_source_x.len() as u32);
        self.body_start.push(self.body_items.len() as u32);

        self.tree_stack[top] = 0;
        top += 1;

        while top > 0 {
            top -= 1;

            let node = self.tree_stack[top] as usize;
            let base = node * 4;
            let node_mass = self.tree_packed[base + 2];

            if node_mass <= 0.0 {
                continue;
            }

            if self.tree_children[base] == -1 {
                let mut other = self.tree_body[node];

                while other != -1 {
                    let slot = other as usize;

                    if slot != index {
                        self.body_items.push(slot as u32);

                        let offset_x = self.x[slot] - particle_x;
                        let offset_y = self.y[slot] - particle_y;
                        let softened = offset_x * offset_x
                            + offset_y * offset_y
                            + PARTICLE_GRAVITY_SOFTENING * PARTICLE_GRAVITY_SOFTENING;
                        let scale = PARTICLE_GRAVITY * PARTICLE_MASS / (softened * softened.sqrt());

                        acceleration_x += offset_x * scale;
                        acceleration_y += offset_y * scale;
                    }

                    other = self.body_next[slot];
                }

                continue;
            }

            let offset_x = self.tree_packed[base] - particle_x;
            let offset_y = self.tree_packed[base + 1] - particle_y;
            let distance_squared = offset_x * offset_x + offset_y * offset_y;
            let node_size = self.tree_packed[base + 3];

            if node_size * node_size < BARNES_HUT_THETA_SQUARED * distance_squared {
                self.node_source_x.push(self.tree_packed[base]);
                self.node_source_y.push(self.tree_packed[base + 1]);
                self.node_source_mass.push(PARTICLE_GRAVITY * node_mass);

                let softened =
                    distance_squared + PARTICLE_GRAVITY_SOFTENING * PARTICLE_GRAVITY_SOFTENING;
                let scale = PARTICLE_GRAVITY * node_mass / (softened * softened.sqrt());

                acceleration_x += offset_x * scale;
                acceleration_y += offset_y * scale;
                continue;
            }

            for quadrant in 0..4 {
                self.tree_stack[top] = self.tree_children[base + quadrant];
                top += 1;
            }
        }

        // Zero-mass padding contributes nothing, and squares off the run so the
        // replay never needs a scalar tail.
        while !self.node_source_x.len().is_multiple_of(4) {
            self.node_source_x.push(0.0);
            self.node_source_y.push(0.0);
            self.node_source_mass.push(0.0);
        }

        (acceleration_x, acceleration_y)
    }

    /// Sources shift far less within a frame than the particle being pulled, so
    /// the acceptance set is reused while the forces are recomputed in full.
    fn replay_tree_gravity(&self, index: usize) -> (f32, f32) {
        let particle_x = self.x[index];
        let particle_y = self.y[index];
        let softening = PARTICLE_GRAVITY_SOFTENING * PARTICLE_GRAVITY_SOFTENING;
        let splat_x = f32x4_splat(particle_x);
        let splat_y = f32x4_splat(particle_y);
        let splat_softening = f32x4_splat(softening);
        let mut lanes_x = f32x4_splat(0.0);
        let mut lanes_y = f32x4_splat(0.0);
        let mut slot = self.node_start[index] as usize;
        let end = self.node_start[index + 1] as usize;

        while slot < end {
            let (offset_x, offset_y, scale) = unsafe {
                let source_x = v128_load(self.node_source_x.as_ptr().add(slot) as *const v128);
                let source_y = v128_load(self.node_source_y.as_ptr().add(slot) as *const v128);
                let mass = v128_load(self.node_source_mass.as_ptr().add(slot) as *const v128);
                let offset_x = f32x4_sub(source_x, splat_x);
                let offset_y = f32x4_sub(source_y, splat_y);
                let softened = f32x4_add(
                    f32x4_add(f32x4_mul(offset_x, offset_x), f32x4_mul(offset_y, offset_y)),
                    splat_softening,
                );

                (
                    offset_x,
                    offset_y,
                    f32x4_div(mass, f32x4_mul(softened, f32x4_sqrt(softened))),
                )
            };

            lanes_x = f32x4_add(lanes_x, f32x4_mul(offset_x, scale));
            lanes_y = f32x4_add(lanes_y, f32x4_mul(offset_y, scale));
            slot += 4;
        }

        let mut acceleration_x = f32x4_extract_lane::<0>(lanes_x)
            + f32x4_extract_lane::<1>(lanes_x)
            + f32x4_extract_lane::<2>(lanes_x)
            + f32x4_extract_lane::<3>(lanes_x);
        let mut acceleration_y = f32x4_extract_lane::<0>(lanes_y)
            + f32x4_extract_lane::<1>(lanes_y)
            + f32x4_extract_lane::<2>(lanes_y)
            + f32x4_extract_lane::<3>(lanes_y);

        // Bodies stay scalar: there are only a handful and their positions are
        // read live so near-field forces do not go stale.
        for slot in self.body_start[index] as usize..self.body_start[index + 1] as usize {
            let other = self.body_items[slot] as usize;
            let offset_x = self.x[other] - particle_x;
            let offset_y = self.y[other] - particle_y;
            let softened = offset_x * offset_x + offset_y * offset_y + softening;
            let scale = PARTICLE_GRAVITY * PARTICLE_MASS / (softened * softened.sqrt());

            acceleration_x += offset_x * scale;
            acceleration_y += offset_y * scale;
        }

        (acceleration_x, acceleration_y)
    }

    #[inline]
    fn pointer_hole_scale(&self, delta_x: f32, delta_y: f32) -> f32 {
        let distance_squared = delta_x * delta_x + delta_y * delta_y;

        if !(0.0001..POINTER_HOLE_RADIUS * POINTER_HOLE_RADIUS).contains(&distance_squared) {
            return 0.0;
        }

        let softened = distance_squared + POINTER_HOLE_SOFTENING * POINTER_HOLE_SOFTENING;
        // Shifted-force truncation keeps the field continuous at the rim.
        let scale = POINTER_HOLE_MASS / (softened * softened.sqrt()) - pointer_hole_rim_scale();

        if scale <= 0.0 {
            0.0
        } else {
            scale * self.hole_polarity
        }
    }

    fn accumulate_field(&mut self) {
        let has_hole = self.hole_polarity != 0.0 && self.has_pointer;
        let has_gravity = self.count > 1;
        let world_gravity = MAX_WORLD_GRAVITY
            * (self.gravity_percent / MAX_CONTROL_PERCENT)
            * GRAVITY_SCALE
            * ACCELERATION_TO_FRAMES;
        let recording = has_gravity && !self.has_interactions;

        if recording {
            self.node_source_x.clear();
            self.node_source_y.clear();
            self.node_source_mass.clear();
            self.node_start.clear();
            self.body_items.clear();
            self.body_start.clear();
        }

        for index in 0..self.count {
            let mut acceleration_x = 0.0;
            let mut acceleration_y = world_gravity;

            if has_gravity {
                let (gravity_x, gravity_y) = if recording {
                    self.accumulate_tree_gravity(index)
                } else {
                    self.replay_tree_gravity(index)
                };

                acceleration_x += gravity_x * ACCELERATION_TO_FRAMES;
                acceleration_y += gravity_y * ACCELERATION_TO_FRAMES;
            }

            if has_hole {
                let delta_x = self.pointer_x - self.x[index];
                let delta_y = self.pointer_y - self.y[index];
                let scale = self.pointer_hole_scale(delta_x, delta_y) * ACCELERATION_TO_FRAMES;

                acceleration_x += delta_x * scale;
                acceleration_y += delta_y * scale;
            }

            self.acceleration_x[index] = acceleration_x;
            self.acceleration_y[index] = acceleration_y;
        }

        if recording {
            // One extra entry so every particle has a closing range bound.
            self.node_start.push(self.node_source_x.len() as u32);
            self.body_start.push(self.body_items.len() as u32);
            self.has_interactions = true;
        }
    }

    // ---- emission ----------------------------------------------------------

    fn is_spawn_position_free(&self, x: f32, y: f32) -> bool {
        for slot in 0..self.batch_x.len() {
            let offset_x = self.batch_x[slot] - x;
            let offset_y = self.batch_y[slot] - y;

            if offset_x * offset_x + offset_y * offset_y < CONTACT_DIAMETER_SQUARED {
                return false;
            }
        }

        let cell_x = (x * INVERSE_CELL_SIZE).floor() as i32;
        let cell_y = (y * INVERSE_CELL_SIZE).floor() as i32;

        for offset_y in -1..=1 {
            for offset_x in -1..=1 {
                let bucket = hash_cell(cell_x + offset_x, cell_y + offset_y) & self.grid_mask;
                let start = self.grid_start[bucket] as usize;
                let end = self.grid_start[bucket + 1] as usize;

                for slot in start..end {
                    let other = self.grid_items[slot] as usize;

                    if other >= self.count {
                        continue;
                    }

                    let delta_x = self.x[other] - x;
                    let delta_y = self.y[other] - y;

                    if delta_x * delta_x + delta_y * delta_y < CONTACT_DIAMETER_SQUARED {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn spawn_particle(&mut self, x: f32, y: f32, heading: f32, pointer_speed: f32, hue: f32) {
        if self.limit == 0 {
            return;
        }

        let max_x = (self.width - PARTICLE_RADIUS).max(PARTICLE_RADIUS);
        let max_y = (self.height - PARTICLE_RADIUS).max(PARTICLE_RADIUS);
        let mut position_x = 0.0;
        let mut position_y = 0.0;
        let mut has_placement = false;

        for attempt in 0..SPAWN_PLACEMENT_ATTEMPTS {
            let jitter = SPAWN_JITTER + attempt as f32 * PARTICLE_RADIUS;

            position_x = clamp(
                x + self.random_between(-jitter, jitter),
                PARTICLE_RADIUS,
                max_x,
            );
            position_y = clamp(
                y + self.random_between(-jitter, jitter),
                PARTICLE_RADIUS,
                max_y,
            );

            if self.is_spawn_position_free(position_x, position_y) {
                has_placement = true;
                break;
            }
        }

        // Nowhere to put it without an overlap, so the field is full here.
        if !has_placement {
            return;
        }

        // The particle being replaced keeps its slot and carries on colliding
        // until its fade runs out, so the newcomer always needs a fresh one.
        if self.alive_count >= self.limit {
            self.retire_oldest_alive();
        }

        while self.count - self.alive_count > MAX_RETIRING {
            self.drop_longest_fade();
        }

        let spread = self.random_between(-0.34, 0.34);
        let speed = (pointer_speed + self.random_between(2.2, 4.8))
            * self.random_between(0.72, 1.12)
            * EMISSION_VELOCITY_SCALE;
        let index = self.count;

        self.ensure_capacity(index + 1);
        self.count += 1;
        self.alive_count += 1;
        self.x[index] = position_x;
        self.y[index] = position_y;
        self.velocity_x[index] = (heading + spread).cos() * speed;
        self.velocity_y[index] = (heading + spread).sin() * speed;
        // Hue spans the whole byte, so the host can shade a continuous wheel.
        self.color[index] = (hue / 360.0 * 256.0) as u8;
        self.respawned[index] = 1;
        self.fade[index] = f32::INFINITY;
        self.birth[index] = self.next_birth;
        self.next_birth = self.next_birth.wrapping_add(1);
        self.batch_x.push(position_x);
        self.batch_y.push(position_y);
    }

    fn emit_from_pointer(&mut self, x: f32, y: f32, time: f64) {
        if !self.has_pointer {
            self.pointer_x = x;
            self.pointer_y = y;
            self.pointer_time = time;
            self.has_pointer = true;
            return;
        }

        let delta_x = x - self.pointer_x;
        let delta_y = y - self.pointer_y;
        let distance = (delta_x * delta_x + delta_y * delta_y).sqrt();

        if distance < 0.8 {
            return;
        }

        let elapsed = clamp((time - self.pointer_time) as f32, 4.0, 40.0);
        let heading = delta_y.atan2(delta_x);
        let speed_per_millisecond = distance / elapsed;
        let pointer_speed = clamp(speed_per_millisecond * FRAME_DURATION, 1.4, 8.0);
        let spawn_ratio = clamp((speed_per_millisecond * 1000.0 - 30.0) / 1000.0, 0.0, 1.0);
        let spawn_rate = 300.0 * spawn_ratio.powf(1.35);

        self.spawn_budget =
            (self.spawn_budget + spawn_rate * (elapsed / 1000.0)).min(MAX_PARTICLES_PER_FRAME);

        let particle_count = self.spawn_budget.floor();

        self.spawn_budget -= particle_count;

        let particle_count = particle_count as usize;

        for slot in 0..particle_count {
            let progress = (slot + 1) as f32 / particle_count as f32;
            let hue = (self.hue + slot as f32 * 17.0) % 360.0;

            self.spawn_particle(
                self.pointer_x + delta_x * progress,
                self.pointer_y + delta_y * progress,
                heading,
                pointer_speed,
                hue,
            );
        }

        self.pointer_x = x;
        self.pointer_y = y;
        self.pointer_time = time;
        self.hue = (self.hue + distance * 1.8 + particle_count as f32 * 8.0) % 360.0;
    }

    // ---- stepping ----------------------------------------------------------

    fn integrate(&mut self, retained: f32, dt: f32) {
        for index in 0..self.count {
            self.velocity_x[index] =
                (self.velocity_x[index] + self.acceleration_x[index] * dt) * retained;
            self.velocity_y[index] =
                (self.velocity_y[index] + self.acceleration_y[index] * dt) * retained;
            self.x[index] += self.velocity_x[index] * dt;
            self.y[index] += self.velocity_y[index] * dt;
        }
    }

    /// A pair passes straight through each other when either crosses a contact
    /// diameter inside one substep, so the fastest particle sets the count.
    fn choose_substeps(&self) -> usize {
        let mut fastest_squared = 0.0f32;

        for index in 0..self.count {
            let speed_squared = self.velocity_x[index] * self.velocity_x[index]
                + self.velocity_y[index] * self.velocity_y[index];

            if speed_squared > fastest_squared {
                fastest_squared = speed_squared;
            }
        }

        let needed = (fastest_squared.sqrt() * SUBSTEP_SPEED_MARGIN / CONTACT_DIAMETER).ceil();

        // Saturating cast, so an infinite speed lands on the ceiling.
        (needed as usize).clamp(MIN_PHYSICS_SUBSTEPS, MAX_PHYSICS_SUBSTEPS)
    }

    fn solve_walls(&mut self) {
        let max_x = (self.width - PARTICLE_RADIUS).max(PARTICLE_RADIUS);
        let max_y = (self.height - PARTICLE_RADIUS).max(PARTICLE_RADIUS);

        for index in 0..self.count {
            if self.x[index] < PARTICLE_RADIUS {
                self.x[index] = PARTICLE_RADIUS;
                if self.velocity_x[index] < 0.0 {
                    self.velocity_x[index] = -self.velocity_x[index];
                }
            } else if self.x[index] > max_x {
                self.x[index] = max_x;
                if self.velocity_x[index] > 0.0 {
                    self.velocity_x[index] = -self.velocity_x[index];
                }
            }

            if self.y[index] < PARTICLE_RADIUS {
                self.y[index] = PARTICLE_RADIUS;
                if self.velocity_y[index] < 0.0 {
                    self.velocity_y[index] = -self.velocity_y[index];
                }
            } else if self.y[index] > max_y {
                self.y[index] = max_y;
                if self.velocity_y[index] > 0.0 {
                    self.velocity_y[index] = -self.velocity_y[index];
                }
            }
        }
    }

    fn step(&mut self) {
        let air_resistance = MAX_AIR_RESISTANCE * (self.air_percent / MAX_CONTROL_PERCENT);
        let substeps = self.choose_substeps();
        let dt = 1.0 / substeps as f32;
        let retained = (1.0 - air_resistance).max(0.0).powf(dt);

        // Ahead of the substeps so the last grid built this step still matches
        // the array, which the spawn overlap check relies on until the next one.
        self.expire_fades();

        for substep in 0..substeps {
            // Sources barely shift within a frame, so one tree serves the whole
            // step and later substeps replay the sets recorded against it.
            if substep == 0 && self.count > 1 {
                self.build_gravity_tree();
            }

            // A field held stale across a frame drains energy from particles
            // orbiting inside a cluster, so it is rebuilt every substep.
            if self.count > 0 {
                self.accumulate_field();
            }

            self.integrate(retained, dt);
            self.solve_contacts();
            self.solve_walls();
        }

        self.batch_x.clear();
        self.batch_y.clear();
        self.step_index = self.step_index.wrapping_add(1);
    }

    fn expire_fades(&mut self) {
        let mut index = 0;

        while index < self.count {
            if self.fade[index].is_infinite() {
                index += 1;
                continue;
            }

            self.fade[index] -= 1.0;

            if self.fade[index] > 0.0 {
                index += 1;
                continue;
            }

            // The swap drops a new particle into this slot, so hold position.
            self.remove_particle(index);
        }
    }

    fn write_snapshot(&mut self) {
        let count = self.count;
        let flags = if self.discontinuous {
            SNAPSHOT_FLAG_DISCONTINUOUS
        } else {
            0
        };

        self.discontinuous = false;
        self.snapshot[0..4].copy_from_slice(&flags.to_le_bytes());
        self.snapshot[4..8].copy_from_slice(&(count as u32).to_le_bytes());
        self.snapshot[8..12].copy_from_slice(&self.step_index.to_le_bytes());
        self.snapshot[12..16].copy_from_slice(&0u32.to_le_bytes());

        let speeds = SNAPSHOT_HEADER_BYTES + count * 8;
        let colors = SNAPSHOT_HEADER_BYTES + count * 12;
        let particle_flags = SNAPSHOT_HEADER_BYTES + count * 13;

        for index in 0..count {
            let position = SNAPSHOT_HEADER_BYTES + index * 8;

            self.snapshot[position..position + 4].copy_from_slice(&self.x[index].to_le_bytes());
            self.snapshot[position + 4..position + 8].copy_from_slice(&self.y[index].to_le_bytes());

            let speed = (self.velocity_x[index] * self.velocity_x[index]
                + self.velocity_y[index] * self.velocity_y[index])
                .sqrt();

            self.snapshot[speeds + index * 4..speeds + index * 4 + 4]
                .copy_from_slice(&speed.to_le_bytes());
            self.snapshot[colors + index] = self.color[index];

            let fade = self.fade[index];
            let level = if fade.is_infinite() {
                RETIRE_FADE_LEVELS
            } else {
                // Floors so the final frame before removal lands on zero and
                // the particle is already black when it disappears.
                (fade / RETIRE_FADE_FRAMES * RETIRE_FADE_LEVELS).floor()
            };
            let respawned = if self.respawned[index] != 0 {
                PARTICLE_FLAG_RESPAWNED
            } else {
                0
            };

            self.snapshot[particle_flags + index] =
                respawned | ((level as u8) << PARTICLE_FADE_SHIFT);
            self.respawned[index] = 0;
        }
    }

    fn snapshot_len(&self) -> usize {
        SNAPSHOT_HEADER_BYTES + self.count * SNAPSHOT_BYTES_PER_PARTICLE
    }
}

/// wasm32-unknown-unknown has no threads, so a plain cell is sound here and
/// avoids the sharp edges of a `static mut`.
struct GlobalWorld(UnsafeCell<Option<World>>);

// SAFETY: the host is single threaded, so no other thread can observe this.
unsafe impl Sync for GlobalWorld {}

static WORLD: GlobalWorld = GlobalWorld(UnsafeCell::new(None));

/// Every export runs to completion before the host can call the next, so the
/// world is only ever borrowed by one caller at a time.
fn world() -> &'static mut World {
    // SAFETY: no other borrow can be live while this one is handed out.
    unsafe { (*WORLD.0.get()).get_or_insert_with(World::new) }
}

#[unsafe(no_mangle)]
pub extern "C" fn init(
    width: f32,
    height: f32,
    seed: u32,
    gravity_percent: f32,
    air_percent: f32,
    limit: u32,
) {
    let world = world();

    world.width = width;
    world.height = height;
    world.rng = if seed == 0 { 0x2545_F491 } else { seed };
    world.gravity_percent = gravity_percent;
    world.air_percent = air_percent;
    world.hue = (seed % 360) as f32;
    world.discontinuous = true;
    world.set_limit(limit as usize);
}

#[unsafe(no_mangle)]
pub extern "C" fn resize(width: f32, height: f32) {
    let world = world();

    world.width = width;
    world.height = height;
}

#[unsafe(no_mangle)]
pub extern "C" fn set_gravity(percent: f32) {
    world().gravity_percent = percent;
}

#[unsafe(no_mangle)]
pub extern "C" fn set_air_resistance(percent: f32) {
    world().air_percent = percent;
}

#[unsafe(no_mangle)]
pub extern "C" fn set_particle_limit(limit: u32) {
    world().set_limit(limit as usize);
}

#[unsafe(no_mangle)]
pub extern "C" fn pointer_move(x: f32, y: f32, time: f64, polarity: f32, emitting: u32) {
    let world = world();

    world.hole_polarity = polarity;

    if emitting != 0 && polarity == 0.0 {
        world.emit_from_pointer(x, y, time);
        return;
    }

    world.pointer_x = x;
    world.pointer_y = y;
    world.pointer_time = time;
    world.has_pointer = true;
    world.spawn_budget = 0.0;
}

#[unsafe(no_mangle)]
pub extern "C" fn pointer_reset() {
    let world = world();

    world.hole_polarity = 0.0;
    world.spawn_budget = 0.0;
    // Forgetting the position stops the next move from emitting a streak
    // spanning the gap from wherever the pointer was last seen.
    world.has_pointer = false;
}

#[unsafe(no_mangle)]
pub extern "C" fn reset_emission() {
    let world = world();

    world.spawn_budget = 0.0;
    world.has_pointer = false;
}

#[unsafe(no_mangle)]
pub extern "C" fn mark_discontinuous() {
    world().discontinuous = true;
}

#[unsafe(no_mangle)]
pub extern "C" fn step() {
    world().step();
}

/// Kept separate from `step` so a snapshot the host cannot transmit does not
/// consume the respawn and discontinuity flags it carries.
#[unsafe(no_mangle)]
pub extern "C" fn write_snapshot() {
    world().write_snapshot();
}

#[unsafe(no_mangle)]
pub extern "C" fn snapshot_ptr() -> *const u8 {
    world().snapshot.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn snapshot_len() -> u32 {
    world().snapshot_len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn particle_count() -> u32 {
    world().count as u32
}
