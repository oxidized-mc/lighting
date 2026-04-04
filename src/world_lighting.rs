//! Persistent world-level lighting state.
//!
//! Owns a single [`LightEngine`] that persists across ticks, accumulating
//! light updates and boundary entries. This matches vanilla's persistent
//! `LevelLightEngine` architecture.

use ahash::AHashMap;
use oxidized_types::ChunkPos;

use super::engine::LightEngine;
use super::propagation::BoundaryEntry;
use super::queue::LightUpdate;

/// Boundary entries pending processing in the next tick.
///
/// Collected when BFS reaches chunk edges during `process_updates()`, then
/// re-queued into the relevant neighbor chunk on the following tick.
#[derive(Debug, Default, Clone)]
pub struct PendingBoundaries {
    /// Block light entries crossing into this chunk.
    pub block: Vec<BoundaryEntry>,
    /// Sky light entries crossing into this chunk.
    pub sky: Vec<BoundaryEntry>,
}

/// Persistent world-level lighting state.
///
/// Holds the [`LightEngine`] (with its internal update queue), pending light
/// updates from block changes, and cross-chunk boundary entries that carry
/// over between ticks.
///
/// # Examples
///
/// ```
/// use oxidized_lighting::world_lighting::WorldLighting;
/// use oxidized_types::ChunkPos;
/// use oxidized_lighting::queue::LightUpdate;
/// use oxidized_mc_types::BlockPos;
///
/// let mut wl = WorldLighting::new();
/// assert!(!wl.has_pending_work());
///
/// wl.queue_update(ChunkPos::new(0, 0), LightUpdate {
///     pos: BlockPos::new(8, 64, 8),
///     old_emission: 0,
///     new_emission: 14,
///     old_opacity: 0,
///     new_opacity: 0,
/// });
/// assert!(wl.has_pending_work());
/// ```
pub struct WorldLighting {
    /// Persistent lighting engine — survives across ticks.
    engine: LightEngine,
    /// Light updates queued by block changes, grouped by chunk.
    pending_updates: Vec<(ChunkPos, LightUpdate)>,
    /// Boundary entries from the previous tick that need processing
    /// in the current tick. Keyed by the *target* chunk position.
    pending_boundaries: AHashMap<ChunkPos, PendingBoundaries>,
}

impl WorldLighting {
    /// Creates a new `WorldLighting` with an empty engine and no pending work.
    #[must_use]
    pub fn new() -> Self {
        Self {
            engine: LightEngine::new(),
            pending_updates: Vec::new(),
            pending_boundaries: AHashMap::new(),
        }
    }

    /// Returns `true` if there are pending updates or boundary entries.
    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        !self.pending_updates.is_empty() || !self.pending_boundaries.is_empty()
    }

    /// Queues a light update caused by a block change.
    pub fn queue_update(&mut self, chunk_pos: ChunkPos, update: LightUpdate) {
        self.pending_updates.push((chunk_pos, update));
    }

    /// Drains all pending light updates, returning them.
    pub fn drain_updates(&mut self) -> Vec<(ChunkPos, LightUpdate)> {
        std::mem::take(&mut self.pending_updates)
    }

    /// Stores boundary entries for processing in the next tick.
    ///
    /// `target` is the chunk position that should receive the boundary
    /// entries (the neighbor, not the source chunk).
    pub fn queue_boundaries(
        &mut self,
        target: ChunkPos,
        block: Vec<BoundaryEntry>,
        sky: Vec<BoundaryEntry>,
    ) {
        if block.is_empty() && sky.is_empty() {
            return;
        }
        let entry = self.pending_boundaries.entry(target).or_default();
        entry.block.extend(block);
        entry.sky.extend(sky);
    }

    /// Drains all pending boundary entries, returning them keyed by chunk.
    pub fn drain_boundaries(&mut self) -> AHashMap<ChunkPos, PendingBoundaries> {
        std::mem::take(&mut self.pending_boundaries)
    }

    /// Returns a mutable reference to the persistent [`LightEngine`].
    pub fn engine_mut(&mut self) -> &mut LightEngine {
        &mut self.engine
    }

    /// Returns a reference to the persistent [`LightEngine`].
    #[must_use]
    pub fn engine(&self) -> &LightEngine {
        &self.engine
    }
}

impl Default for WorldLighting {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::propagation::ALL_DIRECTIONS;
    use crate::queue::LightUpdate;
    use oxidized_mc_types::BlockPos;

    fn sample_update() -> LightUpdate {
        LightUpdate {
            pos: BlockPos::new(8, 64, 8),
            old_emission: 0,
            new_emission: 14,
            old_opacity: 0,
            new_opacity: 0,
        }
    }

    #[test]
    fn test_new_has_no_pending_work() {
        let wl = WorldLighting::new();
        assert!(!wl.has_pending_work());
    }

    #[test]
    fn test_queue_update_marks_pending() {
        let mut wl = WorldLighting::new();
        wl.queue_update(ChunkPos::new(0, 0), sample_update());
        assert!(wl.has_pending_work());
    }

    #[test]
    fn test_drain_updates_returns_all_and_clears() {
        let mut wl = WorldLighting::new();
        wl.queue_update(ChunkPos::new(0, 0), sample_update());
        wl.queue_update(ChunkPos::new(1, 0), sample_update());

        let drained = wl.drain_updates();
        assert_eq!(drained.len(), 2);
        assert!(!wl.has_pending_work());
    }

    #[test]
    fn test_queue_boundaries_stores_entries() {
        let mut wl = WorldLighting::new();
        let entries = vec![BoundaryEntry {
            world_x: 16,
            world_y: 64,
            world_z: 8,
            level: 12,
            directions: ALL_DIRECTIONS,
        }];
        wl.queue_boundaries(ChunkPos::new(1, 0), entries, vec![]);
        assert!(wl.has_pending_work());
    }

    #[test]
    fn test_queue_boundaries_empty_is_noop() {
        let mut wl = WorldLighting::new();
        wl.queue_boundaries(ChunkPos::new(1, 0), vec![], vec![]);
        assert!(!wl.has_pending_work());
    }

    #[test]
    fn test_drain_boundaries_returns_all_and_clears() {
        let mut wl = WorldLighting::new();
        wl.queue_boundaries(
            ChunkPos::new(1, 0),
            vec![BoundaryEntry {
                world_x: 16,
                world_y: 64,
                world_z: 8,
                level: 12,
                directions: ALL_DIRECTIONS,
            }],
            vec![],
        );
        wl.queue_boundaries(
            ChunkPos::new(0, 1),
            vec![],
            vec![BoundaryEntry {
                world_x: 8,
                world_y: 70,
                world_z: 16,
                level: 14,
                directions: ALL_DIRECTIONS,
            }],
        );

        let boundaries = wl.drain_boundaries();
        assert_eq!(boundaries.len(), 2);
        assert!(boundaries.contains_key(&ChunkPos::new(1, 0)));
        assert!(boundaries.contains_key(&ChunkPos::new(0, 1)));
        assert!(!wl.has_pending_work());
    }

    #[test]
    fn test_queue_boundaries_merges_into_same_chunk() {
        let mut wl = WorldLighting::new();
        let target = ChunkPos::new(1, 0);
        wl.queue_boundaries(
            target,
            vec![BoundaryEntry {
                world_x: 16,
                world_y: 64,
                world_z: 8,
                level: 12,
                directions: ALL_DIRECTIONS,
            }],
            vec![],
        );
        wl.queue_boundaries(
            target,
            vec![BoundaryEntry {
                world_x: 16,
                world_y: 65,
                world_z: 8,
                level: 11,
                directions: ALL_DIRECTIONS,
            }],
            vec![],
        );

        let boundaries = wl.drain_boundaries();
        assert_eq!(boundaries[&target].block.len(), 2);
    }

    #[test]
    fn test_engine_persists_across_operations() {
        let mut wl = WorldLighting::new();
        wl.engine_mut().queue_mut().push(sample_update());
        assert_eq!(wl.engine().queue().len(), 1);

        // Drain the engine's queue to simulate process_updates.
        let _ = wl.engine_mut().queue_mut().drain();
        assert!(wl.engine().queue().is_empty());

        // Engine still exists and can accept more updates.
        wl.engine_mut().queue_mut().push(sample_update());
        assert_eq!(wl.engine().queue().len(), 1);
    }

    #[test]
    fn test_boundary_entries_survive_update_drain() {
        let mut wl = WorldLighting::new();
        wl.queue_update(ChunkPos::new(0, 0), sample_update());
        wl.queue_boundaries(
            ChunkPos::new(1, 0),
            vec![BoundaryEntry {
                world_x: 16,
                world_y: 64,
                world_z: 8,
                level: 12,
                directions: ALL_DIRECTIONS,
            }],
            vec![],
        );

        // Drain only updates — boundaries should persist.
        let _ = wl.drain_updates();
        assert!(wl.has_pending_work()); // boundaries still there
    }
}
