//! Lighting engine: batched BFS sky and block light propagation.
//!
//! Processes [`super::queue::LightUpdateQueue`] entries in batched passes,
//! grouping updates by section and propagating cross-section changes.
//! See ADR-017 for the algorithm design.

use std::collections::VecDeque;

use ahash::AHashMap;
use oxidized_chunks::LevelChunk;
use oxidized_chunks::sky_light_sources::ChunkSkyLightSources;
use oxidized_mc_types::SectionPos;
use oxidized_registry::BlockStateId;

use super::block_light::initialize_block_light;
use super::propagation::{
    ALL_DIRECTIONS, BoundaryEntry, DecreaseEntry, LightEntry, propagate_block_light_decrease,
    propagate_block_light_increase, propagate_sky_light_decrease, propagate_sky_light_increase,
};
use super::queue::LightUpdateQueue;
use super::sky::initialize_sky_light;

/// Errors that can occur during light processing.
#[derive(Debug, thiserror::Error)]
pub enum LightingError {
    /// A referenced chunk section is not loaded or available.
    #[error("chunk section unavailable at {section}")]
    SectionUnavailable {
        /// The position of the unavailable section.
        section: SectionPos,
    },
}

/// Result of processing light updates for a single chunk.
///
/// Contains the list of sections whose light data changed, plus any
/// boundary entries that need to be propagated into neighboring chunks.
#[derive(Debug, Default)]
pub struct LightResult {
    /// Sections within this chunk whose light data changed.
    pub changed_sections: Vec<SectionPos>,
    /// Block light entries that crossed chunk boundaries and need
    /// propagation into adjacent chunks.
    pub block_boundary: Vec<BoundaryEntry>,
    /// Sky light entries that crossed chunk boundaries and need
    /// propagation into adjacent chunks.
    pub sky_boundary: Vec<BoundaryEntry>,
}

/// Batched BFS lighting engine.
///
/// Owns a [`LightUpdateQueue`] and processes all pending updates in one pass
/// at the end of each tick. Groups updates by section, runs decrease then
/// increase BFS passes, and propagates across section boundaries.
///
/// # Examples
///
/// ```
/// use oxidized_lighting::engine::LightEngine;
/// use oxidized_types::ChunkPos;
///
/// let engine = LightEngine::new();
/// assert!(engine.queue().is_empty());
/// ```
pub struct LightEngine {
    queue: LightUpdateQueue,
}

impl LightEngine {
    /// Creates a new lighting engine with an empty update queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: LightUpdateQueue::new(),
        }
    }

    /// Returns a reference to the update queue.
    #[must_use]
    pub fn queue(&self) -> &LightUpdateQueue {
        &self.queue
    }

    /// Returns a mutable reference to the update queue.
    pub fn queue_mut(&mut self) -> &mut LightUpdateQueue {
        &mut self.queue
    }

    /// Processes all pending light updates for this tick on a single chunk.
    ///
    /// Groups updates by section, processes each section's decrease and
    /// increase BFS passes, and returns a [`LightResult`] containing the
    /// changed sections and any cross-chunk boundary entries.
    ///
    /// # Errors
    ///
    /// Returns [`LightingError`] if a referenced chunk section is unavailable.
    pub fn process_updates(
        &mut self,
        chunk: &mut LevelChunk,
    ) -> Result<LightResult, LightingError> {
        let updates = self.queue.drain();
        if updates.is_empty() {
            return Ok(LightResult::default());
        }

        let chunk_x = chunk.pos.x;
        let chunk_z = chunk.pos.z;
        let chunk_base_x = chunk_x * 16;
        let chunk_base_z = chunk_z * 16;

        let mut changed_sections: AHashMap<SectionPos, ()> = AHashMap::new();
        let mut block_decrease = VecDeque::new();
        let mut block_increase = VecDeque::new();
        let mut sky_decrease = VecDeque::new();
        let mut sky_increase = VecDeque::new();

        for update in &updates {
            let section_pos = SectionPos::of_block_pos(&update.pos);
            let local_x = update.pos.x & 15;
            let local_z = update.pos.z & 15;
            let y = update.pos.y;

            // Block light: handle emission changes.
            if update.old_emission != update.new_emission {
                if update.old_emission > 0 {
                    // Remove old light source.
                    chunk.set_block_light_at(local_x, y, local_z, 0);
                    block_decrease.push_back(DecreaseEntry {
                        x: local_x,
                        y,
                        z: local_z,
                        old_level: update.old_emission,
                        directions: ALL_DIRECTIONS,
                    });
                }
                if update.new_emission > 0 {
                    // Add new light source.
                    chunk.set_block_light_at(local_x, y, local_z, update.new_emission);
                    block_increase.push_back(LightEntry {
                        x: local_x,
                        y,
                        z: local_z,
                        level: update.new_emission,
                        directions: ALL_DIRECTIONS,
                    });
                }
                changed_sections.insert(section_pos, ());
            }

            // Shape-only change: emission and opacity unchanged but the
            // update was triggered because shape occlusion differs (e.g.,
            // replacing bottom slab with top slab). Clear stale light and
            // re-evaluate from neighbors, matching vanilla's PULL_LIGHT_IN.
            if update.old_emission == update.new_emission
                && update.old_opacity == update.new_opacity
            {
                let block_level = chunk.get_block_light_at(local_x, y, local_z);
                let sky_level = chunk.get_sky_light_at(local_x, y, local_z);

                if block_level > 0 && update.new_emission == 0 {
                    chunk.set_block_light_at(local_x, y, local_z, 0);
                    block_decrease.push_back(DecreaseEntry {
                        x: local_x,
                        y,
                        z: local_z,
                        old_level: block_level,
                        directions: ALL_DIRECTIONS,
                    });
                }
                if sky_level > 0 {
                    chunk.set_sky_light_at(local_x, y, local_z, 0);
                    sky_decrease.push_back(DecreaseEntry {
                        x: local_x,
                        y,
                        z: local_z,
                        old_level: sky_level,
                        directions: ALL_DIRECTIONS,
                    });
                }
                if update.new_emission > 0 {
                    chunk.set_block_light_at(local_x, y, local_z, update.new_emission);
                    block_increase.push_back(LightEntry {
                        x: local_x,
                        y,
                        z: local_z,
                        level: update.new_emission,
                        directions: ALL_DIRECTIONS,
                    });
                }
                changed_sections.insert(section_pos, ());
            }

            // Sky/Block light: handle opacity increases (block placed).
            if update.new_opacity > update.old_opacity {
                let sky_level = chunk.get_sky_light_at(local_x, y, local_z);
                let block_level = chunk.get_block_light_at(local_x, y, local_z);

                if sky_level > 0 {
                    chunk.set_sky_light_at(local_x, y, local_z, 0);
                    sky_decrease.push_back(DecreaseEntry {
                        x: local_x,
                        y,
                        z: local_z,
                        old_level: sky_level,
                        directions: ALL_DIRECTIONS,
                    });
                }
                if block_level > 0 && update.new_emission == 0 {
                    chunk.set_block_light_at(local_x, y, local_z, 0);
                    block_decrease.push_back(DecreaseEntry {
                        x: local_x,
                        y,
                        z: local_z,
                        old_level: block_level,
                        directions: ALL_DIRECTIONS,
                    });
                }

                // Darken the sky column below: blocks that were directly
                // sky-lit lose their source when an opaque block is placed
                // above them. Matches vanilla's column processing in
                // SkyLightEngine.checkNode() → updateSourcesInColumn().
                darken_sky_column(
                    chunk,
                    local_x,
                    y,
                    local_z,
                    &mut sky_decrease,
                    &mut changed_sections,
                );

                changed_sections.insert(section_pos, ());
            }
        }

        let mut block_boundary = Vec::new();
        let mut sky_boundary = Vec::new();

        // Run decrease passes first (per ADR-017).
        // Note: decrease boundary entries are intentionally discarded.
        // They carry the old (removed) light level, but our cross-chunk
        // propagation treats all entries as increases. Feeding decrease
        // entries to neighbors would inject phantom light into dark areas.
        // The increase pass generates correct boundary entries for any
        // light remaining after the decrease+increase cycle.
        // TODO: implement proper cross-chunk decrease propagation (Phase 26).
        if !block_decrease.is_empty() {
            let _decrease_boundary = propagate_block_light_decrease(
                chunk,
                &mut block_decrease,
                &mut block_increase,
                chunk_base_x,
                chunk_base_z,
            );
        }
        if !sky_decrease.is_empty() {
            let _decrease_boundary = propagate_sky_light_decrease(
                chunk,
                &mut sky_decrease,
                &mut sky_increase,
                chunk_base_x,
                chunk_base_z,
            );
        }

        // After decrease completes, handle opacity decreases (block broken)
        // by checking neighbors for light that can now flow through,
        // and re-seeding sky light downward through the column.
        for update in &updates {
            if update.new_opacity < update.old_opacity {
                let local_x = update.pos.x & 15;
                let local_z = update.pos.z & 15;
                let y = update.pos.y;
                let section_pos = SectionPos::of_block_pos(&update.pos);

                check_neighbor_light_sources(
                    chunk,
                    local_x,
                    y,
                    local_z,
                    &mut sky_increase,
                    &mut block_increase,
                );

                // Simplified sky column re-seeding: if the block above has
                // sky light 15, this column was a sky source. Re-fill downward
                // through air until we hit an opaque block.
                reseed_sky_column(
                    chunk,
                    local_x,
                    y,
                    local_z,
                    &mut sky_increase,
                    &mut changed_sections,
                );

                changed_sections.insert(section_pos, ());
            }
        }

        // Then run increase passes.
        if !block_increase.is_empty() {
            let boundary = propagate_block_light_increase(
                chunk,
                &mut block_increase,
                chunk_base_x,
                chunk_base_z,
            );
            block_boundary.extend(boundary);
        }
        if !sky_increase.is_empty() {
            let boundary =
                propagate_sky_light_increase(chunk, &mut sky_increase, chunk_base_x, chunk_base_z);
            sky_boundary.extend(boundary);
        }

        // Mark sections around each changed position as changed.
        for update in &updates {
            let sp = SectionPos::of_block_pos(&update.pos);
            changed_sections.insert(sp, ());
            // Also mark neighboring sections when light could propagate there.
            for dy in [-1i32, 0, 1] {
                let neighbor = SectionPos::new(sp.x, sp.y + dy, sp.z);
                changed_sections.insert(neighbor, ());
            }
        }

        Ok(LightResult {
            changed_sections: changed_sections.into_keys().collect(),
            block_boundary,
            sky_boundary,
        })
    }

    /// Computes full sky + block light for a newly generated chunk.
    ///
    /// Called by the worldgen pipeline at the Light status (ADR-016).
    /// Initializes sky light using per-column source tracking, seeds block light
    /// from emitters, and runs BFS propagation for both light types.
    ///
    /// Returns a [`LightResult`] containing boundary entries for cross-chunk
    /// propagation. The `changed_sections` field is empty since all sections
    /// in a newly generated chunk are implicitly "changed".
    ///
    /// # Errors
    ///
    /// Returns [`LightingError`] if a chunk section is unavailable.
    pub fn light_chunk(&mut self, chunk: &mut LevelChunk) -> Result<LightResult, LightingError> {
        let sky_boundary = initialize_sky_light(chunk);
        let block_boundary = initialize_block_light(chunk);
        Ok(LightResult {
            changed_sections: Vec::new(),
            block_boundary,
            sky_boundary,
        })
    }
}

/// Checks all 6 neighbors of a position for light sources and seeds the
/// increase queues if any neighbor has light that can now flow through.
fn check_neighbor_light_sources(
    chunk: &LevelChunk,
    x: i32,
    y: i32,
    z: i32,
    sky_increase: &mut VecDeque<LightEntry>,
    block_increase: &mut VecDeque<LightEntry>,
) {
    const DIRS: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    for &(dx, dy, dz) in &DIRS {
        let nx = x + dx;
        let ny = y + dy;
        let nz = z + dz;

        if !(0..16).contains(&nx) || !(0..16).contains(&nz) {
            continue;
        }
        if ny < chunk.min_y() || ny >= chunk.max_y() {
            continue;
        }

        let sky = chunk.get_sky_light_at(nx, ny, nz);
        if sky > 1 {
            sky_increase.push_back(LightEntry {
                x: nx,
                y: ny,
                z: nz,
                level: sky,
                directions: ALL_DIRECTIONS,
            });
        }

        let block = chunk.get_block_light_at(nx, ny, nz);
        if block > 1 {
            block_increase.push_back(LightEntry {
                x: nx,
                y: ny,
                z: nz,
                level: block,
                directions: ALL_DIRECTIONS,
            });
        }
    }
}

/// Sky light column re-seeding after a block is broken, using
/// [`ChunkSkyLightSources`] for accurate source tracking.
///
/// When a block is removed, updates the chunk's sky light sources and
/// restores sky light 15 downward through transparent blocks if the
/// column is a sky source. Each restored position is seeded into the
/// BFS increase queue for horizontal propagation into caves.
fn reseed_sky_column(
    chunk: &mut LevelChunk,
    x: i32,
    y: i32,
    z: i32,
    sky_increase: &mut VecDeque<LightEntry>,
    changed_sections: &mut AHashMap<SectionPos, ()>,
) {
    // Update the sky light sources for this column.
    let mut sources = chunk
        .take_sky_light_sources()
        .unwrap_or_else(|| ChunkSkyLightSources::new(chunk.min_y()));
    sources.update(chunk, x as usize, y, z as usize);

    let source_y = sources.get_lowest_source_y(x as usize, z as usize);
    chunk.set_sky_light_sources(sources);

    // If this position is at or above the source height, it should be sky-lit 15.
    if y < source_y {
        return;
    }

    // Walk downward from the broken block's position, restoring sky light 15
    // through transparent blocks until we hit the source boundary or an opaque block.
    let min_y = chunk.min_y();
    let stop_y = source_y.max(min_y);
    for cur_y in (stop_y..=y).rev() {
        let state_id = chunk.get_block_state(x & 15, cur_y, z & 15).unwrap_or(0);
        #[allow(clippy::cast_possible_truncation)]
        let opacity = BlockStateId(state_id as u16).light_opacity();
        if opacity >= 15 {
            break;
        }

        let current = chunk.get_sky_light_at(x, cur_y, z);
        if current < 15 {
            chunk.set_sky_light_at(x, cur_y, z, 15);
            sky_increase.push_back(LightEntry {
                x,
                y: cur_y,
                z,
                level: 15,
                directions: ALL_DIRECTIONS,
            });
            let section_pos = SectionPos::new(chunk.pos.x, cur_y >> 4, chunk.pos.z);
            changed_sections.insert(section_pos, ());
        }
    }
}

/// Darkens sky light in a column after an opaque block is placed.
///
/// Matches vanilla's column processing in `SkyLightEngine.checkNode()`:
/// when a block becomes opaque, updates [`ChunkSkyLightSources`] and
/// clears sky light for all blocks between the old and new source heights,
/// queuing decrease entries for the BFS to re-evaluate from neighbors.
fn darken_sky_column(
    chunk: &mut LevelChunk,
    x: i32,
    y: i32,
    z: i32,
    sky_decrease: &mut VecDeque<DecreaseEntry>,
    changed_sections: &mut AHashMap<SectionPos, ()>,
) {
    let mut sources = chunk
        .take_sky_light_sources()
        .unwrap_or_else(|| ChunkSkyLightSources::new(chunk.min_y()));
    let old_source_y = sources.get_lowest_source_y(x as usize, z as usize);
    sources.update(chunk, x as usize, y, z as usize);
    let new_source_y = sources.get_lowest_source_y(x as usize, z as usize);
    chunk.set_sky_light_sources(sources);

    // If the source didn't move up, no blocks below lost their sky source.
    if new_source_y <= old_source_y {
        return;
    }

    // Iterate from below the placed block down through the old sky column.
    // Clear sky light for blocks that lost their direct sky source.
    let min_y = chunk.min_y();
    let start = y - 1;
    let stop = if old_source_y == i32::MIN {
        min_y
    } else {
        old_source_y
    };

    for cur_y in (stop..=start).rev() {
        let sky = chunk.get_sky_light_at(x, cur_y, z);
        if sky == 0 {
            break;
        }

        chunk.set_sky_light_at(x, cur_y, z, 0);
        sky_decrease.push_back(DecreaseEntry {
            x,
            y: cur_y,
            z,
            old_level: sky,
            directions: ALL_DIRECTIONS,
        });
        let section_pos = SectionPos::new(chunk.pos.x, cur_y >> 4, chunk.pos.z);
        changed_sections.insert(section_pos, ());

        // Stop at fully opaque blocks — light below came from another source.
        #[allow(clippy::cast_possible_truncation)]
        let state_id = chunk.get_block_state(x, cur_y, z).unwrap_or(0);
        let opacity = BlockStateId(state_id as u16).light_opacity();
        if opacity >= 15 {
            break;
        }
    }
}

impl Default for LightEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::queue::LightUpdate;
    use oxidized_chunks::LevelChunk;
    use oxidized_chunks::level_chunk::OVERWORLD_MIN_Y;
    use oxidized_mc_types::BlockPos;
    use oxidized_registry::{BEDROCK, BlockRegistry, BlockStateId, DIRT, GRASS_BLOCK};
    use oxidized_types::ChunkPos;

    fn stone_id() -> u32 {
        u32::from(
            BlockRegistry
                .default_state("minecraft:stone")
                .expect("stone missing")
                .0,
        )
    }

    fn glowstone_id() -> u32 {
        u32::from(
            BlockRegistry
                .default_state("minecraft:glowstone")
                .expect("glowstone missing")
                .0,
        )
    }

    fn flat_chunk() -> LevelChunk {
        let mut chunk = LevelChunk::new(ChunkPos::new(0, 0));
        let bedrock = u32::from(BEDROCK.0);
        let dirt = u32::from(DIRT.0);
        let grass = u32::from(GRASS_BLOCK.0);

        for x in 0..16i32 {
            for z in 0..16i32 {
                chunk
                    .set_block_state(x, OVERWORLD_MIN_Y, z, bedrock)
                    .unwrap();
                chunk
                    .set_block_state(x, OVERWORLD_MIN_Y + 1, z, dirt)
                    .unwrap();
                chunk
                    .set_block_state(x, OVERWORLD_MIN_Y + 2, z, dirt)
                    .unwrap();
                chunk
                    .set_block_state(x, OVERWORLD_MIN_Y + 3, z, grass)
                    .unwrap();
            }
        }

        chunk
    }

    #[test]
    fn test_engine_new_has_empty_queue() {
        let engine = LightEngine::new();
        assert!(engine.queue().is_empty());
    }

    #[test]
    fn test_engine_queue_mut() {
        let mut engine = LightEngine::new();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(0, 64, 0),
            old_emission: 0,
            new_emission: 14,
            old_opacity: 0,
            new_opacity: 0,
        });
        assert_eq!(engine.queue().len(), 1);
    }

    #[test]
    fn test_light_chunk_sky_light() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Above surface.
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 4, 8), 15);
        // Inside solid.
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 2, 8), 0);
    }

    #[test]
    fn test_process_updates_empty_queue() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        let result = engine.process_updates(&mut chunk).unwrap();
        assert!(result.changed_sections.is_empty());
    }

    #[test]
    fn test_process_updates_place_glowstone() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Place glowstone at (8, -56, 8) — above the surface in air.
        let gs = glowstone_id();
        let emission = BlockStateId(gs as u16).light_emission();
        chunk.set_block_state(8, -56, 8, gs).unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(8, -56, 8),
            old_emission: 0,
            new_emission: emission,
            old_opacity: 0,
            new_opacity: BlockStateId(gs as u16).light_opacity(),
        });

        let changed = engine.process_updates(&mut chunk).unwrap();
        assert!(!changed.changed_sections.is_empty());
        assert_eq!(chunk.get_block_light_at(8, -56, 8), emission);
        assert_eq!(chunk.get_block_light_at(9, -56, 8), emission - 1);
    }

    #[test]
    fn test_process_updates_break_glowstone() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Place glowstone.
        let gs = glowstone_id();
        let emission = BlockStateId(gs as u16).light_emission();
        chunk.set_block_state(8, -56, 8, gs).unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(8, -56, 8),
            old_emission: 0,
            new_emission: emission,
            old_opacity: 0,
            new_opacity: BlockStateId(gs as u16).light_opacity(),
        });
        engine.process_updates(&mut chunk).unwrap();

        // Now break it (set to air).
        chunk.set_block_state(8, -56, 8, 0).unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(8, -56, 8),
            old_emission: emission,
            new_emission: 0,
            old_opacity: BlockStateId(gs as u16).light_opacity(),
            new_opacity: 0,
        });
        engine.process_updates(&mut chunk).unwrap();

        assert_eq!(chunk.get_block_light_at(8, -56, 8), 0);
        assert_eq!(chunk.get_block_light_at(9, -56, 8), 0);
    }

    #[test]
    fn test_process_updates_place_opaque_block() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Verify sky light above surface.
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 5, 8), 15);

        // Place stone at the surface+1 position.
        let st = stone_id();
        let opacity = BlockStateId(st as u16).light_opacity();
        chunk
            .set_block_state(8, OVERWORLD_MIN_Y + 4, 8, st)
            .unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(8, OVERWORLD_MIN_Y + 4, 8),
            old_emission: 0,
            new_emission: 0,
            old_opacity: 0,
            new_opacity: opacity,
        });
        let changed = engine.process_updates(&mut chunk).unwrap();
        assert!(!changed.changed_sections.is_empty());
    }

    #[test]
    fn test_break_surface_block_reseeds_sky_column() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Verify sky light above surface is 15 and inside grass is 0.
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 4, 8), 15);
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 3, 8), 0);

        // Break the grass block (opacity → 0).
        let grass_opacity = BlockStateId(GRASS_BLOCK.0).light_opacity();
        chunk.set_block_state(8, OVERWORLD_MIN_Y + 3, 8, 0).unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(8, OVERWORLD_MIN_Y + 3, 8),
            old_emission: 0,
            new_emission: 0,
            old_opacity: grass_opacity,
            new_opacity: 0,
        });
        engine.process_updates(&mut chunk).unwrap();

        // The broken block's position should now have sky light 15.
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 3, 8), 15);
    }

    #[test]
    fn test_process_updates_returns_block_boundary_at_chunk_edge() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Place glowstone at chunk edge (0, -56, 8) — in air above surface.
        let gs = glowstone_id();
        let emission = BlockStateId(gs as u16).light_emission();
        chunk.set_block_state(0, -56, 8, gs).unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(0, -56, 8),
            old_emission: 0,
            new_emission: emission,
            old_opacity: 0,
            new_opacity: BlockStateId(gs as u16).light_opacity(),
        });

        let result = engine.process_updates(&mut chunk).unwrap();
        assert!(
            !result.block_boundary.is_empty(),
            "expected block boundary entries when light source is at chunk edge"
        );
        // Boundary entries should cross into negative X (neighbor chunk).
        assert!(
            result.block_boundary.iter().any(|b| b.world_x < 0),
            "expected boundary entries crossing into negative X"
        );
    }

    #[test]
    fn test_process_updates_returns_sky_boundary_at_chunk_edge() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Place opaque block at chunk edge above surface to trigger sky decrease.
        let st = stone_id();
        let opacity = BlockStateId(st as u16).light_opacity();
        chunk
            .set_block_state(0, OVERWORLD_MIN_Y + 5, 8, st)
            .unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(0, OVERWORLD_MIN_Y + 5, 8),
            old_emission: 0,
            new_emission: 0,
            old_opacity: 0,
            new_opacity: opacity,
        });

        let result = engine.process_updates(&mut chunk).unwrap();
        // After decrease is resolved, the increase pass re-diffuses remaining light
        // near the chunk edge, producing correct boundary entries.
        assert!(
            !result.sky_boundary.is_empty(),
            "expected sky boundary entries from increase pass when opaque block placed at chunk edge"
        );
        // Boundary entries must come from the increase pass (correct current levels),
        // NOT from the decrease pass (stale old_level values).
        for entry in &result.sky_boundary {
            assert!(
                entry.level < 15,
                "boundary entry level should be < 15 (diffused), got {}",
                entry.level
            );
        }
    }

    #[test]
    fn test_place_opaque_darkens_sky_column_below() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Surface at OVERWORLD_MIN_Y + 3 (grass). Air above has sky 15.
        let above_surface = OVERWORLD_MIN_Y + 4;
        assert_eq!(chunk.get_sky_light_at(8, above_surface, 8), 15);
        assert_eq!(chunk.get_sky_light_at(8, above_surface + 3, 8), 15);

        // Place an opaque block 4 above surface, creating a sky gap.
        let place_y = above_surface + 4;
        let st = stone_id();
        let opacity = BlockStateId(st as u16).light_opacity();
        chunk.set_block_state(8, place_y, 8, st).unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(8, place_y, 8),
            old_emission: 0,
            new_emission: 0,
            old_opacity: 0,
            new_opacity: opacity,
        });

        engine.process_updates(&mut chunk).unwrap();

        // The placed block itself should have sky light 0.
        assert_eq!(
            chunk.get_sky_light_at(8, place_y, 8),
            0,
            "placed opaque block should have sky light 0"
        );

        // Blocks below the placed block should no longer have sky light 15
        // because the column's sky source moved above the placed block.
        // They should get light from horizontal neighbors instead (< 15).
        let below = chunk.get_sky_light_at(8, place_y - 1, 8);
        assert!(
            below < 15,
            "block below placed opaque should not have sky 15, got {below}"
        );

        // The block at the surface (grass level + 1) should also lose
        // direct sky source.
        let at_surface = chunk.get_sky_light_at(8, above_surface, 8);
        assert!(
            at_surface < 15,
            "block at surface should not have sky 15 after ceiling placed, got {at_surface}"
        );
    }

    #[test]
    fn test_enclose_in_cube_darkens_interior() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        let st = stone_id();
        let opacity = BlockStateId(st as u16).light_opacity();
        let above_surface = OVERWORLD_MIN_Y + 4;

        // Build a 3×3×3 solid cube from (7, above_surface, 7) to
        // (9, above_surface+2, 9), leaving the center hollow at
        // (8, above_surface+1, 8).
        for x in 7..=9i32 {
            for z in 7..=9i32 {
                for y in above_surface..=above_surface + 2 {
                    if x == 8 && y == above_surface + 1 && z == 8 {
                        continue; // Leave center hollow.
                    }
                    chunk.set_block_state(x, y, z, st).unwrap();
                    engine.queue_mut().push(LightUpdate {
                        pos: BlockPos::new(x, y, z),
                        old_emission: 0,
                        new_emission: 0,
                        old_opacity: 0,
                        new_opacity: opacity,
                    });
                }
            }
        }

        engine.process_updates(&mut chunk).unwrap();

        // The hollow center, surrounded by opaque blocks on all 6 faces,
        // should have sky light 0.
        let center_sky = chunk.get_sky_light_at(8, above_surface + 1, 8);
        assert_eq!(
            center_sky, 0,
            "enclosed air block should have sky light 0, got {center_sky}"
        );
    }

    /// Regression test: placing an opaque block at a chunk edge must NOT
    /// produce boundary entries that inject phantom sky light underground
    /// in the neighboring chunk. Before the fix, decrease boundary entries
    /// carried `old_level` (the removed light value) and were incorrectly
    /// processed as increases, lighting up dark underground areas.
    #[test]
    fn test_decrease_boundary_entries_not_injected_as_increases() {
        let mut engine = LightEngine::new();
        let mut chunk = flat_chunk();
        engine.light_chunk(&mut chunk).unwrap();

        // Surface is at OVERWORLD_MIN_Y + 3 (grass). Air above has sky 15.
        let above_surface = OVERWORLD_MIN_Y + 4;

        // Place opaque block at chunk edge x=0, above surface.
        // This triggers sky decrease for the column at x=0.
        let st = stone_id();
        let opacity = BlockStateId(st as u16).light_opacity();
        chunk.set_block_state(0, above_surface, 8, st).unwrap();
        engine.queue_mut().push(LightUpdate {
            pos: BlockPos::new(0, above_surface, 8),
            old_emission: 0,
            new_emission: 0,
            old_opacity: 0,
            new_opacity: opacity,
        });

        let result = engine.process_updates(&mut chunk).unwrap();

        // All boundary entries must have levels that represent ACTUAL current
        // light, not stale old_level from decrease passes. An underground
        // neighbor receiving these entries should not gain phantom sky light.
        for entry in &result.sky_boundary {
            // Entries with level 15 from the decrease pass would be the bug:
            // they represent removed light, not current light.
            assert!(
                entry.level < 15,
                "boundary entry at y={} has level 15 (stale decrease), would inject phantom light",
                entry.world_y
            );
        }

        // Furthermore, entries should only exist for y-levels where light
        // actually reaches the chunk edge (above or at surface level).
        // No entries should exist for underground y-levels.
        let surface_y = OVERWORLD_MIN_Y + 3;
        for entry in &result.sky_boundary {
            assert!(
                entry.world_y >= surface_y,
                "boundary entry at y={} is underground (surface={}), would inject phantom light",
                entry.world_y,
                surface_y
            );
        }
    }
}
