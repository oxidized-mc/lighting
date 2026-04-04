//! Sky light initialization for newly generated chunks.
//!
//! Computes sky light in two phases:
//! 1. **Vertical pass** — for each (x, z) column, sets sky light to 15 at and
//!    above the sky light source height (from [`ChunkSkyLightSources`]), then
//!    attenuates downward by each block's opacity.
//! 2. **Horizontal BFS** — propagates sky light sideways through transparent
//!    blocks below the source height (caves, overhangs).

use std::collections::VecDeque;

use oxidized_chunks::DataLayer;
use oxidized_chunks::LevelChunk;
use oxidized_chunks::sky_light_sources::ChunkSkyLightSources;
use oxidized_registry::BlockStateId;

use super::propagation::{ALL_DIRECTIONS, BoundaryEntry, LightEntry, propagate_sky_light_increase};

/// Initializes sky light for a newly generated chunk.
///
/// Builds [`ChunkSkyLightSources`] for per-column source tracking, then runs
/// the vertical pass and horizontal BFS. After this call, the chunk's sky
/// light layers are fully populated and the sources are stored in the chunk.
///
/// Returns boundary entries for positions where sky light reached a chunk
/// edge and needs to propagate into a neighbor chunk.
pub fn initialize_sky_light(chunk: &mut LevelChunk) -> Vec<BoundaryEntry> {
    let sources = ChunkSkyLightSources::from_chunk(chunk);
    let min_y = chunk.min_y();
    let max_y = chunk.max_y();

    // Bulk-fill optimization: sections entirely above ALL source heights
    // are guaranteed to have sky light 15 everywhere. Fill them in O(1)
    // per section instead of iterating 4096 blocks.
    let highest_source_y = sources.get_highest_lowest_source_y();
    let bulk_fill_above = bulk_fill_empty_sections(chunk, highest_source_y, min_y, max_y);

    // Phase 1: Vertical pass — top-down per (x, z) column.
    // Only iterate down to `bulk_fill_above` — everything above is already
    // filled with sky light 15.
    let mut bfs_seeds = VecDeque::new();

    for x in 0..16i32 {
        for z in 0..16i32 {
            let source_y = sources.get_lowest_source_y(x as usize, z as usize);
            let mut level: u8 = 15;

            // Iterate from the first block that isn't bulk-filled downward.
            for y in (min_y..bulk_fill_above).rev() {
                if y >= source_y {
                    // At or above the source height: full sky brightness.
                    chunk.set_sky_light_at(x, y, z, 15);
                } else {
                    // Below the source: attenuate by opacity.
                    let state_id = chunk.get_block_state(x & 15, y, z & 15).unwrap_or(0);
                    #[allow(clippy::cast_possible_truncation)]
                    let opacity = BlockStateId(state_id as u16).light_opacity();
                    level = level.saturating_sub(opacity.max(1));
                    chunk.set_sky_light_at(x, y, z, level);
                    if level == 0 {
                        break;
                    }
                }
            }

            // Seed horizontal BFS from blocks just below the source where
            // sky light is still > 1 (light can spread sideways into caves).
            let seed_start = source_y.min(max_y).saturating_sub(1);
            if seed_start >= min_y {
                for y in (min_y..=seed_start).rev() {
                    let sky = chunk.get_sky_light_at(x, y, z);
                    if sky > 1 {
                        bfs_seeds.push_back(LightEntry {
                            x,
                            y,
                            z,
                            level: sky,
                            directions: ALL_DIRECTIONS,
                        });
                    }
                    if sky == 0 {
                        break;
                    }
                }
            }
        }
    }

    // Store the sources in the chunk for incremental updates.
    chunk.set_sky_light_sources(sources);

    // Phase 2: Horizontal BFS for sky light bleeding into caves.
    let chunk_base_x = chunk.pos.x * 16;
    let chunk_base_z = chunk.pos.z * 16;
    propagate_sky_light_increase(chunk, &mut bfs_seeds, chunk_base_x, chunk_base_z)
}

/// Bulk-fills sky light sections that are entirely above all source heights.
///
/// For sections whose minimum Y is at or above `highest_source_y`, the entire
/// section is guaranteed sky light 15. Uses [`DataLayer::filled`] which avoids
/// heap allocation (lazy DataLayer stores only the fill value).
///
/// Returns the world Y coordinate above which all blocks have been filled.
/// The per-block vertical pass should start just below this Y.
fn bulk_fill_empty_sections(
    chunk: &mut LevelChunk,
    highest_source_y: i32,
    min_y: i32,
    max_y: i32,
) -> i32 {
    // If no columns have sky sources (e.g., all-air chunk), fill everything.
    let threshold = if highest_source_y == i32::MIN {
        min_y
    } else {
        highest_source_y
    };

    // Find the first section index whose min Y >= threshold.
    let section_count = chunk.section_count();
    let mut first_bulk_section = section_count;
    for si in 0..section_count {
        let section_min_y = min_y + (si as i32 * 16);
        if section_min_y >= threshold {
            first_bulk_section = si;
            break;
        }
    }

    // Bulk-fill all sections from first_bulk_section to the top.
    for si in first_bulk_section..section_count {
        let light_idx = si + 1; // +1 for below-world border section
        chunk.set_sky_light(light_idx, DataLayer::filled(15));
    }

    if first_bulk_section < section_count {
        min_y + (first_bulk_section as i32 * 16)
    } else {
        max_y
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use oxidized_chunks::LevelChunk;
    use oxidized_chunks::level_chunk::OVERWORLD_MIN_Y;
    use oxidized_registry::{BEDROCK, DIRT, GRASS_BLOCK};
    use oxidized_types::ChunkPos;

    /// Creates a standard flat world chunk: bedrock, 2 dirt, grass, air above.
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
    fn test_flat_world_sky_light_above_surface() {
        let mut chunk = flat_chunk();
        initialize_sky_light(&mut chunk);

        // One block above surface (y = -60).
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 4, 8), 15);
        // High up (y = 100).
        assert_eq!(chunk.get_sky_light_at(8, 100, 8), 15);
    }

    #[test]
    fn test_flat_world_sky_light_at_surface() {
        let mut chunk = flat_chunk();
        initialize_sky_light(&mut chunk);

        // The grass block at y=-61 is opaque — sky light should be 0 inside it.
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 3, 8), 0);
    }

    #[test]
    fn test_flat_world_sky_light_all_zero_below_solid() {
        let mut chunk = flat_chunk();
        initialize_sky_light(&mut chunk);

        // Below the surface layer, everything is opaque. Light should be 0.
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 2, 8), 0);
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 1, 8), 0);
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y, 8), 0);
    }

    #[test]
    fn test_shaft_propagation() {
        let mut chunk = flat_chunk();
        // Dig a 1x1 shaft through all 4 solid layers at (8, 8).
        for y in OVERWORLD_MIN_Y..OVERWORLD_MIN_Y + 4 {
            chunk.set_block_state(8, y, 8, 0).unwrap(); // air
        }
        initialize_sky_light(&mut chunk);

        // Sky light should propagate down the shaft — sources detect the
        // all-air column and set source Y to world bottom.
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 3, 8), 15);
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 2, 8), 15);
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y + 1, 8), 15);
        assert_eq!(chunk.get_sky_light_at(8, OVERWORLD_MIN_Y, 8), 15);
    }

    #[test]
    fn test_sources_stored_in_chunk_after_init() {
        let mut chunk = flat_chunk();
        initialize_sky_light(&mut chunk);

        let sources = chunk.sky_light_sources().expect("sources should be set");
        assert_eq!(sources.get_lowest_source_y(8, 8), OVERWORLD_MIN_Y + 4);
    }

    #[test]
    fn test_empty_air_section_above_surface_gets_bulk_filled() {
        let mut chunk = flat_chunk();
        initialize_sky_light(&mut chunk);

        // Section 1 (y = -48 to -33) is entirely air above the surface
        // (surface is at y = -60). It should have uniform sky light 15.
        for x in 0..16 {
            for z in 0..16 {
                for y in (OVERWORLD_MIN_Y + 16)..(OVERWORLD_MIN_Y + 32) {
                    assert_eq!(
                        chunk.get_sky_light_at(x, y, z),
                        15,
                        "sky light at ({x}, {y}, {z}) should be 15"
                    );
                }
            }
        }

        // High section (y = 304–319, section index 23) should also be 15.
        assert_eq!(chunk.get_sky_light_at(8, 310, 8), 15);
    }

    #[test]
    fn test_bulk_filled_sections_use_lazy_datalayer() {
        let mut chunk = flat_chunk();
        initialize_sky_light(&mut chunk);

        // Flat world: surface at y = -60, section 0 = y[-64, -48).
        // Section 1 (y[-48, -32)) is entirely above the surface.
        // Its sky light DataLayer should be lazy (not materialized).
        let light_idx = 2; // section 1 → light index 2 (1 + 1 for border)
        let layer = chunk.sky_light(light_idx).expect("sky light should exist");
        assert!(
            layer.is_definitely_filled_with(15),
            "bulk-filled sections should use lazy DataLayer::filled(15)"
        );
    }
}
