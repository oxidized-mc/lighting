//! Block light initialization for newly generated chunks.
//!
//! Scans all sections for blocks with `light_emission > 0`, seeds the BFS
//! queue, and propagates outward using the core BFS from `propagation.rs`.

use std::collections::VecDeque;

use oxidized_chunks::LevelChunk;
use oxidized_registry::BlockStateId;

use super::propagation::{
    ALL_DIRECTIONS, BoundaryEntry, LightEntry, propagate_block_light_increase,
};

/// Initializes block light for a newly generated chunk.
///
/// Iterates through all blocks, finds emitters (light_emission > 0), sets
/// their block light to the emission level, and then BFS-propagates outward.
///
/// Returns boundary entries for positions where block light reached a chunk
/// edge and needs to propagate into a neighbor chunk.
pub fn initialize_block_light(chunk: &mut LevelChunk) -> Vec<BoundaryEntry> {
    let min_y = chunk.min_y();
    let section_count = chunk.section_count();
    let mut queue = VecDeque::new();

    // Scan for emitters in all sections.
    for section_idx in 0..section_count {
        let section_base_y = min_y + (section_idx as i32 * 16);

        for local_y in 0..16u32 {
            for local_z in 0..16u32 {
                for local_x in 0..16u32 {
                    let state_id = chunk
                        .section(section_idx)
                        .and_then(|s| {
                            s.get_block_state(local_x as usize, local_y as usize, local_z as usize)
                                .ok()
                        })
                        .unwrap_or(0);

                    #[allow(clippy::cast_possible_truncation)]
                    let emission = BlockStateId(state_id as u16).light_emission();
                    if emission > 0 {
                        let x = local_x as i32;
                        let y = section_base_y + local_y as i32;
                        let z = local_z as i32;

                        chunk.set_block_light_at(x, y, z, emission);
                        queue.push_back(LightEntry {
                            x,
                            y,
                            z,
                            level: emission,
                            directions: ALL_DIRECTIONS,
                        });
                    }
                }
            }
        }
    }

    // BFS propagation from all emitters.
    if !queue.is_empty() {
        let chunk_base_x = chunk.pos.x * 16;
        let chunk_base_z = chunk.pos.z * 16;
        propagate_block_light_increase(chunk, &mut queue, chunk_base_x, chunk_base_z)
    } else {
        Vec::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use oxidized_chunks::LevelChunk;
    use oxidized_registry::BlockRegistry;
    use oxidized_types::ChunkPos;

    fn air_chunk() -> LevelChunk {
        LevelChunk::new(ChunkPos::new(0, 0))
    }

    fn glowstone_id() -> u32 {
        u32::from(
            BlockRegistry
                .default_state("minecraft:glowstone")
                .expect("glowstone missing")
                .0,
        )
    }

    fn torch_id() -> u32 {
        u32::from(
            BlockRegistry
                .default_state("minecraft:torch")
                .expect("torch missing")
                .0,
        )
    }

    fn stone_id() -> u32 {
        u32::from(
            BlockRegistry
                .default_state("minecraft:stone")
                .expect("stone missing")
                .0,
        )
    }

    #[test]
    fn test_no_emitters_no_block_light() {
        let mut chunk = air_chunk();
        initialize_block_light(&mut chunk);
        // All air, no emitters → no block light.
        assert_eq!(chunk.get_block_light_at(8, 64, 8), 0);
    }

    #[test]
    fn test_single_glowstone() {
        let mut chunk = air_chunk();
        chunk.set_block_state(8, 64, 8, glowstone_id()).unwrap();
        initialize_block_light(&mut chunk);

        // Glowstone emits 15.
        assert_eq!(chunk.get_block_light_at(8, 64, 8), 15);
        // Distance 1 = 14.
        assert_eq!(chunk.get_block_light_at(9, 64, 8), 14);
        // Distance 5 = 10.
        assert_eq!(chunk.get_block_light_at(8, 69, 8), 10);
    }

    #[test]
    fn test_single_torch() {
        let mut chunk = air_chunk();
        chunk.set_block_state(8, 64, 8, torch_id()).unwrap();
        initialize_block_light(&mut chunk);

        let emission = oxidized_registry::BlockStateId(torch_id() as u16).light_emission();
        assert_eq!(chunk.get_block_light_at(8, 64, 8), emission);
        if emission > 1 {
            assert_eq!(chunk.get_block_light_at(9, 64, 8), emission - 1);
        }
    }

    #[test]
    fn test_emitter_behind_stone() {
        let mut chunk = air_chunk();
        chunk.set_block_state(8, 64, 8, glowstone_id()).unwrap();
        // Full-height wall of stone at x=9.
        for y in chunk.min_y()..chunk.max_y() {
            for z in 0..16 {
                chunk.set_block_state(9, y, z, stone_id()).unwrap();
            }
        }
        initialize_block_light(&mut chunk);

        // Light should not pass through stone.
        assert_eq!(chunk.get_block_light_at(10, 64, 8), 0);
    }

    #[test]
    fn test_two_emitters_overlap_takes_max() {
        let mut chunk = air_chunk();
        chunk.set_block_state(4, 64, 8, glowstone_id()).unwrap();
        chunk.set_block_state(12, 64, 8, glowstone_id()).unwrap();
        initialize_block_light(&mut chunk);

        // Midpoint (8, 64, 8): distance 4 from both → level 11.
        assert_eq!(chunk.get_block_light_at(8, 64, 8), 11);
    }
}
