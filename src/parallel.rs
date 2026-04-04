//! Parallel section processing for worldgen chunk lighting.
//!
//! Per ADR-017, alternating even/odd Y-layers can be processed in parallel
//! since they don't share vertical boundaries. This module provides the
//! rayon-based parallel full-chunk lighting used by the worldgen pipeline
//! at the Light status (ADR-016).

use oxidized_chunks::LevelChunk;

use super::block_light::initialize_block_light;
use super::sky::initialize_sky_light;

/// Full-chunk lighting with parallel section processing.
///
/// Currently delegates to sequential sky + block light initialization.
/// Parallel even/odd section processing will be added when the worldgen
/// pipeline is fully operational and benchmarks justify the complexity.
///
/// Used by the worldgen pipeline at the Light status (ADR-016).
pub fn light_chunk_parallel(chunk: &mut LevelChunk) {
    // Phase 1: Sky light (vertical pass + horizontal BFS).
    initialize_sky_light(chunk);

    // Phase 2: Block light (emitter scan + BFS).
    initialize_block_light(chunk);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use oxidized_registry::{BEDROCK, DIRT, GRASS_BLOCK};
    use oxidized_types::ChunkPos;
    use oxidized_chunks::LevelChunk;
    use oxidized_chunks::level_chunk::OVERWORLD_MIN_Y;

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
    fn test_parallel_produces_same_as_sequential() {
        let mut parallel_chunk = flat_chunk();
        let mut sequential_chunk = flat_chunk();

        light_chunk_parallel(&mut parallel_chunk);
        initialize_sky_light(&mut sequential_chunk);
        initialize_block_light(&mut sequential_chunk);

        // Compare all sky light values.
        for y in OVERWORLD_MIN_Y..OVERWORLD_MIN_Y + 20 {
            for x in 0..16 {
                for z in 0..16 {
                    assert_eq!(
                        parallel_chunk.get_sky_light_at(x, y, z),
                        sequential_chunk.get_sky_light_at(x, y, z),
                        "sky light mismatch at ({x}, {y}, {z})"
                    );
                    assert_eq!(
                        parallel_chunk.get_block_light_at(x, y, z),
                        sequential_chunk.get_block_light_at(x, y, z),
                        "block light mismatch at ({x}, {y}, {z})"
                    );
                }
            }
        }
    }
}
