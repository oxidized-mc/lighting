//! Cross-chunk light propagation.
//!
//! When BFS reaches x=0/15 or z=0/15 within a chunk, the light must continue
//! into the neighboring chunk. This module handles that boundary propagation.

use std::collections::VecDeque;

use oxidized_registry::BlockStateId;
use oxidized_chunks::LevelChunk;

use super::propagation::{
    BoundaryEntry, LightEntry, propagate_block_light_increase, propagate_sky_light_increase,
};

/// Accessor for the 4 horizontal neighbors of a chunk.
pub struct ChunkNeighbors<'a> {
    /// -Z neighbor.
    pub north: Option<&'a mut LevelChunk>,
    /// +Z neighbor.
    pub south: Option<&'a mut LevelChunk>,
    /// +X neighbor.
    pub east: Option<&'a mut LevelChunk>,
    /// -X neighbor.
    pub west: Option<&'a mut LevelChunk>,
}

/// Propagates block light boundary entries into neighboring chunks.
///
/// For each boundary entry, determines which neighbor chunk owns that
/// position, reads the target block's opacity, and runs BFS increase.
/// Returns any further boundary entries produced by BFS in neighbor chunks.
///
/// `center_chunk_x`/`center_chunk_z` are the chunk coordinates of the
/// chunk that produced the boundary entries.
pub fn propagate_block_light_cross_chunk(
    neighbors: &mut ChunkNeighbors<'_>,
    boundary_entries: &[BoundaryEntry],
    center_chunk_x: i32,
    center_chunk_z: i32,
) -> Vec<BoundaryEntry> {
    let mut further = Vec::new();
    for entry in boundary_entries {
        further.extend(propagate_boundary_block_light(
            neighbors,
            entry,
            center_chunk_x,
            center_chunk_z,
        ));
    }
    further
}

/// Propagates sky light boundary entries into neighboring chunks.
/// Returns any further boundary entries produced by BFS in neighbor chunks.
pub fn propagate_sky_light_cross_chunk(
    neighbors: &mut ChunkNeighbors<'_>,
    boundary_entries: &[BoundaryEntry],
    center_chunk_x: i32,
    center_chunk_z: i32,
) -> Vec<BoundaryEntry> {
    let mut further = Vec::new();
    for entry in boundary_entries {
        further.extend(propagate_boundary_sky_light(
            neighbors,
            entry,
            center_chunk_x,
            center_chunk_z,
        ));
    }
    further
}

fn propagate_boundary_block_light(
    neighbors: &mut ChunkNeighbors<'_>,
    entry: &BoundaryEntry,
    center_chunk_x: i32,
    center_chunk_z: i32,
) -> Vec<BoundaryEntry> {
    let (chunk, local_x, local_z, base_x, base_z) = match resolve_neighbor(
        neighbors,
        entry.world_x,
        entry.world_z,
        center_chunk_x,
        center_chunk_z,
    ) {
        Some(v) => v,
        None => return Vec::new(),
    };

    // Attenuate by target block's opacity (not pre-attenuated).
    let state_id = chunk
        .get_block_state(local_x, entry.world_y, local_z)
        .unwrap_or(0);
    #[allow(clippy::cast_possible_truncation)]
    let opacity = BlockStateId(state_id as u16).light_opacity();
    let attenuated = entry.level.saturating_sub(opacity.max(1));
    if attenuated == 0 {
        return Vec::new();
    }

    let current = chunk.get_block_light_at(local_x, entry.world_y, local_z);
    if attenuated > current {
        chunk.set_block_light_at(local_x, entry.world_y, local_z, attenuated);
        let mut queue = VecDeque::new();
        queue.push_back(LightEntry {
            x: local_x,
            y: entry.world_y,
            z: local_z,
            level: attenuated,
            directions: entry.directions,
        });
        propagate_block_light_increase(chunk, &mut queue, base_x, base_z)
    } else {
        Vec::new()
    }
}

fn propagate_boundary_sky_light(
    neighbors: &mut ChunkNeighbors<'_>,
    entry: &BoundaryEntry,
    center_chunk_x: i32,
    center_chunk_z: i32,
) -> Vec<BoundaryEntry> {
    let (chunk, local_x, local_z, base_x, base_z) = match resolve_neighbor(
        neighbors,
        entry.world_x,
        entry.world_z,
        center_chunk_x,
        center_chunk_z,
    ) {
        Some(v) => v,
        None => return Vec::new(),
    };

    // Attenuate by target block's opacity.
    let state_id = chunk
        .get_block_state(local_x, entry.world_y, local_z)
        .unwrap_or(0);
    #[allow(clippy::cast_possible_truncation)]
    let opacity = BlockStateId(state_id as u16).light_opacity();
    let attenuated = entry.level.saturating_sub(opacity.max(1));
    if attenuated == 0 {
        return Vec::new();
    }

    let current = chunk.get_sky_light_at(local_x, entry.world_y, local_z);
    if attenuated > current {
        chunk.set_sky_light_at(local_x, entry.world_y, local_z, attenuated);
        let mut queue = VecDeque::new();
        queue.push_back(LightEntry {
            x: local_x,
            y: entry.world_y,
            z: local_z,
            level: attenuated,
            directions: entry.directions,
        });
        propagate_sky_light_increase(chunk, &mut queue, base_x, base_z)
    } else {
        Vec::new()
    }
}

/// Resolves which neighbor chunk owns a world position and returns
/// (chunk_ref, local_x, local_z, chunk_base_x, chunk_base_z).
fn resolve_neighbor<'a>(
    neighbors: &'a mut ChunkNeighbors<'_>,
    world_x: i32,
    world_z: i32,
    center_chunk_x: i32,
    center_chunk_z: i32,
) -> Option<(&'a mut LevelChunk, i32, i32, i32, i32)> {
    let chunk_x = world_x.div_euclid(16);
    let chunk_z = world_z.div_euclid(16);
    let local_x = world_x.rem_euclid(16);
    let local_z = world_z.rem_euclid(16);

    // Compute relative offset from center chunk.
    let rel_x = chunk_x - center_chunk_x;
    let rel_z = chunk_z - center_chunk_z;

    let chunk = match (rel_x, rel_z) {
        (1, 0) => neighbors.east.as_deref_mut()?,
        (-1, 0) => neighbors.west.as_deref_mut()?,
        (0, 1) => neighbors.south.as_deref_mut()?,
        (0, -1) => neighbors.north.as_deref_mut()?,
        _ => return None, // Diagonal or same chunk — skip.
    };

    let base_x = chunk_x * 16;
    let base_z = chunk_z * 16;
    Some((chunk, local_x, local_z, base_x, base_z))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::propagation::ALL_DIRECTIONS;
    use oxidized_types::ChunkPos;
    use oxidized_chunks::LevelChunk;

    #[test]
    fn test_missing_neighbor_no_panic() {
        let mut neighbors = ChunkNeighbors {
            north: None,
            south: None,
            east: None,
            west: None,
        };
        let entries = vec![BoundaryEntry {
            world_x: -1,
            world_y: 64,
            world_z: 8,
            level: 13,
            directions: ALL_DIRECTIONS,
        }];
        // Should not panic. Center chunk is (0, 0).
        propagate_block_light_cross_chunk(&mut neighbors, &entries, 0, 0);
    }

    #[test]
    fn test_boundary_propagation_into_east_neighbor() {
        let mut east = LevelChunk::new(ChunkPos::new(1, 0));
        // Source level is 13 (un-attenuated). Air at target → opacity 1 → attenuated to 12.
        let entries = vec![BoundaryEntry {
            world_x: 16, // x=16 → chunk (1,0), local x=0
            world_y: 64,
            world_z: 8,
            level: 13,
            directions: ALL_DIRECTIONS,
        }];

        let mut neighbors = ChunkNeighbors {
            north: None,
            south: None,
            east: Some(&mut east),
            west: None,
        };
        propagate_block_light_cross_chunk(&mut neighbors, &entries, 0, 0);

        // Air has opacity 0, max(1, 0)=1, so 13 - 1 = 12.
        assert_eq!(east.get_block_light_at(0, 64, 8), 12);
        // Should propagate further: 12 - 1 = 11.
        assert_eq!(east.get_block_light_at(1, 64, 8), 11);
    }

    #[test]
    fn test_boundary_propagation_non_origin_chunk() {
        // Center chunk at (5, 3), east neighbor at (6, 3).
        let mut east = LevelChunk::new(ChunkPos::new(6, 3));
        let entries = vec![BoundaryEntry {
            world_x: 96, // 6*16 = 96 → chunk (6,3), local x=0
            world_y: 64,
            world_z: 56, // 3*16 + 8 = 56
            level: 10,
            directions: ALL_DIRECTIONS,
        }];

        let mut neighbors = ChunkNeighbors {
            north: None,
            south: None,
            east: Some(&mut east),
            west: None,
        };
        propagate_block_light_cross_chunk(&mut neighbors, &entries, 5, 3);

        // 10 - max(1, 0) = 9
        assert_eq!(east.get_block_light_at(0, 64, 8), 9);
    }
}
