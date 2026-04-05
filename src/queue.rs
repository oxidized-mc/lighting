//! Batched light update queue.
//!
//! Accumulates block changes that affect lighting during a tick, then feeds
//! them to [`super::engine::LightEngine::process_updates`] in a single batch.

use oxidized_mc_types::BlockPos;

/// A queue of pending light updates for the current tick.
///
/// Block changes that affect light emission or opacity are pushed here during
/// the tick, then processed in bulk by the lighting engine at tick end.
///
/// # Examples
///
/// ```
/// use oxidized_lighting::queue::{LightUpdateQueue, LightUpdate};
/// use oxidized_mc_types::BlockPos;
///
/// let mut queue = LightUpdateQueue::new();
/// assert!(queue.is_empty());
///
/// queue.push(LightUpdate {
///     pos: BlockPos::new(0, 64, 0),
///     old_emission: 0,
///     new_emission: 14,
///     old_opacity: 0,
///     new_opacity: 0,
/// });
/// assert_eq!(queue.len(), 1);
/// ```
#[derive(Debug, Default)]
pub struct LightUpdateQueue {
    pending: Vec<LightUpdate>,
}

impl LightUpdateQueue {
    /// Creates an empty update queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Adds a light update to the queue.
    pub fn push(&mut self, update: LightUpdate) {
        self.pending.push(update);
    }

    /// Returns the number of pending updates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Returns `true` if the queue has no pending updates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Drains all pending updates, returning them as a `Vec`.
    pub fn drain(&mut self) -> Vec<LightUpdate> {
        std::mem::take(&mut self.pending)
    }

    /// Removes all pending updates without returning them.
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

/// A single light update caused by a block change.
///
/// Records both the old and new emission/opacity so the engine can perform
/// the decrease-then-increase BFS passes used by the lighting engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightUpdate {
    /// World position of the changed block.
    pub pos: BlockPos,
    /// Light emission of the old block state (0–15).
    pub old_emission: u8,
    /// Light emission of the new block state (0–15).
    pub new_emission: u8,
    /// Light opacity of the old block state (0–15).
    pub old_opacity: u8,
    /// Light opacity of the new block state (0–15).
    pub new_opacity: u8,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_new_is_empty() {
        let queue = LightUpdateQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_queue_push_and_len() {
        let mut queue = LightUpdateQueue::new();
        queue.push(LightUpdate {
            pos: BlockPos::new(10, 64, 10),
            old_emission: 0,
            new_emission: 14,
            old_opacity: 0,
            new_opacity: 0,
        });
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());
    }

    #[test]
    fn test_queue_drain() {
        let mut queue = LightUpdateQueue::new();
        queue.push(LightUpdate {
            pos: BlockPos::new(0, 0, 0),
            old_emission: 15,
            new_emission: 0,
            old_opacity: 0,
            new_opacity: 15,
        });
        queue.push(LightUpdate {
            pos: BlockPos::new(1, 1, 1),
            old_emission: 0,
            new_emission: 14,
            old_opacity: 0,
            new_opacity: 0,
        });
        let updates = queue.drain();
        assert_eq!(updates.len(), 2);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_clear() {
        let mut queue = LightUpdateQueue::new();
        queue.push(LightUpdate {
            pos: BlockPos::new(5, 5, 5),
            old_emission: 0,
            new_emission: 12,
            old_opacity: 0,
            new_opacity: 0,
        });
        queue.clear();
        assert!(queue.is_empty());
    }
}
