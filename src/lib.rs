//! Lighting engine for sky light and block light propagation.
//!
//! Implements batched BFS with parallel section processing per ADR-017.
//!
//! # Architecture
//!
//! - [`queue::LightUpdateQueue`] accumulates pending light changes during a tick.
//! - [`engine::LightEngine`] processes the queue, propagating light via BFS.
//! - [`propagation`] contains the core BFS increase/decrease algorithms.
//! - [`sky`] initializes sky light for newly generated chunks.
//! - [`block_light`] initializes block light for newly generated chunks.
//! - [`cross_chunk`] handles light that crosses chunk boundaries.
//! - [`parallel`] provides rayon-based parallel chunk lighting for worldgen.

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod block_light;
pub mod cross_chunk;
pub mod engine;
pub mod occlusion;
pub mod parallel;
pub mod propagation;
pub mod queue;
pub mod sky;
pub mod world_lighting;
