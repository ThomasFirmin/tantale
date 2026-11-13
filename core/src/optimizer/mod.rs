//! Sampling interfaces for optimization algorithms.
//!
//! The [`Sampler`] trait proposes candidate solutions for a [`Problem`](crate::Problem),
//! receives evaluated candidates, and maintains the state needed by an
//! experiment. [`Batched`](opt::Batched) extends it with batch operations.
//! This crate defines these interfaces; ready-made algorithms are provided
//! by companion crates.
//! 
//! The [`Prunable`] and [`Pruner`] interfaces support pruning candidate solutions.
//!
//! See the crate-level documentation for how samplers interact with problems
//! and experiment runners.

pub mod opt;
pub use opt::{EmptyInfo, Prunable, Pruner, Sampler};
