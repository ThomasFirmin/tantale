//! # Optimizers
//!
//! This module provides the public optimizer abstractions and re-exports the core optimizer
//! traits and types from the [`opt`] submodule.
//!
//! The [`Optimizer`] combined with [`Objective`](crate::Objective) and [`Searchspace`](crate::Searchspace) defines the core optimization loop:
//!
//! $$ x^\star \in \arg\min_{x \in \mathcal{X}} f(x)\enspace, $$
//! where:
//! * $x^\star$ is the optimal [`Solution`](crate::Solution), or the best-solution
//!   so-far found by the [`Optimizer`](crate::Optimizer)
//! * $\mathcal{X}$ is the [`Searchspace`](crate::Searchspace)
//! * $f$ is the user-define function wrapped in [`Objective`](crate::Objective) or [`Stepped`](crate::Stepped)
//!
//! ## Overview
//!
//! Optimizers are responsible for proposing new candidate solutions and updating their
//! internal state based on evaluated results. Tantale distinguishes between two execution
//! styles:
//!
//! - **Sequential optimization**: generates one solution at a time
//! - **Batch optimization**: generates a batch of solutions at each step
//!
//! These styles are captured by the [`SequentialOptimizer`] and [`BatchOptimizer`] traits.
//!
//! ## Core Traits
//!
//! - [`Optimizer`] - Base trait implemented by all optimizers
//! - [`SequentialOptimizer`] - Optimizers that produce a single solution per step
//! - [`BatchOptimizer`] - Optimizers that produce a batch of solutions per step
//!
//! ## State and Metadata
//!
//! - [`OptState`] - Serializable optimizer state used for checkpointing
//! - [`OptInfo`] - Optional metadata produced by an optimizer (e.g., diagnostics)
//! - [`EmptyInfo`] - Default empty metadata type
//!
//! ## Integration Points
//!
//! Optimizers integrate with:
//! - [`Searchspace`](crate::searchspace::Searchspace) for solution types
//! - [`Objective`](crate::objective::Objective) for evaluation
//! - [`Checkpointer`](crate::Checkpointer) for state persistence
//! - [`Runable`](crate::experiment::Runable) for execution control
//!
//! See the [`opt`] submodule for trait definitions and details.
pub mod opt;
pub use opt::{BatchOptimizer, EmptyInfo, OptInfo, OptState, Optimizer, SequentialOptimizer};
