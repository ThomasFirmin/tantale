//! Stop criteria for an experiment.
//!
//! This module defines the [`Stop`] trait used by a `Runable` to decide when an
//! optimization should terminate. Implementations receive updates via
//! [`Stop::update`], and return their state through [`Stop::stop`].
//!
//! The [`ExpStep`] enum provides a minimal signal of where the experiment is in
//! its progression: distribution, iteration, or after a [`step`](crate::BatchOptimizer::step) (see also sequential [`step`](crate::SequentialOptimizer::step)).

use crate::objective::Step;
use serde::{Deserialize, Serialize};

/// Experiment step used to update a [`Stop`] implementation.
#[derive(Debug, Clone, Copy)]
pub enum ExpStep {
    /// Within an [`Evaluate`](crate::experiment::Evaluate), during evaluation of a batch or a single solution.
    Distribution(Step),
    /// At each full optimization loop.
    Iteration,
    /// After the optimization output
    Optimization,
}

/// Stop criterion for an experiment.
///
/// Implementations should keep any state required to decide termination. The
/// experiment calls [`Stop::init`] before the first update, then repeatedly
/// calls [`Stop::update`] followed by [`Stop::stop`] to check completion.
pub trait Stop
where
    Self: Serialize + for<'a> Deserialize<'a>,
{
    /// Initialize internal state before the experiment starts.
    fn init(&mut self);
    /// Return `true` when the experiment should stop.
    fn stop(&self) -> bool;
    /// Update internal state based on experiment step.
    fn update(&mut self, step: ExpStep);
}

pub mod calls;
pub use calls::Calls;

pub mod time;
pub use time::Time;

pub mod evaluated;
pub use evaluated::Evaluated;
