//! Stop criteria for an experiment.
//!
//! This module defines the [`Stop`] trait used by a `Runable` to decide when an
//! optimization should terminate. Implementations receive updates via
//! [`Stop::update`], and return their state through [`Stop::stop`].
//!
//! The [`ExpStep`] enum tells a stop criterion whether an evaluation, iteration,
//! or sampling event has occurred.

use crate::{objective::Step, utils::has_trait::HasState};
use serde::{Deserialize, Serialize};

/// Experiment step used to update a [`Stop`] implementation.
#[derive(Debug, Clone, Copy)]
pub enum ExpStep {
    /// During evaluation of a batch or a single solution.
    Evaluation(Step),
    /// At each full optimization loop.
    Iteration,
    /// After the sampling step.
    Sampling,
}

/// Stop criterion for an experiment.
///
/// Implementations should keep any state required to decide termination. The
/// experiment calls [`Stop::init`] before the first update, then repeatedly
/// calls [`Stop::update`] followed by [`Stop::stop`] to check completion.
pub trait Stop: HasState
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
