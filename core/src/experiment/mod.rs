//! Experiment orchestration.
//!
//! This module provides runners that execute a [`Problem`] with a [`Sampler`]
//! until its [`Stop`] criterion is met. Runners also coordinate optional
//! recording, checkpointing, result accumulation, and evaluation state.
//!
//! [`Experiment`] runs a problem in one process. [`ParExperiment`] supports
//! parallel execution through the communication abstractions in
//! [`parallel`]; thread communication is available by default, while MPI
//! communication types require the `mpi` feature.
//!
//! The [`Runable`] trait exposes the common execution interface. Its `run`
//! method consumes the runner and returns the accumulator of evaluated
//! solutions. For supported problem construction and execution examples, see
//! the crate-level documentation.
//!
//! [`PoolMode`] selects how multi-step function state is managed. The
//! [`ExpStep`](crate::stop::ExpStep) event type lets stop criteria observe
//! evaluations, iterations, and sampling events.

use crate::{
    Problem,
    aliases::{Accum, Opt},
    checkpointer::Checkpointer,
    optimizer::Sampler,
    recorder::Recorder,
    solution::SolId,
    stop::Stop,
};

// BASICS
pub mod basic;
pub use basic::Experiment;

pub mod exputils;
pub use exputils::PoolMode;

pub mod parallel;
pub use parallel::{ParExperiment, ThreadChannel, ThreadCommunicator};

#[cfg(feature = "mpi")]
pub use parallel::{MPIChannel, MPICommunicator};

/// Common execution interface implemented by concrete experiment runners.
///
/// A runner coordinates problem sampling, evaluation, result accumulation,
/// optional recording and checkpointing, and stop-criterion checks. The core
/// runner types are [`Experiment`] and [`ParExperiment`].
///
/// [`run`](Runable::run) consumes the runner and returns its accumulated
/// results. See the crate-level documentation for complete, tested usage
/// examples.
pub trait Runable: Sized {
    type Id: SolId;
    type Problem: Problem<Id = Self::Id>;
    type Sampler: Sampler<Self::Problem, Opt<Self::Problem>, Info = <Self::Problem as Problem>::Info>;
    type Stop: Stop;
    type Recorder: Recorder<Self::Problem, Self::Context>;
    type Checkpointer: Checkpointer<Self::Problem, Self::Context>;
    type Context;

    /// Runs the optimization loop to completion.
    ///
    /// Executes the full pipeline, generation, evaluation, accumulation, recording,
    /// checkpointing, and stop-criterion checks, until the [`Stop`] criterion is met.
    ///
    /// Returns the accumulated results found during the run.
    fn run(self) -> Accum<Self::Problem>;
}
