//! # Recorder
//!
//! This module defines the recording layer for optimization experiments.
//! A [`Recorder`] saves the different [`Solutions`](crate::Solution), [`Outcomes`](crate::objective::Outcome),
//! metadata from [`OptInfo`](crate::OptInfo) and [`SolInfo`](crate::SolInfo),
//! and [`Codomain`](crate::Codomain) elements.
//!
//! ## Overview
//!
//! Tantale provides concrete recorders such as [`CSVRecorder`] as well as a no-op recorder
//! [`NoSaver`] for experiments where persistence is not required.
//!
//! ## Recorder Traits
//!
//! Tantale provides specialized recorder traits based on the experiment type:
//!
//! ### Single-Process Experiments
//! - [`Recorder`] - Base marker trait for all recorder implementations
//! - [`SeqRecorder`] - For single optimizers (one solution at a time)
//! - [`BatchRecorder`] - For batch optimizers (multiple solutions per iteration)
//!
//! ### Distributed Experiments (MPI)
//! When the `mpi` feature is enabled:
//! - [`DistSeqRecorder`] - For distributed single experiments
//! - [`DistBatchRecorder`] - For distributed batch experiments
//!
//! ## Usage
//!
//! Here a [`CSVRecorder`] is used with a [`FolderConfig`](crate::FolderConfig).
//! ```ignore
//! // Scenario 1: Starting a new optimization
//! let mut recorder = CSVRecorder::new(config)?;
//! recorder.init(); // Done inside the Runable
//! // ... optimization loop with periodic checkpointing ...
//! recorder.save(&computed_solution, &(solution_id, outcome), &searchspace, &codomain, Some(info));
//! ```

use std::path::PathBuf;

use crate::{
    BatchOptimizer, Domain, FuncWrapper, HasVariables, Optimizer, Problem, Runable, SingleOptimizer, SolInfo, 
    objective::Outcome, searchspace::Searchspace, solution::Id,
};

pub mod csv;
pub use csv::CSVRecorder;

/// Marker trait for recorder implementations used by the experiment runner to save evaluation results.
///
/// A [`Recorder`] is invoked by the experiment runner to persist evaluation results. It receives
/// computed solutions and outcomes as they are produced by the optimization loop. Implementations
/// can store results in files (e.g., CSV), databases, or other storage backends.
///
/// This is a base trait that all recorders must implement. The actual recording functionality
/// is provided by specialized traits like [`SeqRecorder`] and [`BatchRecorder`].
///
/// # See Also
///
/// - [`SeqRecorder`] - Sequential experiment recorder trait
/// - [`BatchRecorder`] - Batch experiment recorder trait  
/// - [`CSVRecorder`] - File-based CSV recorder implementation
/// - [`NoSaver`] - No-op recorder for experiments without persistence
pub trait Recorder<R: Runable> {
    /// Initialize recorder storage for a new single experiment.
    ///
    /// This method should create any required files, headers, or database tables.
    /// It is called once before the optimization loop starts.
    ///
    /// # Parameters
    ///
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn init(&mut self, runable: &R);

    /// Prepare the recorder for resuming from a [`load`](crate::load)ed single experiment.
    ///
    /// # Parameters
    ///
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn after_load(&mut self, ctx: &R::Context);

    /// Record a single evaluated solution and its outcome.
    ///
    /// This method is called after each solution evaluation to persist the result.
    ///
    /// # Parameters
    ///
    /// * `computed` - The fully computed [`Solution`](crate::Solution) with all metadata
    /// * `outputed` - Tuple of solution [`Id`] and [`Outcome`]
    /// * `scp` - [`Searchspace`] used to interpret the solution
    /// * `cod` - [`Codomain`](crate::Codomain) used to interpret the outcome
    fn record<T: Tabular<H>, H>(&self, object: &T, header: &H, file: PathBuf);

    /// Same as [`record`](Recorder::record) but for a vector of [`Tabular`] objects.
    fn vec_record<T: Tabular<H>, H>(&self, objects: &[T], header: &H, file: PathBuf);
}