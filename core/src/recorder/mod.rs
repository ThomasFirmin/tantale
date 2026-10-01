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
//! [`Recorder`] is the base trait for all recorders.
//! It defines the interface for initializing storage, recording results, and handling resumption from saved states.
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

use crate::{Problem, errors::RecorderError, utils::record::{Recordable, Schema}};

pub mod csv;
pub use csv::CSVRecorder;



#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecorderTargets {
    XObjective,
    XOptimizer,
    Y,
    Outcome,
    Metadata,
    Other(String),
}

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
/// - [`CSVRecorder`] - A concrete implementation that saves results to CSV files.
pub trait Recorder<P: Problem, Context> {
    /// Initialize recorder storage for a new experiment.
    ///
    /// This method should create any required files, or database.
    /// It is called once before the optimization loop starts.
    fn init(&mut self, problem: &P, context: &Context) -> Result<(), RecorderError>;

    /// Prepare the recorder for resuming from a [`load`](crate::load)ed experiment.
    fn after_load(&mut self, problem: &P, context: &Context) -> Result<(), RecorderError>;

    /// Record a evaluated solution and its outcome.
    ///
    /// This method is called after each solution evaluation to persist the result.
    fn record<T, CR, CS>(&self, target: &RecorderTargets, object: &T, contexts: (&CR, &CS), part: <T as Recordable<CR>>::Parts) -> Result<(), RecorderError>
    where
        T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>;

    /// Same as [`record`](Recorder::record) but for a vector of [`Recordable`] objects.
    fn vec_record<I, T, CR, CS>(&self, target: &RecorderTargets, object: I, contexts: (&CR, &CS), part: <T as Recordable<CR>>::Parts) -> Result<(), RecorderError>
    where
        T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>,
        I: IntoIterator<Item = T>;
}