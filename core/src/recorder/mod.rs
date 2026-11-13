//! Recording interfaces for experiment results.
//!
//! [`Recorder`] defines initialization and record/vec-record operations.
//! [`CSVRecorder`] is the file-based implementation provided by this module.
//! Recorder targets select whether a call writes objective inputs, optimizer
//! inputs, codomain values, outcomes, or metadata. See the crate-level
//! documentation for recorder setup and usage examples.

use crate::{
    Problem,
    errors::RecorderError,
    utils::record::{Recordable, Schema},
};

pub mod csv;
pub use csv::CSVRecorder;

/// Marker enum for selecting which data to record in an experiment.
/// This enum is used to specify the target of recording operations in an experiment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecorderTargets {
    /// The [raw](crate::HasX::X) representation of the solution, as used by the objective function.
    XObjective,
    /// The [raw](crate::HasX::X) representation of the solution, as used by the optimizer.
    XOptimizer,
    /// The [`TypeCodom`](crate::Codomain::TypeCodom) part of the solution.
    Y,
    /// The [`Outcome`](crate::Outcome) associated to a solution..
    Outcome,
    /// The [`Metadata`](crate::Solution::MetaData) associated to a solution.
    Metadata,
    /// A custom string-based target for recording other types of data.
    Other(String),
}

/// Interface for persisting experiment records.
///
/// Experiment runners call this trait to initialize storage and record data.
/// [`CSVRecorder`] is the built-in implementation.
/// The `Context` type parameter allows passing additional information to the recorder, such as experiment configuration or metadata.
pub trait Recorder<P: Problem, Context> {
    /// Initialize recorder storage for a new experiment.
    ///
    /// This method should create any required files, or database.
    /// It is called once before the optimization loop starts.
    fn init(&mut self, problem: &P, context: &Context) -> Result<(), RecorderError>;

    /// Prepare the recorder for resuming from a checkpoint.
    fn before_load(&mut self, problem: &P, context: &Context) -> Result<(), RecorderError>;

    /// Prepare the recorder after resuming from a checkpoint.
    fn after_load(&mut self, problem: &P, context: &Context) -> Result<(), RecorderError>;

    /// Record a evaluated solution and its outcome.
    ///
    /// This method is called after each solution evaluation to persist the result.
    /// The context parameters provides additional information for the [`Recordable`] (`CR`)
    /// and [`Schema`] (`CS`) implementations.
    fn record<T, CR, CS>(
        &self,
        target: &RecorderTargets,
        object: &T,
        contexts: (&CR, &CS),
        part: <T as Recordable<CR>>::Parts,
    ) -> Result<(), RecorderError>
    where
        T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>;

    /// Same as [`record`](Recorder::record) but for a vector of [`Recordable`] objects.
    fn vec_record<I, T, CR, CS>(
        &self,
        target: &RecorderTargets,
        object: I,
        contexts: (&CR, &CS),
        part: <T as Recordable<CR>>::Parts,
    ) -> Result<(), RecorderError>
    where
        T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>,
        I: IntoIterator<Item = T>;
}
