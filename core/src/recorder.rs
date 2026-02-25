//! # Recorder
//!
//! This module defines the recording layer for optimization experiments.
//! A [`Recorder`] saves the different [`Solutions`](crate::Solution), [`Outcomes`](crate::objective::Outcome),
//! meta-data from [`OptInfo`](crate::OptInfo) and [`SolInfo`](crate::SolInfo), [`Codomain`](crate::Codomain) elements...
//!
//! ## Overview
//!
//! Tantale provides concrete recorders such as [`CSVRecorder`] as well as a no-op recorder
//! [`NoSaver`] for experiments where persistence is not required.
//!
//! Tantale provide two traits:
//! - [`Recorder`] - Base trait for single-process experiments
//! - [`DistRecorder`] - Distributed variant for MPI-based experiments (enabled with `mpi` feature)
//!
//! ## Usage
//!
//! Here a [`CSVRecorder`] is used with a [`FolderConfig`](crate::FolderConfig).
//! ```ignore
//! // Scenario 1: Starting a new optimization
//! let mut recorder = CSVRecorder::new(config)?;
//! recorder.init(); // Done inside the Runable
//! // ... optimization loop with periodic checkpointing ...
//! recorder.save_pair(&computed_solution, &(solution_id, outcome), &searchspace, &codomain, Some(info));
//! ```
use std::sync::Arc;

use crate::{
    Optimizer,
    domain::onto::LinkOpt,
    objective::Outcome,
    optimizer::opt::CompBatch,
    searchspace::{CompShape, Searchspace},
    solution::{HasY, Id, OutBatch, SolutionShape, Uncomputed},
};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::utils::MPIProcess;

pub mod csv;
pub use csv::{CSVRecorder, CSVWritable};

pub mod nosaver;
pub use nosaver::NoSaver;

/// Base recorder trait for saving evaluated solutions and outcomes.
///
/// A [`Recorder`] is invoked by the experiment runner to persist evaluation results. It receives
/// computed solutions and outcomes as they are produced by the optimization loop. Implementations
/// can store results in files (e.g. CSV)
///
/// # Lifecycle
///
/// Recorders follow a simple lifecycle:
/// - [`init`](Recorder::init) is called for a new experiment
/// - [`after_load`](Recorder::after_load) is called when resuming an experiment
/// - [`save_pair`](Recorder::save_pair) saves a single evaluated [`SolutionShape`]
/// - [`save_batch`](Recorder::save_batch) saves a [`Batch`](crate::Batch) of evaluated solutions
///
/// # See Also
///
/// - [`CSVRecorder`] - File-based CSV recorder implementation
/// - [`NoSaver`] - No-op recorder
/// - [`DistRecorder`] - MPI distributed recorder
pub trait Recorder<PSol, SolId, Out, Scp, Op>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
{
    /// Initialize recorder storage for a new experiment.
    ///
    /// This method should create any required files, headers, or database tables.
    /// It is called once before the optimization loop starts.
    ///
    /// # Parameters
    ///
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn init(&mut self, scp: &Scp, cod: &Op::Cod);

    /// Prepare the recorder for resuming from a [`load`](crate::load)ed experiment.
    ///
    /// # Parameters
    ///
    /// * `scp` - Searchspace describing the solution structure
    /// * `cod` - Codomain describing objective outputs
    fn after_load(&mut self, scp: &Scp, cod: &Op::Cod);

    /// Save a single evaluated solution and its outcome..
    ///
    /// # Parameters
    ///
    /// * `computed` - The fully computed solution
    /// * `outputed` - Tuple of solution [`Id`] and [`Outcome`](crate::objective::Outcome)
    /// * `scp` - [`Searchspace`] used to interpret the solution
    /// * `cod` - [`Codomain`](crate::Codomain) used to interpret the outcome
    /// * `info` - Optional [`OptInfo`](crate::OptInfo) metadata associated with the evaluation
    fn save_pair(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
        info: Option<Arc<Op::Info>>,
    );

    /// Save a batch of evaluated [`Solution`](crate::Solution)s and [`Outcome`](crate::objective::Outcome)s.
    ///
    /// # Parameters
    ///
    /// * `computed` - The computed [`Batch`](crate::Batch) with optimizer metadata
    /// * `outputed` - The output [`Batch`](crate::Batch) with outcomes
    /// * `scp` - [`Searchspace`] used to interpret the solutions
    /// * `cod` - [`Codomain`](crate::Codomain) used to interpret the outcomes
    fn save_batch(
        &self,
        computed: &CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
        outputed: &OutBatch<SolId, Op::Info, Out>,
        scp: &Scp,
        cod: &Op::Cod,
    );
}

#[cfg(feature = "mpi")]
/// Distributed recorder trait for MPI-based experiments.
///
/// [`DistRecorder`] extends [`Recorder`] and is used when optimization is MPI-distributed across multiple MPI-process.
///
/// # MPI Context
///
/// The `proc: &MPIProcess` parameter provides the rank and communicator needed to
/// manage distributed recorders.
///
/// # See Also
///
/// - [`Recorder`] - Base recorder trait
/// - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) - MPI process context
pub trait DistRecorder<PSol, SolId, Out, Scp, Op>: Recorder<PSol, SolId, Out, Scp, Op>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
{
    /// Similar to [`init`](Recorder::init) but with MPI context for distributed experiments.
    fn init_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);

    /// SImilar to [`after_load`](Recorder::after_load) but with MPI context for distributed experiments.
    fn after_load_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);

    /// Similar to [`save_pair`](Recorder::save_pair) but with MPI context for distributed experiments.
    fn save_pair_dist(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
        info: Option<Arc<Op::Info>>,
    );

    /// Similar to [`save_batch`](Recorder::save_batch) but with MPI context for distributed experiments.
    fn save_batch_dist(
        &self,
        computed: &CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
        outputed: &OutBatch<SolId, Op::Info, Out>,
        scp: &Scp,
        cod: &Op::Cod,
    );
}
