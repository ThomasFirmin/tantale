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
//! - [`SeqRecorder`] - For sequential optimizers (one solution at a time)
//! - [`BatchRecorder`] - For batch optimizers (multiple solutions per iteration)
//!
//! ### Distributed Experiments (MPI)
//! When the `mpi` feature is enabled:
//! - [`DistSeqRecorder`] - For distributed sequential experiments
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

use crate::{
    BatchOptimizer, FuncWrapper, RawObj, SequentialOptimizer,
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
pub trait Recorder {}

/// Recorder trait for sequential optimization experiments.
///
/// [`SeqRecorder`] is used with [`SequentialOptimizer`](crate::SequentialOptimizer)s,
/// which generate and evaluate one solution at a time. Each call to [`save`](SeqRecorder::save)
/// records a single evaluated solution and its associated outcome.
///
/// # Usage
///
/// Implementations must provide three methods:
/// - [`init`](SeqRecorder::init) - Initialize storage for a new experiment
/// - [`after_load`](SeqRecorder::after_load) - Prepare for resuming a loaded experiment
/// - [`save`](SeqRecorder::save) - Record a single evaluated solution
///
/// # See Also
///
/// - [`BatchRecorder`] - For batch optimization experiments
/// - [`DistSeqRecorder`] - For distributed sequential experiments (MPI)
pub trait SeqRecorder<PSol, SolId, Out, Scp, Op, FnWrap>: Recorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    /// Initialize recorder storage for a new sequential experiment.
    ///
    /// This method should create any required files, headers, or database tables.
    /// It is called once before the optimization loop starts.
    ///
    /// # Parameters
    ///
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn init(&mut self, scp: &Scp, cod: &Op::Cod);

    /// Prepare the recorder for resuming from a [`load`](crate::load)ed sequential experiment.
    ///
    /// # Parameters
    ///
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn after_load(&mut self, scp: &Scp, cod: &Op::Cod);

    /// Save a single evaluated solution and its outcome.
    ///
    /// This method is called after each solution evaluation to persist the result.
    ///
    /// # Parameters
    ///
    /// * `computed` - The fully computed [`Solution`](crate::Solution) with all metadata
    /// * `outputed` - Tuple of solution [`Id`] and [`Outcome`](crate::objective::Outcome)
    /// * `scp` - [`Searchspace`] used to interpret the solution
    /// * `cod` - [`Codomain`](crate::Codomain) used to interpret the outcome
    fn save(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
    );
}

/// Recorder trait for batch optimization experiments.
///
/// [`BatchRecorder`] is used with [`BatchOptimizer`](crate::BatchOptimizer)s,
/// which generate and evaluate multiple solutions per iteration. Each call to [`save`](BatchRecorder::save)
/// records a batch of evaluated solutions and their associated outcomes.
///
/// # Usage
///
/// Implementations must provide three methods:
/// - [`init`](BatchRecorder::init) - Initialize storage for a new experiment
/// - [`after_load`](BatchRecorder::after_load) - Prepare for resuming a loaded experiment
/// - [`save`](BatchRecorder::save) - Record a batch of evaluated solutions
///
/// # See Also
///
/// - [`SeqRecorder`] - For sequential optimization experiments
/// - [`DistBatchRecorder`] - For distributed batch experiments (MPI)
pub trait BatchRecorder<PSol, SolId, Out, Scp, Op, FnWrap>: Recorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    /// Initialize recorder storage for a new batched experiment.
    ///
    /// This method should create any required files, headers, or database tables.
    /// It is called once before the optimization loop starts.
    ///
    /// # Parameters
    ///
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn init(&mut self, scp: &Scp, cod: &Op::Cod);

    /// Prepare the recorder for resuming from a [`load`](crate::load)ed batched experiment.
    ///
    /// # Parameters
    ///
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn after_load(&mut self, scp: &Scp, cod: &Op::Cod);

    /// Save a batch of evaluated [`Solution`](crate::Solution)s and [`Outcome`](crate::objective::Outcome)s.
    ///
    /// # Parameters
    ///
    /// * `computed` - The computed [`Batch`](crate::Batch) with optimizer metadata
    /// * `outputed` - The output [`Batch`](crate::Batch) with outcomes
    /// * `scp` - [`Searchspace`] used to interpret the solutions
    /// * `cod` - [`Codomain`](crate::Codomain) used to interpret the outcomes
    fn save(
        &self,
        computed: &CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
        outputed: &OutBatch<SolId, Op::Info, Out>,
        scp: &Scp,
        cod: &Op::Cod,
    );
}

#[cfg(feature = "mpi")]
/// Distributed recorder trait for MPI-based sequential experiments.
///
/// [`DistSeqRecorder`] extends [`Recorder`] for sequential optimizers running in an
/// MPI-distributed environment. It provides the same functionality as [`SeqRecorder`]
/// but with additional MPI context for coordinating recording across multiple processes.
///
/// # MPI Context
///
/// Each method receives a `proc: &MPIProcess` parameter that provides the rank and
/// communicator needed to manage distributed recording, ensuring proper coordination
/// when multiple MPI processes write to shared storage.
///
/// # See Also
///
/// - [`SeqRecorder`] - Single-process sequential recorder
/// - [`DistBatchRecorder`] - Distributed batch recorder
/// - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) - MPI process context
pub trait DistSeqRecorder<PSol, SolId, Out, Scp, Op, FnWrap>: Recorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    /// Initialize recorder storage for a new distributed sequential experiment.
    ///
    /// Similar to [`SeqRecorder::init`] but with MPI process context.
    ///
    /// # Parameters
    ///
    /// * `proc` - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) context for MPI coordination
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn init_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);

    /// Prepare the recorder for resuming from a loaded distributed sequential experiment.
    ///
    /// Similar to [`SeqRecorder::after_load`] but with MPI process context.
    ///
    /// # Parameters
    ///
    /// * `proc` - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) context for MPI coordination
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn after_load_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);

    /// Save a single evaluated solution in a distributed sequential experiment.
    ///
    /// Similar to [`SeqRecorder::save`] but allows MPI-aware implementations to coordinate
    /// writes across processes.
    ///
    /// # Parameters
    ///
    /// * `computed` - The fully computed [`Solution`](crate::Solution) with all metadata
    /// * `outputed` - Tuple of solution [`Id`] and [`Outcome`](crate::objective::Outcome)
    /// * `scp` - [`Searchspace`] used to interpret the solution
    /// * `cod` - [`Codomain`](crate::Codomain) used to interpret the outcome
    fn save_dist(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
    );
}

#[cfg(feature = "mpi")]
/// Distributed recorder trait for MPI-based batch experiments.
///
/// [`DistBatchRecorder`] extends [`Recorder`] for batch optimizers running in an
/// MPI-distributed environment. It provides the same functionality as [`BatchRecorder`]
/// but with additional MPI context for coordinating recording across multiple processes.
///
/// # MPI Context
///
/// Each method receives a `proc: &MPIProcess` parameter that provides the rank and
/// communicator needed to manage distributed recording, ensuring proper coordination
/// when multiple MPI processes write to shared storage.
///
/// # See Also
///
/// - [`BatchRecorder`] - Single-process batch recorder
/// - [`DistSeqRecorder`] - Distributed sequential recorder
/// - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) - MPI process context
pub trait DistBatchRecorder<PSol, SolId, Out, Scp, Op, FnWrap>: Recorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    /// Initialize recorder storage for a new distributed batch experiment.
    ///
    /// Similar to [`BatchRecorder::init`] but with MPI process context.
    ///
    /// # Parameters
    ///
    /// * `proc` - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) context for MPI coordination
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn init_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);

    /// Prepare the recorder for resuming from a loaded distributed batch experiment.
    ///
    /// Similar to [`BatchRecorder::after_load`] but with MPI process context.
    ///
    /// # Parameters
    ///
    /// * `proc` - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) context for MPI coordination
    /// * `scp` - [`Searchspace`] describing the solution structure
    /// * `cod` - [`Codomain`](crate::Codomain) describing objective outputs
    fn after_load_dist(&mut self, proc: &MPIProcess, scp: &Scp, cod: &Op::Cod);

    /// Save a batch of evaluated solutions in a distributed batch experiment.
    ///
    /// Similar to [`BatchRecorder::save`] but allows MPI-aware implementations to coordinate
    /// writes across processes.
    ///
    /// # Parameters
    ///
    /// * `computed` - The computed batch with optimizer metadata
    /// * `outputed` - The output batch with outcomes
    /// * `scp` - [`Searchspace`] used to interpret the solutions
    /// * `cod` - [`Codomain`](crate::Codomain) used to interpret the outcomes
    fn save_dist(
        &self,
        computed: &CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
        outputed: &OutBatch<SolId, Op::Info, Out>,
        scp: &Scp,
        cod: &Op::Cod,
    );
}
