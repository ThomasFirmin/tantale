//! No-op recorder implementation.
//!
//! This module provides [`NoSaver`], a recorder that implements all recording traits
//! but performs no operations. It is useful for optimization experiments where
//! result persistence is not needed, avoiding the overhead of file I/O.
//!
//! # See Also
//!
//! - [`CSVRecorder`](crate::CSVRecorder) - For experiments that need result persistence
//! - [`Recorder`](crate::Recorder) - Base trait for all recorders

use crate::{
    BatchOptimizer, BatchRecorder, CompBatch, FuncWrapper, RawObj, SeqRecorder,
    SequentialOptimizer,
    domain::onto::LinkOpt,
    objective::Outcome,
    recorder::Recorder,
    searchspace::{CompShape, Searchspace},
    solution::{HasY, Id, OutBatch, SolutionShape, Uncomputed},
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "mpi")]
use crate::{DistBatchRecorder, DistSeqRecorder, experiment::mpi::utils::MPIProcess};

/// No-op recorder that performs no operations.
///
/// [`NoSaver`] implements all recorder traits but does nothing when called.
/// This is useful for optimization experiments where persistence is not required,
/// allowing algorithms to run without the overhead of file I/O operations.
///
/// All trait methods are no-ops:
/// - [`init`](crate::SeqRecorder::init) - Does nothing
/// - [`after_load`](crate::SeqRecorder::after_load) - Does nothing
/// - [`save`](crate::SeqRecorder::save) - Does nothing
///
/// # Serialization
///
/// [`NoSaver`] is fully serializable via [`Deserialize`] and [`Serialize`],
/// allowing it to be used in checkpointer-enabled experiments.
///
/// # See Also
///
/// - [`CSVRecorder`](crate::CSVRecorder) - Alternative recorder with file persistence
#[derive(Default, Serialize, Deserialize)]
pub struct NoSaver {}

impl NoSaver {
    /// Create a new [`NoSaver`] recorder.
    ///
    /// # Returns
    ///
    /// `Some(NoSaver)` - Always succeeds (no configuration needed)
    pub fn new() -> Option<NoSaver> {
        Some(NoSaver {})
    }
}

impl Recorder for NoSaver {}

/// Implementation of [`SeqRecorder`] for sequential optimization.
///
/// All methods are no-ops, performing no I/O or output operations.
impl<PSol, SolId, Out, Scp, Op, FnWrap> SeqRecorder<PSol, SolId, Out, Scp, Op, FnWrap> for NoSaver
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
    /// Initialize recorder - no-op.
    fn init(&mut self, _scp: &Scp, _cod: &<Op>::Cod) {}

    /// Prepare for loading - no-op.
    fn after_load(&mut self, _scp: &Scp, _cod: &<Op>::Cod) {}

    /// Save a single solution - no-op.
    fn save(
        &self,
        _computed: &CompShape<Scp, PSol, SolId, <Op>::SInfo, <Op>::Cod, Out>,
        _outputed: &(SolId, Out),
        _scp: &Scp,
        _cod: &<Op>::Cod,
    ) {
    }
}

/// Implementation of [`BatchRecorder`] for batch optimization.
///
/// All methods are no-ops, performing no I/O or output operations.
impl<PSol, SolId, Out, Scp, Op, FnWrap> BatchRecorder<PSol, SolId, Out, Scp, Op, FnWrap> for NoSaver
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    /// Initialize recorder - no-op.
    fn init(&mut self, _scp: &Scp, _cod: &<Op>::Cod) {}

    /// Prepare for loading - no-op.
    fn after_load(&mut self, _scp: &Scp, _cod: &<Op>::Cod) {}

    /// Save a batch of solutions - no-op.
    fn save(
        &self,
        _computed: &crate::CompBatch<SolId, <Op>::SInfo, Op::Info, Scp, PSol, <Op>::Cod, Out>,
        _outputed: &OutBatch<SolId, Op::Info, Out>,
        _scp: &Scp,
        _cod: &<Op>::Cod,
    ) {
    }
}

/// Implementation of [`DistSeqRecorder`] for distributed sequential optimization (MPI).
///
/// All methods are no-ops, performing no I/O or output operations.
#[cfg(feature = "mpi")]
impl<PSol, SolId, Out, Scp, Op, FnWrap> DistSeqRecorder<PSol, SolId, Out, Scp, Op, FnWrap>
    for NoSaver
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
    /// Initialize distributed recorder - no-op.
    fn init_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &<Op>::Cod) {}

    /// Prepare distributed recorder for loading - no-op.
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &<Op>::Cod) {}

    /// Save a single solution in distributed context - no-op.
    fn save_dist(
        &self,
        _computed: &CompShape<Scp, PSol, SolId, <Op>::SInfo, <Op>::Cod, Out>,
        _outputed: &(SolId, Out),
        _scp: &Scp,
        _cod: &<Op>::Cod,
    ) {
    }
}

/// Implementation of [`DistBatchRecorder`] for distributed batch optimization (MPI).
///
/// All methods are no-ops, performing no I/O or output operations.
#[cfg(feature = "mpi")]
impl<PSol, SolId, Out, Scp, Op, FnWrap> DistBatchRecorder<PSol, SolId, Out, Scp, Op, FnWrap>
    for NoSaver
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Out: Outcome,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    /// Initialize distributed recorder - no-op.
    fn init_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &<Op>::Cod) {}

    /// Prepare distributed recorder for loading - no-op.
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &<Op>::Cod) {}

    /// Save a batch of solutions in distributed context - no-op.
    fn save_dist(
        &self,
        _computed: &crate::CompBatch<SolId, <Op>::SInfo, Op::Info, Scp, PSol, <Op>::Cod, Out>,
        _outputed: &OutBatch<SolId, Op::Info, Out>,
        _scp: &Scp,
        _cod: &<Op>::Cod,
    ) {
    }
}
