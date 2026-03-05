use crate::{
    FuncState, Id, Optimizer, Outcome, Searchspace, Stop, ThrCheckpointer,
    checkpointer::{FuncStateCheckpointer, MonoCheckpointer},
    domain::onto::LinkOpt,
    experiment::Evaluate,
    objective::FuncWrapper,
    optimizer::opt::OpSInfType,
    recorder::Recorder,
    searchspace::CompShape,
    solution::{SolutionShape, Uncomputed, shape::RawObj},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer, experiment::mpi::utils::MPIProcess,
    solution::HasY,
};

use indexmap::IndexMap;

//--------------------//
//--- MONOTHREADED ---//
//--------------------//

/// Monothreaded experiment descriptor.
///
/// This is a container that gathers all the components required to
/// execute an optimization loop in a single thread. It does **not** run the
/// optimization by itself; execution is delegated to a [`Runable`](crate::experiment::Runable)
/// implementation and an [`Evaluate`](crate::experiment::Evaluate) evaluator.
///
/// # Type parameters
/// - `PSol`: The uncomputed solution type produced by the [`Searchspace`](crate::searchspace::Searchspace).
/// - `SolId`: The identifier type for solutions, implementing [`Id`](crate::Id).
/// - `Scp`: The [`Searchspace`](crate::searchspace::Searchspace) definition.
/// - `Op`: The [`Optimizer`](crate::optimizer::Optimizer) implementation.
/// - `St`: The stopping criterion implementing [`Stop`](crate::stop::Stop).
/// - `Rec`: A [`Recorder`](crate::recorder::Recorder) implementation (optional).
/// - `Check`: A [`Checkpointer`](crate::checkpointer::Checkpointer) implementation (optional).
/// - `Out`: The [`Outcome`](crate::objective::Outcome) type describing raw outputs.
/// - `Fn`: A wrapped objective implementing [`FuncWrapper`](crate::objective::FuncWrapper).
/// - `Eval`: The evaluation strategy implementing [`Evaluate`](crate::experiment::Evaluate).
///
/// # Notes
/// - `recorder` and `checkpointer` are optional.
/// - `evaluator` is kept `pub(crate)` because it is intended to be used internally
///   by runables, not by end users.
pub struct MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: MonoCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluate,
{
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    pub(crate) evaluator: Option<Eval>,
}

//---------------------//
//--- MULTITHREADED ---//
//---------------------//

/// Multithreaded experiment descriptor.
///
/// This container mirrors [`MonoExperiment`] but is designed for parallel
/// execution contexts. The evaluator will typically distribute the objective
/// evaluations across multiple threads.
///
/// # See also
/// - [`MonoExperiment`] for the single-threaded equivalent.
/// - [`Runable`](crate::experiment::Runable) for the execution loop.
pub struct ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluate,
{
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    pub(crate) evaluator: Option<Eval>,
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
/// Distributed experiment descriptor for MPI backends.
///
/// This structure is enabled with the `mpi` feature and stores an [`MPIProcess`]
/// handle along with distributed variants of the recorder and checkpointer.
/// It is intended for experiments where evaluations are distributed across
/// multiple distributed processes.
///
/// # See also
/// - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess)
/// - [`DistRecorder`](crate::recorder::DistRecorder)
/// - [`DistCheckpointer`](crate::checkpointer::DistCheckpointer)
pub struct MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    St: Stop,
    Rec: Recorder,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluate,
{
    pub proc: &'a MPIProcess,
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    pub(crate) evaluator: Option<Eval>,
}

/// Trait defining the interface for a pool of [`FuncState`]s associated with solution identifiers.
/// This abstraction allows for different implementations of function state management, such as in-memory pools or checkpointer-backed storage.
pub trait FuncStatePool<FnState, SolId>
where
    Self: Default,
    SolId: Id,
    FnState: FuncState,
{
    /// Insert a new [`FuncState`] into the pool, associated with a solution [`Id`].
    fn insert(&mut self, id: SolId, state: FnState);
    /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
    fn remove(&mut self, id: &SolId) -> bool;
    /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
    fn retrieve(&mut self, id: &SolId) -> Option<FnState>;
}

/// An implementation of [`FuncStatePool`] that uses an in-memory [`IndexMap`] to store function states.
/// This pool is suitable for scenarios where memory is sufficient to hold all function states.
/// It also optionally integrates a function state checkpointer for persistence, but does not rely on it for retrieval.
pub struct IdxMapPool<SolId, FnState, FnStCheck>
where
    SolId: Id,
    FnState: FuncState,
    FnStCheck: FuncStateCheckpointer,
{
    pub pool: IndexMap<SolId, Option<FnState>>,
    pub check: Option<FnStCheck>,
}

impl<FnStCheck, SolId, FnState> IdxMapPool<SolId, FnState, FnStCheck>
where
    FnStCheck: FuncStateCheckpointer,
    SolId: Id,
    FnState: FuncState,
{
    /// Create a new [`IdxMapPool`] with an optional function state checkpointer.
    pub fn new(check: Option<FnStCheck>) -> Self {
        Self {
            pool: IndexMap::new(),
            check,
        }
    }
}

impl<SolId: Id, FnState: FuncState, FnStCheck: FuncStateCheckpointer> FromIterator<(SolId, FnState)>
    for IdxMapPool<SolId, FnState, FnStCheck>
{
    fn from_iter<T: IntoIterator<Item = (SolId, FnState)>>(iter: T) -> Self {
        let pool = iter
            .into_iter()
            .map(|(id, state)| (id, Some(state)))
            .collect();
        Self { pool, check: None }
    }
}

impl<FnStCheck, SolId, FnState> Default for IdxMapPool<SolId, FnState, FnStCheck>
where
    FnStCheck: FuncStateCheckpointer,
    SolId: Id,
    FnState: FuncState,
{
    fn default() -> Self {
        Self::new(None)
    }
}

impl<FnStCheck, SolId, FnState> FuncStatePool<FnState, SolId>
    for IdxMapPool<SolId, FnState, FnStCheck>
where
    FnStCheck: FuncStateCheckpointer,
    SolId: Id,
    FnState: FuncState,
{
    /// Insert a new [`FuncState`] into the pool, associated with the given solution [`Id`].
    fn insert(&mut self, id: SolId, state: FnState) {
        if let Some(c) = &self.check {
            c.save_func_state(&id, &state);
        }
        self.pool.insert(id, Some(state));
    }
    /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
    /// Returns `true` if the state was present and removed, `false` otherwise.
    fn remove(&mut self, id: &SolId) -> bool {
        if let Some(c) = &self.check {
            let _ = c.remove_func_state(id);
        }
        self.pool.swap_remove(id).is_some()
    }
    /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
    fn retrieve(&mut self, id: &SolId) -> Option<FnState> {
        if let Some(state_opt) = self.pool.get_mut(id) {
            state_opt.take()
        } else {
            None
        }
    }
}

pub struct LoadPool<FnStCheck>
where
    FnStCheck: FuncStateCheckpointer,
{
    pub check: FnStCheck,
}

impl<FnStCheck> LoadPool<FnStCheck>
where
    FnStCheck: FuncStateCheckpointer,
{
    pub fn new(check: FnStCheck) -> Self {
        Self { check }
    }
}

impl<FnStCheck> Default for LoadPool<FnStCheck>
where
    FnStCheck: FuncStateCheckpointer,
{
    fn default() -> Self {
        panic!("LoadPool cannot be default constructed without a checkpointer")
    }
}

impl<FnStCheck, SolId, FnState> FuncStatePool<FnState, SolId> for LoadPool<FnStCheck>
where
    FnStCheck: FuncStateCheckpointer,
    SolId: Id,
    FnState: FuncState,
{
    fn insert(&mut self, id: SolId, state: FnState) {
        self.check.save_func_state(&id, &state);
    }

    fn remove(&mut self, id: &SolId) -> bool {
        self.check.remove_func_state(id).is_ok()
    }

    fn retrieve(&mut self, id: &SolId) -> Option<FnState> {
        self.check.load_func_state(id)
    }
}
