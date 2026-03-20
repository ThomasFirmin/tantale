use crate::{
    FuncState, Id, Optimizer, Outcome, Searchspace, Stop, ThrCheckpointer,
    checkpointer::{FuncStateCheckpointer, MonoCheckpointer},
    domain::onto::LinkOpt,
    experiment::{CompAcc, Evaluate},
    objective::FuncWrapper,
    optimizer::opt::OpSInfType,
    recorder::Recorder,
    searchspace::CompShape,
    solution::{SolutionShape, Uncomputed, shape::RawObj},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer,
    experiment::{PoolMode, mpi::utils::MPIProcess},
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
    pub accumulator: CompAcc<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
    pub(crate) evaluator: Option<Eval>,
    pub(crate) pool_mode: PoolMode,
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
    pub accumulator: CompAcc<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
    pub(crate) evaluator: Option<Eval>,
    pub(crate) pool_mode: PoolMode,
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
    pub accumulator: CompAcc<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
    pub(crate) evaluator: Option<Eval>,
    pub pool_mode: PoolMode,
}

/// Trait defining the interface for a pool of [`FuncState`]s associated with solution identifiers.
/// This abstraction allows for different implementations of function state management, such as in-memory pools or checkpointer-backed storage.
pub trait FuncStatePool<FnState, SolId>: Default
where
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
pub struct IdxMapPool<FnStCheck, FnState, SolId>
where
    SolId: Id,
    FnState: FuncState,
    FnStCheck: FuncStateCheckpointer,
{
    pub pool: IndexMap<SolId, Option<FnState>>,
    pub check: Option<FnStCheck>,
}

impl<FnStCheck, SolId, FnState> IdxMapPool<FnStCheck, FnState, SolId>
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

impl<SolId, FnState, FnStCheck> Default for IdxMapPool<FnStCheck, FnState, SolId>
where
    SolId: Id,
    FnState: FuncState,
    FnStCheck: FuncStateCheckpointer,
{
    fn default() -> Self {
        Self {
            pool: Default::default(),
            check: Default::default(),
        }
    }
}

impl<SolId: Id, FnState: FuncState, FnStCheck: FuncStateCheckpointer> FromIterator<(SolId, FnState)>
    for IdxMapPool<FnStCheck, FnState, SolId>
{
    fn from_iter<T: IntoIterator<Item = (SolId, FnState)>>(iter: T) -> Self {
        let pool = iter
            .into_iter()
            .map(|(id, state)| (id, Some(state)))
            .collect();
        Self { pool, check: None }
    }
}

impl<FnStCheck, SolId, FnState> FuncStatePool<FnState, SolId>
    for IdxMapPool<FnStCheck, FnState, SolId>
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

pub struct LoadPool<FnStCheck, FnState, SolId>
where
    FnState: FuncState,
    SolId: Id,
    FnStCheck: FuncStateCheckpointer,
{
    pub current: Option<(SolId, FnState)>,
    pub check: FnStCheck,
}

impl<SolId, FnState, FnStCheck> Default for LoadPool<FnStCheck, FnState, SolId>
where
    SolId: Id,
    FnState: FuncState,
    FnStCheck: FuncStateCheckpointer,
{
    fn default() -> Self {
        panic!("LoadPool cannot be created with default.")
    }
}

impl<FnStCheck, FnState, SolId> LoadPool<FnStCheck, FnState, SolId>
where
    FnState: FuncState,
    SolId: Id,
    FnStCheck: FuncStateCheckpointer,
{
    pub fn new(check: FnStCheck) -> Self {
        Self {
            check,
            current: None,
        }
    }
}

impl<FnStCheck, SolId, FnState> FuncStatePool<FnState, SolId>
    for LoadPool<FnStCheck, FnState, SolId>
where
    FnStCheck: FuncStateCheckpointer,
    SolId: Id,
    FnState: FuncState,
{
    /// Insert a new [`FuncState`] into the pool, associated with the given solution [`Id`].
    /// This implementation saves the state using the checkpointer and keeps track of the current state in memory for quick retrieval.
    fn insert(&mut self, id: SolId, state: FnState) {
        self.check.save_func_state(&id, &state);
        self.current = Some((id, state));
    }

    /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
    /// This implementation removes the state from the checkpointer and clears the current state if it matches the given [`Id`].
    fn remove(&mut self, id: &SolId) -> bool {
        if let Some((current_id, _)) = &self.current
            && current_id == id
        {
            self.current = None;
        }
        self.check.remove_func_state(id).is_ok()
    }

    /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
    /// This implementation first checks if the current state in memory matches the given [`Id`] and
    /// returns it if so; otherwise, it attempts to load the state from the checkpointer.
    fn retrieve(&mut self, id: &SolId) -> Option<FnState> {
        if let Some((current_id, _)) = &self.current
            && current_id == id
        {
            Some(self.current.take().unwrap().1)
        } else {
            self.check.load_func_state(id)
        }
    }
}

/// An enum that abstracts over different implementations of a function state pool.
/// It can either be an in-memory [`IdxMapPool`] or a checkpointer-backed [`LoadPool`].
/// This allows the experiment to switch between different pool strategies based on the [`PoolMode`] without changing the underlying logic of function state management.
pub enum Pool<FnStCheck, FnState, SolId>
where
    FnState: FuncState,
    SolId: Id,
    FnStCheck: FuncStateCheckpointer,
{
    IdxMap(IdxMapPool<FnStCheck, FnState, SolId>),
    Load(LoadPool<FnStCheck, FnState, SolId>),
}

impl<FnStCheck, FnState, SolId> Default for Pool<FnStCheck, FnState, SolId>
where
    FnState: FuncState,
    SolId: Id,
    FnStCheck: FuncStateCheckpointer,
{
    fn default() -> Self {
        Pool::IdxMap(IdxMapPool::default())
    }
}

impl<FnStCheck, FnState, SolId> FuncStatePool<FnState, SolId> for Pool<FnStCheck, FnState, SolId>
where
    FnState: FuncState,
    SolId: Id,
    FnStCheck: FuncStateCheckpointer,
{
    /// Insert a new [`FuncState`] into the pool, associated with the given solution [`Id`].
    /// The behavior of this method depends on the underlying pool implementation:
    /// - For [`IdxMapPool`], it will save the state in the in-memory map and optionally persist it using the checkpointer.
    /// - For [`LoadPool`], it will save the state using the checkpointer and keep track of the current state in memory for quick retrieval
    fn insert(&mut self, id: SolId, state: FnState) {
        match self {
            Pool::IdxMap(p) => p.insert(id, state),
            Pool::Load(p) => p.insert(id, state),
        }
    }

    /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
    /// The behavior of this method depends on the underlying pool implementation:
    /// - For [`IdxMapPool`], it will remove the state from the in-memory map and optionally remove it from the checkpointer.
    /// - For [`LoadPool`], it will remove the state from the checkpointer and clear the current state if it matches the given [`Id`].
    fn remove(&mut self, id: &SolId) -> bool {
        match self {
            Pool::IdxMap(p) => p.remove(id),
            Pool::Load(p) => p.remove(id),
        }
    }

    /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
    /// The behavior of this method depends on the underlying pool implementation:
    /// - For [`IdxMapPool`], it will retrieve the state from the in-memory map.
    /// - For [`LoadPool`], it will first check if the current state in memory matches the given [`Id`] and return it if so;
    ///   otherwise, it will attempt to load the state from the checkpointer.
    fn retrieve(&mut self, id: &SolId) -> Option<FnState> {
        match self {
            Pool::IdxMap(p) => p.retrieve(id),
            Pool::Load(p) => p.retrieve(id),
        }
    }
}
