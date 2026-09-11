use crate::{
    Checkpointer, FuncState, Id, Optimizer, Outcome, Problem, Searchspace, Stop, aliases::{Accum, Cod}, checkpointer::FuncStateCheckpointer, evaluator::Evaluator, experiment::PoolMode, objective::FuncWrapper, recorder::Recorder, solution::Uncomputed,
};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::utils::MPIProcess;

use indexmap::IndexMap;

//--------------------//
//--- MONOTHREADED ---//
//--------------------//

/// Monothreaded experiment descriptor.
///
/// This is a container that gathers all the components required to
/// execute an optimization loop in a single thread. It does **not** run the
/// optimization by itself; execution is delegated to a [`Runable`](crate::experiment::Runable)
/// implementation and an [`Evaluate`] evaluator.
///
/// # Type parameters
/// - `PSol`: The uncomputed solution type produced by the [`Searchspace`].
/// - `SolId`: The identifier type for solutions, implementing [`Id`].
/// - `Scp`: The [`Searchspace`] definition.
/// - `Op`: The [`Optimizer`] implementation.
/// - `St`: The stopping criterion implementing [`Stop`].
/// - `Rec`: A [`Recorder`] implementation (optional).
/// - `Check`: A [`Checkpointer`](crate::checkpointer::Checkpointer) implementation (optional).
/// - `Out`: The [`Outcome`] type describing raw outputs.
/// - `Fn`: A wrapped objective implementing [`FuncWrapper`].
/// - `Eval`: The evaluation strategy implementing [`Evaluate`].
///
/// # Notes
/// - `recorder` and `checkpointer` are optional.
/// - `evaluator` is kept `pub(crate)` because it is intended to be used internally
///   by runables, not by end users.
pub struct MonoExperiment<P, O, S, R, C, E>
where
    P: Problem,
    O: Optimizer<P>,
    S: Stop,
    R: Recorder<Self>,
    C: Checkpointer<Self>,
    E: Evaluator,
{
    pub searchspace: P::Scp,
    pub codomain: Cod<P>,
    pub objective: P::Func,
    pub optimizer: O,
    pub stop: S,
    pub recorder: Option<R>,
    pub checkpointer: Option<C>,
    pub accumulator: Accum<P>,
    pub(crate) evaluator: Option<E>,
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
pub struct ThrExperiment<P, O, S, R, C, E>
where
    P: Problem,
    O: Optimizer<P>,
    S: Stop,
    R: Recorder<Self>,
    C: Checkpointer<Self>,
    E: Evaluator,
{
    pub searchspace: P::Scp,
    pub codomain: Cod<P>,
    pub objective: P::Func,
    pub optimizer: O,
    pub stop: S,
    pub recorder: Option<R>,
    pub checkpointer: Option<C>,
    pub accumulator: Accum<P>,
    pub(crate) evaluator: Option<E>,
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
/// - [`MPIProcess`]
/// - [`DistCheckpointer`]
pub struct MPIExperiment<'a, P, O, S, R, C, E>
where
    P: Problem,
    O: Optimizer<P>,
    S: Stop,
    R: Recorder<Self>,
    C: Checkpointer<Self>,
    E: Evaluator,
{
    pub proc: &'a MPIProcess,
    pub searchspace: P::Scp,
    pub codomain: Cod<P>,
    pub objective: P::Func,
    pub optimizer: O,
    pub stop: S,
    pub recorder: Option<R>,
    pub checkpointer: Option<C>,
    pub accumulator: Accum<P>,
    pub(crate) evaluator: Option<E>,
    pub(crate) pool_mode: PoolMode,
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
    pub pool: IndexMap<SolId, FnState>,
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
        let pool = iter.into_iter().collect();
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
        self.pool.insert(id, state);
    }
    /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
    /// Returns `true` if the state was present and removed, `false` otherwise.
    fn remove(&mut self, id: &SolId) -> bool {
        let mut check_rmv = true;
        if let Some(c) = &self.check {
            check_rmv = c.remove_func_state(id).is_ok();
        }
        self.pool.swap_remove(id).is_some() && check_rmv
    }
    /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
    fn retrieve(&mut self, id: &SolId) -> Option<FnState> {
        self.pool.swap_remove(id)
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
        } else if let Some((_, fnstate)) = self.check.load_func_state(id) {
            Some(fnstate)
        } else {
            None
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



/// Creates a single-threaded experiment ([`MonoExperiment`]).
///
/// This is the primary entry point for creating experiments that execute in a single thread.
/// The experiment combines all necessary components and can be immediately run or inspected.
///
/// # Parameters
///
/// * `space` - A [`Searchspace`], $f: X \rightarrow Y$
/// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped), $f$
/// * `optimizer` - An optimizer implementing [`Optimizer`]
/// * `stop` - A stopping criterion implementing [`Stop`]
/// * `saver` - A tuple of (optional [`Recorder`], optional [`Checkpointer`])
///
/// # Returns
///
/// An experiment implementing [`Runable`] ready to be executed
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::mono, Objective, stop::Calls};
/// use tantale::algos::RandomSearch;
///
/// let sp = my_searchspace();
/// let obj = Objective::new(my_function);
/// let opt = RandomSearch::new();
/// let stop = Calls::new(100);
///
/// let experiment = mono(sp, obj, opt, stop, (None, None)); // No recording or checkpointing
/// experiment.run();
/// ```
///
/// # See Also
///
/// * [`mono_with_pool`] - For single-threaded execution with function state management.
/// * [`threaded`] - For multi-threaded execution
/// * [`distributed`] - For MPI-based distributed execution
/// * [`MonoExperiment`] - The underlying experiment type
pub fn mono<Scp, Func, Opt, St, Rec, Check>(
    space: Scp,
    objective: Func,
    optimizer: Opt,
    stop: St,
    rec : Option<Rec>,
    check: Option<Check>,
) -> impl Runable<BasicProblem<Scp, Func>>
where
    Scp: Searchspace<Func>,
    Func: FuncWrapper,
    Opt: Optimizer<BasicProblem<Scp, Func>>,
    St: Stop,
    Rec: Recorder,
    Check: Checkpointer,
{
    <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new(
        space, objective, optimizer, stop, saver,
    )
}

/// Similar to [`mono`] but allows specifying a [`PoolMode`] for function state management.
pub fn mono_with_pool<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    space: Scp,
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
    pool_mode: PoolMode,
) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: MonoCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
    SolId: Id,
{
    <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new_with_pool(
        space, objective, optimizer, stop, saver, pool_mode,
    )
}

/// Creates a multi-threaded experiment ([`ThrExperiment`]).
///
/// This function creates experiments that execute using multiple OS threads, enabling
/// parallel evaluation of solutions. Particularly effective for batch optimizers or
/// sequential optimizers with varying evaluation times.
///
/// # Parameters
///
/// * `space` - A [`Searchspace`]
/// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped)
/// * `optimizer` - An optimizer implementing [`Optimizer`]
/// * `stop` - A stopping criterion implementing [`Stop`]
/// * `saver` - A tuple of (optional [`Recorder`], optional [`ThrCheckpointer`])
///
/// # Thread Safety
///
/// All components must implement `Send + Sync` for thread-safe sharing across threads.
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::threaded, Objective, stop::Calls};
/// use tantale::algos::BatchRandomSearch;
///
/// let sp = my_searchspace();
/// let obj = Objective::new(my_function);
/// let opt = BatchRandomSearch::new(10);  // Batch size 10
/// let stop = Calls::new(100);
///
/// let experiment = threaded(sp, obj, opt, stop, (None, None)); // No recording or checkpointing
/// experiment.run();  // Evaluates batches in parallel
/// ```
///
/// # See Also
///
/// * [`threaded_with_pool`] - For multi-threaded execution with function state management.
/// * [`mono`] - For single-threaded execution
/// * [`distributed`] - For MPI-based distributed execution
/// * [`ThrExperiment`] - The underlying experiment type
pub fn threaded<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    space: Scp,
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
    SolId: Id,
{
    <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new(
        space, objective, optimizer, stop, saver,
    )
}

/// Similar to [`threaded`] but allows specifying a [`PoolMode`] for function state management.
pub fn threaded_with_pool<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    space: Scp,
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
    pool_mode: PoolMode,
) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
    SolId: Id,
{
    <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new_with_pool(
        space, objective, optimizer, stop, saver, pool_mode,
    )
}

#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
/// Creates an MPI-distributed experiment ([`MPIExperiment`]).
///
/// This function creates experiments that execute across multiple processes using MPI, enabling
/// parallel evaluation of solutions in a distributed environment.
/// Requires the `mpi` feature and an initialized MPI environment.
/// Each process will run an instance of the experiment, with one process acting as the master coordinating the optimization
/// and others as workers evaluating solutions.
///
/// # Parameters
///
/// * `proc` - A reference to an initialized [`MPIProcess`] containing rank, communicator, etc.
/// * `space` - A [`Searchspace`]
/// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped)
/// * `optimizer` - An optimizer implementing [`Optimizer`]
/// * `stop` - A stopping criterion implementing [`Stop`]
/// * `saver` - A tuple of (optional [`Recorder`], optional [`ThrCheckpointer`])
///
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::threaded, Objective, stop::Calls};
/// use tantale::algos::BatchRandomSearch;
///
/// let sp = my_searchspace();
/// let obj = Objective::new(my_function);
/// let opt = BatchRandomSearch::new(10);  // Batch size 10
/// let stop = Calls::new(100);
///
/// let experiment = distributed(&proc, sp, obj, opt, stop, (None, None)); // No recording or checkpointing
/// experiment.run();  // Evaluates batches in parallel
/// ```
///
/// # See Also
///
/// * [`distributed_with_pool`] - For MPI-based distributed execution with function state management.
/// * [`mono`] - For single-threaded execution
/// * [`threaded`] - For multi-threaded execution
/// * [`MPIExperiment`] - The underlying experiment type
pub fn distributed<'a, SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    proc: &'a MPIProcess,
    space: Scp,
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
) -> MasterWorker<
    'a,
    MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
    PSol,
    SolId,
    Scp,
    Op,
    St,
    Rec,
    Check,
    Out,
    Fn,
>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
    SolId: Id,
{
    <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::new(
        proc,
        space, objective, optimizer, stop, saver
    )
}

#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
/// Similar to [`distributed`] but allows specifying a [`PoolMode`] for function state management.
pub fn distributed_with_pool<'a, SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    proc: &'a MPIProcess,
    space: Scp,
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
    pool_mode: PoolMode,
) -> MasterWorker<
    'a,
    MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
    PSol,
    SolId,
    Scp,
    Op,
    St,
    Rec,
    Check,
    Out,
    Fn,
>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
    SolId: Id,
{
    <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::new_with_pool(
        proc,
        space, objective, optimizer, stop, saver, pool_mode
    )
}

/// Loads a single-threaded experiment from a checkpoint.
///
/// Restores a [`MonoExperiment`] from saved state, allowing optimization to be resumed
/// or inspected. All optimizer state, stopping criterion progress, and relevant data
/// are recovered from the checkpoint.
///
/// # Parameters
///
/// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
/// * `objective` - The objective function wrapper  
/// * `saver` - Tuple of (optional [`Recorder`], **required** [`Checkpointer`])
///
/// # Type Parameters
///
/// Unlike [`mono`], you must explicitly specify the optimizer and stop types:
/// - `Op` - The optimizer type
/// - `St` - The stopping criterion type
///
/// # Returns
///
/// A loaded experiment implementing [`Runable`], ready to continue or be inspected
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::mono_load, MessagePack, FolderConfig};
/// use tantale::algos::RandomSearch;
/// use tantale::core::stop::Calls;
///
/// // Load from checkpoint
/// let config = FolderConfig::new("my_experiment").init();
/// let checkpointer = MessagePack::new(config).unwrap();
///
/// let mut exp = mono_load::<RandomSearch, Calls, _, _, _, _, _, _, _>(
///     searchspace,
///     objective,
///     (None, checkpointer)
/// );
///
/// // Inspect or modify before continuing
/// println!("Completed {} evaluations", exp.get_stop().calls());
/// exp.get_mut_stop().add(100);  // Add 100 more evaluations
///
/// // Continue optimization
/// exp.run();
/// ```
///
/// # Note
///
/// Prefer using the `load!` macro for cleaner syntax and better type inference.
///
/// # See Also
///
/// * [`mono`] - For creating new experiments
/// * [`threaded_load`] - For loading multi-threaded experiments
/// * `load!` - Macro for simpler loading syntax
pub fn mono_load<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    space: Scp,
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    Rec: Recorder,
    Check: MonoCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
{
    <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load(
        space, objective, saver,
    )
}

/// Similar to [`mono_load`] but allows specifying a [`PoolMode`] for function state management.
pub fn mono_load_with_pool<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    space: Scp,
    objective: Fn,
    saver: (Option<Rec>, Check),
    pool_mode: PoolMode,
) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    Rec: Recorder,
    Check: MonoCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
{
    <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load_with_pool(
        space, objective, saver, pool_mode
    )
}

/// Loads a multi-threaded experiment from a checkpoint.
///
/// Restores a [`ThrExperiment`] from saved state. The checkpoint must have been created
/// with a thread-safe checkpointer ([`ThrCheckpointer`]).
///
/// # Parameters
///
/// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
/// * `objective` - The objective function wrapper
/// * `saver` - Tuple of (optional [`Recorder`], **required** [`ThrCheckpointer`])
///
/// # Type Parameters
///
/// - `Op` - The optimizer type
/// - `St` - The stopping criterion type
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::threaded_load, MessagePack, FolderConfig};
/// use tantale::algos::BatchRandomSearch;
/// use tantale::core::stop::Calls;
///
/// let config = FolderConfig::new("my_experiment").init();
/// let checkpointer = MessagePack::new(config).unwrap();
///
/// let mut exp = threaded_load::<BatchRandomSearch, Calls, _, _, _, _, _, _, _>(
///     searchspace,
///     objective,
///     (None, checkpointer)
/// );
///
/// exp.run();
/// ```
///
/// # Note
///
/// Prefer using the `load!` macro for cleaner syntax.
///
/// # See Also
///
/// * [`threaded`] - For creating new multi-threaded experiments
/// * [`mono_load`] - For loading single-threaded experiments
/// * `load!` - Macro for simpler loading syntax
pub fn threaded_load<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    space: Scp,
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    Rec: Recorder,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
{
    <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load(
        space, objective, saver,
    )
}

/// Similar to [`threaded_load`] but allows specifying a [`PoolMode`] for function state management.
pub fn threaded_load_with_pool<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    space: Scp,
    objective: Fn,
    saver: (Option<Rec>, Check),
    pool_mode: PoolMode,
) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    Rec: Recorder,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
{
    <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load_with_pool(
        space, objective, saver, pool_mode
    )
}

#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
/// Loads an MPI-distributed experiment from a checkpoint.
///
/// Restores a [`MPIExperiment`] from saved state. The checkpoint must have been created
/// with a distributed checkpointer ([`DistCheckpointer`]) and requires the `mpi`
/// feature and an initialized MPI environment. Each process will load its own instance of the experiment,
/// with one process acting as the master coordinating the optimization and others as workers evaluating solutions.
///
/// # Parameters
///
/// * `proc` - A reference to an initialized [`MPIProcess`] containing rank, communicator, etc.
/// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
/// * `objective` - The objective function wrapper
/// * `saver` - Tuple of (optional [`Recorder`], **required** [`DistCheckpointer`])
///
/// # Type Parameters
///
/// - `Op` - The optimizer type
/// - `St` - The stopping criterion type
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::distributed_load, MessagePack, FolderConfig};
/// use tantale::core::mpi::utils::MPIProcess;
/// use tantale::algos::BatchRandomSearch;
/// use tantale::core::stop::Calls;
///
/// let config = FolderConfig::new("my_experiment").init();
/// let checkpointer = MessagePack::new(config).unwrap();
/// let proc = MPIProcess::new();
///
/// let mut exp = distributed_load::<BatchRandomSearch, Calls, _, _, _, _, _, _, _>(
///     &proc,
///     searchspace,
///     objective,
///     (None, checkpointer)
/// );
///
/// exp.run();
/// ```
///
/// # Note
///
/// Prefer using the `load!` macro for cleaner syntax.
///
/// # See Also
///
/// * [`distributed`] - For creating new multi-threaded experiments
/// * [`mono_load`] - For loading single-threaded experiments
/// * `load!` - Macro for simpler loading syntax
pub fn distributed_load<'a, SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    proc: &'a MPIProcess,
    space: Scp,
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> MasterWorker<
    'a,
    MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
    PSol,
    SolId,
    Scp,
    Op,
    St,
    Rec,
    Check,
    Out,
    Fn,
>
where
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    Rec: Recorder,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
    SolId: Id,
{
    <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::load(
        proc,
        space,
        objective,
        saver
    )
}

#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
/// Similar to [`distributed_load`] but allows specifying a [`PoolMode`] for function state management.
pub fn distributed_load_with_pool<'a, SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    proc: &'a MPIProcess,
    space: Scp,
    objective: Fn,
    saver: (Option<Rec>, Check),
    pool_mode: PoolMode,
) -> MasterWorker<
    'a,
    MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
    PSol,
    SolId,
    Scp,
    Op,
    St,
    Rec,
    Check,
    Out,
    Fn,
>
where
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
    MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    Rec: Recorder,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    Eval: Evaluator,
    SolId: Id,
{
    <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::load_with_pool(
        proc,
        space,
        objective,
        saver,
        pool_mode
    )
}

/// Macro for loading experiments from checkpoints with simplified syntax.
///
/// This macro provides a convenient interface for calling the `*_load` functions
/// while requiring only the optimizer and stop criterion types to be specified.
/// All other generic types are inferred from the provided arguments.
///
/// # Syntax
///
/// ```text
/// load!(mono, OptimizerType, StopType, space, objective, saver)
/// load!(threaded, OptimizerType, StopType, space, objective, saver)
/// load!(distributed, mpi_proc, OptimizerType, StopType, space, objective, saver)  // MPI feature only
/// ```
///
/// # Parameters
///
/// * First argument: Execution context (`mono`, `threaded`, or `distributed`)
/// * For distributed: `mpi_proc` - Reference to [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess)
/// * `OptimizerType` - The concrete optimizer type (e.g., `RandomSearch`)
/// * `StopType` - The concrete stopping criterion type (e.g., `Calls`)
/// * `space` - Tuple of searchspace
/// * `objective` - The objective function wrapper
/// * `saver` - Tuple of (optional recorder, required checkpointer)
///
/// # Alternative with PoolMode
///
/// You can also specify a [`PoolMode`] for function state management by using the following syntax:
/// ```text
/// load!(mono, PoolMode::Persistent, OptimizerType, StopType, space, objective, saver)
/// ```
///
/// # Examples
///
/// ## Basic Loading
///
/// ```rust,ignore
/// use tantale::core::{load, MessagePack, FolderConfig};
/// use tantale::algos::RandomSearch;
/// use tantale::core::stop::Calls;
///
/// let config = FolderConfig::new("results").init();
/// let checkpointer = MessagePack::new(config).unwrap();
///
/// let exp = load!(
///     mono,                    // Execution context
///     RandomSearch,            // Optimizer type
///     Calls,                   // Stop criterion type
///     searchspace, // Space
///     objective,               // Objective function
///     (None, checkpointer)     // Saver (no recorder, with checkpointer)
/// );
///
/// exp.run();
/// ```
///
/// ## Extending Optimization Budget
///
/// ```rust,ignore
/// let mut exp = load!(mono, MyOpt, Calls, searchspace, obj, (None, checkpointer));
///
/// let config = FolderConfig::new("results[`FuncWrapper``] / ").init();
/// let checkpointer = MessagePack::new(config).unwrap();
///
/// // Inspect current progress
/// println!("Completed: {}", exp.get_stop().calls());
///
/// // Add more evaluations
/// exp.get_mut_stop().add(100);
///
/// // Continue optimization
/// exp.run();
/// ```
///
/// ## Multi-Threaded Loading
///
/// ```rust,ignore
/// let exp = load!(threaded, BatchRandomSearch, Calls, space, obj, saver);
/// exp.run();
/// ```
///
/// ## Distributed Loading (MPI feature)
///
/// ```rust,ignore
/// #[cfg(feature = "mpi")]
/// {
///     use tantale::core::{load, MessagePack, FolderConfig, experiment::mpi::utils::MPIProcess};
///     use tantale::algos::RandomSearch;
///     use tantale::core::stop::Calls;
///     
///     let mpi_process = MPIProcess::new();
///
///     let config = FolderConfig::new("results").init(&mpi_process);
///     let checkpointer = MessagePack::new(config).unwrap();
///
///     let exp = load!(
///         distributed,
///         &mpi_process,
///         MyOptimizer,
///         Calls,
///         searchspace,
///         obj,
///         (None, checkpointer)
///     );
///     exp.run();
/// }
/// ```
///
/// # See Also
///
/// * [`mono_load`] - Direct function for single-threaded loading
/// * [`threaded_load`] - Direct function for multi-threaded loading  
/// * [`distributed_load`] - Direct function for distributed loading (MPI feature)
/// * [`mono`], [`threaded`], [`distributed`] - For creating new experiments
#[macro_export]
#[cfg(not(feature = "mpi"))]
macro_rules! load {
    (mono, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::mono_load::<_, $Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver,
        )
    };
    (threaded, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::threaded_load::<_, $Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver,
        )
    };
    (mono, $pool_mode:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::mono_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver, $pool_mode,
        )
    };
    (threaded, $pool_mode:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::threaded_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver, $pool_mode,
        )
    };
}

/// Macro for loading experiments from checkpoints with simplified syntax (MPI-enabled version).
///
/// This is the MPI-enabled version of the `load!` macro. It provides the same functionality
/// as the non-MPI version but with additional support for distributed experiments.
///
/// # See Also
///
/// See the non-MPI `load!` macro documentation for detailed usage and examples.
/// The only difference is that this version supports the `distributed` variant.
#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! load {
    (mono, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::mono_load::<_, $Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver,
        )
    };
    (threaded, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::threaded_load::<_, $Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver,
        )
    };
    (distributed, $proc:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::distributed_load::<_, $Op, $St, _, _, _, _, _, _, _>(
            $proc, $space, $objective, $saver,
        )
    };
    (mono, $pool_mode:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::mono_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver, $pool_mode,
        )
    };
    (threaded, $pool_mode:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::threaded_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver, $pool_mode,
        )
    };
    (distributed, $pool_mode:expr, $proc:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::distributed_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
            $proc, $space, $objective, $saver, $pool_mode,
        )
    };
}