use crate::{
    BaseSol, Checkpointer, FuncState, OneStep, Optimizer, 
    Outcome, Problem, Runable, SId, Searchspace, SimplePb, SolInfo, SolutionShape, StepId, Stop,
    aliases::{Accum, Cod, LinkObj, LinkOpt}, checkpointer::CheckpointerTargets, experiment::PoolMode, recorder::Recorder,
};

use indexmap::IndexMap;


pub struct Experiment<P, O, S, R, C>
where
    Self: Runable,
    P: Problem<Info = O::Info>,
    O: Optimizer<P>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
{
    pub problem: P,
    pub optimizer: O,
    pub stop: S,
    pub recorder: Option<R>,
    pub checkpointer: Option<C>,
    pub accumulator: Accum<P>,
    pub(crate) pool_mode: PoolMode,
}

impl<P, O, S, R, C> Experiment<P, O, S, R, C>
where
    Self: Runable,
    P:Problem<Info = O::Info>,
    P::Scp: Searchspace<P::Func>,
    O:Optimizer<P>,
    S:Stop,
    R:Recorder<P, ()>,
    C:Checkpointer<P, ()>,
{
    pub fn new(
        problem: P,
        optimizer: O,
        stop: S,
        recorder: Option<R>,
        checkpointer: Option<C>,
    ) -> Self {
        let accumulator = Accum::<P>::default();
        Self {
            problem,
            optimizer,
            stop,
            recorder,
            checkpointer,
            accumulator,
            pool_mode: PoolMode::InMemory,
        }
    }

    pub fn load_with_pool(problem: P, recorder: Option<R>, mut checkpointer: C, pool_mode: PoolMode) -> Self {
        checkpointer.before_load(&problem, &()).unwrap();

        let optimizer = checkpointer.load(&CheckpointerTargets::Optimizer, &()).unwrap();
        let stop = checkpointer.load(&CheckpointerTargets::Stop, &()).unwrap();
        let accumulator = checkpointer.load(&CheckpointerTargets::Accumulator, &()).unwrap();

        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&problem, &()).unwrap();
                Some(rec)
            }
            None => None,
        };
        checkpointer.after_load(&problem, &()).unwrap();

        Self {
            problem,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            pool_mode,
        }
    }
}

impl<O, S, R, C, Scp, Out, Info> Runable for Experiment<SimplePb<Info, Scp, Out>, O, S, R, C>
where
    O:Optimizer<SimplePb<Info, Scp, Out>, Info = Info>,
    S:Stop,
    R:Recorder<SimplePb<Info, Scp, Out>, ()>,
    C:Checkpointer<SimplePb<Info, Scp, Out>, ()>,
    Scp: Searchspace<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>>,
    Scp::Shape: SolutionShape<
        Id = SId,
        Info = Info,
        ObjSol = BaseSol<SId, LinkObj<Scp>, Info>,
        OptSol = BaseSol<SId, LinkOpt<Scp>, Info>,
        Obj = LinkObj<Scp>,
        Opt = LinkOpt<Scp>,
    >,
    Out: Outcome,
    Info: SolInfo,
{
    type SolId  = SId;
    type Context = ();

    type Problem = SimplePb<Info, Scp, Out>;
    type Optimizer = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;
    
    fn run(self) -> Accum<Self::Problem> {
        todo!()
    }
    
    fn get_stop(&self) -> &Self::Stop {
        &self.stop
    }
    
    fn get_searchspace(&self) -> &crate::aliases::Scp<Self::Problem> {
        self.problem.get_searchspace()
    }
    
    fn get_codomain(&self) -> &Cod<Self::Problem> {
        self.problem.get_codomain()
    }
    
    fn get_objective(&self) -> &crate::aliases::Func<Self::Problem> {
        self.problem.get_objective()
    }
    
    fn get_optimizer(&self) -> &Self::Optimizer {
        &self.optimizer
    }
    
    fn get_recorder(&self) -> Option<&Self::Recorder> {
        self.recorder.as_ref()
    }
    
    fn get_checkpointer(&self) -> Option<&Self::Checkpointer> {
        self.checkpointer.as_ref()
    }
    
    fn get_mut_stop(&mut self) -> &mut Self::Stop {
        &mut self.stop
    }
    
    fn get_mut_searchspace(&mut self) -> &mut crate::aliases::Scp<Self::Problem> {
        self.problem.get_mut_searchspace()
    }
    
    fn get_mut_codomain(&mut self) -> &mut Cod<Self::Problem> {
        self.problem.get_mut_codomain()
    }
    
    fn get_mut_objective(&mut self) -> &mut crate::aliases::Func<Self::Problem> {
        self.problem.get_mut_objective()
    }
    
    fn get_mut_optimizer(&mut self) -> &mut Self::Optimizer {
        &mut self.optimizer
    }
    
    fn get_mut_recorder(&mut self) -> Option<&mut Self::Recorder> {
        self.recorder.as_mut()
    }
    
    fn get_mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer> {
        self.checkpointer.as_mut()
    }
    
    fn get_accumalator(&self) -> &Accum<Self::Problem> {
        &self.accumulator
    }
    
    fn get_mut_accumalator(&mut self) -> &mut Accum<Self::Problem> {
        &mut self.accumulator
    }
}

/// Trait defining the interface for a pool of [`FuncState`]s associated with solution identifiers.
/// This abstraction allows for different implementations of function state management, such as in-memory pools or checkpointer-backed storage.
pub trait FuncStatePool<Id: StepId, FnState: FuncState>: Default
{
    /// Insert a new [`FuncState`] into the pool, associated with a solution [`Id`].
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx);
    /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool;
    /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
    fn retrieve(&mut self, id: &Id) -> Option<FnState>;
}

/// An implementation of [`FuncStatePool`] that uses an in-memory [`IndexMap`] to store function states.
/// This pool is suitable for scenarios where memory is sufficient to hold all function states.
pub struct InMemoryPool<Id: StepId, FnState: FuncState>(IndexMap<Id, FnState>);

impl<Id: StepId, FnState: FuncState> InMemoryPool<Id, FnState>
{
    /// Create a new [`InMemoryPool`] with an optional function state checkpointer.
    pub fn new() -> Self {
        Self(IndexMap::new())
    }
}

impl<Id: StepId, FnState: FuncState> Default for InMemoryPool<Id, FnState>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Id: StepId, FnState: FuncState> FromIterator<(Id, FnState)> for InMemoryPool<Id, FnState>
{
    fn from_iter<T: IntoIterator<Item = (Id, FnState)>>(iter: T) -> Self {
        let pool = iter.into_iter().collect();
        Self(pool)
    }
}

impl<Id: StepId, FnState: FuncState> FuncStatePool<Id, FnState> for InMemoryPool<Id, FnState>
{
    /// Insert a new [`FuncState`] into the pool, associated with the given solution [`Id`].
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx) {
        if let Some(c) = check {
            let _ = c.save_fn_state(&id, &state, ctx);
        }
        self.0.insert(id, state);
    }
    /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
    /// Returns `true` if the state was present and removed, `false` otherwise.
    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool {
        if let Some(c) = check {
            let _ = c.remove_all_previous_fn_states::<Id, FnState>(id, ctx);
        }
        self.0.swap_remove(id).is_some()
    }
    /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
    fn retrieve(&mut self, id: &Id) -> Option<FnState> {
        self.0.swap_remove(id)
    }
}

// pub struct LoadPool<FnStCheck, FnState, SolId>
// where
//     FnState: FuncState,
//     SolId: Id,
//     FnStCheck: FuncStateCheckpointer,
// {
//     pub current: Option<(SolId, FnState)>,
//     pub check: FnStCheck,
// }

// impl<SolId, FnState, FnStCheck> Default for LoadPool<FnStCheck, FnState, SolId>
// where
//     SolId: Id,
//     FnState: FuncState,
//     FnStCheck: FuncStateCheckpointer,
// {
//     fn default() -> Self {
//         panic!("LoadPool cannot be created with default.")
//     }
// }

// impl<FnStCheck, FnState, SolId> LoadPool<FnStCheck, FnState, SolId>
// where
//     FnState: FuncState,
//     SolId: Id,
//     FnStCheck: FuncStateCheckpointer,
// {
//     pub fn new(check: FnStCheck) -> Self {
//         Self {
//             check,
//             current: None,
//         }
//     }
// }

// impl<FnStCheck, SolId, FnState> FuncStatePool<FnState, SolId>
//     for LoadPool<FnStCheck, FnState, SolId>
// where
//     FnStCheck: FuncStateCheckpointer,
//     SolId: Id,
//     FnState: FuncState,
// {
//     /// Insert a new [`FuncState`] into the pool, associated with the given solution [`Id`].
//     /// This implementation saves the state using the checkpointer and keeps track of the current state in memory for quick retrieval.
//     fn insert(&mut self, id: SolId, state: FnState) {
//         self.check.save_func_state(&id, &state);
//         self.current = Some((id, state));
//     }

//     /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
//     /// This implementation removes the state from the checkpointer and clears the current state if it matches the given [`Id`].
//     fn remove(&mut self, id: &SolId) -> bool {
//         if let Some((current_id, _)) = &self.current
//             && current_id == id
//         {
//             self.current = None;
//         }
//         self.check.remove_func_state(id).is_ok()
//     }

//     /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
//     /// This implementation first checks if the current state in memory matches the given [`Id`] and
//     /// returns it if so; otherwise, it attempts to load the state from the checkpointer.
//     fn retrieve(&mut self, id: &SolId) -> Option<FnState> {
//         if let Some((current_id, _)) = &self.current
//             && current_id == id
//         {
//             Some(self.current.take().unwrap().1)
//         } else if let Some((_, fnstate)) = self.check.load_func_state(id) {
//             Some(fnstate)
//         } else {
//             None
//         }
//     }
// }

// /// An enum that abstracts over different implementations of a function state pool.
// /// It can either be an in-memory [`IdxMapPool`] or a checkpointer-backed [`LoadPool`].
// /// This allows the experiment to switch between different pool strategies based on the [`PoolMode`] without changing the underlying logic of function state management.
// pub enum Pool<FnStCheck, FnState, SolId>
// where
//     FnState: FuncState,
//     SolId: Id,
//     FnStCheck: FuncStateCheckpointer,
// {
//     IdxMap(IdxMapPool<FnStCheck, FnState, SolId>),
//     Load(LoadPool<FnStCheck, FnState, SolId>),
// }

// impl<FnStCheck, FnState, SolId> Default for Pool<FnStCheck, FnState, SolId>
// where
//     FnState: FuncState,
//     SolId: Id,
//     FnStCheck: FuncStateCheckpointer,
// {
//     fn default() -> Self {
//         Pool::IdxMap(IdxMapPool::default())
//     }
// }

// impl<FnStCheck, FnState, SolId> FuncStatePool<FnState, SolId> for Pool<FnStCheck, FnState, SolId>
// where
//     FnState: FuncState,
//     SolId: Id,
//     FnStCheck: FuncStateCheckpointer,
// {
//     /// Insert a new [`FuncState`] into the pool, associated with the given solution [`Id`].
//     /// The behavior of this method depends on the underlying pool implementation:
//     /// - For [`IdxMapPool`], it will save the state in the in-memory map and optionally persist it using the checkpointer.
//     /// - For [`LoadPool`], it will save the state using the checkpointer and keep track of the current state in memory for quick retrieval
//     fn insert(&mut self, id: SolId, state: FnState) {
//         match self {
//             Pool::IdxMap(p) => p.insert(id, state),
//             Pool::Load(p) => p.insert(id, state),
//         }
//     }

//     /// Remove the [`FuncState`] associated with the given solution [`Id`], if it exists.
//     /// The behavior of this method depends on the underlying pool implementation:
//     /// - For [`IdxMapPool`], it will remove the state from the in-memory map and optionally remove it from the checkpointer.
//     /// - For [`LoadPool`], it will remove the state from the checkpointer and clear the current state if it matches the given [`Id`].
//     fn remove(&mut self, id: &SolId) -> bool {
//         match self {
//             Pool::IdxMap(p) => p.remove(id),
//             Pool::Load(p) => p.remove(id),
//         }
//     }

//     /// Retrieve the [`FuncState`] associated with the given solution [`Id`], if it exists.
//     /// The behavior of this method depends on the underlying pool implementation:
//     /// - For [`IdxMapPool`], it will retrieve the state from the in-memory map.
//     /// - For [`LoadPool`], it will first check if the current state in memory matches the given [`Id`] and return it if so;
//     ///   otherwise, it will attempt to load the state from the checkpointer.
//     fn retrieve(&mut self, id: &SolId) -> Option<FnState> {
//         match self {
//             Pool::IdxMap(p) => p.retrieve(id),
//             Pool::Load(p) => p.retrieve(id),
//         }
//     }
// }



// /// Creates a single-threaded experiment ([`MonoExperiment`]).
// ///
// /// This is the primary entry point for creating experiments that execute in a single thread.
// /// The experiment combines all necessary components and can be immediately run or inspected.
// ///
// /// # Parameters
// ///
// /// * `space` - A [`Searchspace`], $f: X \rightarrow Y$
// /// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped), $f$
// /// * `optimizer` - An optimizer implementing [`Optimizer`]
// /// * `stop` - A stopping criterion implementing [`Stop`]
// /// * `saver` - A tuple of (optional [`Recorder`], optional [`Checkpointer`])
// ///
// /// # Returns
// ///
// /// An experiment implementing [`Runable`] ready to be executed
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::mono, Objective, stop::Calls};
// /// use tantale::algos::RandomSearch;
// ///
// /// let sp = my_searchspace();
// /// let obj = Objective::new(my_function);
// /// let opt = RandomSearch::new();
// /// let stop = Calls::new(100);
// ///
// /// let experiment = mono(sp, obj, opt, stop, (None, None)); // No recording or checkpointing
// /// experiment.run();
// /// ```
// ///
// /// # See Also
// ///
// /// * [`mono_with_pool`] - For single-threaded execution with function state management.
// /// * [`threaded`] - For multi-threaded execution
// /// * [`distributed`] - For MPI-based distributed execution
// /// * [`MonoExperiment`] - The underlying experiment type
// pub fn mono<Scp, Func, Opt, St, Rec, Check>(
//     space: Scp,
//     objective: Func,
//     optimizer: Opt,
//     stop: St,
//     rec : Option<Rec>,
//     check: Option<Check>,
// ) -> impl Runable<BasicProblem<Scp, Func>>
// where
//     Scp: Searchspace<Func>,
//     Func: FuncWrapper,
//     Opt: Optimizer<BasicProblem<Scp, Func>>,
//     St: Stop,
//     Rec: Recorder,
//     Check: Checkpointer,
// {
//     <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new(
//         space, objective, optimizer, stop, saver,
//     )
// }

// /// Similar to [`mono`] but allows specifying a [`PoolMode`] for function state management.
// pub fn mono_with_pool<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
//     pool_mode: PoolMode,
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: MonoCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new_with_pool(
//         space, objective, optimizer, stop, saver, pool_mode,
//     )
// }

// /// Creates a multi-threaded experiment ([`ThrExperiment`]).
// ///
// /// This function creates experiments that execute using multiple OS threads, enabling
// /// parallel evaluation of solutions. Particularly effective for batch optimizers or
// /// sequential optimizers with varying evaluation times.
// ///
// /// # Parameters
// ///
// /// * `space` - A [`Searchspace`]
// /// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped)
// /// * `optimizer` - An optimizer implementing [`Optimizer`]
// /// * `stop` - A stopping criterion implementing [`Stop`]
// /// * `saver` - A tuple of (optional [`Recorder`], optional [`ThrCheckpointer`])
// ///
// /// # Thread Safety
// ///
// /// All components must implement `Send + Sync` for thread-safe sharing across threads.
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::threaded, Objective, stop::Calls};
// /// use tantale::algos::BatchRandomSearch;
// ///
// /// let sp = my_searchspace();
// /// let obj = Objective::new(my_function);
// /// let opt = BatchRandomSearch::new(10);  // Batch size 10
// /// let stop = Calls::new(100);
// ///
// /// let experiment = threaded(sp, obj, opt, stop, (None, None)); // No recording or checkpointing
// /// experiment.run();  // Evaluates batches in parallel
// /// ```
// ///
// /// # See Also
// ///
// /// * [`threaded_with_pool`] - For multi-threaded execution with function state management.
// /// * [`mono`] - For single-threaded execution
// /// * [`distributed`] - For MPI-based distributed execution
// /// * [`ThrExperiment`] - The underlying experiment type
// pub fn threaded<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: ThrCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new(
//         space, objective, optimizer, stop, saver,
//     )
// }

// /// Similar to [`threaded`] but allows specifying a [`PoolMode`] for function state management.
// pub fn threaded_with_pool<SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
//     pool_mode: PoolMode,
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: ThrCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new_with_pool(
//         space, objective, optimizer, stop, saver, pool_mode,
//     )
// }

// #[cfg(feature = "mpi")]
// #[allow(clippy::type_complexity)]
// /// Creates an MPI-distributed experiment ([`MPIExperiment`]).
// ///
// /// This function creates experiments that execute across multiple processes using MPI, enabling
// /// parallel evaluation of solutions in a distributed environment.
// /// Requires the `mpi` feature and an initialized MPI environment.
// /// Each process will run an instance of the experiment, with one process acting as the master coordinating the optimization
// /// and others as workers evaluating solutions.
// ///
// /// # Parameters
// ///
// /// * `proc` - A reference to an initialized [`MPIProcess`] containing rank, communicator, etc.
// /// * `space` - A [`Searchspace`]
// /// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped)
// /// * `optimizer` - An optimizer implementing [`Optimizer`]
// /// * `stop` - A stopping criterion implementing [`Stop`]
// /// * `saver` - A tuple of (optional [`Recorder`], optional [`ThrCheckpointer`])
// ///
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::threaded, Objective, stop::Calls};
// /// use tantale::algos::BatchRandomSearch;
// ///
// /// let sp = my_searchspace();
// /// let obj = Objective::new(my_function);
// /// let opt = BatchRandomSearch::new(10);  // Batch size 10
// /// let stop = Calls::new(100);
// ///
// /// let experiment = distributed(&proc, sp, obj, opt, stop, (None, None)); // No recording or checkpointing
// /// experiment.run();  // Evaluates batches in parallel
// /// ```
// ///
// /// # See Also
// ///
// /// * [`distributed_with_pool`] - For MPI-based distributed execution with function state management.
// /// * [`mono`] - For single-threaded execution
// /// * [`threaded`] - For multi-threaded execution
// /// * [`MPIExperiment`] - The underlying experiment type
// pub fn distributed<'a, SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     proc: &'a MPIProcess,
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
// ) -> MasterWorker<
//     'a,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
//     PSol,
//     SolId,
//     Scp,
//     Op,
//     St,
//     Rec,
//     Check,
//     Out,
//     Fn,
// >
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: DistCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::new(
//         proc,
//         space, objective, optimizer, stop, saver
//     )
// }

// #[cfg(feature = "mpi")]
// #[allow(clippy::type_complexity)]
// /// Similar to [`distributed`] but allows specifying a [`PoolMode`] for function state management.
// pub fn distributed_with_pool<'a, SolId, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
//     proc: &'a MPIProcess,
//     space: Scp,
//     objective: Fn,
//     optimizer: Op,
//     stop: St,
//     saver: (Option<Rec>, Option<Check>),
//     pool_mode: PoolMode,
// ) -> MasterWorker<
//     'a,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
//     PSol,
//     SolId,
//     Scp,
//     Op,
//     St,
//     Rec,
//     Check,
//     Out,
//     Fn,
// >
// where
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     St: Stop,
//     Rec: Recorder,
//     Check: DistCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::new_with_pool(
//         proc,
//         space, objective, optimizer, stop, saver, pool_mode
//     )
// }

// /// Loads a single-threaded experiment from a checkpoint.
// ///
// /// Restores a [`MonoExperiment`] from saved state, allowing optimization to be resumed
// /// or inspected. All optimizer state, stopping criterion progress, and relevant data
// /// are recovered from the checkpoint.
// ///
// /// # Parameters
// ///
// /// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
// /// * `objective` - The objective function wrapper  
// /// * `saver` - Tuple of (optional [`Recorder`], **required** [`Checkpointer`])
// ///
// /// # Type Parameters
// ///
// /// Unlike [`mono`], you must explicitly specify the optimizer and stop types:
// /// - `Op` - The optimizer type
// /// - `St` - The stopping criterion type
// ///
// /// # Returns
// ///
// /// A loaded experiment implementing [`Runable`], ready to continue or be inspected
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::mono_load, MessagePack, FolderConfig};
// /// use tantale::algos::RandomSearch;
// /// use tantale::core::stop::Calls;
// ///
// /// // Load from checkpoint
// /// let config = FolderConfig::new("my_experiment").init();
// /// let checkpointer = MessagePack::new(config).unwrap();
// ///
// /// let mut exp = mono_load::<RandomSearch, Calls, _, _, _, _, _, _, _>(
// ///     searchspace,
// ///     objective,
// ///     (None, checkpointer)
// /// );
// ///
// /// // Inspect or modify before continuing
// /// println!("Completed {} evaluations", exp.get_stop().calls());
// /// exp.get_mut_stop().add(100);  // Add 100 more evaluations
// ///
// /// // Continue optimization
// /// exp.run();
// /// ```
// ///
// /// # Note
// ///
// /// Prefer using the `load!` macro for cleaner syntax and better type inference.
// ///
// /// # See Also
// ///
// /// * [`mono`] - For creating new experiments
// /// * [`threaded_load`] - For loading multi-threaded experiments
// /// * `load!` - Macro for simpler loading syntax
// pub fn mono_load<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     SolId: Id,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: MonoCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
// {
//     <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load(
//         space, objective, saver,
//     )
// }

// /// Similar to [`mono_load`] but allows specifying a [`PoolMode`] for function state management.
// pub fn mono_load_with_pool<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
//     pool_mode: PoolMode,
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     SolId: Id,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MonoExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: MonoCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
// {
//     <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load_with_pool(
//         space, objective, saver, pool_mode
//     )
// }

// /// Loads a multi-threaded experiment from a checkpoint.
// ///
// /// Restores a [`ThrExperiment`] from saved state. The checkpoint must have been created
// /// with a thread-safe checkpointer ([`ThrCheckpointer`]).
// ///
// /// # Parameters
// ///
// /// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
// /// * `objective` - The objective function wrapper
// /// * `saver` - Tuple of (optional [`Recorder`], **required** [`ThrCheckpointer`])
// ///
// /// # Type Parameters
// ///
// /// - `Op` - The optimizer type
// /// - `St` - The stopping criterion type
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::threaded_load, MessagePack, FolderConfig};
// /// use tantale::algos::BatchRandomSearch;
// /// use tantale::core::stop::Calls;
// ///
// /// let config = FolderConfig::new("my_experiment").init();
// /// let checkpointer = MessagePack::new(config).unwrap();
// ///
// /// let mut exp = threaded_load::<BatchRandomSearch, Calls, _, _, _, _, _, _, _>(
// ///     searchspace,
// ///     objective,
// ///     (None, checkpointer)
// /// );
// ///
// /// exp.run();
// /// ```
// ///
// /// # Note
// ///
// /// Prefer using the `load!` macro for cleaner syntax.
// ///
// /// # See Also
// ///
// /// * [`threaded`] - For creating new multi-threaded experiments
// /// * [`mono_load`] - For loading single-threaded experiments
// /// * `load!` - Macro for simpler loading syntax
// pub fn threaded_load<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     SolId: Id,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: ThrCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
// {
//     <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load(
//         space, objective, saver,
//     )
// }

// /// Similar to [`threaded_load`] but allows specifying a [`PoolMode`] for function state management.
// pub fn threaded_load_with_pool<SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
//     pool_mode: PoolMode,
// ) -> impl Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
// where
//     SolId: Id,
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     ThrExperiment<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: ThrCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
// {
//     <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load_with_pool(
//         space, objective, saver, pool_mode
//     )
// }

// #[cfg(feature = "mpi")]
// #[allow(clippy::type_complexity)]
// /// Loads an MPI-distributed experiment from a checkpoint.
// ///
// /// Restores a [`MPIExperiment`] from saved state. The checkpoint must have been created
// /// with a distributed checkpointer ([`DistCheckpointer`]) and requires the `mpi`
// /// feature and an initialized MPI environment. Each process will load its own instance of the experiment,
// /// with one process acting as the master coordinating the optimization and others as workers evaluating solutions.
// ///
// /// # Parameters
// ///
// /// * `proc` - A reference to an initialized [`MPIProcess`] containing rank, communicator, etc.
// /// * `space` - A [`Searchspace`] (must be the same as used for the original experiment)
// /// * `objective` - The objective function wrapper
// /// * `saver` - Tuple of (optional [`Recorder`], **required** [`DistCheckpointer`])
// ///
// /// # Type Parameters
// ///
// /// - `Op` - The optimizer type
// /// - `St` - The stopping criterion type
// ///
// /// # Example
// ///
// /// ```rust,ignore
// /// use tantale::core::{experiment::distributed_load, MessagePack, FolderConfig};
// /// use tantale::core::mpi::utils::MPIProcess;
// /// use tantale::algos::BatchRandomSearch;
// /// use tantale::core::stop::Calls;
// ///
// /// let config = FolderConfig::new("my_experiment").init();
// /// let checkpointer = MessagePack::new(config).unwrap();
// /// let proc = MPIProcess::new();
// ///
// /// let mut exp = distributed_load::<BatchRandomSearch, Calls, _, _, _, _, _, _, _>(
// ///     &proc,
// ///     searchspace,
// ///     objective,
// ///     (None, checkpointer)
// /// );
// ///
// /// exp.run();
// /// ```
// ///
// /// # Note
// ///
// /// Prefer using the `load!` macro for cleaner syntax.
// ///
// /// # See Also
// ///
// /// * [`distributed`] - For creating new multi-threaded experiments
// /// * [`mono_load`] - For loading single-threaded experiments
// /// * `load!` - Macro for simpler loading syntax
// pub fn distributed_load<'a, SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     proc: &'a MPIProcess,
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
// ) -> MasterWorker<
//     'a,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
//     PSol,
//     SolId,
//     Scp,
//     Op,
//     St,
//     Rec,
//     Check,
//     Out,
//     Fn,
// >
// where
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: DistCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::load(
//         proc,
//         space,
//         objective,
//         saver
//     )
// }

// #[cfg(feature = "mpi")]
// #[allow(clippy::type_complexity)]
// /// Similar to [`distributed_load`] but allows specifying a [`PoolMode`] for function state management.
// pub fn distributed_load_with_pool<'a, SolId, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
//     proc: &'a MPIProcess,
//     space: Scp,
//     objective: Fn,
//     saver: (Option<Rec>, Check),
//     pool_mode: PoolMode,
// ) -> MasterWorker<
//     'a,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
//     PSol,
//     SolId,
//     Scp,
//     Op,
//     St,
//     Rec,
//     Check,
//     Out,
//     Fn,
// >
// where
//     Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
//     St: Stop,
//     PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
//     PSol::Twin<Scp::Obj>: Uncomputed<SolId, Scp::Obj, Op::SInfo, Twin<Scp::Opt> = PSol>,
//     MPIExperiment<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
//         MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
//     Scp: Searchspace<PSol, SolId, Op::SInfo>,
//     Rec: Recorder,
//     Check: DistCheckpointer,
//     Out: Outcome,
//     Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
//     Eval: Evaluator,
//     SolId: Id,
// {
//     <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::load_with_pool(
//         proc,
//         space,
//         objective,
//         saver,
//         pool_mode
//     )
// }

// /// Macro for loading experiments from checkpoints with simplified syntax.
// ///
// /// This macro provides a convenient interface for calling the `*_load` functions
// /// while requiring only the optimizer and stop criterion types to be specified.
// /// All other generic types are inferred from the provided arguments.
// ///
// /// # Syntax
// ///
// /// ```text
// /// load!(mono, OptimizerType, StopType, space, objective, saver)
// /// load!(threaded, OptimizerType, StopType, space, objective, saver)
// /// load!(distributed, mpi_proc, OptimizerType, StopType, space, objective, saver)  // MPI feature only
// /// ```
// ///
// /// # Parameters
// ///
// /// * First argument: Execution context (`mono`, `threaded`, or `distributed`)
// /// * For distributed: `mpi_proc` - Reference to [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess)
// /// * `OptimizerType` - The concrete optimizer type (e.g., `RandomSearch`)
// /// * `StopType` - The concrete stopping criterion type (e.g., `Calls`)
// /// * `space` - Tuple of searchspace
// /// * `objective` - The objective function wrapper
// /// * `saver` - Tuple of (optional recorder, required checkpointer)
// ///
// /// # Alternative with PoolMode
// ///
// /// You can also specify a [`PoolMode`] for function state management by using the following syntax:
// /// ```text
// /// load!(mono, PoolMode::Persistent, OptimizerType, StopType, space, objective, saver)
// /// ```
// ///
// /// # Examples
// ///
// /// ## Basic Loading
// ///
// /// ```rust,ignore
// /// use tantale::core::{load, MessagePack, FolderConfig};
// /// use tantale::algos::RandomSearch;
// /// use tantale::core::stop::Calls;
// ///
// /// let config = FolderConfig::new("results").init();
// /// let checkpointer = MessagePack::new(config).unwrap();
// ///
// /// let exp = load!(
// ///     mono,                    // Execution context
// ///     RandomSearch,            // Optimizer type
// ///     Calls,                   // Stop criterion type
// ///     searchspace, // Space
// ///     objective,               // Objective function
// ///     (None, checkpointer)     // Saver (no recorder, with checkpointer)
// /// );
// ///
// /// exp.run();
// /// ```
// ///
// /// ## Extending Optimization Budget
// ///
// /// ```rust,ignore
// /// let mut exp = load!(mono, MyOpt, Calls, searchspace, obj, (None, checkpointer));
// ///
// /// let config = FolderConfig::new("results[`FuncWrapper``] / ").init();
// /// let checkpointer = MessagePack::new(config).unwrap();
// ///
// /// // Inspect current progress
// /// println!("Completed: {}", exp.get_stop().calls());
// ///
// /// // Add more evaluations
// /// exp.get_mut_stop().add(100);
// ///
// /// // Continue optimization
// /// exp.run();
// /// ```
// ///
// /// ## Multi-Threaded Loading
// ///
// /// ```rust,ignore
// /// let exp = load!(threaded, BatchRandomSearch, Calls, space, obj, saver);
// /// exp.run();
// /// ```
// ///
// /// ## Distributed Loading (MPI feature)
// ///
// /// ```rust,ignore
// /// #[cfg(feature = "mpi")]
// /// {
// ///     use tantale::core::{load, MessagePack, FolderConfig, experiment::mpi::utils::MPIProcess};
// ///     use tantale::algos::RandomSearch;
// ///     use tantale::core::stop::Calls;
// ///     
// ///     let mpi_process = MPIProcess::new();
// ///
// ///     let config = FolderConfig::new("results").init(&mpi_process);
// ///     let checkpointer = MessagePack::new(config).unwrap();
// ///
// ///     let exp = load!(
// ///         distributed,
// ///         &mpi_process,
// ///         MyOptimizer,
// ///         Calls,
// ///         searchspace,
// ///         obj,
// ///         (None, checkpointer)
// ///     );
// ///     exp.run();
// /// }
// /// ```
// ///
// /// # See Also
// ///
// /// * [`mono_load`] - Direct function for single-threaded loading
// /// * [`threaded_load`] - Direct function for multi-threaded loading  
// /// * [`distributed_load`] - Direct function for distributed loading (MPI feature)
// /// * [`mono`], [`threaded`], [`distributed`] - For creating new experiments
// #[macro_export]
// #[cfg(not(feature = "mpi"))]
// macro_rules! load {
//     (mono, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::mono_load::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $space, $objective, $saver,
//         )
//     };
//     (threaded, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::threaded_load::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $space, $objective, $saver,
//         )
//     };
//     (mono, $pool_mode:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::mono_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $space, $objective, $saver, $pool_mode,
//         )
//     };
//     (threaded, $pool_mode:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::threaded_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $space, $objective, $saver, $pool_mode,
//         )
//     };
// }

// /// Macro for loading experiments from checkpoints with simplified syntax (MPI-enabled version).
// ///
// /// This is the MPI-enabled version of the `load!` macro. It provides the same functionality
// /// as the non-MPI version but with additional support for distributed experiments.
// ///
// /// # See Also
// ///
// /// See the non-MPI `load!` macro documentation for detailed usage and examples.
// /// The only difference is that this version supports the `distributed` variant.
// #[macro_export]
// #[cfg(feature = "mpi")]
// macro_rules! load {
//     (mono, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::mono_load::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $space, $objective, $saver,
//         )
//     };
//     (threaded, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::threaded_load::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $space, $objective, $saver,
//         )
//     };
//     (distributed, $proc:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::distributed_load::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $proc, $space, $objective, $saver,
//         )
//     };
//     (mono, $pool_mode:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::mono_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $space, $objective, $saver, $pool_mode,
//         )
//     };
//     (threaded, $pool_mode:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::threaded_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $space, $objective, $saver, $pool_mode,
//         )
//     };
//     (distributed, $pool_mode:expr, $proc:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
//         $crate::experiment::distributed_load_with_pool::<_, $Op, $St, _, _, _, _, _, _, _>(
//             $proc, $space, $objective, $saver, $pool_mode,
//         )
//     };
// }