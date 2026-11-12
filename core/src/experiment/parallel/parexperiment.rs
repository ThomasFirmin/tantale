use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::Ordering},
};

use serde::{Deserialize, Serialize};

use crate::{
    BaseSol, Checkpointer, FidOutcome, FidPb, FidelitySol, FuncState, FuncWrapper,
    GlobalParameters, HasFidelity, HasState, HasStep, HasStepId, HasVariables, MultiStep, OneStep,
    Outcome, Problem, RUN_ID, Recorder, Runable, SId, SOL_ID, Sampler, Searchspace, SimplePb,
    SolInfo, SolutionShape, State, StepSId, Stop,
    aliases::{Accum, CompShape, LinkObj, LinkOpt, Opt, Out, Shape},
    checkpointer::CheckpointerTargets,
    errors::WorkerError,
    experiment::{
        exputils::{Pool, PoolMode},
        parallel::parutils::{
            fidelity_master, simple_master, start_fidelity_workers, start_simple_workers,
        },
    },
    utils::Communicator,
};

#[derive(Debug, Serialize, Deserialize)]
/// [`PriorityList`] defines a [`Vec`] of [`Vec`].
/// Each inner [`Vec`] holds elements of type `T`.
/// It allows to add, push, pop elements associated to an index,
/// and to check if the whole structure is empty via [`count`](PriorityList::count).
pub struct PriorityList<T> {
    pub list: Vec<Vec<T>>,
    pub count: usize,
}

impl<T> PriorityList<T> {
    /// Create a new [`PriorityList`] of a given size.
    pub fn new(size: usize) -> Self {
        PriorityList {
            list: (0..size).map(|_| Vec::new()).collect(),
            count: 0,
        }
    }
    /// Add an element to the inner [`Vec`] associated to a given [`usize`].
    pub fn add(&mut self, elem: T, index: usize) {
        self.list[index].push(elem);
        self.count += 1;
    }
    /// Push a [`Vec`] of elements to the inner [`Vec`] associated to a given [`usize`].
    pub fn push(&mut self, elem: Vec<T>, index: usize) {
        self.list[index].extend(elem);
        self.count += 1;
    }
    /// Pop an element from the inner [`Vec`] associated to a given [`usize`].
    pub fn pop(&mut self, index: usize) -> Option<T> {
        let res = self.list[index].pop();
        if res.is_some() {
            self.count -= 1
        }
        res
    }
    /// Return `true` if the whole [`PriorityList`] is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Extend the current [`PriorityList`] with another one.
    pub fn extend(&mut self, other: PriorityList<T>) {
        if self.count != other.count {
            panic!("The two PriorityList have different lengthes.");
        } else {
            self.list.extend(other.list);
            self.count += other.count;
        }
    }
}

/// Trait for a master-worker communication pattern.
pub trait MasterWorkerCom: Communicator {
    /// Return `true` if the current process is the master.
    fn is_master(&self) -> bool;

    /// Return `true` if the current process is a worker.
    fn is_worker(&self) -> bool {
        !self.is_master()
    }

    /// Run the master routine.
    /// The master routine is a function that takes a [`MasterWorkerCom`] and returns an [`Accum`].
    fn master<P, F>(self, f: F) -> Accum<P>
    where
        P: Problem,
        F: FnOnce(Self) -> Accum<P>;

    /// Handle on the running workers, used to wait for their termination.
    type Handle: WorkerHandle;

    /// Start the worker side from the current process.
    /// The worker routine is a function that takes a [`FuncWrapper`], a [`Channel`](Communicator::Channel), and returns a [`Result`].
    ///
    /// This method must not block when the workers live in the same process as the master
    /// (e.g. threads), so that the master routine can run afterwards.
    /// When the current process is the master of a multi-process communicator (e.g. MPI), it does nothing.
    fn worker<R, F>(&mut self, func: F, routine: R) -> Result<Self::Handle, WorkerError>
    where
        F: FuncWrapper + 'static,
        R: Fn(Arc<F>, &mut Self::Channel) -> Result<(), WorkerError> + Send + Sync + 'static;
}

/// Handle on workers started by [`MasterWorkerCom::worker`].
pub trait WorkerHandle {
    /// Wait for all workers to terminate.
    fn join(self) -> Result<(), WorkerError>;
}

impl WorkerHandle for () {
    fn join(self) -> Result<(), WorkerError> {
        Ok(())
    }
}

impl WorkerHandle for Vec<std::thread::JoinHandle<Result<(), WorkerError>>> {
    fn join(self) -> Result<(), WorkerError> {
        for handle in self {
            handle.join().expect("A worker thread panicked.")?;
        }
        Ok(())
    }
}

/// Trait for a worker.
/// The worker is a process that runs a routine and communicates with the master.
pub trait WorkerTrait<P: Problem> {
    /// Handle on the launched workers.
    type Handle: WorkerHandle;
    /// Launch the worker routine. Does not block for in-process workers (threads).
    fn launch(self) -> Result<Self::Handle, WorkerError>;
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Shape<P>: SolutionShape, Out<P>: Outcome")]
pub struct ParState<P: Problem> {
    /// The current solution being evaluated.
    pub currently: Vec<Option<Shape<P>>>,
    #[serde(skip)] // Skip because 'where' a solution can vary after loading, and we don't want to serialize it.
    /// Maps each [`SolId`](crate::SolId) to its corresponding worker where the [`FuncState`] is stored.
    pub where_is: HashMap<P::Id, usize>,
    /// A list of [`FuncState`] that are awaiting processing by each worker.
    pub awaiting: Vec<Vec<Shape<P>>>,
    /// A list of [`FuncState`] that are awaiting processing by any worker, when the worker is not known.
    pub awaiting_unknown: Vec<Shape<P>>,
}

impl<P: Problem> State for ParState<P> {}

impl<P: Problem> ParState<P> {
    pub fn new(size: usize) -> Self {
        Self {
            currently: vec![None; size],
            where_is: HashMap::new(),
            awaiting: vec![Vec::new(); size],
            awaiting_unknown: Vec::new(),
        }
    }

    pub fn set_currently(&mut self, index: usize, currently: Shape<P>) {
        if index < self.currently.len() {
            self.currently[index] = Some(currently);
        } else {
            log::warn!(
                "Index {} is out of bounds for currently vector. No update performed.",
                index
            );
        }
    }

    /// Add a [`FuncState`] to the awaiting list of a specific worker.
    pub fn where_is(&self, id: &P::Id) -> Option<usize> {
        self.where_is.get(id).copied()
    }

    /// Get the next [`FuncState`] awaiting processing for a specific worker.
    pub fn pop_awaiting(&mut self, peer: usize) -> Option<Shape<P>> {
        if let Some(sol) = self.awaiting[peer].pop() {
            Some(sol)
        } else {
            self.awaiting_unknown.pop()
        }
    }
}

impl<P: Problem> HasState for ParState<P> {
    type State = Self;
    fn get_state(&self) -> &Self::State {
        self
    }

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

pub struct Worker<P, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    /// The [`Problem`] being solved in the experiment.
    pub problem: P,
    /// The [`Checkpointer`] used to save and load experiment state.
    pub checkpointer: Arc<Mutex<Option<C>>>,
    /// The [`PoolMode`] used for managing function states during the experiment.
    pub pool_mode: PoolMode,
    /// The [`Communicator`] used for send and receiving tasks.
    pub com: Com,
    /// Flag indicating whether to consider Parallel File System (PFS).
    /// Meaning that all processes can access the same file system.
    pub pfs: bool,
    /// Flag indicating whether the experiment state has been loaded from a checkpoint.
    pub is_loaded: bool,
}

impl<P, C, Com> Worker<P, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    /// Creates a new `Worker` instance if the current process is not the master.
    /// If it is the master, returns `None`.
    pub fn new(
        problem: P,
        mut checkpointer: Option<C>,
        pool_mode: PoolMode,
        com: Com,
    ) -> Option<Self> {
        if com.is_master() {
            None
        } else {
            if let Some(c) = checkpointer.as_mut() {
                c.set_pfs(true)
            }
            let checkpointer = Arc::new(Mutex::new(checkpointer));
            Some(Self {
                problem,
                checkpointer,
                pool_mode,
                com,
                is_loaded: false,
                pfs: true,
            })
        }
    }

    fn load(problem: P, checkpointer: C, pool_mode: PoolMode, com: Com) -> Option<Self> {
        if com.is_master() {
            None
        } else {
            let checkpointer = Arc::new(Mutex::new(Some(checkpointer)));
            Some(Self {
                problem,
                checkpointer,
                pool_mode,
                com,
                is_loaded: true,
                pfs: true,
            })
        }
    }

    /// Set the Parallel File System (PFS) flag for the worker.
    pub fn with_pfs(self, pfs: bool) -> Self {
        if let Some(c) = self.checkpointer.lock().unwrap().as_mut() {
            c.set_pfs(pfs)
        }
        Self { pfs, ..self }
    }

    /// Set the [`PoolMode`] for the worker,
    /// determining how function states are managed during the experiment.
    pub fn with_pool_mode(self, pool_mode: PoolMode) -> Self {
        Self { pool_mode, ..self }
    }
}

impl<Scp, Info, Out, C, Com> WorkerTrait<SimplePb<Info, Scp, Out>>
    for Worker<SimplePb<Info, Scp, Out>, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
    Scp: Searchspace<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>> + HasVariables,
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
    C: Checkpointer<SimplePb<Info, Scp, Out>, ()>,
    Com: MasterWorkerCom,
{
    type Handle = Com::Handle;

    fn launch(mut self) -> Result<Self::Handle, WorkerError> {
        start_simple_workers(&self.problem, &mut self.com)
    }
}

impl<Scp, Info, Out, C, Com, FnState> WorkerTrait<FidPb<Info, Scp, Out, FnState>>
    for Worker<FidPb<Info, Scp, Out, FnState>, C, Com>
where
    FidPb<Info, Scp, Out, FnState>: 'static,
    C: Checkpointer<FidPb<Info, Scp, Out, FnState>, ()> + Send + Sync + 'static,
    Scp: Searchspace<MultiStep<FidelitySol<StepSId, LinkObj<Scp>, Info>, Out, FnState>>
        + HasVariables,
    Scp::Shape: SolutionShape<
            Id = StepSId,
            Info = Info,
            ObjSol = FidelitySol<StepSId, LinkObj<Scp>, Info>,
            OptSol = FidelitySol<StepSId, LinkOpt<Scp>, Info>,
            Obj = LinkObj<Scp>,
            Opt = LinkOpt<Scp>,
        > + HasFidelity
        + HasStep
        + HasStepId,
    CompShape<FidPb<Info, Scp, Out, FnState>>: HasFidelity + HasStep + HasStepId,
    Out: FidOutcome,
    Info: SolInfo,
    FnState: FuncState + Send + Sync,
    Com: MasterWorkerCom,
{
    type Handle = Com::Handle;

    fn launch(mut self) -> Result<Self::Handle, WorkerError> {
        let pool = Arc::new(Mutex::new(Pool::<StepSId, FnState>::new(self.pool_mode)));
        let pb = Arc::new(self.problem);
        start_fidelity_workers(
            pb.clone(),
            &mut self.com,
            self.checkpointer,
            pool,
            self.is_loaded,
        )
    }
}

pub struct Master<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    /// The [`Problem`] being solved in the experiment.
    pub problem: Arc<P>,
    /// The [`Sampler`] used to generate new candidate solutions.sed to generate new candidate solutions.
    pub sampler: O,
    /// The [stopping](Stop) criterion for the experiment.
    pub stop: S,
    /// The [`Recorder`] used to log experiment data.
    pub recorder: Option<R>,
    /// The [`Checkpointer`] used to save and load experiment state.
    pub checkpointer: Arc<Mutex<Option<C>>>,
    /// The [`Accumulator`] that collects best evaluated solutions and their outcomes.
    pub accumulator: Accum<P>,
    /// The [`Communicator`] used for send and receiving tasks.
    pub com: Com,
    /// Flag indicating whether to consider Parallel File System (PFS).
    /// Meaning that all processes can access the same file system.
    pub pfs: bool,
    /// The [`PoolMode`] used for managing function states during the experiment.
    pub pool_mode: PoolMode,
    /// Flag indicating whether the experiment state has been loaded from a checkpoint.
    is_loaded: bool,
}

impl<P, O, S, R, C, Com> Master<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    pub fn new(
        problem: P,
        sampler: O,
        stop: S,
        recorder: Option<R>,
        mut checkpointer: Option<C>,
        com: Com,
    ) -> Option<Self> {
        if com.is_worker() {
            None
        } else {
            let accumulator = Accum::<P>::default();
            if let Some(c) = checkpointer.as_mut() {
                c.set_pfs(true)
            }
            let checkpointer = Arc::new(Mutex::new(checkpointer));
            Some(Self {
                problem: Arc::new(problem),
                sampler,
                stop,
                recorder,
                checkpointer,
                accumulator,
                com,
                pfs: true,
                is_loaded: false,
                pool_mode: PoolMode::InMemory, // Default pool mode, can be changed later
            })
        }
    }

    pub fn load(
        problem: P,
        mut recorder: Option<R>,
        mut checkpointer: C,
        com: Com,
    ) -> Result<Self, String> {
        if com.is_worker() {
            Err("Cannot load experiment state on a worker process.".to_string())
        } else {
            checkpointer.set_pfs(true);

            if let Some(rec) = recorder.as_mut() {
                rec.before_load(&problem, &()).expect(
                    " 
                        Recorder failed to prepare for loading the experiment state. 
                        Please ensure that the recorder is properly configured and that the necessary files are accessible.
                    "
                );
            }
            checkpointer.before_load(&problem, &()).unwrap();

            let sampler: O = checkpointer
                .load(&CheckpointerTargets::Optimizer, &())
                .unwrap();
            let stop = checkpointer.load(&CheckpointerTargets::Stop, &()).unwrap();
            let accumulator = checkpointer
                .load(&CheckpointerTargets::Accumulator, &())
                .unwrap();
            let parameters: GlobalParameters = checkpointer
                .load(&CheckpointerTargets::Config, &())
                .unwrap();
            // if the parameters.id is greater than the current SOL_ID,
            // we replace SOl_ID by the parameters.id
            if parameters.sold_id > SOL_ID.load(Ordering::SeqCst) {
                SOL_ID.store(parameters.sold_id, Ordering::SeqCst);
            }
            RUN_ID.store(parameters.run_id + 1, Ordering::SeqCst);

            let checkpointer = Arc::new(Mutex::new(Some(checkpointer)));
            Ok(Master {
                problem: Arc::new(problem),
                sampler,
                stop,
                recorder,
                checkpointer,
                accumulator,
                com,
                pfs: true,
                is_loaded: true,
                pool_mode: PoolMode::InMemory, // Default pool mode when loading, can be changed later
            })
        }
    }

    /// Set the Parallel File System (PFS) flag for the master process.
    pub fn with_pfs(self, pfs: bool) -> Self {
        if let Some(c) = self.checkpointer.lock().unwrap().as_mut() {
            c.set_pfs(pfs)
        }
        Self { pfs, ..self }
    }

    pub fn with_pool_mode(self, pool_mode: PoolMode) -> Self {
        Self { pool_mode, ..self }
    }
}

impl<O, S, R, C, Scp, Out, Info, Com> Runable for Master<SimplePb<Info, Scp, Out>, O, S, R, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
    O: Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S: Stop,
    R: Recorder<SimplePb<Info, Scp, Out>, ()>,
    C: Checkpointer<SimplePb<Info, Scp, Out>, ()> + Send + Sync,
    Scp: Searchspace<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>> + HasVariables,
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
    Com: MasterWorkerCom,
{
    type Id = SId;
    type Context = ();

    type Problem = SimplePb<Info, Scp, Out>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;

    fn run(mut self) -> Accum<Self::Problem> {
        let state: ParState<SimplePb<Info, Scp, Out>> = if self.is_loaded {
            if let Some(rec) = &mut self.recorder {
                rec.after_load(&self.problem, &()).unwrap();
            }

            if let Some(check) = self.checkpointer.lock().unwrap().as_mut() {
                let state = check.load(&CheckpointerTargets::Experiment, &());
                check.after_load(&self.problem, &()).unwrap();
                match state {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("Failed to load experiment state: {}.", e);
                        ParState::new(self.com.size())
                    }
                }
            } else {
                ParState::new(self.com.size())
            }
        } else {
            if let Some(rec) = &mut self.recorder {
                rec.init(&self.problem, &()).unwrap();
            }
            if let Some(check) = self.checkpointer.lock().unwrap().as_mut() {
                check.init(&self.problem, &()).unwrap();
            }
            ParState::new(self.com.size())
        };

        let problem = self.problem;
        let accumulator = self.accumulator;
        let sampler = self.sampler;
        let stop = self.stop;
        let recorder = self.recorder;
        let checkpointer = self.checkpointer;

        self.com.master::<SimplePb<Info, Scp, Out>, _>(move |com| {
            simple_master(
                com,
                problem,
                accumulator,
                sampler,
                stop,
                recorder,
                checkpointer,
                state,
            )
        })
    }
}

impl<O, S, R, C, Scp, Out, Info, Com, FnState> Runable
    for Master<FidPb<Info, Scp, Out, FnState>, O, S, R, C, Com>
where
    FidPb<Info, Scp, Out, FnState>: 'static,
    O: Sampler<FidPb<Info, Scp, Out, FnState>, Scp::Opt, Info = Info>,
    S: Stop,
    R: Recorder<FidPb<Info, Scp, Out, FnState>, ()>,
    C: Checkpointer<FidPb<Info, Scp, Out, FnState>, ()> + Send + Sync,
    Scp: Searchspace<MultiStep<FidelitySol<StepSId, LinkObj<Scp>, Info>, Out, FnState>>
        + HasVariables,
    Scp::Shape: SolutionShape<
            Id = StepSId,
            Info = Info,
            ObjSol = FidelitySol<StepSId, LinkObj<Scp>, Info>,
            OptSol = FidelitySol<StepSId, LinkOpt<Scp>, Info>,
            Obj = LinkObj<Scp>,
            Opt = LinkOpt<Scp>,
        > + HasFidelity
        + HasStep
        + HasStepId,
    CompShape<FidPb<Info, Scp, Out, FnState>>: HasFidelity + HasStep + HasStepId,
    Out: FidOutcome,
    Info: SolInfo,
    FnState: FuncState + Send + Sync,
    Com: MasterWorkerCom,
{
    type Id = StepSId;
    type Context = ();

    type Problem = FidPb<Info, Scp, Out, FnState>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;

    fn run(mut self) -> Accum<Self::Problem> {
        let state: ParState<FidPb<Info, Scp, Out, FnState>> = if self.is_loaded {
            if let Some(rec) = &mut self.recorder {
                rec.after_load(&self.problem, &()).unwrap();
            }

            if let Some(check) = self.checkpointer.lock().unwrap().as_mut() {
                let state = check.load(&CheckpointerTargets::Experiment, &());
                check.after_load(&self.problem, &()).unwrap();
                match state {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("Failed to load experiment state: {}.", e);
                        ParState::new(self.com.size())
                    }
                }
            } else {
                ParState::new(self.com.size())
            }
        } else {
            if let Some(rec) = &mut self.recorder {
                rec.init(&self.problem, &()).unwrap();
            }
            if let Some(check) = self.checkpointer.lock().unwrap().as_mut() {
                check.init(&self.problem, &()).unwrap();
            }
            ParState::new(self.com.size())
        };

        let problem = self.problem;
        let accumulator = self.accumulator;
        let sampler = self.sampler;
        let stop = self.stop;
        let recorder = self.recorder;
        let checkpointer = self.checkpointer;

        self.com
            .master::<FidPb<Info, Scp, Out, FnState>, _>(move |com| {
                fidelity_master(
                    com,
                    problem,
                    accumulator,
                    sampler,
                    stop,
                    recorder,
                    checkpointer,
                    state,
                )
            })
    }
}

enum Role<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    Master(Master<P, O, S, R, C, Com>),
    Worker(Worker<P, C, Com>),
}

/// A parallel experiment, with the same user experience as [`Experiment`].
///
/// It hides the [`Worker`] / [`Master`] split and guarantees that the workers are started
/// before the master routine runs, whatever the [`MasterWorkerCom`] backend:
/// - threaded: the worker threads are spawned, the master routine runs in the calling thread,
///   then the threads are joined.
/// - MPI: the same code is executed by every rank. Worker ranks run the worker routine
///   and never return, the master rank runs the master routine.
///
/// ```ignore
/// let exp = ParExperiment::new(pb, opt, stop, rec, check, ThreadCommunicator::new());
/// let accumulator = exp.run(); // `Some(_)` on the master, `None` on MPI worker ranks.
///
/// let exp = ParExperiment::<_, RandomSearch, Evaluated, _, _, _>::load(pb, rec, check, com);
/// ```
pub struct ParExperiment<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    role: Role<P, O, S, R, C, Com>,
}

impl<P, O, S, R, C, Com> ParExperiment<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    /// Create a new parallel experiment.
    pub fn new(
        problem: P,
        sampler: O,
        stop: S,
        recorder: Option<R>,
        checkpointer: Option<C>,
        com: Com,
    ) -> Self {
        let role = if com.is_master() {
            Role::Master(
                Master::new(problem, sampler, stop, recorder, checkpointer, com)
                    .expect("A master process must be able to create a Master."),
            )
        } else {
            Role::Worker(
                Worker::new(problem, checkpointer, PoolMode::InMemory, com)
                    .expect("A worker process must be able to create a Worker."),
            )
        };
        Self { role }
    }

    /// Load a parallel experiment from a [`Checkpointer`].
    /// Only the master process reads the checkpoint.
    pub fn load(problem: P, recorder: Option<R>, checkpointer: C, com: Com) -> Self {
        let role = if com.is_master() {
            Role::Master(Master::load(problem, recorder, checkpointer, com).unwrap())
        } else {
            Role::Worker(
                Worker::load(problem, checkpointer, PoolMode::InMemory, com)
                    .expect("A worker process must be able to create a Worker."),
            )
        };
        Self { role }
    }

    pub fn with_pool(mut self, pool_mode: PoolMode) -> Self {
        match &mut self.role {
            Role::Master(m) => m.pool_mode = pool_mode,
            Role::Worker(w) => w.pool_mode = pool_mode,
        }
        self
    }

    /// Set the Parallel File System (PFS) flag.
    pub fn with_pfs(mut self, pfs: bool) -> Self {
        match &mut self.role {
            Role::Master(m) => m.pfs = pfs,
            Role::Worker(w) => w.pfs = pfs,
        }
        self
    }

    /// Return the [`Master`] if the current process is the master, to inspect or modify the experiment.
    pub fn master_mut(&mut self) -> Option<&mut Master<P, O, S, R, C, Com>> {
        match &mut self.role {
            Role::Master(m) => Some(m),
            Role::Worker(_) => None,
        }
    }

    /// Return the [`Master`] if the current process is the master, to inspect the experiment.
    pub fn master(&self) -> Option<&Master<P, O, S, R, C, Com>> {
        match &self.role {
            Role::Master(m) => Some(m),
            Role::Worker(_) => None,
        }
    }
}

impl<O, S, R, C, Scp, Out, Info, Com> ParExperiment<SimplePb<Info, Scp, Out>, O, S, R, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
    O: Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S: Stop,
    R: Recorder<SimplePb<Info, Scp, Out>, ()>,
    C: Checkpointer<SimplePb<Info, Scp, Out>, ()> + Send + Sync,
    Scp: Searchspace<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>> + HasVariables,
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
    Com: MasterWorkerCom,
{
    /// Run the experiment. Workers are always started before the master routine.
    ///
    /// Returns the [`Accumulator`] on the master, and `None` on worker processes.
    pub fn run(self) -> Option<Accum<SimplePb<Info, Scp, Out>>> {
        match self.role {
            Role::Master(mut master) => {
                let handle = start_simple_workers(master.problem.as_ref(), &mut master.com)
                    .map_err(|e| log::error!("Failed to start workers: {}", e))
                    .ok()?;
                let accumulator = master.run();
                if let Err(e) = handle.join() {
                    log::error!("A worker failed: {}", e);
                }
                Some(accumulator)
            }
            Role::Worker(worker) => {
                if let Err(e) = worker.launch().and_then(|h| h.join()) {
                    log::error!("A worker failed: {}", e);
                }
                None
            }
        }
    }
}

impl<O, S, R, C, Scp, Out, Info, Com, FnState>
    ParExperiment<FidPb<Info, Scp, Out, FnState>, O, S, R, C, Com>
where
    FidPb<Info, Scp, Out, FnState>: 'static,
    O: Sampler<FidPb<Info, Scp, Out, FnState>, Scp::Opt, Info = Info>,
    S: Stop,
    R: Recorder<FidPb<Info, Scp, Out, FnState>, ()>,
    C: Checkpointer<FidPb<Info, Scp, Out, FnState>, ()> + Send + Sync + 'static,
    Scp: Searchspace<MultiStep<FidelitySol<StepSId, LinkObj<Scp>, Info>, Out, FnState>>
        + HasVariables,
    Scp::Shape: SolutionShape<
            Id = StepSId,
            Info = Info,
            ObjSol = FidelitySol<StepSId, LinkObj<Scp>, Info>,
            OptSol = FidelitySol<StepSId, LinkOpt<Scp>, Info>,
            Obj = LinkObj<Scp>,
            Opt = LinkOpt<Scp>,
        > + HasFidelity
        + HasStep
        + HasStepId,
    CompShape<FidPb<Info, Scp, Out, FnState>>: HasFidelity + HasStep + HasStepId,
    Out: FidOutcome,
    Info: SolInfo,
    FnState: FuncState + Send + Sync,
    Com: MasterWorkerCom,
{
    /// Run the experiment. Workers are always started before the master routine.
    ///
    /// Returns the [`Accumulator`] on the master, and `None` on worker processes.
    pub fn run(self) -> Option<Accum<FidPb<Info, Scp, Out, FnState>>> {
        match self.role {
            Role::Master(mut master) => {
                let pool = Arc::new(Mutex::new(Pool::<StepSId, FnState>::new(master.pool_mode)));
                let handle = start_fidelity_workers(
                    master.problem.clone(),
                    &mut master.com,
                    master.checkpointer.clone(),
                    pool,
                    master.is_loaded,
                )
                .map_err(|e| log::error!("Failed to start workers: {}", e))
                .ok()?;
                let accumulator = master.run();
                if let Err(e) = handle.join() {
                    log::error!("A worker failed: {}", e);
                }
                Some(accumulator)
            }
            Role::Worker(worker) => {
                if let Err(e) = worker.launch().and_then(|h| h.join()) {
                    log::error!("A worker failed: {}", e);
                }
                None
            }
        }
    }
}
