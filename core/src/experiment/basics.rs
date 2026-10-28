use std::{marker::PhantomData, process::exit, sync::{Arc, Mutex, atomic::Ordering}};

use crate::{
    Accumulator, BaseSol, Checkpointer, Codomain, Computed, FidOutcome, FidPb, FidelitySol, FuncState, FuncWrapper, GlobalParameters, HasFidelity, HasId, HasState, HasStep, HasStepId, HasVariables, HasX, MultiStep, OneStep, Outcome, Problem, RUN_ID, Runable, SId, SOL_ID, Sampler, Searchspace, SimplePb, SolId, SolInfo, SolutionShape, State, Step, StepId, StepSId, Stop, Var, aliases::{Accum, Cod, CompShape, Elem, LinkObj, LinkOpt, Obj, Opt, Out, SObj, SOpt, Shape, TrueOpt, XObj}, checkpointer::CheckpointerTargets, errors::WorkerError, experiment::{PoolMode, parallel::{MasterWorker, threaded::Envelope}}, objective::outcome::IdOut, recorder::{Recorder, RecorderTargets}, solution::shape::IntoComputedShape, stop::ExpStep, utils::{Communicator, communicator::{AskMsg, Channel, Envelope, MessageWithTag, Msg, OutMsg, StopMsg, XMsg}, record::{CompParts, Recordable, Schema, SolParts}}
};

use indexmap::IndexMap;
use log::warn;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Shape<P>: SolutionShape, Out<P>: Outcome")]
pub struct ExpState<P:Problem> {
    /// The current solution being evaluated.
    pub currently: Shape<P>,
    /// The current output of the optimization process, if any.
    pub output: Option<Out<P>>,
}

impl<P:Problem> State for ExpState<P> { }

impl<P:Problem> ExpState<P> {
    pub fn new(currently: Shape<P>) -> Self {
        Self { currently, output: None }
    }

    pub fn set_output(&mut self, output: Out<P>) {
        self.output = Some(output);
    }

    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    pub fn to_computed(self, codomain: &Cod<P>) -> (CompShape<P>, Out<P>) {
        let output = self.output.expect("No output available to convert to computed shape.");
        (self.currently.into_computed(codomain.get_elem(&output).into()), output)
    }
}

impl<P:Problem> HasState for ExpState<P> {
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

pub struct Experiment<P, O, S, R, C>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
{
    /// The [`Problem`] being solved in the experiment.
    pub problem: P,
    /// The [`Sampler`] used to generate new candidate solutions.sed to generate new candidate solutions.
    pub sampler: O,
    /// The [stopping](Stop) criterion for the experiment.
    pub stop: S,
    /// The [`Recorder`] used to log experiment data.
    pub recorder: Option<R>,
    /// The [`Checkpointer`] used to save and load experiment state.
    pub checkpointer: Option<C>,
    /// The [`Accumulator`] that collects best evaluated solutions and their outcomes.
    pub accumulator: Accum<P>,
    /// The [`PoolMode`] used for managing function states during the experiment.
    pub pool_mode: PoolMode,
    /// The current state of the experiment.
    state: ExpState<P>,
}

impl<P, O, S, R, C> Experiment<P, O, S, R, C>
where
    P:Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S:Stop,
    R:Recorder<P, ()>,
    C:Checkpointer<P, ()>,
{

    pub fn new(
        problem: P,
        mut sampler: O,
        stop: S,
        mut recorder: Option<R>,
        mut checkpointer: Option<C>,
    ) -> Self 
    {
        let accumulator = Accum::<P>::default();
        let sol = sampler.sample(problem.get_searchspace(), &accumulator);
        if sol.is_none() {
            log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate.");
            exit(1);
        }
        let state = ExpState::new(sol.unwrap());

        if let Some(rec) = recorder.as_mut(){
            rec.init(&problem, &()).unwrap();
        }

        if let Some(check) = checkpointer.as_mut() {
            check.init(&problem, &()).unwrap();
        }

        Self {
            problem,
            sampler,
            stop,
            recorder,
            checkpointer,
            accumulator,
            pool_mode: PoolMode::InMemory,
            state,
        }
    }

    pub fn with_pool(self, pool: PoolMode) -> Self {
        Self {
            pool_mode: pool,
            ..self
        }
    }

    pub fn load(problem: P, mut recorder: Option<R>, mut checkpointer: C) -> Self
    {
        if let Some(rec) = recorder.as_mut() {
            rec.before_load(&problem, &()).expect(
                " 
                    Recorder failed to prepare for loading the experiment state. 
                    Please ensure that the recorder is properly configured and that the necessary files are accessible.
                "
            );
        }
        checkpointer.before_load(&problem, &()).unwrap();

        let mut sampler: O = checkpointer.load(&CheckpointerTargets::Optimizer, &()).unwrap();
        let stop = checkpointer.load(&CheckpointerTargets::Stop, &()).unwrap();
        let accumulator = checkpointer.load(&CheckpointerTargets::Accumulator, &()).unwrap();
        let parameters: GlobalParameters = checkpointer.load(&CheckpointerTargets::Config, &()).unwrap();
        // if the parameters.id is greater than the current SOL_ID, 
        // we replace SOl_ID by the parameters.id 
        if parameters.sold_id > SOL_ID.load(Ordering::SeqCst) {
            SOL_ID.store(parameters.sold_id, Ordering::SeqCst);
        }
        RUN_ID.store(parameters.run_id + 1, Ordering::SeqCst);
        let exp = checkpointer.load(&CheckpointerTargets::Experiment, &());

        if let Some(rec) = recorder.as_mut() {
            rec.after_load(&problem, &()).unwrap();
        }
        checkpointer.after_load(&problem, &()).unwrap();


        let state = match exp {
            Ok(e) => e,
            Err(e) => {
                log::warn!("
                    Failed to load experiment state: {}. Because of an error, or because
                    the experiment unintentionally stopped after an evaluation and before sampling a new solution.
                    The experiment will now attempt to recover by sampling a new solution
                    "
                    , e
                );
                let sol = sampler.sample(problem.get_searchspace(), &accumulator);
                if sol.is_none() {
                    log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate.");
                    exit(1);
                }
                ExpState::new(sol.unwrap())
            },
        };

        Self {
            problem,
            sampler,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            pool_mode: PoolMode::InMemory,
            state,
        }
    }
}

fn record<R: Recorder<P,Ctx>, P: Problem, Ctx>
(
    recorder: &Option<R>,
    problem: &P,
    computed: &CompShape<P>,
    outputed: &Out<P>,
)
where
    P::Id: SolId,
    P::Info: SolInfo,
    P::Scp: HasVariables,
    P::Func: FuncWrapper<Dom = LinkObj<P::Scp>>,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts> + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>] ,&'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    if let Some(rec) = recorder {
        let codomain = problem.get_codomain();
        let vars = problem.get_searchspace().get_vars();
        let contexts = (codomain, &(vars, codomain));
        rec.record(&RecorderTargets::XObjective, computed.get_sobj(), contexts, CompParts::Sol(SolParts::X)).unwrap();
        rec.record(&RecorderTargets::XOptimizer, computed.get_sopt(), contexts, CompParts::Sol(SolParts::X)).unwrap();
        rec.record(&RecorderTargets::Y, computed.get_sobj(), contexts, CompParts::Y).unwrap();
        rec.record(&RecorderTargets::Outcome, &IdOut(computed.ref_id(),outputed), (&(),&()), ()).unwrap();
        rec.record(&RecorderTargets::Metadata, computed.get_sobj(), contexts, CompParts::Sol(SolParts::Info)).unwrap();
    }
}

impl<O, S, R, C, Scp, Out, Info> Runable for Experiment<SimplePb<Info, Scp, Out>, O, S, R, C>
where
    O:Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S:Stop,
    R:Recorder<SimplePb<Info, Scp, Out>, ()>,
    C:Checkpointer<SimplePb<Info, Scp, Out>, ()>,
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
{
    type Id  = SId;
    type Context = ();

    type Problem = SimplePb<Info, Scp, Out>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;
    
    fn run(mut self) -> Accum<Self::Problem> 
    {
        if self.state.has_output() 
        {
            // If the experiment state has an output, it means that the last evlauation was recorded
            // but not yet recorded, and accumulated. So we need to record it and update the accumulator.
            let (computed, output) = self.state.to_computed(self.problem.get_codomain());
            record(&self.recorder, &self.problem, &computed, &output);

            self.accumulator.accumulate(&computed);
            let new_sol = self.sampler.update_and_sample(&computed, self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };
            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
        }
        
        loop {
            // Check if stop
            if let Some(ref check) = self.checkpointer {
                check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.sampler, &CheckpointerTargets::Optimizer, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.stop, &CheckpointerTargets::Stop, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &()).map_err(|e| log::warn!("{}", e)).ok();
                let parameters = GlobalParameters::new();
                check.save_state(&parameters, &CheckpointerTargets::Config, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            if self.stop.stop() { break; };

            // Compute the output of the current solution
            let output = self.problem.get_objective().compute(self.state.currently.get_sobj().clone_x(), ());
            
            self.state.set_output(output);
            if let Some(ref check) = self.checkpointer {
                check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            
            let (computed, output) = self.state.to_computed(self.problem.get_codomain());

            record(&self.recorder, &self.problem, &computed, &output);
            self.accumulator.accumulate(&computed);

            if let Some(ref check) = self.checkpointer {
                check.remove(&CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            self.stop.update(ExpStep::Evaluation(Step::Evaluated));

            let new_sol = self.sampler.update_and_sample(&computed, self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };

            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
            self.stop.update(ExpStep::Iteration); // New iteration
        }

        self.accumulator
    }
    
    fn get_stop(&self) -> &Self::Stop {
        &self.stop
    }
    
    fn searchspace(&self) -> &crate::aliases::Scp<Self::Problem> {
        self.problem.get_searchspace()
    }
    
    fn codomain(&self) -> &Cod<Self::Problem> {
        self.problem.get_codomain()
    }
    
    fn objective(&self) -> &crate::aliases::Func<Self::Problem> {
        self.problem.get_objective()
    }
    
    fn sampler(&self) -> &Self::Sampler {
        &self.sampler
    }
    
    fn recorder(&self) -> Option<&Self::Recorder> {
        self.recorder.as_ref()
    }
    
    fn checkpointer(&self) -> Option<&Self::Checkpointer> {
        self.checkpointer.as_ref()
    }

    fn accumulator(&self) -> &Accum<Self::Problem> {
        &self.accumulator
    }
    
    fn mut_stop(&mut self) -> &mut Self::Stop {
        &mut self.stop
    }
    
    fn mut_searchspace(&mut self) -> &mut crate::aliases::Scp<Self::Problem> {
        self.problem.get_mut_searchspace()
    }
    
    fn mut_codomain(&mut self) -> &mut Cod<Self::Problem> {
        self.problem.get_mut_codomain()
    }
    
    fn mut_objective(&mut self) -> &mut crate::aliases::Func<Self::Problem> {
        self.problem.get_mut_objective()
    }
    
    fn mut_sampler(&mut self) -> &mut Self::Sampler {
        &mut self.sampler
    }
    
    fn mut_recorder(&mut self) -> Option<&mut Self::Recorder> {
        self.recorder.as_mut()
    }
    
    fn mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer> {
        self.checkpointer.as_mut()
    }
    
    fn mut_accumulator(&mut self) -> &mut Accum<Self::Problem> {
        &mut self.accumulator
    }
}

/// Trait defining the interface for a pool of [`FuncState`]s associated with solution identifiers.
/// This abstraction allows for different implementations of function state management, such as in-memory pools or checkpointer-backed storage.
pub trait FuncStatePool<Id: StepId, FnState: FuncState>: Default
{
    /// Insert a new [`FuncState`] into the pool, associated with a solution [`StepId`].
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx);
    /// Remove the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool;
    /// Retrieve the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    fn retrieve<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> Option<FnState>;
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
    /// Insert a new [`FuncState`] into the pool, associated with the given solution [`StepId`].
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx) {
        if let Some(c) = check {
            let _ = c.save_fn_state(&id, &state, ctx);
        }
        self.0.insert(id, state);
    }
    /// Remove the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    /// Returns `true` if the state was present and removed, `false` otherwise.
    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool {
        if let Some(c) = check {
            let _ = c.remove_all_previous_fn_states::<Id, FnState>(id, ctx);
        }
        self.0.swap_remove(id).is_some()
    }
    /// Retrieve the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    fn retrieve<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> Option<FnState> {
        let res = self.0.swap_remove(id);
        if let Some(c) = check && res.is_none() {
            let loaded = c.load_fn_state::<Id, FnState>(*id, ctx);
            loaded.map_or(None, |a| Some(a.1))
        } else {
            res
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadPool<Id: StepId, FnState: FuncState>(PhantomData<(Id, FnState)>);

impl<Id: StepId, FnState: FuncState> LoadPool<Id, FnState>
{
    /// Create a new [`LoadPool`] with an optional function state checkpointer.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Id: StepId, FnState: FuncState> Default for LoadPool<Id, FnState>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Id: StepId, FnState: FuncState> FuncStatePool<Id, FnState> for LoadPool<Id, FnState>
{
    /// Insert a new [`FuncState`] into the pool, associated with the given solution [`StepId`].
    /// This implementation saves the state using the checkpointer and keeps track of the current state in memory for quick retrieval.
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx) {
        if let Some(c) = check {
            let _ = c.save_fn_state(&id, &state, ctx);
        } else {
            panic!("LoadPool requires a checkpointer to save function states.");
        }
    }

    /// Remove the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    /// This implementation removes the state from the checkpointer and clears the current state if it matches the given [`StepId`].
    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool {
        if let Some(c) = check {
            let _ = c.remove_all_previous_fn_states::<Id, FnState>(id, ctx);
            true
        } else {
            panic!("LoadPool requires a checkpointer to remove function states.");
        }
    }

    /// Retrieve the [`FuncState`] associated with the given solution [`StepId`], if it exists.
    /// This implementation first checks if the current state in memory matches the given [`StepId`] and
    /// returns it if so; otherwise, it attempts to load the state from the checkpointer.
    fn retrieve<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> Option<FnState> {
        if let Some(c) = check {
            let loaded = c.load_fn_state::<Id, FnState>(*id, ctx);
            loaded.map_or(None, |a| Some(a.1))
        } else {
            panic!("LoadPool requires a checkpointer to retrieve function states.");
        }
    }
}

pub enum Pool<Id: StepId, FnState: FuncState>
{
    InMemory(InMemoryPool<Id, FnState>),
    Persistent(LoadPool<Id, FnState>),
}


impl<Id: StepId, FnState: FuncState> Pool<Id, FnState> {
    pub fn new(mode: PoolMode) -> Self {
        match mode {
            PoolMode::InMemory => Self::InMemory(InMemoryPool::default()),
            PoolMode::Persistent => Self::Persistent(LoadPool::default()),
        }
    }
}

impl<Id: StepId, FnState: FuncState> Default for Pool<Id, FnState> {
    fn default() -> Self {
        Self::InMemory(InMemoryPool::default())
    }
}

impl<Id: StepId, FnState: FuncState> FuncStatePool<Id, FnState> for Pool<Id, FnState> {
    fn insert<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: Id, state: FnState, check: &Option<C>,ctx: &Ctx) {
        match self {
            Pool::InMemory(p) => p.insert(id, state, check, ctx),
            Pool::Persistent(p) => p.insert(id, state, check, ctx),
        }
    }

    fn remove<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> bool {
        match self {
            Pool::InMemory(p) => p.remove(id, check, ctx),
            Pool::Persistent(p) => p.remove(id, check, ctx),
        }
    }

    fn retrieve<P: Problem<Id=Id>, Ctx, C: Checkpointer<P, Ctx>>(&mut self, id: &Id, check: &Option<C>,ctx: &Ctx) -> Option<FnState> {
        match self {
            Pool::InMemory(p) => p.retrieve(id, check, ctx),
            Pool::Persistent(p) => p.retrieve(id, check, ctx),
        }
    }
}


impl<O, S, R, C, Scp, Out, Info, FnState> Runable for Experiment<FidPb<Info, Scp, Out, FnState>, O, S, R, C>
where
    O:Sampler<FidPb<Info, Scp, Out, FnState>, Scp::Opt, Info = Info>,
    S:Stop,
    R:Recorder<FidPb<Info, Scp, Out, FnState>, ()>,
    C:Checkpointer<FidPb<Info, Scp, Out, FnState>, ()>,
    Scp: Searchspace<MultiStep<FidelitySol<StepSId, LinkObj<Scp>, Info>, Out, FnState>> + HasVariables,
    Scp::Shape: SolutionShape<
        Id = StepSId,
        Info = Info,
        ObjSol = FidelitySol<StepSId, LinkObj<Scp>, Info>,
        OptSol = FidelitySol<StepSId, LinkOpt<Scp>, Info>,
        Obj = LinkObj<Scp>,
        Opt = LinkOpt<Scp>,
    > + HasFidelity + HasStep,
    CompShape<FidPb<Info, Scp, Out, FnState>>: HasStep + HasStepId,
    Out: FidOutcome,
    Info: SolInfo,
    FnState: FuncState,
{
    type Id  = StepSId;
    type Context = ();

    type Problem = FidPb<Info, Scp, Out, FnState>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;
    
    fn run(mut self) -> Accum<Self::Problem> 
    {
        if self.state.has_output() 
        {
            // If the experiment state has an output, it means that the last evlauation was recorded
            // but not yet recorded, and accumulated. So we need to record it and update the accumulator.
            let (computed, output) = self.state.to_computed(self.problem.get_codomain());
            record(&self.recorder, &self.problem, &computed, &output);

            self.accumulator.accumulate(&computed);
            let new_sol = self.sampler.update_and_sample(&computed, self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };
            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
        }

        let mut pool = match self.pool_mode {
            PoolMode::InMemory => Pool::InMemory(InMemoryPool::default()),
            PoolMode::Persistent => Pool::Persistent(LoadPool::default()),
        };
        
        loop {
            if let Some(ref check) = self.checkpointer {
                check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.sampler, &CheckpointerTargets::Optimizer, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.stop, &CheckpointerTargets::Stop, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &()).map_err(|e| log::warn!("{}", e)).ok();
                let parameters = GlobalParameters::new();
                check.save_state(&parameters, &CheckpointerTargets::Config, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            // Check if stop
            if self.stop.stop() { break; };
            
            let result = match self.state.currently.step() {
                Step::Pending => {
                    let fidelity = self.state.currently.fidelity();
                    Some(
                        self.problem.get_objective().compute(self.state.currently.get_sobj().clone_x(), (fidelity, None))
                    )
                },
                Step::Partially(_) => {
                    // Compute the output of the current solution
                    let fn_state = pool.retrieve(self.state.currently.ref_id(), &self.checkpointer, &());
                    let fidelity = self.state.currently.fidelity();
                    Some(
                        self.problem.get_objective().compute(self.state.currently.get_sobj().clone_x(), (fidelity, fn_state))
                    )
                },
                Step::Discard => {
                    pool.remove(self.state.currently.ref_id(), &self.checkpointer, &());
                    None
                },
                Step::Evaluated => {
                    warn!("Solution with ID: {:?} has already been evaluated. Skipping evaluation.", self.state.currently.ref_id());
                    pool.remove(self.state.currently.ref_id(), &self.checkpointer, &());
                    None
                },
                Step::Error => {
                    warn!("Solution with ID: {:?} encountered an error during evaluation. Skipping evaluation.",
                    self.state.currently.ref_id());
                    pool.remove(self.state.currently.ref_id(), &self.checkpointer, &());
                    None
                },
            };
            
            let computed = if let Some((output, state)) = result{
                self.state.set_output(output);        
                if let Some(ref check) = self.checkpointer {
                    check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
                }
                pool.remove(self.state.currently.ref_id(), &self.checkpointer, &());

                let (mut computed, output) = self.state.to_computed(self.problem.get_codomain());
                computed.increment();
                computed.set_raw_step(output.get_step());
    
                if matches!(computed.step(), Step::Partially(_)) {
                    pool.insert(computed.id(), state, &self.checkpointer, &());
                }
                record(&self.recorder, &self.problem, &computed, &output);
                self.accumulator.accumulate(&computed);
                self.stop.update(ExpStep::Evaluation(computed.step()));
                Some(computed)
            } else {
                None
            };

            if let Some(ref check) = self.checkpointer {
                check.remove(&CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            }

            if let Some(comp) = &computed {
                self.sampler.update(comp, self.problem.get_searchspace(), &self.accumulator);
            }
            let new_sol = self.sampler.sample(self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };

            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
            self.stop.update(ExpStep::Iteration); // New iteration
        }

        self.accumulator
    }
    
    fn get_stop(&self) -> &Self::Stop {
        &self.stop
    }
    
    fn searchspace(&self) -> &crate::aliases::Scp<Self::Problem> {
        self.problem.get_searchspace()
    }
    
    fn codomain(&self) -> &Cod<Self::Problem> {
        self.problem.get_codomain()
    }
    
    fn objective(&self) -> &crate::aliases::Func<Self::Problem> {
        self.problem.get_objective()
    }
    
    fn sampler(&self) -> &Self::Sampler {
        &self.sampler
    }
    
    fn recorder(&self) -> Option<&Self::Recorder> {
        self.recorder.as_ref()
    }
    
    fn checkpointer(&self) -> Option<&Self::Checkpointer> {
        self.checkpointer.as_ref()
    }

    fn accumulator(&self) -> &Accum<Self::Problem> {
        &self.accumulator
    }
    
    fn mut_stop(&mut self) -> &mut Self::Stop {
        &mut self.stop
    }
    
    fn mut_searchspace(&mut self) -> &mut crate::aliases::Scp<Self::Problem> {
        self.problem.get_mut_searchspace()
    }
    
    fn mut_codomain(&mut self) -> &mut Cod<Self::Problem> {
        self.problem.get_mut_codomain()
    }
    
    fn mut_objective(&mut self) -> &mut crate::aliases::Func<Self::Problem> {
        self.problem.get_mut_objective()
    }
    
    fn mut_sampler(&mut self) -> &mut Self::Sampler {
        &mut self.sampler
    }
    
    fn mut_recorder(&mut self) -> Option<&mut Self::Recorder> {
        self.recorder.as_mut()
    }
    
    fn mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer> {
        self.checkpointer.as_mut()
    }
    
    fn mut_accumulator(&mut self) -> &mut Accum<Self::Problem> {
        &mut self.accumulator
    }
}


pub struct Parallel<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: Communicator + MasterWorker
{
    /// The [`Problem`] being solved in the experiment.
    pub problem: P,
    /// The [`Sampler`] used to generate new candidate solutions.sed to generate new candidate solutions.
    pub sampler: O,
    /// The [stopping](Stop) criterion for the experiment.
    pub stop: S,
    /// The [`Recorder`] used to log experiment data.
    pub recorder: Option<R>,
    /// The [`Checkpointer`] used to save and load experiment state.
    pub checkpointer: Option<C>,
    /// The [`Accumulator`] that collects best evaluated solutions and their outcomes.
    pub accumulator: Accum<P>,
    /// The [`PoolMode`] used for managing function states during the experiment.
    pub pool_mode: PoolMode,
    /// The [`Communicator`] used for send and receiving tasks.
    pub com: Com,
    /// Flag indicating whether to consider Parallel File System (PFS).
    /// Meaning that all processes can access the same file system.
    pub pfs: bool,
    state: ExpState<P>,
}

/// A simple worker struct that holds an optional tuple of a solution shape and its corresponding output.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "Shape<P>: SolutionShape, Out<P>: Outcome")]
pub struct SimpleWorker<P:Problem>(Option<Shape<P>>, Option<Out<P>>);

impl<P:Problem> SimpleWorker<P> {
    /// Creates a new `SimpleWorker` with the given solution shape and output.
    pub fn new(sol: Shape<P>) -> Self {
        Self(Some(sol), None)
    }

    pub fn set_output(&mut self, output: Out<P>) {
        self.1 = Some(output);
    }
}

impl<P:Problem> State for SimpleWorker<P>{ }

impl<P:Problem> HasState for SimpleWorker<P>{
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

impl<P, O, S, R, C, Com> Parallel<P, O, S, R, C, Com>
where
    P: Problem + 'static,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S:Stop,
    R:Recorder<P, ()>,
    C:Checkpointer<P, ()>,
    Com: Communicator + MasterWorker,
{

    pub fn new(
        problem: P,
        mut sampler: O,
        stop: S,
        mut recorder: Option<R>,
        mut checkpointer: Option<C>,
        mut com: Com
    ) -> Self 
    {
        let accumulator = Accum::<P>::default();
        let sol = sampler.sample(problem.get_searchspace(), &accumulator);
        if sol.is_none() {
            log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate.");
            exit(1);
        }
        let state = ExpState::new(sol.unwrap());

        if let Some(rec) = recorder.as_mut(){
            rec.init(&problem, &()).unwrap();
        }

        if let Some(check) = checkpointer.as_mut() {
            check.init(&problem, &()).unwrap();
        }

        Self {
            problem,
            sampler,
            stop,
            recorder,
            checkpointer,
            accumulator,
            pool_mode: PoolMode::InMemory,
            pfs: true,
            com,
            state,
        }
    }

    pub fn with_pool(self, pool: PoolMode) -> Self {
        Self {
            pool_mode: pool,
            ..self
        }
    }

    pub fn with_pfs(self, pfs: bool) -> Self {
        Self {
            pfs,
            ..self
        }
    }

    pub fn load(problem: P, mut recorder: Option<R>, mut checkpointer: C, com: Com) -> Self
    {
        if let Some(rec) = recorder.as_mut() {
            rec.before_load(&problem, &()).expect(
                " 
                    Recorder failed to prepare for loading the experiment state. 
                    Please ensure that the recorder is properly configured and that the necessary files are accessible.
                "
            );
        }
        checkpointer.before_load(&problem, &()).unwrap();

        let mut sampler: O = checkpointer.load(&CheckpointerTargets::Optimizer, &()).unwrap();
        let stop = checkpointer.load(&CheckpointerTargets::Stop, &()).unwrap();
        let accumulator = checkpointer.load(&CheckpointerTargets::Accumulator, &()).unwrap();
        let parameters: GlobalParameters = checkpointer.load(&CheckpointerTargets::Config, &()).unwrap();
        // if the parameters.id is greater than the current SOL_ID, 
        // we replace SOl_ID by the parameters.id 
        if parameters.sold_id > SOL_ID.load(Ordering::SeqCst) {
            SOL_ID.store(parameters.sold_id, Ordering::SeqCst);
        }
        RUN_ID.store(parameters.run_id + 1, Ordering::SeqCst);
        let exp = checkpointer.load(&CheckpointerTargets::Experiment, &());

        if let Some(rec) = recorder.as_mut() {
            rec.after_load(&problem, &()).unwrap();
        }
        checkpointer.after_load(&problem, &()).unwrap();


        let state = match exp {
            Ok(e) => e,
            Err(e) => {
                log::warn!("
                    Failed to load experiment state: {}. Because of an error, or because
                    the experiment unintentionally stopped after an evaluation and before sampling a new solution.
                    The experiment will now attempt to recover by sampling a new solution
                    "
                    , e
                );
                let sol = sampler.sample(problem.get_searchspace(), &accumulator);
                if sol.is_none() {
                    log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate.");
                    exit(1);
                }
                ExpState::new(sol.unwrap())
            },
        };

        Self {
            problem,
            sampler,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            accumulator,
            com,
            pool_mode: PoolMode::InMemory,
            pfs: true,
            state,
        }
    }
}

fn simple_worker<Scp, Info, Out, C>(obj: Arc<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>>, chan: C, state : SimpleWorker<SimplePb<Info, Scp, Out>>) -> Result<(), WorkerError>
where
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
    C: Channel
{
    let me = chan.me();
    let mut stop = false;
    while !stop {
        let task = chan.receive();
        if let Ok(msg) = task {
            match msg.tag {
                StopMsg::TAG => stop = true,
                XMsg::<SimplePb<Info, Scp, Out>>::TAG => {
                    let msg = msg.content.into_any().downcast::<XMsg::<SimplePb<Info, Scp, Out>>>().unwrap();
                    let x = msg.as_ref().1;
                    let id = msg.as_ref().0;
                    let outcome = obj.compute(x, ());
                    let outmsg = OutMsg(id, outcome);
                    let envelope = Envelope::new(me, 0, outmsg);
                    chan.send(envelope).unwrap();

                },
                _ => {
                    warn!("Received unexpected message: {:?}", msg.tag);
                    let envelope = Envelope::new(me, 0, AskMsg(0));
                    chan.send(envelope).unwrap();
                },
            }
        } else {
            return Err(WorkerError(format!("Failed to receive a message: {:?}", task.err())));
        }
    }
    Ok(())
}

impl<O, S, R, C, Scp, Out, Info> Runable for Parallel<SimplePb<Info, Scp, Out>, O, S, R, C>
where
    O:Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S:Stop,
    R:Recorder<SimplePb<Info, Scp, Out>, ()>,
    C:Checkpointer<SimplePb<Info, Scp, Out>, ()>,
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
{
    type Id  = SId;
    type Context = ();

    type Problem = SimplePb<Info, Scp, Out>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;
    
    fn run(mut self) -> Accum<Self::Problem> 
    {
        if self.state.has_output() 
        {
            // If the experiment state has an output, it means that the last evlauation was recorded
            // but not yet recorded, and accumulated. So we need to record it and update the accumulator.
            let (computed, output) = self.state.to_computed(self.problem.get_codomain());
            record(&self.recorder, &self.problem, &computed, &output);

            self.accumulator.accumulate(&computed);
            let new_sol = self.sampler.update_and_sample(&computed, self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };
            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
        }
        
        loop {
            // Check if stop
            if let Some(ref check) = self.checkpointer {
                check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.sampler, &CheckpointerTargets::Optimizer, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.stop, &CheckpointerTargets::Stop, &()).map_err(|e| log::warn!("{}", e)).ok();
                check.save_state(&self.accumulator, &CheckpointerTargets::Accumulator, &()).map_err(|e| log::warn!("{}", e)).ok();
                let parameters = GlobalParameters::new();
                check.save_state(&parameters, &CheckpointerTargets::Config, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            if self.stop.stop() { break; };

            // Compute the output of the current solution
            let output = self.problem.get_objective().compute(self.state.currently.get_sobj().clone_x(), ());
            
            self.state.set_output(output);
            if let Some(ref check) = self.checkpointer {
                check.save_state(&self.state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            
            let (computed, output) = self.state.to_computed(self.problem.get_codomain());

            record(&self.recorder, &self.problem, &computed, &output);
            self.accumulator.accumulate(&computed);

            if let Some(ref check) = self.checkpointer {
                check.remove(&CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            }
            self.stop.update(ExpStep::Evaluation(Step::Evaluated));

            let new_sol = self.sampler.update_and_sample(&computed, self.problem.get_searchspace(), &self.accumulator);
            let Some(sol) = new_sol else { log::warn!("Optimizer returned None, stopping experiment. No new solution to evaluate."); return self.accumulator; };

            self.state = ExpState::new(sol);
            self.stop.update(ExpStep::Sampling);
            self.stop.update(ExpStep::Iteration); // New iteration
        }

        self.accumulator
    }
    
    fn get_stop(&self) -> &Self::Stop {
        &self.stop
    }
    
    fn searchspace(&self) -> &crate::aliases::Scp<Self::Problem> {
        self.problem.get_searchspace()
    }
    
    fn codomain(&self) -> &Cod<Self::Problem> {
        self.problem.get_codomain()
    }
    
    fn objective(&self) -> &crate::aliases::Func<Self::Problem> {
        self.problem.get_objective()
    }
    
    fn sampler(&self) -> &Self::Sampler {
        &self.sampler
    }
    
    fn recorder(&self) -> Option<&Self::Recorder> {
        self.recorder.as_ref()
    }
    
    fn checkpointer(&self) -> Option<&Self::Checkpointer> {
        self.checkpointer.as_ref()
    }

    fn accumulator(&self) -> &Accum<Self::Problem> {
        &self.accumulator
    }
    
    fn mut_stop(&mut self) -> &mut Self::Stop {
        &mut self.stop
    }
    
    fn mut_searchspace(&mut self) -> &mut crate::aliases::Scp<Self::Problem> {
        self.problem.get_mut_searchspace()
    }
    
    fn mut_codomain(&mut self) -> &mut Cod<Self::Problem> {
        self.problem.get_mut_codomain()
    }
    
    fn mut_objective(&mut self) -> &mut crate::aliases::Func<Self::Problem> {
        self.problem.get_mut_objective()
    }
    
    fn mut_sampler(&mut self) -> &mut Self::Sampler {
        &mut self.sampler
    }
    
    fn mut_recorder(&mut self) -> Option<&mut Self::Recorder> {
        self.recorder.as_mut()
    }
    
    fn mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer> {
        self.checkpointer.as_mut()
    }
    
    fn mut_accumulator(&mut self) -> &mut Accum<Self::Problem> {
        &mut self.accumulator
    }
}